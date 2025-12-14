"""
Batch evaluation script for evaluating all epoch snapshots on AFLW2000 and BIWI datasets.
Outputs MAE per epoch and generates a plot.
"""
import os
import re
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import SixDRepNet
import utils
import datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluation of 6DRepNet snapshots')
    parser.add_argument('--gpu', dest='gpu_id', default=0, type=int)
    parser.add_argument('--snapshot_dir', dest='snapshot_dir',
                        default='output/snapshots/SixDRepNet_1765605639_bs64',
                        type=str, help='Directory containing epoch snapshots')
    parser.add_argument('--aflw2000_dir', dest='aflw2000_dir',
                        default='datasets/AFLW2000', type=str)
    parser.add_argument('--aflw2000_filelist', dest='aflw2000_filelist',
                        default='datasets/AFLW2000/files.txt', type=str)
    parser.add_argument('--biwi_file', dest='biwi_file',
                        default='datasets/BIWI/BIWI_noTrack.npz', type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output/evaluation', type=str)
    return parser.parse_args()


def evaluate_model(model, test_loader, gpu):
    """Evaluate model and return MAE for yaw, pitch, roll."""
    model.eval()
    total = 0
    yaw_error = pitch_error = roll_error = 0.0

    with torch.no_grad():
        for images, r_label, cont_labels, name in test_loader:
            images = images.cuda(gpu)
            total += cont_labels.size(0)

            y_gt_deg = cont_labels[:, 0].float() * 180 / np.pi
            p_gt_deg = cont_labels[:, 1].float() * 180 / np.pi
            r_gt_deg = cont_labels[:, 2].float() * 180 / np.pi

            R_pred = model(images)
            euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            pitch_error += torch.sum(torch.min(torch.stack((
                torch.abs(p_gt_deg - p_pred_deg),
                torch.abs(p_pred_deg + 360 - p_gt_deg),
                torch.abs(p_pred_deg - 360 - p_gt_deg),
                torch.abs(p_pred_deg + 180 - p_gt_deg),
                torch.abs(p_pred_deg - 180 - p_gt_deg)
            )), 0)[0])

            yaw_error += torch.sum(torch.min(torch.stack((
                torch.abs(y_gt_deg - y_pred_deg),
                torch.abs(y_pred_deg + 360 - y_gt_deg),
                torch.abs(y_pred_deg - 360 - y_gt_deg),
                torch.abs(y_pred_deg + 180 - y_gt_deg),
                torch.abs(y_pred_deg - 180 - y_gt_deg)
            )), 0)[0])

            roll_error += torch.sum(torch.min(torch.stack((
                torch.abs(r_gt_deg - r_pred_deg),
                torch.abs(r_pred_deg + 360 - r_gt_deg),
                torch.abs(r_pred_deg - 360 - r_gt_deg),
                torch.abs(r_pred_deg + 180 - r_gt_deg),
                torch.abs(r_pred_deg - 180 - r_gt_deg)
            )), 0)[0])

    yaw_mae = (yaw_error / total).item()
    pitch_mae = (pitch_error / total).item()
    roll_mae = (roll_error / total).item()
    total_mae = (yaw_error + pitch_error + roll_error) / (total * 3)

    return {
        'yaw': yaw_mae,
        'pitch': pitch_mae,
        'roll': roll_mae,
        'mae': total_mae.item()
    }


def main():
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id

    os.makedirs(args.output_dir, exist_ok=True)

    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load AFLW2000 dataset
    print('Loading AFLW2000 dataset...')
    aflw2000_dataset = datasets.getDataset(
        'AFLW2000', args.aflw2000_dir, args.aflw2000_filelist,
        transformations, train_mode=False
    )
    aflw2000_loader = DataLoader(
        dataset=aflw2000_dataset,
        batch_size=args.batch_size,
        num_workers=4
    )

    # Load BIWI dataset
    print('Loading BIWI dataset...')
    biwi_dataset = datasets.getDataset(
        'BIWI', args.biwi_file, args.biwi_file,
        transformations, train_mode=False
    )
    biwi_loader = DataLoader(
        dataset=biwi_dataset,
        batch_size=args.batch_size,
        num_workers=4
    )

    # Find all epoch snapshots
    snapshot_files = []
    for f in os.listdir(args.snapshot_dir):
        match = re.match(r'_epoch_(\d+)\.tar', f)
        if match:
            epoch = int(match.group(1))
            snapshot_files.append((epoch, os.path.join(args.snapshot_dir, f)))

    snapshot_files.sort(key=lambda x: x[0])
    print(f'Found {len(snapshot_files)} epoch snapshots')

    results = {
        'epochs': [],
        'aflw2000': {'yaw': [], 'pitch': [], 'roll': [], 'mae': []},
        'biwi': {'yaw': [], 'pitch': [], 'roll': [], 'mae': []}
    }

    for epoch, snapshot_path in snapshot_files:
        print(f'\n=== Evaluating Epoch {epoch} ===')

        # Create model (deploy=False for training checkpoints)
        model = SixDRepNet(
            backbone_name='RepVGG-B1g2',
            backbone_file='',
            deploy=False,
            pretrained=False
        )

        # Load snapshot
        saved_state_dict = torch.load(snapshot_path, map_location='cpu')
        if 'model_state_dict' in saved_state_dict:
            model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            model.load_state_dict(saved_state_dict)
        model.cuda(gpu)

        # Evaluate on AFLW2000
        print(f'  Evaluating on AFLW2000...')
        aflw2000_results = evaluate_model(model, aflw2000_loader, gpu)
        print(f'  AFLW2000 - Yaw: {aflw2000_results["yaw"]:.4f}, '
              f'Pitch: {aflw2000_results["pitch"]:.4f}, '
              f'Roll: {aflw2000_results["roll"]:.4f}, '
              f'MAE: {aflw2000_results["mae"]:.4f}')

        # Evaluate on BIWI
        print(f'  Evaluating on BIWI...')
        biwi_results = evaluate_model(model, biwi_loader, gpu)
        print(f'  BIWI - Yaw: {biwi_results["yaw"]:.4f}, '
              f'Pitch: {biwi_results["pitch"]:.4f}, '
              f'Roll: {biwi_results["roll"]:.4f}, '
              f'MAE: {biwi_results["mae"]:.4f}')

        # Store results
        results['epochs'].append(epoch)
        for key in ['yaw', 'pitch', 'roll', 'mae']:
            results['aflw2000'][key].append(aflw2000_results[key])
            results['biwi'][key].append(biwi_results[key])

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Save results to JSON
    json_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {json_path}')

    # Create plots
    create_plots(results, args.output_dir)


def create_plots(results, output_dir):
    """Create MAE plots for both datasets."""
    epochs = results['epochs']

    # Plot 1: Overall MAE comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, results['aflw2000']['mae'], 'b-o', label='AFLW2000', markersize=4)
    ax.plot(epochs, results['biwi']['mae'], 'r-s', label='BIWI', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MAE (degrees)', fontsize=12)
    ax.set_title('Mean Absolute Error per Epoch', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_comparison.png'), dpi=150)
    plt.close()
    print(f'Saved: mae_comparison.png')

    # Plot 2: Detailed MAE for AFLW2000
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, results['aflw2000']['yaw'], 'b-o', label='Yaw', markersize=4)
    ax.plot(epochs, results['aflw2000']['pitch'], 'g-s', label='Pitch', markersize=4)
    ax.plot(epochs, results['aflw2000']['roll'], 'r-^', label='Roll', markersize=4)
    ax.plot(epochs, results['aflw2000']['mae'], 'k-d', label='MAE', markersize=4, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Error (degrees)', fontsize=12)
    ax.set_title('AFLW2000 - Error per Epoch', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aflw2000_detailed.png'), dpi=150)
    plt.close()
    print(f'Saved: aflw2000_detailed.png')

    # Plot 3: Detailed MAE for BIWI
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, results['biwi']['yaw'], 'b-o', label='Yaw', markersize=4)
    ax.plot(epochs, results['biwi']['pitch'], 'g-s', label='Pitch', markersize=4)
    ax.plot(epochs, results['biwi']['roll'], 'r-^', label='Roll', markersize=4)
    ax.plot(epochs, results['biwi']['mae'], 'k-d', label='MAE', markersize=4, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Error (degrees)', fontsize=12)
    ax.set_title('BIWI - Error per Epoch', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'biwi_detailed.png'), dpi=150)
    plt.close()
    print(f'Saved: biwi_detailed.png')

    # Plot 4: Combined subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AFLW2000
    axes[0].plot(epochs, results['aflw2000']['yaw'], 'b-o', label='Yaw', markersize=3)
    axes[0].plot(epochs, results['aflw2000']['pitch'], 'g-s', label='Pitch', markersize=3)
    axes[0].plot(epochs, results['aflw2000']['roll'], 'r-^', label='Roll', markersize=3)
    axes[0].plot(epochs, results['aflw2000']['mae'], 'k-d', label='MAE', markersize=4, linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Error (degrees)', fontsize=11)
    axes[0].set_title('AFLW2000', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # BIWI
    axes[1].plot(epochs, results['biwi']['yaw'], 'b-o', label='Yaw', markersize=3)
    axes[1].plot(epochs, results['biwi']['pitch'], 'g-s', label='Pitch', markersize=3)
    axes[1].plot(epochs, results['biwi']['roll'], 'r-^', label='Roll', markersize=3)
    axes[1].plot(epochs, results['biwi']['mae'], 'k-d', label='MAE', markersize=4, linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Error (degrees)', fontsize=11)
    axes[1].set_title('BIWI', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('6DRepNet Training Evaluation', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_evaluation.png'), dpi=150)
    plt.close()
    print(f'Saved: combined_evaluation.png')


if __name__ == '__main__':
    main()
