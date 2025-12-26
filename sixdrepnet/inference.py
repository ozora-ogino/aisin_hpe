"""
Inference and visualization script for 6DRepNet head pose estimation.

Usage:
    python sixdrepnet/inference.py \
        --snapshot output/snapshots/run_name/epoch_10.pth \
        --data_dir datasets/AFLW2000 \
        --output_dir output/inference \
        --gpu 0

    # With file list:
    python sixdrepnet/inference.py \
        --snapshot output/snapshots/run_name/epoch_10.pth \
        --data_dir datasets/AFLW2000 \
        --filename_list datasets/AFLW2000/files.txt \
        --output_dir output/inference
"""

import os
import argparse
import glob

import numpy as np
import cv2
import torch
from torch.backends import cudnn
from torchvision import transforms
from PIL import Image

from model import SixDRepNet
import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference and visualization for 6DRepNet head pose estimation.')
    parser.add_argument('--snapshot', required=True, type=str,
                        help='Path to trained model snapshot (.pth file)')
    parser.add_argument('--data_dir', required=True, type=str,
                        help='Directory containing input images')
    parser.add_argument('--filename_list', type=str, default=None,
                        help='Optional text file with image filenames (without extension)')
    parser.add_argument('--output_dir', type=str, default='output/inference',
                        help='Directory to save visualized results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id to use, set -1 for CPU')
    parser.add_argument('--img_ext', type=str, default='.jpg',
                        help='Image file extension')
    parser.add_argument('--viz_mode', type=str, default='cube', choices=['cube', 'axis'],
                        help='Visualization mode: "cube" for 3D pose cube, "axis" for axis arrows')
    parser.add_argument('--viz_size', type=int, default=100,
                        help='Size of visualization overlay')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save visualized images (display only)')
    return parser.parse_args()


def get_image_files(data_dir, filename_list=None, img_ext='.jpg'):
    """Get list of image files to process."""
    if filename_list is not None:
        with open(filename_list, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]
        image_files = [os.path.join(data_dir, fn + img_ext) for fn in filenames]
    else:
        # Find all images with the given extension
        pattern = os.path.join(data_dir, '**', f'*{img_ext}')
        image_files = glob.glob(pattern, recursive=True)
        if not image_files:
            # Try non-recursive
            pattern = os.path.join(data_dir, f'*{img_ext}')
            image_files = glob.glob(pattern)
    return sorted(image_files)


def detect_deploy_mode(state_dict):
    """Detect if the checkpoint is in deploy mode or train mode.

    Deploy mode: uses rbr_reparam layers
    Train mode: uses rbr_dense, rbr_1x1, rbr_identity layers
    """
    for key in state_dict.keys():
        if 'rbr_reparam' in key:
            return True
        if 'rbr_dense' in key or 'rbr_1x1' in key:
            return False
    return True  # Default to deploy mode


def load_model(snapshot_path, device):
    """Load trained 6DRepNet model.

    Automatically detects whether the checkpoint is in train or deploy mode.
    """
    saved_state_dict = torch.load(snapshot_path, map_location='cpu')
    if 'model_state_dict' in saved_state_dict:
        model_state = saved_state_dict['model_state_dict']
    else:
        model_state = saved_state_dict

    # Detect if checkpoint is in deploy mode or train mode
    deploy = detect_deploy_mode(model_state)
    print(f'Detected model mode: {"deploy" if deploy else "train"}')

    model = SixDRepNet(
        backbone_name='RepVGG-B1g2',
        backbone_file='',
        deploy=deploy,
        pretrained=False
    )

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def run_inference(model, image_path, transform, device):
    """Run inference on a single image."""
    img_pil = Image.open(image_path).convert('RGB')
    original_size = img_pil.size  # (width, height)

    img_tensor = transform(img_pil)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        R_pred = model(img_tensor)
        euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
        pitch = euler[0, 0].cpu().item()
        yaw = euler[0, 1].cpu().item()
        roll = euler[0, 2].cpu().item()

    return yaw, pitch, roll, original_size


def visualize_pose(image, yaw, pitch, roll, viz_mode='cube', size=100):
    """Draw pose visualization on image."""
    h, w = image.shape[:2]
    tdx, tdy = w // 2, h // 2

    if viz_mode == 'cube':
        utils.plot_pose_cube(image, yaw, pitch, roll, tdx, tdy, size=size)
    else:
        utils.draw_axis(image, yaw, pitch, roll, tdx, tdy, size=size)

    return image


def add_text_overlay(image, yaw, pitch, roll):
    """Add text showing Euler angles on image."""
    text = f'Yaw: {yaw:.1f}  Pitch: {pitch:.1f}  Roll: {roll:.1f}'
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return image


def main():
    args = parse_args()

    # Setup device
    cudnn.enabled = True
    if args.gpu < 0:
        device = torch.device('cpu')
        print('Using CPU')
    else:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'Using GPU: {args.gpu}')

    # Create output directory
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f'Output directory: {args.output_dir}')

    # Load model
    print(f'Loading model from: {args.snapshot}')
    model = load_model(args.snapshot, device)

    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get image files
    image_files = get_image_files(args.data_dir, args.filename_list, args.img_ext)
    print(f'Found {len(image_files)} images to process')

    if len(image_files) == 0:
        print('No images found. Check data_dir and img_ext arguments.')
        return

    # Process each image
    results = []
    for i, image_path in enumerate(image_files):
        if not os.path.exists(image_path):
            print(f'[{i+1}/{len(image_files)}] Skipping (not found): {image_path}')
            continue

        # Run inference
        yaw, pitch, roll, original_size = run_inference(model, image_path, transform, device)

        # Load image for visualization
        image = cv2.imread(image_path)
        if image is None:
            print(f'[{i+1}/{len(image_files)}] Failed to load: {image_path}')
            continue

        # Visualize
        viz_size = min(args.viz_size, min(image.shape[:2]) // 2)
        image = visualize_pose(image, yaw, pitch, roll, args.viz_mode, viz_size)
        image = add_text_overlay(image, yaw, pitch, roll)

        # Store results
        rel_path = os.path.relpath(image_path, args.data_dir)
        results.append({
            'filename': rel_path,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        })

        print(f'[{i+1}/{len(image_files)}] {rel_path}: Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f}')

        # Save or display
        if not args.no_save:
            output_filename = os.path.basename(image_path).replace(args.img_ext, f'_pose{args.img_ext}')
            output_path = os.path.join(args.output_dir, output_filename)
            cv2.imwrite(output_path, image)

    # Save results summary
    if not args.no_save and results:
        import json
        summary_path = os.path.join(args.output_dir, 'results.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to: {summary_path}')

    print(f'\nProcessed {len(results)} images')


if __name__ == '__main__':
    main()
