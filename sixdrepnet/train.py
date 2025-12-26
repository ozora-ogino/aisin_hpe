import time
import math
import re
import sys
import os
import argparse

import numpy as np
from numpy.lib.function_base import _quantile_unchecked
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils import model_zoo
import torchvision
from torchvision import transforms
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
matplotlib.use('Agg')

from model import SixDRepNet, SixDRepNet2
import datasets
import utils
from loss import GeodesicLoss


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=30, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=64, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.0001, type=float)
    parser.add_argument('--scheduler', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        default='Pose_300W_LP', type=str) #Pose_300W_LP
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/300W_LP', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/300W_LP/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--log_dir', dest='log_dir',
        help='TensorBoard log directory.',
        default='output/logs', type=str)
    # Validation dataset arguments
    parser.add_argument(
        '--val_dataset', dest='val_dataset',
        help='Validation dataset type (AFLW2000, BIWI, etc.). Leave empty to skip validation.',
        default='', type=str)
    parser.add_argument(
        '--val_data_dir', dest='val_data_dir',
        help='Directory path for validation data.',
        default='', type=str)
    parser.add_argument(
        '--val_filename_list', dest='val_filename_list',
        help='Path to validation file list.',
        default='', type=str)

    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def evaluate(model, val_loader, gpu):
    """Evaluate model and return MAE for yaw, pitch, roll."""
    model.eval()
    total = 0
    yaw_error = pitch_error = roll_error = 0.0

    with torch.no_grad():
        for images, r_label, cont_labels, name in val_loader:
            images = images.cuda(gpu)
            total += cont_labels.size(0)

            # Ground truth in radians, convert to degrees
            y_gt_deg = cont_labels[:, 0].float() * 180 / np.pi
            p_gt_deg = cont_labels[:, 1].float() * 180 / np.pi
            r_gt_deg = cont_labels[:, 2].float() * 180 / np.pi

            R_pred = model(images)
            euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            # Calculate errors with wrap-around handling
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

    model.train()

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


if __name__ == '__main__':

    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    b_scheduler = args.scheduler

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    summary_name = '{}_{}_bs{}'.format(
        'SixDRepNet', int(time.time()), args.batch_size)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name))

    # TensorBoard init
    log_dir = os.path.join(args.log_dir, summary_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'TensorBoard logs: {log_dir}')

    # Log hyperparameters
    hparams = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "filename_list": args.filename_list,
        "batch_size": batch_size,
        "lr": args.lr,
        "num_epochs": num_epochs,
        "scheduler": b_scheduler,
        "backbone": "RepVGG-B1g2",
    }
    writer.add_text('hparams', str(hparams), 0)

    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='RepVGG-B1g2-train.pth',
                        deploy=False,
                        pretrained=True)

    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict['model_state_dict'])

    print('Loading data.')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.RandomResizedCrop(size=224,scale=(0.8,1)),
                                          transforms.ToTensor(),
                                          normalize])

    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations)

    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    # Validation data loader (optional)
    val_loader = None
    if args.val_dataset:
        print('Loading validation data.')
        val_transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        val_dataset = datasets.getDataset(
            args.val_dataset, args.val_data_dir, args.val_filename_list,
            val_transformations, train_mode=False)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4)
        print(f'Validation dataset: {args.val_dataset}, {len(val_dataset)} samples')

    model.cuda(gpu)
    crit = GeodesicLoss().cuda(gpu) #torch.nn.MSELoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)


    #milestones = np.arange(num_epochs)
    milestones = [10, 20]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)

    print('Starting training.')
    global_step = 0
    best_loss = float('inf')
    best_epoch = -1
    best_path = os.path.join('output', 'snapshots', summary_name, f'{args.output_string}_best_model.tar')

    for epoch in range(num_epochs):
        loss_sum = .0
        iter = 0
        for i, (images, gt_mat, _, _) in enumerate(train_loader):
            iter += 1
            global_step += 1
            images = torch.Tensor(images).cuda(gpu)

            # Forward pass
            pred_mat = model(images)

            # Calc loss
            loss = crit(gt_mat.cuda(gpu), pred_mat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            # log per-iteration loss
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: '
                      '%.6f' % (
                          epoch+1,
                          num_epochs,
                          i+1,
                          len(pose_dataset)//batch_size,
                          loss.item(),
                      )
                      )

        if b_scheduler:
            scheduler.step()

        epoch_loss = loss_sum / iter if iter > 0 else 0.0

        # epoch-level logging
        writer.add_scalar('train/epoch_loss', epoch_loss, epoch + 1)
        writer.add_scalar('train/lr_epoch_end', optimizer.param_groups[0]['lr'], epoch + 1)

        # Validation
        val_mae = None
        if val_loader is not None:
            val_results = evaluate(model, val_loader, gpu)
            val_mae = val_results['mae']
            print(f'Epoch [{epoch+1}/{num_epochs}] Validation - '
                  f'Yaw: {val_results["yaw"]:.4f}, Pitch: {val_results["pitch"]:.4f}, '
                  f'Roll: {val_results["roll"]:.4f}, MAE: {val_mae:.4f}')
            writer.add_scalar('val/yaw_mae', val_results['yaw'], epoch + 1)
            writer.add_scalar('val/pitch_mae', val_results['pitch'], epoch + 1)
            writer.add_scalar('val/roll_mae', val_results['roll'], epoch + 1)
            writer.add_scalar('val/mae', val_mae, epoch + 1)

        # Best model tracking (use val_mae if available, otherwise train loss)
        current_metric = val_mae if val_mae is not None else epoch_loss
        if current_metric < best_loss:
            best_loss = current_metric
            best_epoch = epoch + 1
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, best_path)
            metric_name = 'val MAE' if val_mae is not None else 'train loss'
            print(f'New best model at epoch {best_epoch} with {metric_name} {best_loss:.6f}. Saved to {best_path}')

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'output/snapshots/' + summary_name + '/' + args.output_string +
                '_epoch_' + str(epoch+1) + '.tar')

    metric_name = 'val MAE' if val_loader is not None else 'train loss'
    print(f'Best model was at epoch {best_epoch} with {metric_name} {best_loss:.6f}, saved to {best_path}')
    writer.close()
