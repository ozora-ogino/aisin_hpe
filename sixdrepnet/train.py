import time
import os
import argparse
import json
from datetime import datetime

import numpy as np
import torch
from torch.backends import cudnn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich import box

from model import SixDRepNet
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
        '--output_dir', dest='output_dir',
        help='Base output directory for checkpoints and logs.',
        default='output', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot to resume.',
        default='', type=str)
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
    parser.add_argument(
        '--seed', dest='seed',
        help='Random seed for reproducibility.',
        default=42, type=int)
    parser.add_argument(
        '--milestones', dest='milestones',
        help='Learning rate scheduler milestones (comma-separated).',
        default='10,20', type=str)
    parser.add_argument(
        '--amp', dest='amp',
        help='Use mixed precision training.',
        default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--save_interval', dest='save_interval',
        help='Save checkpoint every N epochs (0 to save every epoch).',
        default=1, type=int)

    args = parser.parse_args()
    return args

console = Console()


def format_time(seconds):
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_gpu_memory_usage(gpu_id):
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(gpu_id) / 1024**2
    return 0


def print_config(args, summary_name, model, train_size, val_size=None):
    """Print training configuration in a nice table."""
    table = Table(title="Training Configuration", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Run Name", summary_name)
    table.add_row("Model", "SixDRepNet (RepVGG-B1g2)")
    table.add_row("Dataset", args.dataset)
    table.add_row("Train Samples", f"{train_size:,}")
    if val_size:
        table.add_row("Val Samples", f"{val_size:,}")
    table.add_row("Batch Size", str(args.batch_size))
    table.add_row("Learning Rate", f"{args.lr:.6f}")
    table.add_row("Epochs", str(args.num_epochs))
    table.add_row("Scheduler", "Enabled" if args.scheduler else "Disabled")
    table.add_row("Milestones", args.milestones)
    table.add_row("Mixed Precision", "Enabled" if args.amp else "Disabled")
    table.add_row("GPU", f"cuda:{args.gpu_id}")
    table.add_row("Seed", str(args.seed))

    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    table.add_row("Total Parameters", f"{total_params:,}")
    table.add_row("Trainable Parameters", f"{trainable_params:,}")

    console.print(table)


def print_epoch_summary(epoch, num_epochs, train_loss, val_results, lr, epoch_time, best_loss, is_best):
    """Print epoch summary in a nice format."""
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Metric", style="dim")
    table.add_column("Value")

    # Epoch header with progress
    progress_pct = (epoch + 1) / num_epochs * 100
    epoch_str = f"[bold cyan]Epoch {epoch+1}/{num_epochs}[/] ({progress_pct:.0f}%)"

    table.add_row("Train Loss", f"[yellow]{train_loss:.6f}[/]")

    if val_results:
        table.add_row("Val MAE", f"[magenta]{val_results['mae']:.4f}[/]")
        table.add_row("  Yaw", f"{val_results['yaw']:.4f}")
        table.add_row("  Pitch", f"{val_results['pitch']:.4f}")
        table.add_row("  Roll", f"{val_results['roll']:.4f}")

    table.add_row("LR", f"{lr:.6f}")
    table.add_row("Time", format_time(epoch_time))

    if is_best:
        table.add_row("", "[bold green]* New Best Model[/]")

    panel = Panel(table, title=epoch_str, border_style="blue" if is_best else "dim")
    console.print(panel)


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

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    cudnn.enabled = True
    cudnn.benchmark = True  # Enable cuDNN auto-tuner for faster training

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id
    b_scheduler = args.scheduler
    use_amp = args.amp
    milestones = [int(m) for m in args.milestones.split(',')]

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler(enabled=use_amp)

    # Create unified output directory structure
    run_name = '{}_{}_bs{}'.format(
        'SixDRepNet', int(time.time()), args.batch_size)
    run_dir = os.path.join(args.output_dir, run_name)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # TensorBoard init (logs directly in run_dir)
    writer = SummaryWriter(log_dir=run_dir)

    # Save hyperparameters as config.json
    config = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "filename_list": args.filename_list,
        "val_dataset": args.val_dataset,
        "val_data_dir": args.val_data_dir,
        "batch_size": batch_size,
        "lr": args.lr,
        "num_epochs": num_epochs,
        "scheduler": b_scheduler,
        "milestones": milestones,
        "backbone": "RepVGG-B1g2",
        "seed": args.seed,
        "amp": use_amp,
        "gpu": gpu,
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    writer.add_text('config', json.dumps(config, indent=2), 0)

    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='RepVGG-B1g2-train.pth',
                        deploy=False,
                        pretrained=True)

    if args.snapshot:
        saved_state_dict = torch.load(args.snapshot, map_location=f'cuda:{gpu}', weights_only=True)
        model.load_state_dict(saved_state_dict['model_state_dict'])
        console.print(f"[green]Loaded checkpoint:[/] {args.snapshot}")

    console.print("[dim]Loading data...[/]")

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
        num_workers=4,
        pin_memory=True)

    # Validation data loader (optional)
    val_loader = None
    val_dataset = None
    if args.val_dataset:
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
            num_workers=4,
            pin_memory=True)

    model.cuda(gpu)
    crit = GeodesicLoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)

    # Print configuration
    console.print()
    print_config(args, run_name, model, len(pose_dataset),
                 len(val_dataset) if val_dataset else None)
    console.print()
    console.print(f"[dim]Output:[/] {run_dir}")
    console.print()

    console.rule("[bold blue]Training Started")
    training_start_time = time.time()
    global_step = 0
    best_loss = float('inf')
    best_epoch = -1
    best_path = os.path.join(checkpoints_dir, 'best.pt')

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        loss_sum = .0
        num_iters = 0

        # Training loop with tqdm progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        for images, gt_mat, _, _ in pbar:
            num_iters += 1
            global_step += 1
            images = images.cuda(gpu, non_blocking=True)
            gt_mat = gt_mat.cuda(gpu, non_blocking=True)

            # Forward pass with optional mixed precision
            with autocast(enabled=use_amp):
                pred_mat = model(images)
                loss = crit(gt_mat, pred_mat)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()

            # Update progress bar with current loss
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'gpu': f'{get_gpu_memory_usage(gpu):.0f}MB'
            })

            # Log per-iteration loss
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        pbar.close()

        if b_scheduler:
            scheduler.step()

        epoch_loss = loss_sum / num_iters if num_iters > 0 else 0.0
        epoch_time = time.time() - epoch_start_time

        # Epoch-level logging
        writer.add_scalar('train/epoch_loss', epoch_loss, epoch + 1)
        writer.add_scalar('train/lr_epoch_end', optimizer.param_groups[0]['lr'], epoch + 1)
        writer.add_scalar('train/epoch_time', epoch_time, epoch + 1)
        writer.add_scalar('train/gpu_memory_mb', get_gpu_memory_usage(gpu), epoch + 1)

        # Validation
        val_results = None
        if val_loader is not None:
            val_results = evaluate(model, val_loader, gpu)
            writer.add_scalar('val/yaw_mae', val_results['yaw'], epoch + 1)
            writer.add_scalar('val/pitch_mae', val_results['pitch'], epoch + 1)
            writer.add_scalar('val/roll_mae', val_results['roll'], epoch + 1)
            writer.add_scalar('val/mae', val_results['mae'], epoch + 1)

        # Best model tracking (use val_mae if available, otherwise train loss)
        current_metric = val_results['mae'] if val_results else epoch_loss
        is_best = current_metric < best_loss
        if is_best:
            best_loss = current_metric
            best_epoch = epoch + 1
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, best_path)

        # Print epoch summary
        print_epoch_summary(
            epoch, num_epochs, epoch_loss, val_results,
            optimizer.param_groups[0]['lr'], epoch_time, best_loss, is_best
        )

        # Save models at specified intervals
        save_interval = args.save_interval if args.save_interval > 0 else 1
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

    # Training complete
    total_time = time.time() - training_start_time
    console.print()
    console.rule("[bold green]Training Complete")
    console.print()

    # Final summary table
    summary_table = Table(title="Training Summary", box=box.ROUNDED)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total Time", format_time(total_time))
    summary_table.add_row("Best Epoch", str(best_epoch))
    metric_name = 'Val MAE' if val_loader is not None else 'Train Loss'
    summary_table.add_row(f"Best {metric_name}", f"{best_loss:.6f}")
    summary_table.add_row("Best Model", best_path)

    console.print(summary_table)
    writer.close()
