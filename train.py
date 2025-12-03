import os
import time
import math
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# Adjust these imports to match your actual package structure
from models import DepthUNet, DepthUNetMonoFusion
from data import KittiDepthCropped
from losses import (
    global_l1_loss,
    lidar_anchor_loss,
    tv_smoothness,
    mae_rmse_psnr,
)

""" 
Example usages:

python train.py --data_root /path/to/val_selection_cropped --model_type lidar_mono
python train.py --model_type lidar_only
python train.py --model_type lidar_mono_rgb
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Depth Completion Training")

    # Paths
    parser.add_argument(
        "--data_root",
        type=str,
        default="val_selection_cropped",
        help="Path to KITTI val_selection_cropped root",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    

    # Model / ablations
    parser.add_argument(
        "--model_type",
        type=str,
        default="lidar_mono",
        choices=["lidar_only", "lidar_mono", "lidar_mono_rgb"],
        help="Which input combination to use",
    )
    parser.add_argument("--base_ch", type=int, default=32)

    # Loss weights
    parser.add_argument("--w_all", type=float, default=1.0)
    parser.add_argument("--w_lidar", type=float, default=0.2)
    parser.add_argument("--w_tv", type=float, default=0.001)

    # If time for quick tuning:
    # If outputs look too noisy / speckled : increase w_tv (0.02, 0.05).
    # If outputs look too smooth / washed out : decrease w_tv (0.005 or 0.001).
    # If LiDAR points are clearly being ignored (pred doesn’t match sparse at anchors) : bump w_lidar to 0.2.

    parser.add_argument(
        "--residual",
        action="store_true",
        help="Use residual learning if set, otherwise predict depth directly",
    )

    args = parser.parse_args()
    return args



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args):
    """
    Build train/val dataloaders by splitting val_selection_cropped
    into an 80/20 train/val split.
    """
    full_dataset = KittiDepthCropped(root=args.data_root)

    n = len(full_dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(0.8 * n)

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_dataset = KittiDepthCropped(root=args.data_root, split_indices=train_idx)
    val_dataset = KittiDepthCropped(root=args.data_root, split_indices=val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def build_model(args, device):
    """
    Build model with appropriate in_ch and residual flag based on the chosen ablation.
    """
    if args.model_type == "lidar_only":
        # [sparse, sparse_mask] -> DepthUNet
        in_ch = 2
        residual = False
        model = DepthUNet(in_ch=in_ch, base_ch=args.base_ch, residual=args.residual)

    elif args.model_type == "lidar_mono":
        # Use special fusion architecture for mono + LiDAR
        # Here we start with ABSOLUTE depth; can flip residual=True later for the residual variant.
        model = DepthUNetMonoFusion(base_ch=args.base_ch, residual=args.residual)

    elif args.model_type == "lidar_mono_rgb":
        # [rgb(3), mono, sparse, sparse_mask] -> 6 channels
        in_ch = 6
        residual = False
        model = DepthUNet(in_ch=in_ch, base_ch=args.base_ch, residual=args.residual)

    else:
        raise ValueError(f"Unknown model_type {args.model_type}")

    return model.to(device)



def train_one_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    device,
    args,
):
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(train_loader):
        # Move tensors to device
        rgb = batch["rgb"].to(device)           # (B,3,H,W)
        mono = batch["mono"].to(device)         # (B,1,H,W)
        sparse = batch["sparse"].to(device)     # (B,1,H,W)
        sparse_mask = batch["sparse_mask"].to(device)
        gt = batch["gt"].to(device)
        gt_mask = batch["gt_mask"].to(device)

        # Build model input based on ablation type
        if args.model_type == "lidar_only":
            x = torch.cat([sparse, sparse_mask], dim=1)
            pred = model(x)  # absolute depth prediction

        elif args.model_type == "lidar_mono":
            # DO NOT concat here; fusion model expects separate sparse / mask / mono
            pred = model(sparse, sparse_mask, mono) # predict depth directly

        elif args.model_type == "lidar_mono_rgb":
            # [rgb(3), mono, sparse, sparse_mask]
            x = torch.cat([rgb, mono, sparse, sparse_mask], dim=1)
            pred = model(x, mono=mono)
        else:
            raise ValueError(f"Unknown model_type {args.model_type}")

        # Compute losses
        loss_all = global_l1_loss(pred, gt, gt_mask)  # masked L1 vs GT
        loss_lidar = lidar_anchor_loss(pred, sparse, sparse_mask)  # anchor LiDAR points
        loss_tv = tv_smoothness(pred, focus_mask=gt_mask)  # smoothness

        loss = (
            args.w_all * loss_all +
            args.w_lidar * loss_lidar +
            args.w_tv * loss_tv
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % 20 == 0:
            print(
                f"Epoch [{epoch}] Step [{step+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"(main={loss_all.item():.4f}, lidar={loss_lidar.item():.4f}, tv={loss_tv.item():.4f})"
            )

    avg_loss = running_loss / len(train_loader)
    return avg_loss


@torch.no_grad()
def validate(epoch, model, val_loader, device, args):
    model.eval()

    total_mae = 0.0
    total_rmse = 0.0
    total_psnr = 0.0
    total_val_loss = 0.0
    count = 0

    for batch in val_loader:
        # Move tensors to device
        rgb = batch["rgb"].to(device)
        mono = batch["mono"].to(device)
        sparse = batch["sparse"].to(device)
        sparse_mask = batch["sparse_mask"].to(device)
        gt = batch["gt"].to(device)
        gt_mask = batch["gt_mask"].to(device)

        # Forward pass depending on ablation
        if args.model_type == "lidar_only":
            x = torch.cat([sparse, sparse_mask], dim=1)
            pred = model(x)

        elif args.model_type == "lidar_mono":
            pred = model(sparse, sparse_mask, mono)
        elif args.model_type == "lidar_mono_rgb":
            x = torch.cat([rgb, mono, sparse, sparse_mask], dim=1)
            pred = model(x)
        else:
            raise ValueError(f"Unknown model_type {args.model_type}")


        # ----- Compute validation LOSS (same components as training) -----
        loss_main = global_l1_loss(pred, gt, gt_mask)          # masked L1 vs GT
        loss_lidar = lidar_anchor_loss(pred, sparse, sparse_mask)  # LiDAR anchor
        loss_tv = tv_smoothness(pred, focus_mask=gt_mask)      # smoothness

        val_loss = (
            args.w_all * loss_main +
            args.w_lidar * loss_lidar +
            args.w_tv * loss_tv
        )

        # Aggregate validation loss
        total_val_loss += val_loss.item()

        # ----- Compute evaluation metrics only on valid GT pixels -----
        mae, rmse, psnr = mae_rmse_psnr(pred, gt, gt_mask, max_val=80.0)

        # Skip NaN batches (shouldn't happen, but safe)
        if not (math.isnan(mae) or math.isnan(rmse)):
            total_mae += mae
            total_rmse += rmse
            total_psnr += psnr
            count += 1

    if count == 0:
        print("Warning: no valid GT pixels in validation set?")
        return math.nan, math.nan, math.nan, math.nan

    avg_mae = total_mae / count
    avg_rmse = total_rmse / count
    avg_psnr = total_psnr / count
    avg_val_loss = total_val_loss / count

    print(
        f"[Val Epoch {epoch}] "
        f"Loss: {avg_val_loss:.4f} | "
        f"MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, PSNR: {avg_psnr:.2f} dB"
    )

    return avg_mae, avg_rmse, avg_psnr, avg_val_loss



def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Mode:
    if args.residual:
        mode = "residual"
    else:
        mode = "direct-depth"

    # Data loaders
    train_loader, val_loader = build_dataloaders(args)

    # Model
    model = build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_rmse = float("inf")
    best_ckpt_path = None

    for epoch in tqdm(range(1, args.epochs + 1)):
        t0 = time.time()
        train_loss = train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            device,
            args,
        )
        print(f"Epoch {epoch} train loss: {train_loss:.4f} (took {time.time() - t0:.1f}s)")

        # Validation
        mae, rmse, psnr, val_loss = validate(epoch, model, val_loader, device, args)
        print(f"val loss: {val_loss:.4f}, val RMSE: {rmse:.4f}")


        # Save best model by best val RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_ckpt_path = os.path.join(
                args.save_dir,
        f"best_{args.model_type}_epoch{epoch}_rmse{rmse:.3f}_{mode}.pth",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "rmse": rmse,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"  -> New best model saved to {best_ckpt_path}")

    print(f"Training complete. Best RMSE: {best_rmse:.4f} at {best_ckpt_path}")


if __name__ == "__main__":
    main()
