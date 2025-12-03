import torch
import matplotlib.pyplot as plt

from models import DepthUNet, DepthUNetMonoFusion
from data import KittiDepthCropped

def compute_metrics(pred, gt_np, mask_np, vmax=80.0):
    """
    pred: 2D numpy array or torch tensor (H, W)
    gt_np: 2D numpy array of ground-truth depth (H, W)
    mask_np: 2D boolean numpy array, True where GT is valid
    vmax: max depth range used for PSNR computation
    """
    import numpy as np
    import torch

    # convert to numpy if tensor
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    # ensure numpy arrays
    if isinstance(gt_np, torch.Tensor):
        gt_np = gt_np.detach().cpu().numpy()
    if isinstance(mask_np, torch.Tensor):
        mask_np = mask_np.detach().cpu().numpy().astype(bool)

    diff = pred[mask_np] - gt_np[mask_np]
    mse = (diff ** 2).mean()
    rmse = float(mse ** 0.5)
    psnr = float("inf") if rmse == 0 else 20 * torch.log10(torch.tensor(vmax / rmse)).item()
    return rmse, psnr

@torch.no_grad()
def visualize_depth_sample(
    model,
    dataset,
    idx=0,
    device="cuda",
    model_type="lidar_mono",  # "lidar_only" | "lidar_mono" | "lidar_mono_rgb"
    vmax=80.0,
    save_path=None,
):
    """
    Visualize a single sample (idx) from KittiDepthCropped with:
      - RGB
      - Sparse LiDAR
      - Monocular prior
      - Predicted depth
      - Ground-truth depth
      - |pred - gt| error map (masked)
    """
    model.eval()

    # Load sample & to device
    sample = dataset[idx]

    rgb = sample["rgb"].unsqueeze(0).to(device)          # (1,3,H,W)
    mono = sample["mono"].unsqueeze(0).to(device)        # (1,1,H,W)
    sparse = sample["sparse"].unsqueeze(0).to(device)    # (1,1,H,W)
    sparse_mask = sample["sparse_mask"].unsqueeze(0).to(device)
    gt = sample["gt"].unsqueeze(0).to(device)
    gt_mask = sample["gt_mask"].unsqueeze(0).to(device)

    # Forward pass 
    if model_type == "lidar_only":
        # DepthUNet: input = [sparse, sparse_mask]
        x = torch.cat([sparse, sparse_mask], dim=1)  # (B,2,H,W)
        pred = model(x)                              # residual=False ⇒ absolute depth

    elif model_type == "lidar_mono":
        # DepthUNetMonoFusion: forward(sparse, sparse_mask, mono)
        pred = model(sparse, sparse_mask, mono=mono)

    elif model_type == "lidar_mono_rgb":
        # DepthUNet: input = [rgb(3), mono(1), sparse(1), sparse_mask(1)] = 6 channels
        x = torch.cat([rgb, mono, sparse, sparse_mask], dim=1)  # (B,6,H,W)
        if getattr(model, "residual", False):
            pred = model(x, mono=mono)
        else:
            pred = model(x)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # -----------------------
    # Move to CPU and squeeze
    # -----------------------
    rgb_np = rgb[0].cpu()          # (3,H,W)
    mono_np = mono[0, 0].cpu()     # (H,W)
    sparse_np = sparse[0, 0].cpu() # (H,W)
    sparse_mask_np = sparse_mask[0, 0].cpu()
    gt_np = gt[0, 0].cpu()
    gt_mask_np = gt_mask[0, 0].cpu().bool()
    pred_np = pred[0, 0].cpu()

    # -------------------------
    # Unnormalize RGB for visualization
    # -------------------------
    if hasattr(dataset, "rgb_mean") and dataset.rgb_mean is not None:
        mean = dataset.rgb_mean.view(3, 1, 1)
        std = dataset.rgb_std.view(3, 1, 1)
        rgb_vis = rgb_np * std + mean
    else:
        rgb_vis = rgb_np

    rgb_vis = rgb_vis.clamp(0.0, 1.0).permute(1, 2, 0).numpy()  # (H,W,3)

    # -----------------------
    # Prepare depth-like maps
    # -----------------------
    sparse_vis = sparse_np.clone()
    sparse_vis[~(sparse_mask_np > 0.5)] = 0.0

    err = torch.zeros_like(gt_np)
    err[gt_mask_np] = (pred_np[gt_mask_np] - gt_np[gt_mask_np]).abs()

    mono_vis = mono_np.numpy()
    sparse_vis = sparse_vis.numpy()
    gt_vis = gt_np.numpy()
    pred_vis = pred_np.numpy()
    err_vis = err.numpy()
    gt_mask_vis = gt_mask_np.numpy()

    # mask error outside GT
    err_display = err_vis.copy()
    err_display[~gt_mask_vis] = 0.0

    # Compute metrics (masked)
    gt_np_arr = gt_np.numpy()
    mask_np_arr = gt_mask_np.numpy().astype(bool)
    pred_np_arr = pred_np.numpy()
    rmse, psnr = compute_metrics(pred_np_arr, gt_np_arr, mask_np_arr, vmax)

    # ------
    # Plot
    # ------
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: RGB, sparse LiDAR, mono
    axs[0, 0].imshow(rgb_vis)
    axs[0, 0].set_title("RGB")
    axs[0, 0].axis("off")

    im_sparse = axs[0, 1].imshow(sparse_vis, vmin=0, vmax=vmax, cmap="binary")
    axs[0, 1].set_title("Sparse LiDAR")
    axs[0, 1].axis("off")
    fig.colorbar(im_sparse, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im_mono = axs[0, 2].imshow(mono_vis, cmap="gray")
    axs[0, 2].set_title("Monocular prior (relative)")
    axs[0, 2].axis("off")
    fig.colorbar(im_mono, ax=axs[0, 2], fraction=0.046, pad=0.04)

    # Row 2: prediction (with metrics), GT, error
    im_pred = axs[1, 0].imshow(pred_vis, vmin=0, vmax=vmax, cmap="magma")
    axs[1, 0].set_title(
        f"Prediction ({model_type})\nRMSE={rmse:.3f}, PSNR={psnr:.2f} dB"
    )
    axs[1, 0].axis("off")
    fig.colorbar(im_pred, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im_gt = axs[1, 1].imshow(gt_vis, vmin=0, vmax=vmax, cmap="magma")
    axs[1, 1].set_title("Ground truth (semi-dense)")
    axs[1, 1].axis("off")
    fig.colorbar(im_gt, ax=axs[1, 1], fraction=0.046, pad=0.04)

    im_err = axs[1, 2].imshow(err_display, cmap="viridis")
    axs[1, 2].set_title("|Pred - GT| (masked)")
    axs[1, 2].axis("off")
    fig.colorbar(im_err, ax=axs[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Sample idx={idx}, model_type={model_type}", fontsize=14)
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def visualize_comparison_scene(dataset, idx=0, device="cuda", vmax=80.0):
    """
    Visualize a single scene with:
      RGB | Mono | Sparse LiDAR mask
      LiDAR-only | LiDAR+Mono Direct | LiDAR+Mono Residual

    Also prints RMSE + PSNR for each prediction.
    """

    # -------------------------
    # Load sample
    # -------------------------
    sample = dataset[idx]
    rgb         = sample["rgb"].unsqueeze(0).to(device)
    mono        = sample["mono"].unsqueeze(0).to(device)
    sparse      = sample["sparse"].unsqueeze(0).to(device)
    sparse_mask = sample["sparse_mask"].unsqueeze(0).to(device)
    gt          = sample["gt"].unsqueeze(0).to(device)
    gt_mask     = sample["gt_mask"].unsqueeze(0).to(device)

    # -------------------------
    # Unnormalize RGB
    # -------------------------
    rgb_np = rgb[0].cpu()
    if hasattr(dataset, "rgb_mean") and dataset.rgb_mean is not None:
        mean = dataset.rgb_mean.view(3, 1, 1)
        std  = dataset.rgb_std.view(3, 1, 1)
        rgb_vis = rgb_np * std + mean
    else:
        rgb_vis = rgb_np
    rgb_vis = rgb_vis.clamp(0, 1).permute(1, 2, 0).numpy()

    # -------------------------
    # Mono and binary LiDAR mask
    # -------------------------
    mono_vis = mono[0, 0].cpu().numpy()

    # 1 where LiDAR present, 0 elsewhere
    sparse_mask_vis = (sparse_mask[0, 0].cpu().numpy() > 0.5).astype(float)

    # -------------------------
    # Load models + predictions
    # -------------------------
    # LiDAR only
    lidar_only_model = DepthUNet(in_ch=2, base_ch=32, residual=False).to(device)
    ck1 = torch.load(
        "checkpoints/best_lidar_only_epoch29_rmse2.798_direct-depth.pth",
        map_location=device,
    )
    lidar_only_model.load_state_dict(ck1["model_state"])
    lidar_only_model.eval()
    pred_lidar_only = (
        lidar_only_model(torch.cat([sparse, sparse_mask], dim=1))[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    # LiDAR + Mono Direct
    direct_model = DepthUNetMonoFusion(base_ch=32, residual=False).to(device)
    ck2 = torch.load(
        "checkpoints/mono_skips/best_lidar_mono_epoch30_rmse2.431_direct-depth.pth",
        map_location=device,
    )
    direct_model.load_state_dict(ck2["model_state"])
    direct_model.eval()
    pred_direct = (
        direct_model(sparse, sparse_mask, mono=mono)[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    # LiDAR + Mono Residual
    residual_model = DepthUNetMonoFusion(base_ch=32, residual=True).to(device)
    ck3 = torch.load(
        "checkpoints/mono_skips/best_lidar_mono_epoch30_rmse2.405_residual.pth",
        map_location=device,
    )
    residual_model.load_state_dict(ck3["model_state"])
    residual_model.eval()
    pred_residual = (
        residual_model(sparse, sparse_mask, mono=mono)[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    # Metrics
    gt_np   = gt[0, 0].cpu().numpy()
    mask_np = gt_mask[0, 0].cpu().numpy().astype(bool)

    rmse_L, psnr_L = compute_metrics(pred_lidar_only, gt_np, mask_np, vmax)
    rmse_D, psnr_D = compute_metrics(pred_direct,      gt_np, mask_np, vmax)
    rmse_R, psnr_R = compute_metrics(pred_residual,    gt_np, mask_np, vmax)

    print(f"\n=== Metrics for idx={idx} ===")
    print(f"LiDAR-only        -> RMSE={rmse_L:.3f}, PSNR={psnr_L:.2f} dB")
    print(f"LiDAR+Mono Direct -> RMSE={rmse_D:.3f}, PSNR={psnr_D:.2f} dB")
    print(f"LiDAR+Mono Resid  -> RMSE={rmse_R:.3f}, PSNR={psnr_R:.2f} dB")
    print("================================\n")

    # Plot 2×3 grid tightly packed
    # Make the figure less tall so there isn't extra blank space
    fig, axs = plt.subplots(2, 3, figsize=(12, 4))

    axs[0, 0].imshow(rgb_vis);            axs[0, 0].axis("off")
    axs[0, 1].imshow(mono_vis, cmap="gray");        axs[0, 1].axis("off")
    axs[0, 2].imshow(sparse_mask_vis, cmap="gray", vmin=0, vmax=1); axs[0, 2].axis("off")

    axs[1, 0].imshow(pred_lidar_only, vmin=0, vmax=vmax, cmap="magma"); axs[1, 0].axis("off")
    axs[1, 1].imshow(pred_direct,     vmin=0, vmax=vmax, cmap="magma"); axs[1, 1].axis("off")
    axs[1, 2].imshow(pred_residual,   vmin=0, vmax=vmax, cmap="magma"); axs[1, 2].axis("off")

    # Let each axis stretch to fill its cell (prevents extra vertical white space)
    for ax in axs.ravel():
        ax.set_aspect("auto")

    # Remove all spacing and margins
    plt.subplots_adjust(
        left=0, right=1, top=1, bottom=0,
        wspace=0, hspace=0
    )

    plt.show()



if __name__ == "__main__":
    # -------------------------
    # Config: choose model + checkpoint + type
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example options:
    # ckpt_path = "checkpoints/depth-only-models/lidar-only/best_lidar_only_epoch30_rmse2.684.pth"
    # model_type = "lidar_only"

    # ckpt_path = "checkpoints/best_lidar_mono_epoch30_rmse2.583.pth"
    # model_type = "lidar_mono"

    ckpt_path = "checkpoints/mono_skips/best_lidar_mono_epoch30_rmse2.405_residual.pth"
    model_type = "lidar_mono"  # "lidar_only", "lidar_mono", or "lidar_mono_rgb"

    # -------------------------
    # Build model (must match training)
    # -------------------------
    if model_type == "lidar_only":
        # Input: [sparse, sparse_mask]
        model = DepthUNet(in_ch=2, base_ch=32, residual=False)

    elif model_type == "lidar_mono":
        # Use DepthUNetMonoFusion for lidar+mono
        model = DepthUNetMonoFusion(base_ch=32, residual=True)

    elif model_type == "lidar_mono_rgb":
        # Input: [rgb(3), mono(1), sparse(1), sparse_mask(1)] -> 6 channels
        # Choose residual according to how you trained
        model = DepthUNet(in_ch=6, base_ch=32, residual=False)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # -------------------------
    # Load dataset (same root as training)
    # -------------------------
    dataset = KittiDepthCropped(root="val_selection_cropped")

    # -------------------------
    # Visualize a few indices
    # -------------------------
    for idx in [181, 195, 300]:
        # visualize_depth_sample(
        #     model,
        #     dataset,
        #     idx=idx,
        #     device=device,
        #     model_type=model_type,
        #     vmax=80.0,
        #     save_path=None,  # or "viz_idx5.png"
        # )

        visualize_comparison_scene(dataset=dataset, idx=idx, device=device)
