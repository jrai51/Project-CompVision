import math
import numpy as np
import torch

def global_l1_loss(pred, gt, mask=None):
    """
    Mean absolute error between final prediction and ground truth.

    Args:
        pred: (B, 1, H, W) predicted depth (meters)
        gt:   (B, 1, H, W) ground truth depth (meters)
        mask: (B, 1, H, W) optional validity mask for gt (1 = valid, 0 = invalid)

    Returns:
        Scalar tensor (L1 loss).
    """
    # Sanity: ensure shapes match
    assert pred.shape == gt.shape, f"pred {pred.shape} vs gt {gt.shape}"

    if mask is None:
        return (pred - gt).abs().mean()

    # Convert mask to boolean
    if mask.dtype.is_floating_point:
        valid = mask > 0.5
    else:
        valid = mask.bool()

    if valid.sum() == 0:
        # Return a properly zero tensor on correct device / dtype
        return (pred - pred).sum() * 0.0

    diff = (pred - gt).abs()
    return diff[valid].mean()

def lidar_anchor_loss(pred, lidar, mask):
    """
    LiDAR anchoring loss.

    Encourages the network to respect sparse LiDAR measurements by penalizing
    disagreement between prediction and LiDAR at LiDAR pixels.

    Args:
        pred:  (B, 1, H, W) predicted depth (meters)
        lidar: (B, 1, H, W) sparse LiDAR depth (meters)
        mask:  (B, 1, H, W) LiDAR validity mask (1 where LiDAR depth exists, 0 otherwise)

    Returns:
        Scalar tensor (L1 loss on LiDAR points).
    """
    assert pred.shape == lidar.shape == mask.shape, "Shape mismatch in lidar_anchor_loss"

    # Convert mask to boolean
    if mask.dtype.is_floating_point:
        valid = mask > 0.5
    else:
        valid = mask.bool()

    if valid.sum() == 0:
        # No LiDAR points in this batch
        return (pred - pred).sum() * 0.0

    diff = (pred - lidar).abs()
    return diff[valid].mean()



def tv_smoothness(pred, focus_mask=None):
    """
    Total variation smoothness on prediction.

    Encourages spatial smoothness in depth:
      - penalizes differences between neighboring pixels
      - if focus_mask is provided, only applies where mask is valid
        (and where both neighboring pixels are valid).

    Args:
        pred:       (B, 1, H, W) predicted depth
        focus_mask: (B, 1, H, W) optional mask (float 0/1 or bool)

    Returns:
        Scalar tensor (TV loss).
    """
    # Horizontal and vertical finite differences
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # (B,1,H,W-1)
    dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # (B,1,H-1,W)

    if focus_mask is not None:
        # Ensure boolean mask
        if focus_mask.dtype.is_floating_point:
            fm = focus_mask > 0.5
        else:
            fm = focus_mask.bool()

        # Neighbor-valid masks
        m_x = fm[:, :, :, 1:] & fm[:, :, :, :-1]   # (B,1,H,W-1)
        m_y = fm[:, :, 1:, :] & fm[:, :, :-1, :]   # (B,1,H-1,W)

        # If there are no valid neighbor pairs, return 0
        if (not m_x.any()) and (not m_y.any()):
            return (pred - pred).sum() * 0.0

        loss_x = dx.abs()[m_x].mean() if m_x.any() else (pred - pred).sum() * 0.0
        loss_y = dy.abs()[m_y].mean() if m_y.any() else (pred - pred).sum() * 0.0

        return 0.5 * (loss_x + loss_y)

    else:
        # Unmasked TV over full image
        return 0.5 * (dx.abs().mean() + dy.abs().mean())

import math
import torch

def mae_rmse_psnr(pred, tgt, region_mask, max_val=80.0):
    """
    Compute MAE, RMSE, PSNR over the region where region_mask is valid.

    Args:
        pred:        (B,1,H,W) predicted depth
        tgt:         (B,1,H,W) ground-truth depth
        region_mask: (B,1,H,W) mask of valid GT pixels (float 0/1 or bool)
        max_val:     max depth value for PSNR (e.g., 80m for KITTI)

    Returns:
        (mae, rmse, psnr) as Python floats
    """
    # Ensure boolean mask
    if region_mask.dtype.is_floating_point:
        m = region_mask > 0.5
    else:
        m = region_mask.bool()

    # No valid pixels -> return NaNs
    if m.sum() == 0:
        return (math.nan, math.nan, math.nan)

    # Collect errors only over valid pixels
    diff = pred - tgt  # (B,1,H,W)
    e = diff[m]        # 1D tensor of errors on valid pixels

    mae = e.abs().mean().item()
    rmse = torch.sqrt((e ** 2).mean()).item()

    if rmse == 0 or math.isnan(rmse):
        psnr = float("inf")
    else:
        psnr = 20.0 * math.log10(max_val / rmse)

    return mae, rmse, psnr


