# set up data loaders

import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class KittiDepthCropped(Dataset):
    def __init__(self, root, split_indices=None, rgb_mean=None, rgb_std=None):
        """
        root: path to val_selection_cropped
        split_indices: optional list of indices (for manual train/val split)
        rgb_mean/std: optional normalization constants (e.g. ImageNet)
        """
        self.root = root
        self.rgb_dir = os.path.join(root, "image")
        self.sparse_dir = os.path.join(root, "velodyne_raw")
        self.gt_dir = os.path.join(root, "groundtruth_depth")
        self.mono_dir = os.path.join(root, "depth_estimates")

        # list rgb files as canonical index
        rgb_paths = sorted(glob.glob(os.path.join(self.rgb_dir, "*.png")))
        if split_indices is not None:
            self.rgb_paths = [rgb_paths[i] for i in split_indices]
        else:
            self.rgb_paths = rgb_paths

        # store RGB normalization params as tensors
        if rgb_mean is not None:
            self.rgb_mean = torch.as_tensor(rgb_mean, dtype=torch.float32)
        else:
            self.rgb_mean = None

        if rgb_std is not None:
            self.rgb_std = torch.as_tensor(rgb_std, dtype=torch.float32)
        else:
            self.rgb_std = None

    def __len__(self):
        return len(self.rgb_paths)

    def _load_depth_png(self, path):
        """
        Load 16-bit KITTI depth PNG and convert to float32 in meters.
        KITTI encodes depth as uint16 with a scale of 1/256.
        """
        depth_png = np.array(Image.open(path), dtype=np.uint16)
        depth = depth_png.astype(np.float32) / 256.0  # meters
        return depth

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        basename = os.path.basename(rgb_path)  # e.g. 2011_09_26_drive_0002_sync_image_0000000005_image_02.png

        # Derive corresponding filenames:
        # Replace the first 'image_' with 'velodyne_raw_' / 'groundtruth_depth_'
        sparse_basename = basename.replace("image_", "velodyne_raw_", 1)
        gt_basename = basename.replace("image_", "groundtruth_depth_", 1)
        mono_basename = basename.replace(".png", "_depth_estimate.npy")

        sparse_path = os.path.join(self.sparse_dir, sparse_basename)
        gt_path = os.path.join(self.gt_dir, gt_basename)
        mono_path = os.path.join(self.mono_dir, mono_basename)

        # --- load modalities ---

        # RGB: [0,1] then optional mean/std normalization
        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0  # (H, W, 3)

        # Sparse LiDAR depth (meters, NOT normalized)
        sparse = self._load_depth_png(sparse_path)  # (H, W)

        # Ground-truth depth (meters, NOT normalized)
        gt = self._load_depth_png(gt_path)  # (H, W)

        # Monocular depth estimate (relative depth, already normalized per image to [0,1])
        mono = np.load(mono_path).astype(np.float32)  # (H, W)

        # masks
        sparse_mask = (sparse > 0).astype(np.float32)
        gt_mask = (gt > 0).astype(np.float32)

        # to tensors, CHW
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)          # (3, H, W)
        sparse = torch.from_numpy(sparse).unsqueeze(0)        # (1, H, W)
        sparse_mask = torch.from_numpy(sparse_mask).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)
        gt_mask = torch.from_numpy(gt_mask).unsqueeze(0)
        mono = torch.from_numpy(mono).unsqueeze(0)            # (1, H, W)

        # optional RGB normalization (e.g., ImageNet stats)
        if self.rgb_mean is not None and self.rgb_std is not None:
            # reshape to (3, 1, 1) for broadcasting
            mean = self.rgb_mean.view(3, 1, 1)
            std = self.rgb_std.view(3, 1, 1)
            rgb = (rgb - mean) / std

        return {
            "rgb": rgb,                     # (3, H, W), normalized
            "sparse": sparse,               # (1, H, W), meters, NOT normalized
            "sparse_mask": sparse_mask,     # (1, H, W), 0/1
            "mono": mono,                   # (1, H, W), relative monocular depth estimate, normalized per-image 
            "gt": gt,                       # (1, H, W), meters
            "gt_mask": gt_mask,             # (1, H, W), 0/1
            "id": basename,
        }

        # return {
        #     "rgb": rgb,                     # rgb image 
        #     "sparse": sparse,               # velodyne_raw lidar PNG 
        #     "sparse_mask": sparse_mask,     # mask indicating where LiDAR depth exists
        #     "mono": mono,                   # monocular depth estimate
        #     "gt": gt,                       # ground truth depth map 
        #     "gt_mask": gt_mask,             # mask indicating where ground truth depth exists
        #     "id": basename,                 # 
        # }


import numpy as np
from torch.utils.data import DataLoader

root = "val_selection_cropped"

# First, create a "base" dataset to know how many samples there are
base_dataset = KittiDepthCropped(root)

num_samples = len(base_dataset)
indices = np.arange(num_samples)
np.random.seed(42)
np.random.shuffle(indices)

train_size = int(0.8 * num_samples)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# ImageNet RGB stats (if you want them)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_dataset = KittiDepthCropped(
    root,
    split_indices=train_indices,
    rgb_mean=imagenet_mean,
    rgb_std=imagenet_std,
)

val_dataset = KittiDepthCropped(
    root,
    split_indices=val_indices,
    rgb_mean=imagenet_mean,
    rgb_std=imagenet_std,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

