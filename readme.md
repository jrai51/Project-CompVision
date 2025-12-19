# Depth Completion on KITTI (Sparse LiDAR + Monocular Depth Estimates)

This repository contains a full PyTorch pipeline for depth completion on the KITTI Depth Completion benchmark (val_selection_cropped). The project implements several ablation models using sparse LiDAR input and monocular depth priors, with a training script and visualization script.

## Installation
```
conda env create -f environment.yml
conda activate depth-env
```

Ensure CUDA is available if you want GPU training.

## Training

```
python train.py --model_type lidar_only
python train.py --model_type lidar_mono
python train.py --model_type lidar_mono_rgb
```

| Arg            | Meaning                                      |
| -------------- | -------------------------------------------- |
| `--epochs`     | Default: 15                                  |
| `--batch_size` | Default: 4                                   |
| `--lr`         | Learning rate                                |
| `--w_all`      | Main L1 loss weight                          |
| `--w_lidar`    | LiDAR anchor penalty                         |
| `--w_tv`       | Total variation smoothness                   |
| `--model_type` | `lidar_only`, `lidar_mono`, `lidar_mono_rgb` |
| `--residual`   | Uses residual learning                       |


## Visualization
`python visualize.py`
Visualize a single sample (idx) from KittiDepthCropped with:

    - RGB
    - Sparse LiDAR
    - Monocular prior
    - Predicted depth
    - Ground-truth depth
    - |pred - gt| error map (masked)

## Folder Structure:
```
PROJECT-COMPVISION/
│
├── train.py                   # Main training script
├── visualize.py               # Visualization & model comparison utilities
├── models.py                  # DepthUNet + DepthUNetMonoFusion architectures
├── data.py                    # KITTI dataloader (val_selection_cropped)
├── losses.py                  # Loss functions: L1, LiDAR anchor, TV, metrics
│
├── val_selection_cropped/     # KITTI dataset (cropped validation split)
│   ├── image/
│   ├── velodyne_raw/
│   ├── groundtruth_depth/
│   ├── depth_estimates/
│   ├── intrinsics/
│   └── ...
│
├── checkpoints/               # Saved model weights (auto-created)
│
├── figs/                      # Architecture diagrams & comparison figures
│
├── environment.yml            # Conda environment
└── readme.md

```




