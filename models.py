# U-net Depth Completion Architectures 

# U-Net only LiDAR 
# input: [sparse, sparse_mask]
# prediction: dense depth directly from sparse lidar input 

# U-Net with LiDAR data + monocular depth estimation 
# input: [mono, sparse, sparse_mask]
# prediction = mono + residual
# "Start from the monocular map, then pull it into metric consistency wherever LiDAR and GT say so"

# U-Net with LiDAR data + monocular depth estimation + RGB image 
# input : [rgb, mono, sparse, sparse_mask]
# prediction = mono + residual
# "Does adding appearance cues help refine depth beyond what mono+LiDAR already give?""

# Can just use variable input channels to cover most of the bases

import math, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# U-Net building blocks
# -----------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch)  # cheap downsample
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed (in case of odd dims)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, (0, diffX, 0, diffY))
        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return x
    

######################
# CREATE U-NET MODEL #
######################

class DepthUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, residual=False):
        """
        in_ch: number of input channels
            e.g. 2  -> [sparse, sparse_mask]
                  3  -> [mono, sparse, sparse_mask]
                  6  -> [rgb(3), mono, sparse, sparse_mask]
        base_ch: number of base channels for the U-Net
        residual: if True, model predicts a residual to add on top of a monocular prior
                  forward(x, mono) -> mono + residual
                  if False, model predicts absolute depth directly:
                  forward(x) -> depth
        """
        super().__init__()
        self.residual = residual

        c1, c2, c3, c4, c5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16
        
        # Monocular Prior learned embedding
        self.mono_embed = nn.Conv2d(1, 4, kernel_size=1)  

        # Encoder
        self.enc1 = ConvBlock(in_ch, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        # Decoder
        self.up1 = Up(c5 + c4, c4)
        self.up2 = Up(c4 + c3, c3)
        self.up3 = Up(c3 + c2, c2)
        self.up4 = Up(c2 + c1, c1)

        # Final 1x1 conv to single-channel depth (residual or absolute)
        self.head = nn.Conv2d(c1, 1, 1)

        # Kaiming init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, mono=None):
        """
        x: input tensor of shape (B, in_ch, H, W)
        mono: optional monocular prior of shape (B, 1, H, W), required if residual=True

        Returns:
            If residual == False:
                out_depth: (B, 1, H, W)
            If residual == True:
                out_depth = mono + residual: (B, 1, H, W)
        """
        # Encoder
        e1 = self.enc1(x)         # c1
        e2 = self.down1(e1)       # c2
        e3 = self.down2(e2)       # c3
        e4 = self.down3(e3)       # c4
        e5 = self.down4(e4)       # c5

        # Decoder
        d1 = self.up1(e5, e4)     # c4
        d2 = self.up2(d1, e3)     # c3
        d3 = self.up3(d2, e2)     # c2
        d4 = self.up4(d3, e1)     # c1

        residual = self.head(d4)  # (B,1,H,W)

        if self.residual:
            if mono is None:
                raise ValueError("Monocular estimate prior (mono) must be provided when residual=True")
            # We assume mono is already aligned & normalized
            # The network learns a correction on top of mono
            return mono + residual
        else:
            return residual


# example:
# # inputs: [mono, sparse, sparse_mask] -> 3 channels
# model_lidar_mono = DepthUNet(in_ch=3, base_ch=32, residual=True)

# # Training loop:
# x = torch.cat([batch["mono"], batch["sparse"], batch["sparse_mask"]], dim=1)  # (B,3,H,W)
# mono = batch["mono"]  # (B,1,H,W)
# pred = model_lidar_mono(x, mono=mono)  # outputs mono + residual





class DepthUNetMonoFusion(nn.Module):
    def __init__(self, base_ch=32, residual=False):
        """
        U-Net for Monocular Estimate and Sparse LiDAR Fusion. 

        Two-branch encoder at the first level:
          - sparse branch: [sparse, sparse_mask]    (2 ch)
          - mono branch:   [mono]                   (1 ch)

        They are encoded separately, then concatenated to form c1 channels
        and passed through a standard U-Net backbone.

        residual:
          - False: predict absolute depth
          - True:  predict residual to add on top of mono (requires mono in forward)
        """
        super().__init__()
        self.residual = residual

        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16

        # --- First level: separate encoders ---
        # we split c1 in half for sparse vs mono
        c1_sparse = c1 // 2
        c1_mono   = c1 - c1_sparse  # in case c1 not divisible by 2

        self.enc1_sparse = ConvBlock(in_ch=2, out_ch=c1_sparse)  # [sparse, mask]
        self.enc1_mono   = ConvBlock(in_ch=1, out_ch=c1_mono)    # [mono]

        # --- Shared backbone from level 2 onward ---
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.up1 = Up(c5 + c4, c4)
        self.up2 = Up(c4 + c3, c3)
        self.up3 = Up(c3 + c2, c2)
        self.up4 = Up(c2 + c1, c1)
        
        # self.head = nn.Conv2d(c1, 1, kernel_size=1)

        # NEW: 1x1 conv to embed mono at full resolution
        self.mono_skip = nn.Conv2d(1, c1, kernel_size=1)
        # head now takes [decoder features + mono_embed]
        self.head = nn.Conv2d(c1 * 2, 1, kernel_size=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, sparse, sparse_mask, mono=None):
        """
        sparse:      (B,1,H,W)   metric LiDAR depth
        sparse_mask: (B,1,H,W)   1 where LiDAR valid
        mono:        (B,1,H,W)   relative monocular prior (required if residual=True)

        Returns:
            depth_pred: (B,1,H,W)
        """
        # --- level 1: separate branches ---
        x_sparse = torch.cat([sparse, sparse_mask], dim=1)  # (B,2,H,W)
        f_sparse = self.enc1_sparse(x_sparse)               # (B,c1_sparse,H,W)
        f_mono   = self.enc1_mono(mono)                     # (B,c1_mono,H,W)

        e1 = torch.cat([f_sparse, f_mono], dim=1)           # (B,c1,H,W)

        # --- shared encoder ---
        e2 = self.down1(e1) # (B,c2,H/2,W/2)
        e3 = self.down2(e2) # (B,c3,...)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        # --- decoder with skips ---
        d1 = self.up1(e5, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)

        if mono is not None:
            # NEW: inject *raw* mono structure at full resolution
            mono_embed = self.mono_skip(mono)                   # (B,c1,H,W)
            d4_cat = torch.cat([d4, mono_embed], dim=1)         # (B,2*c1,H,W)
            residual = self.head(d4_cat)                        # (B,1,H,W)
        else:
            residual = self.head(d4)    # (B,1,H,W)


        if self.residual:
            if mono is None:
                raise ValueError("mono must be provided when residual=True")
            return mono + residual
        else:
            return residual