"""HeatmapNet: encoder-decoder for court keypoint heatmap prediction.

Architecture
------------
Encoder : MobileNetV3-Small pretrained on ImageNet (fast, ~2.5M params)
Decoder : 4× bilinear-upsample + conv blocks → (14, H, W) sigmoid output
Loss    : Weighted MSE (penalise visible keypoints only, weight=10 vs background)
Input   : (B, 3, 360, 640)  normalised ImageNet
Output  : (B, 14, 360, 640) heatmaps in [0, 1]
Postprocess: per-channel softmax-weighted centroid (differentiable argmax)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .schema import NUM_KP
from .torch_dataset import HEATMAP_H, HEATMAP_W, CourtDataset


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = _ConvBnRelu(in_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class HeatmapNet(nn.Module):
    def __init__(self, num_kp: int = NUM_KP):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        features = backbone.features  # 16 sequential blocks

        # Extract skip levels:  stride 2, 4, 8, 16, 16(final)
        # MobileNetV3-Small channel map (verified):
        #   features[:2]   → 16ch, /4
        #   features[2:4]  → 24ch, /8
        #   features[4:9]  → 48ch, /16
        #   features[9:]   → 576ch, /32
        self.enc0 = features[:2]    # /4,  16ch
        self.enc1 = features[2:4]   # /8,  24ch
        self.enc2 = features[4:9]   # /16, 48ch
        self.enc3 = features[9:]    # /32, 576ch

        # Bottleneck channel reduction
        self.bridge = _ConvBnRelu(576, 256)

        # Decoder
        self.up3 = _UpBlock(256, 48, 128)  # →/16
        self.up2 = _UpBlock(128, 24, 64)   # →/8
        self.up1 = _UpBlock(64,  16, 32)   # →/4
        self.up0 = _UpBlock(32,   0, 32)   # →/2  (final upsample to full res via head)

        self.head = nn.Conv2d(32, num_kp, kernel_size=1)

    def forward(self, x):
        s0 = self.enc0(x)    # /2
        s1 = self.enc1(s0)   # /4
        s2 = self.enc2(s1)   # /8
        s3 = self.enc3(s2)   # /16

        b = self.bridge(s3)
        d = self.up3(b, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        d = self.up0(d)
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)  # /2 → full res
        return torch.sigmoid(self.head(d))  # (B, 14, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Postprocess: softmax-weighted centroid
# ─────────────────────────────────────────────────────────────────────────────

def heatmap_to_coords(heatmaps: torch.Tensor) -> torch.Tensor:
    """Convert (B, K, H, W) heatmaps → (B, K, 2) pixel coords [x, y]."""
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    weights = torch.softmax(flat * 100.0, dim=-1).view(B, K, H, W)

    ys = torch.arange(H, device=heatmaps.device, dtype=torch.float32)
    xs = torch.arange(W, device=heatmaps.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    cx = (weights * grid_x).sum(dim=[2, 3])  # (B, K)
    cy = (weights * grid_y).sum(dim=[2, 3])
    return torch.stack([cx, cy], dim=-1)     # (B, K, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def heatmap_loss(pred: torch.Tensor, target: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """Weighted MSE: weight=10 on keypoint region, 1 elsewhere."""
    weight = 1.0 + 9.0 * target  # (B, K, H, W)
    loss = (weight * (pred - target) ** 2).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_heatmap(
    dataset_root: Path,
    save_path: Path,
    epochs: int = 60,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "",
    val_interval: int = 5,
    patience: int = 10,
) -> None:
    from .paths import COURT_DATASET

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[heatmap] training on {dev}")

    train_ds = CourtDataset(
        COURT_DATASET / "images" / "train",
        COURT_DATASET / "labels" / "train",
        mode="heatmap", augment=True,
        img_w=HEATMAP_W, img_h=HEATMAP_H,
    )
    val_ds = CourtDataset(
        COURT_DATASET / "images" / "valid",
        COURT_DATASET / "labels" / "valid",
        mode="heatmap", augment=False,
        img_w=HEATMAP_W, img_h=HEATMAP_H,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = HeatmapNet().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    no_improve = 0
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, hms, vis in train_loader:
            imgs, hms, vis = imgs.to(dev), hms.to(dev), vis.to(dev)
            pred = model(imgs)
            loss = heatmap_loss(pred, hms, vis)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        if epoch % val_interval == 0 or epoch == epochs:
            model.eval()
            val_loss = 0.0
            val_rmse = 0.0
            val_n = 0
            with torch.no_grad():
                for imgs, hms, vis in val_loader:
                    imgs, hms, vis = imgs.to(dev), hms.to(dev), vis.to(dev)
                    pred = model(imgs)
                    val_loss += heatmap_loss(pred, hms, vis).item()
                    # RMSE in pixels
                    coords_pred = heatmap_to_coords(pred)   # (B, K, 2)
                    coords_gt = heatmap_to_coords(hms)
                    mask = vis.unsqueeze(-1).expand_as(coords_pred)  # (B, K, 2) bool-like
                    diff = ((coords_pred - coords_gt) ** 2).sum(-1)  # (B, K)
                    vis_flat = vis.bool()
                    if vis_flat.any():
                        val_rmse += diff[vis_flat].sqrt().sum().item()
                        val_n += vis_flat.sum().item()
            val_loss /= len(val_loader)
            val_rmse = val_rmse / val_n if val_n else 0.0
            print(f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_rmse={val_rmse:.2f}px")

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ saved best → {save_path}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  early stop at epoch {epoch}")
                    break

    print(f"[heatmap] training done. best val_loss={best_val:.4f}")
