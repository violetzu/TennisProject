"""ResNet50Regressor: direct coordinate regression for court keypoints.

Architecture
------------
Backbone : ResNet50 pretrained on ImageNet
Head     : AdaptiveAvgPool → FC(2048→512) → ReLU → Dropout → FC(512→28)
           28 = 14 keypoints × 2 (x, y) normalised to [0, 1]
Loss     : Smooth-L1 (Huber) on visible keypoints only
Input    : (B, 3, 224, 224)  normalised ImageNet
Output   : (B, 14, 2)  coords in [0, 1]
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .schema import NUM_KP
from .torch_dataset import REGRESSOR_SIZE, CourtDataset


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class ResNet50Regressor(nn.Module):
    def __init__(self, num_kp: int = NUM_KP):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_kp * 2),
            nn.Sigmoid(),   # clamp to [0, 1]
        )
        self.num_kp = num_kp

    def forward(self, x):
        feat = self.encoder(x)       # (B, 2048, 1, 1)
        out = self.head(feat)        # (B, 28)
        return out.view(-1, self.num_kp, 2)  # (B, 14, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def regressor_loss(pred: torch.Tensor, target: torch.Tensor,
                   vis: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 loss on visible keypoints only.

    pred, target: (B, K, 2) normalised [0,1]
    vis          : (B, K) float mask
    """
    mask = vis.bool()
    if not mask.any():
        return pred.sum() * 0.0
    p = pred[mask]   # (N, 2)
    t = target[mask]
    return nn.functional.smooth_l1_loss(p, t)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_regressor(
    dataset_root: Path,
    save_path: Path,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "",
    val_interval: int = 5,
    patience: int = 10,
) -> None:
    from .paths import COURT_DATASET

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[regressor] training on {dev}")

    train_ds = CourtDataset(
        COURT_DATASET / "images" / "train",
        COURT_DATASET / "labels" / "train",
        mode="regressor", augment=True,
        img_w=REGRESSOR_SIZE, img_h=REGRESSOR_SIZE,
    )
    val_ds = CourtDataset(
        COURT_DATASET / "images" / "valid",
        COURT_DATASET / "labels" / "valid",
        mode="regressor", augment=False,
        img_w=REGRESSOR_SIZE, img_h=REGRESSOR_SIZE,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = ResNet50Regressor().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    no_improve = 0
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, coords, vis in train_loader:
            imgs, coords, vis = imgs.to(dev), coords.to(dev), vis.to(dev)
            pred = model(imgs)
            loss = regressor_loss(pred, coords, vis)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        if epoch % val_interval == 0 or epoch == epochs:
            model.eval()
            val_loss = 0.0
            val_rmse_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for imgs, coords, vis in val_loader:
                    imgs, coords, vis = imgs.to(dev), coords.to(dev), vis.to(dev)
                    pred = model(imgs)
                    val_loss += regressor_loss(pred, coords, vis).item()

                    # RMSE in pixels (coords are normalised; multiply back)
                    mask = vis.bool()
                    if mask.any():
                        p_px = pred[mask] * torch.tensor(
                            [REGRESSOR_SIZE, REGRESSOR_SIZE], device=dev, dtype=torch.float32)
                        t_px = coords[mask] * torch.tensor(
                            [REGRESSOR_SIZE, REGRESSOR_SIZE], device=dev, dtype=torch.float32)
                        val_rmse_sum += ((p_px - t_px) ** 2).sum(-1).sqrt().sum().item()
                        val_n += mask.sum().item()
            val_loss /= len(val_loader)
            val_rmse = val_rmse_sum / val_n if val_n else 0.0
            print(f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.5f}  "
                  f"val_loss={val_loss:.5f}  val_rmse={val_rmse:.2f}px")

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

    print(f"[regressor] training done. best val_loss={best_val:.5f}")
