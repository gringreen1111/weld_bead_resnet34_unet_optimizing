"""
Stage 1 U-Net Training: Weld Bead Segmentation
========================================================================
Based on: Closed-Loop Robotic Grinding of Weld Beads (IJPEM Special Issue)

Architecture : U-Net with ResNet34 encoder (ImageNet pretrained)
Input        : RGB 3-channel, 1280 x 1280 (longest-side resize + pad)
Loss         : BCEWithLogitsLoss + Dice Loss
Optimizer    : Adam (lr=1e-4) + ReduceLROnPlateau
Threshold    : 0.4 (for binary mask)

사용법: 아래 CONFIG 값 수정 후 → python3 train_unet.py
"""

import os
import sys
import random
import glob
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==================================================================
# ★ CONFIG — 여기만 수정하면 됨 ★
# ==================================================================
CONFIG = {
    # --- 데이터 ---
    "data_dir":     "/home/kim/combined",   # 합성 데이터 폴더
    "save_dir":     "./checkpoints",                        # 모델 저장 폴더
    "model_name":   "april_resumed5.pth",                    # 저장할 모델 파일명

    # --- 이어서 학습 (None이면 처음부터) ---                   
    "resume_from":  "best_stage2_unet.pth",                # 기존 pth 파일명 (save_dir 기준)

    # --- 학습 파라미터 ---
    "epochs":       50,
    "batch_size":   2,
    "lr":           1e-4,    
    "img_size":     1280,
    "patience":     15,         # Early stopping (val_loss 기준)

    # --- 기타 ---
    "seed":         42,     
    "num_workers":  8,
}
# ==================================================================


# ==================================================================
# 0. Reproducibility
# ==================================================================
def set_seed(seed: int = 42):
    """전체 seed 고정 (재현성 확보)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================================================================
# 1. Dataset
# ==================================================================
class WeldBeadDataset(Dataset):
    """RGB 이미지 + 이진 마스크 데이터셋"""

    def __init__(self, rgb_paths, mask_paths, transform=None):
        assert len(rgb_paths) == len(mask_paths)
        self.rgb_paths = rgb_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # RGB 로드
        rgb = np.array(Image.open(self.rgb_paths[idx]).convert("RGB"))

        # 마스크 로드 (흰=255 → 1, 검=0 → 0)
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 127).astype(np.float32)

        # Augmentation 적용
        if self.transform:
            transformed = self.transform(image=rgb, mask=mask)
            rgb = transformed["image"]
            mask = transformed["mask"]

        # mask dtype & shape 정규화: (H,W) → (1,H,W), float32 보장
        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return rgb, mask


# ==================================================================
# 2. Loss Function (BCE + Dice)
# ==================================================================
class BCEDiceLoss(nn.Module):
    """논문과 동일: BCEWithLogitsLoss + Dice Loss"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1.0):
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice_loss(pred, target)


# ==================================================================
# 3. Metrics (이미지별 계산 후 평균)
# ==================================================================
@torch.no_grad()
def compute_iou(pred, target, threshold: float = 0.4):
    """IoU: 이미지별(B 차원별)로 계산해서 평균"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = torch.where(
        union > 0,
        intersection / union.clamp(min=1e-6),
        torch.ones_like(union),
    )
    return iou.mean().item()


@torch.no_grad()
def compute_dice(pred, target, threshold: float = 0.4):
    """Dice: 이미지별로 계산해서 평균"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    total = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = torch.where(
        total > 0,
        2.0 * intersection / total.clamp(min=1e-6),
        torch.ones_like(total),
    )
    return dice.mean().item()


# ==================================================================
# 4. Augmentation (논문 기반)
# ==================================================================
def _safe_gauss_noise(p: float = 0.3):
    """Albumentations 버전에 따라 인자 이름이 다름 → 자동 분기"""
    try:
        return A.GaussNoise(std_range=(0.02, 0.1), p=p)
    except TypeError:
        return A.GaussNoise(var_limit=(10.0, 50.0), p=p)


def get_train_transform(img_size: int = 1280):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=0, value=0, mask_value=0,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.03, scale_limit=0.05, rotate_limit=3, p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.5
        ),
        # ↓ 추가되는 부분
        A.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.3, hue=0.1, p=0.5
        ),
        A.RandomGamma(gamma_limit=(60, 140), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(30, 100),
            hole_width_range=(30, 100),
            fill=0, p=0.2
        ),
        # ↑ 추가되는 부분
        _safe_gauss_noise(p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = 1280):
    """검증/테스트용: resize + padding + normalize만"""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=0, value=0, mask_value=0,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ==================================================================
# 5. Training Loop
# ==================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_iou = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(outputs, masks)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}")

    n = len(loader)
    return total_loss / n, total_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()
        total_iou += compute_iou(outputs, masks)
        total_dice += compute_dice(outputs, masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


# ==================================================================
# 6. Main
# ==================================================================
def main():
    # --- CONFIG에서 값 가져오기 ---
    cfg = CONFIG

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n--- CONFIG ---")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    os.makedirs(cfg["save_dir"], exist_ok=True)

    # --- 데이터 로드 ---
    rgb_paths = sorted(glob.glob(os.path.join(cfg["data_dir"], "rgb_*.png")))
    mask_paths = sorted(glob.glob(
        os.path.join(cfg["data_dir"], "binary_masks", "mask_*.png")
    ))

    if len(rgb_paths) == 0:
        print(f"[ERROR] No RGB images found in {cfg['data_dir']}")
        sys.exit(1)

    assert len(rgb_paths) == len(mask_paths), \
        f"RGB({len(rgb_paths)})와 마스크({len(mask_paths)}) 수가 다름!"
    print(f"Total images: {len(rgb_paths)}")

    # --- Train/Val/Test 분리 (70:15:15) ---
    n_total = len(rgb_paths)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)

    indices = np.random.RandomState(cfg["seed"]).permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_rgb = [rgb_paths[i] for i in train_idx]
    train_mask = [mask_paths[i] for i in train_idx]
    val_rgb = [rgb_paths[i] for i in val_idx]
    val_mask = [mask_paths[i] for i in val_idx]
    test_rgb = [rgb_paths[i] for i in test_idx]
    test_mask = [mask_paths[i] for i in test_idx]

    print(f"Train: {len(train_rgb)}, Val: {len(val_rgb)}, Test: {len(test_rgb)}")

    # --- Test split 저장 ---
    split_info = {
        "seed": cfg["seed"],
        "n_total": n_total,
        "train": [[r, m] for r, m in zip(train_rgb, train_mask)],
        "val": [[r, m] for r, m in zip(val_rgb, val_mask)],
        "test": [[r, m] for r, m in zip(test_rgb, test_mask)],
    }
    split_path = os.path.join(cfg["save_dir"], "split.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Split info saved: {split_path}")

    # --- Dataset & DataLoader ---
    train_dataset = WeldBeadDataset(
        train_rgb, train_mask, transform=get_train_transform(cfg["img_size"])
    )
    val_dataset = WeldBeadDataset(
        val_rgb, val_mask, transform=get_val_transform(cfg["img_size"])
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"], pin_memory=True,
    )

    # --- 모델 (논문: ResNet34 U-Net, ImageNet pretrained) ---
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: U-Net + ResNet34 encoder (ImageNet pretrained)")
    print(f"Total parameters: {total_params:,}")

    # --- 기존 체크포인트에서 이어서 학습 ---
    if cfg["resume_from"]:
        resume_path = os.path.join(cfg["save_dir"], cfg["resume_from"])
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"\n[RESUME] 기존 모델 로드 완료: {resume_path}")
            print(f"  이전 Epoch:    {checkpoint.get('epoch', 'N/A')}")
            print(f"  이전 Val Loss: {checkpoint.get('val_loss', 'N/A')}")
            print(f"  이전 Val IoU:  {checkpoint.get('val_iou', 'N/A')}")
            print(f"  이전 Val Dice: {checkpoint.get('val_dice', 'N/A')}")
        else:
            print(f"\n[WARNING] resume_from 파일 없음: {resume_path}")
            print(f"  → ImageNet pretrained 가중치로 처음부터 학습합니다.")
    else:
        print(f"\n[INFO] resume_from = None → 처음부터 학습")

    # --- Loss, Optimizer, Scheduler ---
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # --- 학습 ---
    best_val_loss = float("inf")
    best_val_iou = 0.0
    no_improve_count = 0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{cfg['epochs']} "
              f"(lr={optimizer.param_groups[0]['lr']:.6f})")
        print(f"{'=' * 60}")

        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, "
              f"Dice: {val_dice:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Best 모델 저장 기준: val_loss (논문과 동일)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_iou = val_iou
            no_improve_count = 0

            save_path = os.path.join(cfg["save_dir"], cfg["model_name"])
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_dice": val_dice,
                "config": cfg,
            }, save_path)
            print(f"  *** Best model saved! "
                  f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f} ***")
        else:
            no_improve_count += 1
            print(f"  No improvement for {no_improve_count} epoch(s)")

            if no_improve_count >= cfg["patience"]:
                print(f"\n[Early Stopping] at epoch {epoch} "
                      f"(patience={cfg['patience']})")
                break

        # History 저장 (매 에폭)
        history_path = os.path.join(cfg["save_dir"], "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val IoU : {best_val_iou:.4f}")
    print(f"Model saved  : {cfg['save_dir']}/{cfg['model_name']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()