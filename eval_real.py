import os
import re
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# ==================================================================
# ★ CONFIG — 여기만 수정 ★
# ==================================================================
CONFIG = {
    "image_dir":  os.path.expanduser("~/Downloads/beadlearn/fitimage"),
    "mask_dir":   os.path.expanduser("~/Downloads/beadlearn/fitmask"),
    "model_path": "./checkpoints_3000/april_resumed7.pth",
    "save_dir":   "./eval_results_resumed7",
    "img_size":   1280,
    "batch_size": 1,
    "threshold":  0.4,
    "num_workers": 4,
}
# ==================================================================


# ==================================================================
# 숫자 기준 정렬
# ==================================================================
def numeric_sort(path):
    num = re.search(r'(\d+)', os.path.basename(path))
    return int(num.group(1)) if num else 0


# ==================================================================
# Dataset
# ==================================================================
class WeldBeadDataset(Dataset):
    def __init__(self, rgb_paths, mask_paths, transform=None):
        assert len(rgb_paths) == len(mask_paths)
        self.rgb_paths  = rgb_paths
        self.mask_paths = mask_paths
        self.transform  = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb  = np.array(Image.open(self.rgb_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 127).astype(np.float32)

        original_rgb = rgb.copy()

        # 마스크를 이미지 크기에 맞춤
        if rgb.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        if self.transform:
            transformed = self.transform(image=rgb, mask=mask)
            rgb  = transformed["image"]
            mask = transformed["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return rgb, mask, self.rgb_paths[idx], original_rgb


# ==================================================================
# Transform
# ==================================================================
def get_eval_transform(img_size):
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
    ], is_check_shapes=False)  # ← 이것만 추가

# ==================================================================
# Metrics
# ==================================================================
@torch.no_grad()
def compute_dice_per_image(pred, target, threshold=0.4):
    pred_binary  = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    total        = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = torch.where(
        total > 0,
        2.0 * intersection / total.clamp(min=1e-6),
        torch.ones_like(total),
    )
    return dice.cpu().tolist()


@torch.no_grad()
def compute_iou_per_image(pred, target, threshold=0.4):
    pred_binary  = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union        = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = torch.where(
        union > 0,
        intersection / union.clamp(min=1e-6),
        torch.ones_like(union),
    )
    return iou.cpu().tolist()


# ==================================================================
# 시각화 저장
# ==================================================================
def save_visualization(original_rgb, gt_mask, pred_mask, save_path, dice, iou):
    """
    4분할 저장:
    [원본] [GT 마스크(초록)] [예측 마스크(빨강)] [오버레이]
    """
    H, W = original_rgb.shape[:2]

    # GT 마스크 → 초록
    gt_vis = np.zeros((H, W, 3), dtype=np.uint8)
    gt_vis[gt_mask > 0] = [0, 255, 0]

    # 예측 마스크 → 빨강
    pred_vis = np.zeros((H, W, 3), dtype=np.uint8)
    pred_vis[pred_mask > 0] = [0, 0, 255]

    # 오버레이
    overlay = original_rgb.copy()
    overlay[gt_mask > 0] = (
        overlay[gt_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)
    overlay[pred_mask > 0] = (
        overlay[pred_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    ).astype(np.uint8)

    # 4분할 합치기
    panel = np.concatenate([
        original_rgb,
        gt_vis,
        pred_vis,
        overlay,
    ], axis=1)

    # 텍스트
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    label = f"Dice: {dice:.4f}  IoU: {iou:.4f}  |  Green=GT  Red=Pred"
    cv2.putText(
        panel_bgr, label,
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2, cv2.LINE_AA
    )

    cv2.imwrite(save_path, panel_bgr)


# ==================================================================
# Main
# ==================================================================
def main():
    cfg    = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    os.makedirs(cfg["save_dir"], exist_ok=True)

    # --- 데이터 로드 (숫자 기준 정렬) ---
    rgb_paths = sorted(
        glob.glob(os.path.join(cfg["image_dir"], "*.jpg")),
        key=numeric_sort
    )
    mask_paths = sorted(
        glob.glob(os.path.join(cfg["mask_dir"], "*.png")),
        key=numeric_sort
    )

    # 매칭 확인
    print("매칭 확인 (앞 5개):")
    for img, mask in zip(rgb_paths[:5], mask_paths[:5]):
        print(f"  {os.path.basename(img)}  ↔  {os.path.basename(mask)}")
    print()

    assert len(rgb_paths) == len(mask_paths), \
        f"이미지 {len(rgb_paths)}장 vs 마스크 {len(mask_paths)}장 불일치!"
    print(f"총 {len(rgb_paths)}쌍 로드 완료\n")

    # --- Dataset & DataLoader ---
    dataset = WeldBeadDataset(
        rgb_paths, mask_paths,
        transform=get_eval_transform(cfg["img_size"])
    )
    loader = DataLoader(
        dataset,
        batch_size=1,       # 시각화는 1장씩
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    # --- 모델 로드 ---
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)

    checkpoint = torch.load(cfg["model_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"모델 로드 완료: {cfg['model_path']}")
    print(f"  학습 epoch:    {checkpoint.get('epoch', 'N/A')}")
    print(f"  학습 val Dice: {checkpoint.get('val_dice', 'N/A'):.4f}\n")

    # --- 평가 ---
    model.eval()
    all_dice  = []
    all_iou   = []
    all_paths = []

    with torch.no_grad():
        for i, (images, masks, paths, original_rgbs) in enumerate(loader):
            images = images.to(device)
            masks  = masks.to(device)

            outputs = model(images)

            dice_list = compute_dice_per_image(outputs, masks, cfg["threshold"])
            iou_list  = compute_iou_per_image(outputs,  masks, cfg["threshold"])

            all_dice.extend(dice_list)
            all_iou.extend(iou_list)
            all_paths.extend(paths)

            # 시각화 저장
            pred_mask = (torch.sigmoid(outputs[0, 0]) > cfg["threshold"]).cpu().numpy()
            gt_mask   = masks[0, 0].cpu().numpy()
            orig_rgb  = original_rgbs[0].numpy()

            # GT 마스크 해상도에 맞게 리사이즈
            h, w = gt_mask.shape
            orig_rgb = cv2.resize(orig_rgb, (w, h))

            fname     = f"{i+1:03d}_dice{dice_list[0]:.4f}.png"
            save_path = os.path.join(cfg["save_dir"], fname)
            save_visualization(orig_rgb, gt_mask, pred_mask,
                               save_path, dice_list[0], iou_list[0])

            print(f"  [{i+1:3d}/{len(loader)}] "
                  f"Dice={dice_list[0]:.4f}  IoU={iou_list[0]:.4f}  → {fname}")

    # --- 최종 결과 ---
    all_dice = np.array(all_dice)
    all_iou  = np.array(all_iou)

    print("\n" + "=" * 50)
    print(f"{'실제 데이터 평가 결과 (189장)':^50}")
    print("=" * 50)
    print(f"  평균 Dice    : {all_dice.mean():.4f}")
    print(f"  평균 IoU     : {all_iou.mean():.4f}")
    print(f"  Dice 표준편차: {all_dice.std():.4f}")
    print(f"  Dice 최솟값  : {all_dice.min():.4f}")
    print(f"  Dice 최댓값  : {all_dice.max():.4f}")
    print("=" * 50)

    print("\n▼ Dice 하위 10개")
    for rank, i in enumerate(np.argsort(all_dice)[:10]):
        print(f"  {rank+1:2d}. Dice={all_dice[i]:.4f}  "
              f"{os.path.basename(all_paths[i])}")

    print("\n▲ Dice 상위 10개")
    for rank, i in enumerate(np.argsort(all_dice)[::-1][:10]):
        print(f"  {rank+1:2d}. Dice={all_dice[i]:.4f}  "
              f"{os.path.basename(all_paths[i])}")

    print(f"\n시각화 저장 완료: {cfg['save_dir']}/")


if __name__ == "__main__":
    main()