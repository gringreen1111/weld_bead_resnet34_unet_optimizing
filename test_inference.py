"""
Stage 1 U-Net Inference: 학습된 모델로 실제 이미지 테스트
========================================================================
학습한 best_stage1_unet.pth로 실제 D405 이미지에 추론
- 단일 이미지 또는 폴더 전체 처리
- 결과 시각화: 원본 / mask / overlay 3-panel
- IoU 등 평가 지표 (GT mask 있는 경우)

사용 예:
  # 단일 이미지
  python test_inference.py \
      --checkpoint ./checkpoints/best_stage1_unet.pth \
      --input ./real_test/img.png \
      --output ./inference_results

  # 폴더 전체
  python test_inference.py \
      --checkpoint ./checkpoints/best_stage1_unet.pth \
      --input ./real_test/ \
      --output ./inference_results

  # GT mask와 비교 (IoU 계산)
  python test_inference.py \
      --checkpoint ./checkpoints/best_stage1_unet.pth \
      --input ./real_test/ \
      --gt_dir ./real_test/masks/ \
      --output ./inference_results
"""

import os
import sys
import argparse
import glob
import time
import numpy as np
from PIL import Image

import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==================================================================
# 1. Transform (학습 코드의 val_transform과 동일해야 함!)
# ==================================================================
def get_inference_transform(img_size: int = 1280):
    """추론용: 학습 시 val_transform과 정확히 동일"""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=0, value=0,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ==================================================================
# 2. 모델 로드
# ==================================================================
def load_model(checkpoint_path: str, device: torch.device):
    """학습된 모델 로드"""
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 모델 구조: 학습 때와 동일하게 생성
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Trained at epoch : {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss         : {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Val IoU          : {checkpoint.get('val_iou', 'N/A'):.4f}")
    print(f"  Val Dice         : {checkpoint.get('val_dice', 'N/A'):.4f}")

    return model


# ==================================================================
# 3. 추론 (1장)
# ==================================================================
@torch.no_grad()
def predict(model, rgb_path: str, transform, device: torch.device,
            threshold: float = 0.4, img_size: int = 1280):
    """
    이미지 1장에 대한 segmentation mask 추론
    Returns:
        rgb_orig    : 원본 RGB (H_orig, W_orig, 3)
        mask_pred   : 원본 해상도로 복원된 binary mask (H_orig, W_orig) - {0, 255}
        prob_map    : 원본 해상도 sigmoid 확률 맵 (H_orig, W_orig) - [0, 1]
        infer_ms    : 모델 forward 시간 (ms)
    """
    # 원본 로드
    rgb_orig = np.array(Image.open(rgb_path).convert("RGB"))
    H_orig, W_orig = rgb_orig.shape[:2]

    # Transform 적용 (LongestMaxSize + Pad → img_size x img_size)
    transformed = transform(image=rgb_orig)
    img_tensor = transformed["image"].unsqueeze(0).to(device)  # (1, 3, S, S)

    # 추론
    t0 = time.time()
    logits = model(img_tensor)  # (1, 1, S, S)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (S, S)
    infer_ms = (time.time() - t0) * 1000

    # 원본 해상도로 복원
    # 1) Padding 제거: LongestMaxSize 후 작은 변에 padding 들어감
    # 2) 원본 크기로 resize
    scale = img_size / max(H_orig, W_orig)
    H_resized = int(round(H_orig * scale))
    W_resized = int(round(W_orig * scale))

    # 패딩이 위/아래(또는 좌/우) 균등하게 들어감 (PadIfNeeded 기본)
    pad_top = (img_size - H_resized) // 2
    pad_left = (img_size - W_resized) // 2

    # 패딩 영역 잘라내기
    prob_unpadded = prob[pad_top:pad_top + H_resized, pad_left:pad_left + W_resized]

    # 원본 해상도로 resize
    prob_pil = Image.fromarray((prob_unpadded * 255).astype(np.uint8))
    prob_resized = np.array(
        prob_pil.resize((W_orig, H_orig), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # Binary mask
    mask_pred = (prob_resized > threshold).astype(np.uint8) * 255

    return rgb_orig, mask_pred, prob_resized, infer_ms


# ==================================================================
# 4. 시각화
# ==================================================================
def make_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5,
                 color: tuple = (0, 255, 0)) -> np.ndarray:
    """
    RGB 위에 mask를 컬러로 overlay
    rgb  : (H, W, 3) uint8
    mask : (H, W) uint8 {0, 255}
    """
    overlay = rgb.copy()
    mask_bool = mask > 127
    color_layer = np.zeros_like(rgb)
    color_layer[..., 0] = color[0]
    color_layer[..., 1] = color[1]
    color_layer[..., 2] = color[2]
    overlay[mask_bool] = (
        rgb[mask_bool] * (1 - alpha) + color_layer[mask_bool] * alpha
    ).astype(np.uint8)
    return overlay


def save_visualization(rgb: np.ndarray, mask: np.ndarray, save_path: str,
                       gt_mask: np.ndarray = None):
    """
    3-panel 시각화 저장: 원본 | predicted mask | overlay
    GT가 있으면 4-panel: 원본 | GT mask | predicted mask | overlay
    """
    H, W = rgb.shape[:2]

    # Mask를 3채널로 (시각화용)
    mask_rgb = np.stack([mask] * 3, axis=-1)
    overlay = make_overlay(rgb, mask, alpha=0.5, color=(0, 255, 0))

    if gt_mask is not None:
        gt_rgb = np.stack([gt_mask] * 3, axis=-1)
        # 4-panel: rgb | gt | pred | overlay
        sep = np.ones((H, 5, 3), dtype=np.uint8) * 128  # 회색 구분선
        combined = np.concatenate(
            [rgb, sep, gt_rgb, sep, mask_rgb, sep, overlay], axis=1
        )
    else:
        sep = np.ones((H, 5, 3), dtype=np.uint8) * 128
        combined = np.concatenate([rgb, sep, mask_rgb, sep, overlay], axis=1)

    Image.fromarray(combined).save(save_path)


# ==================================================================
# 5. 평가 지표 (GT가 있을 때)
# ==================================================================
def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """IoU, Dice, Precision, Recall 계산"""
    pred = (pred_mask > 127).astype(np.float32)
    gt = (gt_mask > 127).astype(np.float32)

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection

    iou = intersection / union if union > 0 else 1.0
    dice = (2.0 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 1.0
    precision = intersection / pred.sum() if pred.sum() > 0 else 0.0
    recall = intersection / gt.sum() if gt.sum() > 0 else 0.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


# ==================================================================
# 6. Main
# ==================================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 1 U-Net Inference")
    parser.add_argument("--checkpoint", required=True, help="best_stage1_unet.pth 경로")
    parser.add_argument("--input", required=True, help="입력 이미지 또는 폴더")
    parser.add_argument("--output", default="./inference_results", help="결과 저장 폴더")
    parser.add_argument("--gt_dir", default=None,
                        help="GT mask 폴더 (있으면 IoU 계산). 파일명은 입력과 동일해야 함")
    parser.add_argument("--img_size", type=int, default=1280, help="학습 시와 동일하게")
    parser.add_argument("--threshold", type=float, default=0.4, help="논문 0.4")
    parser.add_argument("--save_mask_only", action="store_true",
                        help="3-panel 시각화 안 만들고 mask만 저장")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- 모델 로드 ---
    model = load_model(args.checkpoint, device)
    transform = get_inference_transform(args.img_size)

    # --- 출력 폴더 ---
    os.makedirs(args.output, exist_ok=True)
    mask_dir = os.path.join(args.output, "masks")
    vis_dir = os.path.join(args.output, "visualizations")
    os.makedirs(mask_dir, exist_ok=True)
    if not args.save_mask_only:
        os.makedirs(vis_dir, exist_ok=True)

    # --- 입력 파일 목록 ---
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        # 일반적인 이미지 확장자 모두 지원
        exts = ["png", "jpg", "jpeg", "bmp"]
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(args.input, f"*.{ext}")))
            image_paths.extend(glob.glob(os.path.join(args.input, f"*.{ext.upper()}")))
        image_paths = sorted(set(image_paths))
    else:
        print(f"[ERROR] Input not found: {args.input}")
        sys.exit(1)

    if len(image_paths) == 0:
        print(f"[ERROR] No images found in {args.input}")
        sys.exit(1)

    print(f"\nFound {len(image_paths)} image(s) to process")
    print(f"Threshold: {args.threshold}, Image size: {args.img_size}")
    print(f"Output: {args.output}\n")

    # --- 추론 루프 ---
    all_metrics = []
    total_infer_ms = 0.0

    for idx, rgb_path in enumerate(image_paths):
        fname = os.path.basename(rgb_path)
        stem = os.path.splitext(fname)[0]

        # 추론
        rgb, mask_pred, prob, infer_ms = predict(
            model, rgb_path, transform, device,
            threshold=args.threshold, img_size=args.img_size,
        )
        total_infer_ms += infer_ms

        # GT mask 로드 (있으면)
        gt_mask = None
        metrics = None
        if args.gt_dir:
            # 같은 파일명으로 찾기
            gt_candidates = [
                os.path.join(args.gt_dir, fname),
                os.path.join(args.gt_dir, f"{stem}.png"),
                os.path.join(args.gt_dir, f"mask_{stem}.png"),
            ]
            gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)
            if gt_path:
                gt_mask = np.array(Image.open(gt_path).convert("L"))
                # 크기가 다르면 pred에 맞춰 resize
                if gt_mask.shape != mask_pred.shape:
                    gt_mask = np.array(
                        Image.fromarray(gt_mask).resize(
                            (mask_pred.shape[1], mask_pred.shape[0]),
                            Image.NEAREST,
                        )
                    )
                metrics = compute_metrics(mask_pred, gt_mask)
                all_metrics.append(metrics)

        # Mask 저장
        mask_save_path = os.path.join(mask_dir, f"{stem}_mask.png")
        Image.fromarray(mask_pred, mode="L").save(mask_save_path)

        # 시각화 저장
        if not args.save_mask_only:
            vis_save_path = os.path.join(vis_dir, f"{stem}_vis.png")
            save_visualization(rgb, mask_pred, vis_save_path, gt_mask=gt_mask)

        # 진행 상황
        bead_pct = (mask_pred > 127).mean() * 100
        log = f"[{idx+1}/{len(image_paths)}] {fname} | infer: {infer_ms:.0f}ms | bead: {bead_pct:.1f}%"
        if metrics:
            log += (f" | IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}, "
                    f"P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}")
        print(log)

    # --- 요약 ---
    n = len(image_paths)
    print(f"\n{'=' * 60}")
    print(f"Inference complete: {n} images")
    print(f"Avg inference time: {total_infer_ms / n:.1f} ms/image")
    print(f"Masks saved        : {mask_dir}")
    if not args.save_mask_only:
        print(f"Visualizations     : {vis_dir}")

    if all_metrics:
        avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        std = {k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print(f"\n--- Metrics (mean ± std over {len(all_metrics)} images) ---")
        print(f"  IoU       : {avg['iou']:.4f} ± {std['iou']:.4f}")
        print(f"  Dice      : {avg['dice']:.4f} ± {std['dice']:.4f}")
        print(f"  Precision : {avg['precision']:.4f} ± {std['precision']:.4f}")
        print(f"  Recall    : {avg['recall']:.4f} ± {std['recall']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()