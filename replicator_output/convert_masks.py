"""
이진 마스크 변환 스크립트
- colorize=True로 생성된 segmentation 마스크를
- 비드=흰색(255), 나머지=검은색(0) 이진 마스크로 변환

사용법:
  cd ~/replicator_output/weld_bead_synth
  python3 convert_masks.py

또는:
  python3 convert_masks.py --input ~/replicator_output/synth_color
"""

from PIL import Image
import numpy as np
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=".", help="segmentation PNG가 있는 폴더")
parser.add_argument("--bead_color", default="243,69,141", help="weld_bead RGBA 색상 (R,G,B)")
args = parser.parse_args()

INPUT_DIR = args.input
OUTPUT_DIR = os.path.join(INPUT_DIR, "binary_masks")
os.makedirs(OUTPUT_DIR, exist_ok=True)

r, g, b = map(int, args.bead_color.split(","))

count = 0
for f in sorted(glob.glob(os.path.join(INPUT_DIR, "semantic_segmentation_*.png"))):
    img = np.array(Image.open(f))
    mask = (img[:,:,0] == r) & (img[:,:,1] == g) & (img[:,:,2] == b)
    binary = np.uint8(mask) * 255

    fname = os.path.basename(f).replace("semantic_segmentation", "mask")
    Image.fromarray(binary).save(os.path.join(OUTPUT_DIR, fname))

    bead_pct = mask.sum() / mask.size * 100
    count += 1
    print(f"[{count}] {os.path.basename(f)} -> {fname} (bead: {bead_pct:.1f}%)")

print(f"\nDone! {count} binary masks -> {OUTPUT_DIR}")
