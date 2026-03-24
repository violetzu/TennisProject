#!/usr/bin/env python3
"""
將 TrackNet 官方資料集轉換為 YOLO 物件偵測格式。

來源結構 (tracknet_dataset/):
    game1/ ~ game10/
    └── Clip1/ ~ ClipN/
        ├── 0000.jpg, 0001.jpg, ...
        └── Label.csv

CSV 表頭: file name, visibility, x-coordinate, y-coordinate, status
  visibility: 0=不可見, 1=清楚可見, 2=模糊可定位, 3=不確定
  只取 visibility ∈ {1, 2} 且座標 > 0 的幀

輸出結構 (tracknet_yolo_dataset/):
    data.yaml
    train/images/  train/labels/
    valid/images/  valid/labels/

用法:
    cd backend/models
    python tracknet_to_yolo.py [--bbox-size 16] [--val-ratio 0.15]
"""

import argparse
import csv
import random
import shutil
from pathlib import Path


IMG_W, IMG_H = 1280, 720  # TrackNet 固定解析度


def parse_args():
    p = argparse.ArgumentParser(description="TrackNet → YOLO 轉換")
    p.add_argument("--src", default="tracknet_dataset")
    p.add_argument("--dst", default="tracknet_yolo_dataset")
    p.add_argument("--bbox-size", type=int, default=16, help="bbox 邊長 (px)")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", default=False,
                   help="複製圖片（預設 symlink）")
    return p.parse_args()


def collect_records(src: Path) -> list[dict]:
    """掃描所有 game/Clip/Label.csv，回傳有效標註。"""
    records = []
    for label_csv in sorted(src.rglob("Label.csv")):
        clip_dir = label_csv.parent
        with open(label_csv, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vis = int(row["visibility"])
                if vis not in (1, 2):
                    continue
                x = float(row["x-coordinate"])
                y = float(row["y-coordinate"])
                if x <= 0 or y <= 0:
                    continue
                img_path = clip_dir / row["file name"].strip()
                if not img_path.exists():
                    continue
                records.append({"img": img_path, "x": x, "y": y})
    return records


def convert(args):
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    bbox = args.bbox_size
    w_norm = bbox / IMG_W
    h_norm = bbox / IMG_H

    if not src.exists():
        print(f"[ERROR] 來源不存在: {src}")
        return

    # 收集標註
    records = collect_records(src)
    print(f"[INFO] 有效標註: {len(records)} 幀")
    if not records:
        return

    # 分 train/val
    random.seed(args.seed)
    random.shuffle(records)
    val_n = int(len(records) * args.val_ratio)
    splits = {"train": records[val_n:], "valid": records[:val_n]}
    print(f"[INFO] train={len(splits['train'])}, valid={len(splits['valid'])}")

    # 建目錄
    for s in splits:
        (dst / s / "images").mkdir(parents=True, exist_ok=True)
        (dst / s / "labels").mkdir(parents=True, exist_ok=True)

    # 轉換
    for split, recs in splits.items():
        for i, r in enumerate(recs):
            # 檔名: game1_Clip1_0000
            parts = r["img"].relative_to(src).parts
            stem = "_".join(parts).replace(".jpg", "").replace(".png", "")

            # YOLO label
            xc = max(w_norm / 2, min(1 - w_norm / 2, r["x"] / IMG_W))
            yc = max(h_norm / 2, min(1 - h_norm / 2, r["y"] / IMG_H))
            lbl_path = dst / split / "labels" / f"{stem}.txt"
            lbl_path.write_text(f"0 {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            # 圖片
            img_dst = dst / split / "images" / f"{stem}.jpg"
            if not img_dst.exists():
                if args.copy:
                    shutil.copy2(r["img"], img_dst)
                else:
                    try:
                        img_dst.symlink_to(r["img"])
                    except OSError:
                        shutil.copy2(r["img"], img_dst)

            if (i + 1) % 5000 == 0:
                print(f"  [{split}] {i+1}/{len(recs)}")

    # data.yaml
    (dst / "data.yaml").write_text(
        f"train: ../train/images\n"
        f"val: ../valid/images\n\n"
        f"nc: 1\n"
        f"names: ['tennis ball']\n\n"
        f"# TrackNet dataset, {IMG_W}x{IMG_H}, bbox={bbox}px\n"
    )

    print(f"\n[OK] 輸出: {dst}")
    print(f"     train={len(splits['train'])}, valid={len(splits['valid'])}, bbox={bbox}px")


if __name__ == "__main__":
    convert(parse_args())
