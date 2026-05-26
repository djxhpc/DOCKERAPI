"""
classify_train.py — YOLOv8 影像分類訓練腳本

設定 4 個類別資料夾後，一鍵完成:
  1. 自動切分 train / val
  2. 建立 YOLOv8 classify 格式資料集
  3. 訓練並輸出 best.pt
"""

import os
import shutil
import random
from pathlib import Path

# =============================================
# 設定區 — 請修改以下路徑與參數
# =============================================

# 4 個類別：key=類別名稱（同時作為資料夾名），value=原始影像所在路徑
CLASS_DIRS = {
    "埋深":   r"C:\path\to\class_burial_depth",
    "水準點": r"C:\path\to\class_benchmark",
    "座標":   r"C:\path\to\class_coordinate",
    "其他":   r"C:\path\to\class_other",
}

# 所有輸出（資料集 + 訓練結果）都放在這個資料夾下
OUTPUT_DIR = r"D:\work2\0526ocr\classify_output"

# 模型大小: n(最小/最快) | s | m | l | x(最準/最慢)
# 50 張/類建議用 n 或 s，避免過擬合
MODEL_SIZE = "n"

# 訓練超參數
EPOCHS    = 100
IMG_SIZE  = 224
BATCH     = 16
VAL_RATIO = 0.2   # 驗證集比例（0.2 = 20%）

# =============================================

_IMG_EXT = {'.jpg', '.jpeg', '.jpe', '.png', '.bmp', '.webp', '.tiff'}


def prepare_dataset(class_dirs: dict, output_dir: str, val_ratio: float) -> Path:
    """
    將各類別資料夾的影像複製成 YOLOv8 classify 格式：
      dataset/
        train/<class>/
        val/<class>/
    """
    dataset_dir = Path(output_dir) / "dataset"

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for split in ("train", "val"):
        for cls in class_dirs:
            (dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)

    print("\n[步驟 1] 準備資料集")
    for cls, src in class_dirs.items():
        src = Path(src)
        if not src.exists():
            print(f"  [錯誤] 找不到資料夾: {src}，跳過 {cls}")
            continue

        imgs = [f for f in src.iterdir() if f.suffix.lower() in _IMG_EXT]
        if not imgs:
            print(f"  [警告] {cls}: 資料夾內無影像")
            continue

        random.shuffle(imgs)
        n_val      = max(1, int(len(imgs) * val_ratio))
        val_imgs   = imgs[:n_val]
        train_imgs = imgs[n_val:]

        for img in train_imgs:
            shutil.copy2(img, dataset_dir / "train" / cls / img.name)
        for img in val_imgs:
            shutil.copy2(img, dataset_dir / "val"   / cls / img.name)

        print(f"  {cls:8s}: train={len(train_imgs):3d}  val={len(val_imgs):2d}  (共 {len(imgs)} 張)")

    return dataset_dir


def train_model(dataset_dir: Path, output_dir: str) -> Path:
    from ultralytics import YOLO

    model_name = f"yolov8{MODEL_SIZE}-cls.pt"
    print(f"\n[步驟 2] 訓練開始")
    print(f"  模型   : {model_name}")
    print(f"  epochs : {EPOCHS}")
    print(f"  imgsz  : {IMG_SIZE}")
    print(f"  batch  : {BATCH}")

    model = YOLO(model_name)
    model.train(
        data=str(dataset_dir),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(Path(output_dir) / "runs"),
        name="classify",
        exist_ok=True,
    )

    best_pt = Path(output_dir) / "runs" / "classify" / "weights" / "best.pt"
    return best_pt


def main():
    random.seed(42)

    print("=" * 40)
    print(" YOLOv8 影像分類訓練")
    print("=" * 40)
    print(f"類別: {list(CLASS_DIRS.keys())}")
    print(f"輸出: {OUTPUT_DIR}")

    dataset_dir = prepare_dataset(CLASS_DIRS, OUTPUT_DIR, VAL_RATIO)
    best_pt     = train_model(dataset_dir, OUTPUT_DIR)

    print("\n" + "=" * 40)
    print(" 訓練完成")
    print("=" * 40)
    print(f"最佳模型: {best_pt}")
    print()
    print("推理範例:")
    print("  from ultralytics import YOLO")
    print(f"  model  = YOLO(r'{best_pt}')")
    print("  result = model('image.jpg')[0]")
    print("  label  = result.names[result.probs.top1]")
    print("  conf   = result.probs.top1conf.item()")


if __name__ == "__main__":
    main()
