"""
scripts/import_downloaded_rdd.py
================================
Automated importer for pre-downloaded RDD2022 datasets.

Usage:
  python scripts/import_downloaded_rdd.py
  python scripts/import_downloaded_rdd.py --source "C:/Users/kakum/Downloads/RD2022/RDD_SPLIT"
  python scripts/import_downloaded_rdd.py --max-images 2000

This script:
  1. Locates the downloaded RDD2022 folder (defaulting to C:/Users/kakum/Downloads/RD2022/RDD_SPLIT)
  2. Identifies images with pothole annotations (Class 3 / D40)
  3. Maps class 3 -> class 0 (single-class YOLO pothole model)
  4. Organizes them into train (80%) and val (20%) splits under data/processed/split/
  5. Generates the dataset.yaml file required for `python ml/train_yolo.py`
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import setup_logger

logger = setup_logger(__name__)

POTHOLE_CLASS_ID = "3"  # In RDD2022 YOLO format, Class 3 is D40 (Pothole)


def import_dataset(source_dir: str, max_images: int = 0):
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"ERROR: Source directory not found: {source_path}")
        return False

    print("=" * 65)
    print(" RDD2022 DATASET EXPORT & IMPORT INTO PROJECT")
    print("=" * 65)
    print(f"Source: {source_path}")

    # Destination directories
    dest_base = PROJECT_ROOT / "data" / "processed" / "split"
    train_img_dir = dest_base / "train" / "images"
    train_lbl_dir = dest_base / "train" / "labels"
    val_img_dir = dest_base / "val" / "images"
    val_lbl_dir = dest_base / "val" / "labels"
    test_img_dir = dest_base / "test" / "images"
    test_lbl_dir = dest_base / "test" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, test_img_dir, test_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Find image and label pairs
    print("\nScanning for pothole (D40) records...")
    pothole_samples = []

    # Look in train and test of source
    search_dirs = [
        (source_path / "train" / "images", source_path / "train" / "labels"),
        (source_path / "test" / "images", source_path / "test" / "labels"),
        (source_path / "images", source_path / "labels"),
    ]

    for img_folder, lbl_folder in search_dirs:
        if not (img_folder.exists() and lbl_folder.exists()):
            continue
        for lbl_file in lbl_folder.glob("*.txt"):
            pothole_lines = []
            try:
                with open(lbl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and (parts[0] == POTHOLE_CLASS_ID or parts[0] == "D40"):
                            # Remap to class 0 for single-class pothole model
                            new_line = "0 " + " ".join(parts[1:]) + "\n"
                            pothole_lines.append(new_line)
            except Exception:
                continue

            if pothole_lines:
                # O(1) instant lookup
                stem = lbl_file.stem
                found_img = None
                for ext in [".jpg", ".png", ".jpeg"]:
                    candidate = img_folder / f"{stem}{ext}"
                    if candidate.is_file():
                        found_img = candidate
                        break
                if found_img:
                    pothole_samples.append((found_img, pothole_lines))
                    if max_images > 0 and len(pothole_samples) >= max_images:
                        break
        if max_images > 0 and len(pothole_samples) >= max_images:
            break

    if not pothole_samples:
        print("No annotated images could be paired.")
        return False

    random.seed(42)
    random.shuffle(pothole_samples)

    if max_images > 0 and len(pothole_samples) > max_images:
        print(f"Limiting to first {max_images} images as requested...")
        pothole_samples = pothole_samples[:max_images]

    # Split: 70% train, 20% val, 10% test (matching config.yaml)
    n_total = len(pothole_samples)
    n_train = int(n_total * 0.70)
    n_val   = int(n_total * 0.20)
    train_samples = pothole_samples[:n_train]
    val_samples   = pothole_samples[n_train : n_train + n_val]
    test_samples  = pothole_samples[n_train + n_val :]

    print(f"\nImporting {len(train_samples)} training, {len(val_samples)} validation, and {len(test_samples)} test samples...")

    def copy_samples(samples, img_dest, lbl_dest):
        for img_path, label_lines in samples:
            # Copy image
            target_img = img_dest / img_path.name
            if not target_img.exists():
                shutil.copy2(img_path, target_img)
            # Write remap label
            target_lbl = lbl_dest / f"{img_path.stem}.txt"
            with open(target_lbl, "w", encoding="utf-8") as f:
                f.writelines(label_lines)

    copy_samples(train_samples, train_img_dir, train_lbl_dir)
    copy_samples(val_samples, val_img_dir, val_lbl_dir)
    copy_samples(test_samples, test_img_dir, test_lbl_dir)

    # Also copy to raw dataset folder so pipeline scripts can find it
    raw_dest = PROJECT_ROOT / "dataset" / "rdd2022" / "India" / "train"
    raw_dest_img = raw_dest / "images"
    raw_dest_lbl = raw_dest / "labels"
    raw_dest_img.mkdir(parents=True, exist_ok=True)
    raw_dest_lbl.mkdir(parents=True, exist_ok=True)

    print("Populating dataset/rdd2022/ folder for reference...")
    for img_path, label_lines in pothole_samples[:100]:
        shutil.copy2(img_path, raw_dest_img / img_path.name)
        with open(raw_dest_lbl / f"{img_path.stem}.txt", "w", encoding="utf-8") as f:
            f.writelines(label_lines)

    # Create dataset.yaml
    dataset_yaml_content = f"""# Ultralytics YOLOv8 Dataset Configuration
path: {dest_base.resolve().as_posix()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: pothole
"""
    yaml_path = dest_base / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(dataset_yaml_content)

    print(f"\n[OK] dataset.yaml generated at: {yaml_path}")
    print("\n" + "=" * 65)
    print(" DATASET READY FOR YOLO TRAINING!")
    print("=" * 65)
    print(f"  Training Images:   {len(train_samples)}")
    print(f"  Validation Images: {len(val_samples)}")
    print(f"  Classes:           1 ('pothole')")
    print(f"  Config YAML:       {yaml_path}")
    print("\nYou can now start training with:")
    print("  python ml/train_yolo.py")
    print("=" * 65)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import downloaded RDD2022 dataset")
    parser.add_argument(
        "--source",
        default=r"C:\Users\kakum\Downloads\RD2022\RDD_SPLIT",
        help="Path to downloaded RDD2022 folder",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional limit on images to import (0 = all pothole images)",
    )
    args = parser.parse_args()
    import_dataset(args.source, args.max_images)
