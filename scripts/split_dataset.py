"""
scripts/split_dataset.py
========================
Step 6 of the pipeline: Split augmented images into train / val / test sets.

Ratios (from config.yaml):
  Train : 70%
  Val   : 20%
  Test  : 10%

IMPORTANT — No data leakage:
  Augmented images are variants of original images. If we naively split
  all files randomly, augmented variants of the same original might end
  up in both train and test sets, which inflates performance metrics.

  Solution: We group files by their *base* image stem (i.e., the part of
  the filename before '__aug_name'), then split the groups. All augmented
  variants of the same original always stay in the same split.

  Example:
    img001.jpg, img001__flip.jpg, img001__brightness.jpg
    → all three go to the SAME split (train / val / test)

Output structure:
  data/processed/split/
    train/images/  train/labels/
    val/images/    val/labels/
    test/images/   test/labels/

Also creates:
  data/processed/split/dataset.yaml  (for YOLOv8 training)

Run from project root:
  python scripts/split_dataset.py
"""

import sys
import shutil
import random
import logging
import yaml
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── Project root on path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import (
    load_config,
    resolve_path,
    ensure_dir,
    save_report,
    setup_logging,
    list_images,
)


# ──────────────────────────────────────────────────────────────
# GROUPING HELPER
# ──────────────────────────────────────────────────────────────

def get_base_stem(filename: str) -> str:
    """
    Extract the base image stem from an augmented filename.

    Augmented files are named:  originalname__augtype.jpg
    Original files are named:   originalname.jpg

    The separator '__' (double underscore) is used to distinguish
    augmented variants from base images.

    Parameters
    ----------
    filename : str
        The image filename (without directory).

    Returns
    -------
    str
        The base image identifier (before '__').

    Examples
    --------
    >>> get_base_stem("img001.jpg")
    'img001'
    >>> get_base_stem("img001__flip.jpg")
    'img001'
    >>> get_base_stem("img001__brightness.jpg")
    'img001'
    """
    stem = Path(filename).stem           # Remove extension
    if "__" in stem:
        return stem.split("__")[0]       # Take part before first '__'
    return stem


# ──────────────────────────────────────────────────────────────
# COPY HELPERS
# ──────────────────────────────────────────────────────────────

def copy_split_files(
    file_list: list,
    src_images_dir: Path,
    src_labels_dir: Path,
    dest_images_dir: Path,
    dest_labels_dir: Path,
    logger: logging.Logger,
) -> int:
    """
    Copy images and their YOLO label files to a split destination directory.

    Parameters
    ----------
    file_list : list[Path]
        List of image Paths to copy.
    src_images_dir : Path
        Source directory for images.
    src_labels_dir : Path
        Source directory for YOLO .txt label files.
    dest_images_dir : Path
        Destination images/ directory.
    dest_labels_dir : Path
        Destination labels/ directory.
    logger : logging.Logger

    Returns
    -------
    int
        Number of images successfully copied.
    """
    ensure_dir(dest_images_dir)
    ensure_dir(dest_labels_dir)

    count = 0
    for img_path in file_list:
        # Copy image
        try:
            shutil.copy2(img_path, dest_images_dir / img_path.name)
        except Exception as e:
            logger.warning(f"Could not copy image {img_path.name}: {e}")
            continue

        # Copy label (if it exists)
        label_src = src_labels_dir / (img_path.stem + ".txt")
        if label_src.exists():
            try:
                shutil.copy2(label_src, dest_labels_dir / label_src.name)
            except Exception as e:
                logger.warning(f"Could not copy label {label_src.name}: {e}")

        count += 1

    return count


# ──────────────────────────────────────────────────────────────
# DATASET.YAML GENERATOR
# ──────────────────────────────────────────────────────────────

def create_dataset_yaml(split_dir: Path, config: dict) -> None:
    """
    Create a dataset.yaml file for YOLOv8 training.

    YOLOv8 reads this file to know where the train/val/test images are
    and what classes to detect.

    Parameters
    ----------
    split_dir : Path
        The root of the split directory (contains train/, val/, test/).
    config : dict
        Loaded config.yaml.
    """
    dataset_yaml = {
        "path": str(split_dir),                          # Absolute dataset root
        "train": "train/images",                         # Relative to path
        "val":   "val/images",
        "test":  "test/images",
        "nc":    1,                                       # Number of classes
        "names": ["pothole"],                             # Class names
    }

    yaml_path = split_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"  dataset.yaml created at: {yaml_path}")


# ──────────────────────────────────────────────────────────────
# MAIN SPLIT FUNCTION
# ──────────────────────────────────────────────────────────────

def split_dataset(config: dict, logger: logging.Logger) -> dict:
    """
    Split augmented (or resized) images into train/val/test without leakage.

    Parameters
    ----------
    config : dict
        Loaded config.yaml.
    logger : logging.Logger

    Returns
    -------
    dict
        Split report with counts per subset.
    """
    processed_path = resolve_path(config["dataset"]["processed_path"])

    # ── Determine source (augmented preferred, resized as fallback) ───
    aug_images_dir    = processed_path / "augmented" / "images"
    aug_labels_dir    = processed_path / "augmented" / "labels"
    resized_images_dir = processed_path / "resized" / "images"
    resized_labels_dir = processed_path / "resized" / "labels"

    if aug_images_dir.exists() and list_images(aug_images_dir):
        src_images_dir = aug_images_dir
        src_labels_dir = aug_labels_dir
        logger.info("Using AUGMENTED images as source.")
    elif resized_images_dir.exists() and list_images(resized_images_dir):
        src_images_dir = resized_images_dir
        src_labels_dir = resized_labels_dir
        logger.info("Augmented directory empty — falling back to RESIZED images.")
    else:
        logger.warning(
            "No source images found in augmented/ or resized/ directories.\n"
            "  Run augment_dataset.py (or resize_images.py) first."
        )
        return {
            "status": "NO_SOURCE_IMAGES",
            "train": 0, "val": 0, "test": 0, "total": 0,
            "timestamp": datetime.now().isoformat(),
        }

    split_dir = processed_path / "split"

    # ── Read split ratios ─────────────────────────────────────
    train_ratio = config["dataset"]["split"]["train"]
    val_ratio   = config["dataset"]["split"]["val"]
    # test_ratio is implicitly 1 - train - val

    # ── Collect images ────────────────────────────────────────
    all_images = list_images(src_images_dir)
    logger.info(f"Total images to split: {len(all_images)}")

    if not all_images:
        logger.warning("No images found in source directory.")
        return {
            "status": "NO_IMAGES",
            "train": 0, "val": 0, "test": 0, "total": 0,
            "timestamp": datetime.now().isoformat(),
        }

    # ── Group images by base stem (anti-leakage) ──────────────
    groups = defaultdict(list)  # base_stem → [Path, Path, ...]
    for img_path in all_images:
        base = get_base_stem(img_path.name)
        groups[base].append(img_path)

    base_stems = sorted(groups.keys())
    logger.info(f"Unique base images (groups): {len(base_stems)}")

    # ── Shuffle deterministically ─────────────────────────────
    random.seed(42)              # Fixed seed for reproducibility
    random.shuffle(base_stems)

    # ── Calculate split indices ───────────────────────────────
    n = len(base_stems)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    # Remainder goes to test

    train_stems = base_stems[:n_train]
    val_stems   = base_stems[n_train : n_train + n_val]
    test_stems  = base_stems[n_train + n_val :]

    logger.info(f"Base groups → train: {len(train_stems)}, val: {len(val_stems)}, test: {len(test_stems)}")

    # ── Build per-split image lists ───────────────────────────
    def stems_to_images(stems):
        imgs = []
        for s in stems:
            imgs.extend(groups[s])
        return imgs

    train_images = stems_to_images(train_stems)
    val_images   = stems_to_images(val_stems)
    test_images  = stems_to_images(test_stems)

    # ── Copy files ────────────────────────────────────────────
    logger.info("Copying train split ...")
    train_count = copy_split_files(
        train_images, src_images_dir, src_labels_dir,
        split_dir / "train" / "images", split_dir / "train" / "labels", logger
    )
    logger.info("Copying val split ...")
    val_count = copy_split_files(
        val_images, src_images_dir, src_labels_dir,
        split_dir / "val" / "images", split_dir / "val" / "labels", logger
    )
    logger.info("Copying test split ...")
    test_count = copy_split_files(
        test_images, src_images_dir, src_labels_dir,
        split_dir / "test" / "images", split_dir / "test" / "labels", logger
    )

    total = train_count + val_count + test_count
    logger.info(f"Split complete → train: {train_count}, val: {val_count}, test: {test_count}, total: {total}")

    # ── Create dataset.yaml ───────────────────────────────────
    create_dataset_yaml(split_dir, config)

    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "source_directory": str(src_images_dir),
        "split_directory": str(split_dir),
        "dataset_yaml": str(split_dir / "dataset.yaml"),
        "unique_base_images": n,
        "train": train_count,
        "val": val_count,
        "test": test_count,
        "total": total,
    }


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    """
    Entry point: load config, split dataset, save report, print summary.
    """
    logger = setup_logging("split_dataset")
    logger.info("=" * 60)
    logger.info("  STEP 6: SPLIT DATASET (TRAIN / VAL / TEST)")
    logger.info("=" * 60)

    try:
        config = load_config()
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    report = split_dataset(config, logger)

    output_path = resolve_path(config["dataset"]["outputs_path"]) / "split_report.json"
    save_report(report, output_path)

    print("\n" + "=" * 50)
    print("  DATASET SPLIT SUMMARY")
    print("=" * 50)
    print(f"  Status        : {report['status']}")
    print(f"  Train images  : {report.get('train', 0)}")
    print(f"  Val images    : {report.get('val', 0)}")
    print(f"  Test images   : {report.get('test', 0)}")
    print(f"  Total         : {report.get('total', 0)}")
    if report.get("dataset_yaml"):
        print(f"  dataset.yaml  : {report['dataset_yaml']}")
    print(f"  Report saved  : {output_path}")
    print("=" * 50 + "\n")

    return report


if __name__ == "__main__":
    main()
