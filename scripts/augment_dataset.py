"""
scripts/augment_dataset.py
==========================
Step 5 of the pipeline: Apply data augmentations to resized images.

Why augment?
  More training examples → better generalisation.
  Instead of collecting more data, we synthetically vary existing images
  using transformations that preserve the semantic content (it is still
  a pothole, just under different conditions).

Augmentations applied (from config.yaml):
  1. flip       — horizontal mirror          (YOLO label: flip x_center)
  2. brightness — random brightness jitter   (YOLO label: unchanged)
  3. scale      — random zoom in / out       (YOLO label: unchanged *)
  4. crop       — crop 10% from each edge    (YOLO label: unchanged *)

  * Scale and crop augmentations in this script are applied image-only.
    The YOLO label files are copied as-is, which is an approximation
    valid for small scale/crop amounts (±20% / 10%).

Label handling for flip:
  If the original YOLO box is (class_id, x_c, y_c, w, h), then after a
  horizontal flip: x_c becomes (1.0 - x_c). All other values stay the same.

Run from project root:
  python scripts/augment_dataset.py

Input  : data/processed/resized/images/     + labels/
Output : data/processed/augmented/images/   + labels/
"""

import sys
import shutil
import logging
import random
from pathlib import Path
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
from utils.preprocessing import flip_image, adjust_brightness, scale_image, crop_image

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install pillow")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# YOLO LABEL HELPERS
# ──────────────────────────────────────────────────────────────

def flip_yolo_labels(label_path: Path, output_path: Path) -> None:
    """
    Read a YOLO label file, flip x_center for each box, and write output.

    For a horizontal flip:
      new_x_center = 1.0 - original_x_center
      y_center, width, height are unchanged.

    Parameters
    ----------
    label_path : Path
        Source YOLO .txt file.
    output_path : Path
        Destination .txt file for flipped labels.
    """
    if not label_path.exists():
        return  # No label file → nothing to flip

    flipped_lines = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue  # Malformed line — skip
                class_id = parts[0]
                x_c   = 1.0 - float(parts[1])   # ← only change!
                y_c   = float(parts[2])
                w     = float(parts[3])
                h     = float(parts[4])
                flipped_lines.append(
                    f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                )
    except Exception:
        return

    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(flipped_lines) + "\n")


def _read_yolo_labels(label_path: Path) -> list:
    if not label_path.exists():
        return []
    rows = []
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    except Exception:
        return []
    return rows


def _write_yolo_labels(rows: list, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        for cls, xc, yc, w, h in rows:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def _transform_labels(rows: list, transform: str, factor: float = 1.0, crop_fraction: float = 0.1) -> list:
    """Transform normalized YOLO boxes to match the image augmentation."""
    out = []
    for cls, xc, yc, w, h in rows:
        if transform == "flip":
            out.append([cls, 1.0 - xc, yc, w, h])
            continue

        if transform == "brightness":
            out.append([cls, xc, yc, w, h])
            continue

        if transform == "scale":
            # Source and destination are square target_size canvases.
            if factor >= 1.0:
                shift = (factor - 1.0) / 2.0
                nx = xc * factor - shift
                ny = yc * factor - shift
            else:
                shift = (1.0 - factor) / 2.0
                nx = xc * factor + shift
                ny = yc * factor + shift
            nw = w * factor
            nh = h * factor
        elif transform == "crop":
            keep = 1.0 - 2.0 * crop_fraction
            if keep <= 0:
                continue
            nx = (xc - crop_fraction) / keep
            ny = (yc - crop_fraction) / keep
            nw = w / keep
            nh = h / keep
        else:
            out.append([cls, xc, yc, w, h])
            continue

        # Convert center/size to corners, clip to the image, then back to YOLO.
        x1 = max(0.0, nx - nw / 2.0)
        y1 = max(0.0, ny - nh / 2.0)
        x2 = min(1.0, nx + nw / 2.0)
        y2 = min(1.0, ny + nh / 2.0)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append([cls, (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])
    return out


# ──────────────────────────────────────────────────────────────
# MAIN AUGMENTATION FUNCTION
# ──────────────────────────────────────────────────────────────

def augment_dataset(config: dict, logger: logging.Logger) -> dict:
    """
    Apply all augmentations to resized images and save results.

    Parameters
    ----------
    config : dict
        Loaded config.yaml.
    logger : logging.Logger

    Returns
    -------
    dict
        Augmentation report.
    """
    processed_path = resolve_path(config["dataset"]["processed_path"])
    target_size    = config["model"]["image_size"]

    src_images_dir  = processed_path / "resized" / "images"
    src_labels_dir  = processed_path / "resized" / "labels"
    dest_images_dir = processed_path / "augmented" / "images"
    dest_labels_dir = processed_path / "augmented" / "labels"

    # ── Check source exists ───────────────────────────────────
    if not src_images_dir.exists():
        logger.warning(
            f"Source directory not found: {src_images_dir}\n"
            "  Run resize_images.py first."
        )
        return {
            "status": "NO_SOURCE_DIR",
            "original_count": 0,
            "augmented_count": 0,
            "total_count": 0,
            "timestamp": datetime.now().isoformat(),
        }

    ensure_dir(dest_images_dir)
    ensure_dir(dest_labels_dir)

    src_images = list_images(src_images_dir)
    logger.info(f"Source images found : {len(src_images)}")

    if not src_images:
        logger.warning("No images found in resized directory.")
        return {
            "status": "NO_IMAGES",
            "original_count": 0,
            "augmented_count": 0,
            "total_count": 0,
            "timestamp": datetime.now().isoformat(),
        }

    original_count  = 0
    augmented_count = 0
    failed_count    = 0

    for i, img_path in enumerate(src_images, start=1):
        stem = img_path.stem
        src_label = src_labels_dir / (stem + ".txt")

        # ── 1. Copy original image and label ──────────────────
        dest_orig_img   = dest_images_dir / img_path.name
        dest_orig_label = dest_labels_dir / (stem + ".txt")
        try:
            shutil.copy2(img_path, dest_orig_img)
            _write_yolo_labels(_read_yolo_labels(src_label), dest_orig_label)
            original_count += 1
        except Exception as e:
            logger.warning(f"Could not copy original {img_path.name}: {e}")
            failed_count += 1
            continue

        # ── 2. Apply augmentations ────────────────────────────
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                aug_results = {}
                aug_cfg = config.get("augmentation", {})
                if aug_cfg.get("horizontal_flip", True):
                    aug_results["flip"] = (flip_image(img), None)
                aug_results["brightness"] = (
                    adjust_brightness(img, tuple(aug_cfg.get("brightness_range", [0.7, 1.3]))), None
                )
                scale_range = tuple(aug_cfg.get("scale_range", [0.8, 1.2]))
                scale_factor = random.uniform(*scale_range)
                aug_results["scale"] = (scale_image(img, scale_range=scale_range, target_size=target_size, factor=scale_factor), scale_factor)
                crop_fraction = float(aug_cfg.get("crop_fraction", 0.1))
                aug_results["crop"] = (crop_image(img, crop_fraction=crop_fraction, target_size=target_size), crop_fraction)
        except Exception as e:
            logger.warning(f"Could not open {img_path.name} for augmentation: {e}")
            failed_count += 1
            continue

        source_labels = _read_yolo_labels(src_label)

        # ── 3. Save each augmented variant ───────────────────
        for aug_name, (aug_img, meta) in aug_results.items():
            aug_img_name   = f"{stem}__{aug_name}.jpg"
            aug_label_name = f"{stem}__{aug_name}.txt"
            dest_aug_img   = dest_images_dir / aug_img_name
            dest_aug_label = dest_labels_dir / aug_label_name

            try:
                aug_img.save(dest_aug_img, "JPEG", quality=90)
                if aug_name == "flip":
                    transformed = _transform_labels(source_labels, "flip")
                elif aug_name == "scale":
                    transformed = _transform_labels(source_labels, "scale", factor=float(meta))
                elif aug_name == "crop":
                    transformed = _transform_labels(source_labels, "crop", crop_fraction=float(meta))
                else:
                    transformed = _transform_labels(source_labels, "brightness")
                _write_yolo_labels(transformed, dest_aug_label)
            except Exception as e:
                logger.warning(f"Could not save augmented pair {aug_img_name}: {e}")
                continue

            augmented_count += 1

        if i % 100 == 0 or i == len(src_images):
            logger.info(f"  Processed {i}/{len(src_images)} originals ...")

    total_count = original_count + augmented_count
    logger.info(f"Original images : {original_count}")
    logger.info(f"Augmented images: {augmented_count}")
    logger.info(f"Total in output : {total_count}")
    logger.info(f"Failed          : {failed_count}")

    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "source_directory": str(src_images_dir),
        "output_directory": str(dest_images_dir),
        "original_count": original_count,
        "augmented_count": augmented_count,
        "total_count": total_count,
        "failed": failed_count,
        "augmentations_applied": ["flip", "brightness", "scale", "crop"] if config.get("augmentation", {}).get("horizontal_flip", True) else ["brightness", "scale", "crop"],
    }


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    """
    Entry point: load config, augment dataset, save report, print summary.
    """
    logger = setup_logging("augment_dataset")
    logger.info("=" * 60)
    logger.info("  STEP 5: DATA AUGMENTATION")
    logger.info("=" * 60)

    try:
        config = load_config()
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    report = augment_dataset(config, logger)

    output_path = resolve_path(config["dataset"]["outputs_path"]) / "augment_report.json"
    save_report(report, output_path)

    print("\n" + "=" * 50)
    print("  AUGMENTATION REPORT SUMMARY")
    print("=" * 50)
    print(f"  Status             : {report['status']}")
    print(f"  Original images    : {report.get('original_count', 0)}")
    print(f"  Augmented images   : {report.get('augmented_count', 0)}")
    print(f"  Total in output    : {report.get('total_count', 0)}")
    print(f"  Report saved to    : {output_path}")
    print("=" * 50 + "\n")

    return report


if __name__ == "__main__":
    main()
