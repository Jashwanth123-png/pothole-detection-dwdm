"""
scripts/resize_images.py
========================
Step 4 of the pipeline: Resize all D40 images to 416 × 416 using letterboxing.

Why letterboxing?
  A simple stretch resize would distort objects (potholes could look thinner
  or fatter than reality). Letterboxing maintains the original aspect ratio
  by:
    1. Scaling the image so its longest side = 416 px
    2. Padding the shorter side with grey pixels (128, 128, 128)

This is the exact preprocessing YOLO models expect during training.

Run from project root:
  python scripts/resize_images.py

Input  : data/processed/d40/images/     (original D40 images)
Output : data/processed/resized/images/ (416×416 letterboxed images)

Note: YOLO .txt label files are copied unchanged because letterboxing
does NOT change bounding box normalised coordinates — the padding area
simply has no labels.
"""

import sys
import shutil
import logging
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
from utils.preprocessing import letterbox_resize

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install pillow")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# MAIN RESIZE FUNCTION
# ──────────────────────────────────────────────────────────────

def resize_images(config: dict, logger: logging.Logger) -> dict:
    """
    Letterbox-resize all images in data/processed/d40/images/ to 416×416.

    Also copies matching YOLO .txt label files to the output directory.

    Parameters
    ----------
    config : dict
        Loaded config.yaml.
    logger : logging.Logger

    Returns
    -------
    dict
        Resize report with success/failure counts.
    """
    processed_path = resolve_path(config["dataset"]["processed_path"])
    target_size    = config["model"]["image_size"]  # 416 from config

    src_images_dir  = processed_path / "d40" / "images"
    src_labels_dir  = processed_path / "d40" / "labels"
    dest_images_dir = processed_path / "resized" / "images"
    dest_labels_dir = processed_path / "resized" / "labels"

    # ── Check source exists ───────────────────────────────────
    if not src_images_dir.exists():
        logger.warning(
            f"Source images directory not found: {src_images_dir}\n"
            "  Run filter_d40.py first."
        )
        return {
            "status": "NO_SOURCE_DIR",
            "resized": 0,
            "failed": 0,
            "timestamp": datetime.now().isoformat(),
        }

    # ── Create output directories ─────────────────────────────
    ensure_dir(dest_images_dir)
    ensure_dir(dest_labels_dir)
    logger.info(f"Target size         : {target_size} × {target_size}")
    logger.info(f"Source images       : {src_images_dir}")
    logger.info(f"Destination images  : {dest_images_dir}")

    # ── Collect source images ─────────────────────────────────
    src_images = list_images(src_images_dir)
    logger.info(f"Images to resize    : {len(src_images)}")

    if not src_images:
        logger.warning("No images found in source directory.")
        return {
            "status": "NO_IMAGES",
            "resized": 0,
            "failed": 0,
            "timestamp": datetime.now().isoformat(),
        }

    # ── Resize loop ───────────────────────────────────────────
    resized_count = 0
    failed_count  = 0
    failed_files  = []

    for i, img_path in enumerate(src_images, start=1):
        dest_img = dest_images_dir / img_path.name

        try:
            with Image.open(img_path) as img:
                resized = letterbox_resize(img, target_size=target_size)
            resized.save(dest_img, "JPEG", quality=95)
            resized_count += 1
        except Exception as e:
            logger.warning(f"Failed to resize {img_path.name}: {e}")
            failed_count += 1
            failed_files.append({"file": img_path.name, "error": str(e)})
            continue

        # ── Copy matching label file (YOLO .txt) ──────────────
        label_src = src_labels_dir / (img_path.stem + ".txt")
        label_dst = dest_labels_dir / (img_path.stem + ".txt")
        if label_src.exists():
            shutil.copy2(label_src, label_dst)

        if i % 200 == 0 or i == len(src_images):
            logger.info(f"  Resized {i}/{len(src_images)} ...")

    logger.info(f"Resized  : {resized_count} images")
    logger.info(f"Failed   : {failed_count} images")

    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "source_directory": str(src_images_dir),
        "output_directory": str(dest_images_dir),
        "target_size": target_size,
        "total_source_images": len(src_images),
        "resized": resized_count,
        "failed": failed_count,
        "failed_files": failed_files,
    }


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    """
    Entry point: load config, resize images, save report, print summary.
    """
    logger = setup_logging("resize_images")
    logger.info("=" * 60)
    logger.info("  STEP 4: RESIZE IMAGES (LETTERBOX → 416×416)")
    logger.info("=" * 60)

    try:
        config = load_config()
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    report = resize_images(config, logger)

    output_path = resolve_path(config["dataset"]["outputs_path"]) / "resize_report.json"
    save_report(report, output_path)

    print("\n" + "=" * 50)
    print("  RESIZE REPORT SUMMARY")
    print("=" * 50)
    print(f"  Status         : {report['status']}")
    print(f"  Source images  : {report.get('total_source_images', 0)}")
    print(f"  Successfully resized: {report.get('resized', 0)}")
    print(f"  Failed         : {report.get('failed', 0)}")
    print(f"  Target size    : {report.get('target_size', 416)}×{report.get('target_size', 416)}")
    print(f"  Report saved to: {output_path}")
    print("=" * 50 + "\n")

    return report


if __name__ == "__main__":
    main()
