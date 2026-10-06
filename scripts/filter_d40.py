"""
scripts/filter_d40.py
=====================
Step 2 of the pipeline: Filter images that contain D40 (Pothole) annotations.

The RDD2022 dataset contains multiple road damage categories:
  D00 - Longitudinal Crack
  D10 - Transverse Crack
  D20 - Alligator Crack
  D40 - Pothole  ← This is what we want!
  (and others depending on country)

What this script does:
  1. Scans the RDD2022 dataset for annotation files (XML or TXT format)
  2. Parses each annotation to check if it contains a D40 label
  3. Copies qualifying images + annotations to data/processed/d40/
  4. Reports statistics

Run from the project root:
  python scripts/filter_d40.py

Output:
  - data/processed/d40/images/   ← filtered JPG images
  - data/processed/d40/labels/   ← corresponding annotation files
"""

import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime

# ── Add project root to path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import (
    load_config,
    resolve_path,
    ensure_dir,
    save_report,
    setup_logging,
    parse_voc_xml,
    IMAGE_EXTENSIONS,
)

# The target damage class we are filtering for
D40_CLASS_NAME = "D40"


# ──────────────────────────────────────────────────────────────
# ANNOTATION FINDERS
# ──────────────────────────────────────────────────────────────

def find_annotation_for_image(image_path: Path) -> tuple:
    """
    Given an image path, look for a matching annotation file.

    RDD2022 stores annotations in Annotations/ folders alongside Images/.
    This function tries both XML (Pascal VOC) and TXT formats.

    Parameters
    ----------
    image_path : Path
        Full path to the image file.

    Returns
    -------
    tuple[Path or None, str]
        (annotation_path, format_string) where format_string is "xml" or "txt",
        or (None, "") if no annotation is found.
    """
    stem = image_path.stem  # filename without extension

    # Common RDD2022 layout: Images/ and Annotations/ are siblings
    parent = image_path.parent
    ann_candidates = [
        # Same folder
        parent / f"{stem}.xml",
        parent / f"{stem}.txt",
        # Sibling 'Annotations' folder at same level as 'images'
        parent.parent / "Annotations" / f"{stem}.xml",
        parent.parent / "Annotations" / f"{stem}.txt",
        parent.parent / "annotations" / f"{stem}.xml",
        parent.parent / "annotations" / f"{stem}.txt",
        parent.parent / "labels" / f"{stem}.txt",
        parent.parent / "Labels" / f"{stem}.txt",
    ]

    for ann_path in ann_candidates:
        if ann_path.exists():
            fmt = "xml" if ann_path.suffix.lower() == ".xml" else "txt"
            return ann_path, fmt

    return None, ""


def has_d40_xml(xml_path: Path) -> tuple:
    """
    Check if an XML annotation file contains at least one D40 object.

    Parameters
    ----------
    xml_path : Path
        Path to the Pascal VOC XML file.

    Returns
    -------
    tuple[bool, int]
        (has_d40, d40_count) — True if D40 found, and count of D40 boxes.
    """
    data = parse_voc_xml(xml_path)
    if not data:
        return False, 0

    d40_objects = [obj for obj in data.get("objects", []) if obj["name"] == D40_CLASS_NAME]
    return len(d40_objects) > 0, len(d40_objects)


def has_d40_txt(txt_path: Path) -> tuple:
    """
    Check if a plain-text annotation file mentions D40 or has class 0/3.

    Supports both formats:
      - Named: D40 xmin ymin xmax ymax
      - YOLO:  <cls_id> x_center y_center width height (cls_id 3 in RDD2022 or 0 remapped)

    Parameters
    ----------
    txt_path : Path
        Path to the text annotation file.

    Returns
    -------
    tuple[bool, int]
        (has_d40, d40_count)
    """
    count = 0
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                first = parts[0].upper()
                if first == D40_CLASS_NAME or first in {"0", "3"}:
                    count += 1
    except Exception:
        return False, 0

    return count > 0, count


# ──────────────────────────────────────────────────────────────
# MAIN FILTERING FUNCTION
# ──────────────────────────────────────────────────────────────

def filter_d40(config: dict, logger: logging.Logger) -> dict:
    """
    Filter all images containing D40 annotations and copy them to the
    processed directory.

    Parameters
    ----------
    config : dict
        Loaded config.yaml dictionary.
    logger : logging.Logger
        Logger for console output.

    Returns
    -------
    dict
        Report with counts and list of copied files.
    """
    rdd_path = resolve_path(config["dataset"]["rdd2022_path"])
    out_images = resolve_path(config["dataset"]["processed_path"]) / "d40" / "images"
    out_labels = resolve_path(config["dataset"]["processed_path"]) / "d40" / "labels"

    # ── Validate dataset ──────────────────────────────────────
    if not rdd_path.exists():
        logger.error(
            f"\n{'='*60}\n"
            f"  DATASET NOT FOUND: {rdd_path}\n"
            f"{'='*60}\n"
            "  Please place the RDD2022 dataset folder here.\n"
            f"{'='*60}"
        )
        return {
            "status": "DATASET_NOT_FOUND",
            "total_images": 0,
            "images_with_d40": 0,
            "d40_annotation_count": 0,
            "timestamp": datetime.now().isoformat(),
        }

    # ── Create output directories ─────────────────────────────
    ensure_dir(out_images)
    ensure_dir(out_labels)
    logger.info(f"Output (images)     : {out_images}")
    logger.info(f"Output (labels/ann) : {out_labels}")

    # ── Find all image files recursively ─────────────────────
    all_images = sorted(
        p for p in rdd_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info(f"Total images scanned: {len(all_images)}")

    # ── Filter loop ───────────────────────────────────────────
    copied_images = []
    total_d40_count = 0
    no_annotation = 0
    processed = 0

    for img_path in all_images:
        # Find the matching annotation file
        ann_path, ann_fmt = find_annotation_for_image(img_path)

        if ann_path is None:
            no_annotation += 1
            continue

        # Check if it contains D40
        if ann_fmt == "xml":
            found, count = has_d40_xml(ann_path)
        else:  # "txt"
            found, count = has_d40_txt(ann_path)

        if not found:
            continue

        # ── Copy image ──
        dest_img = out_images / img_path.name
        try:
            shutil.copy2(img_path, dest_img)
        except Exception as e:
            logger.warning(f"Could not copy image {img_path.name}: {e}")
            continue

        # ── Copy annotation ──
        dest_ann = out_labels / ann_path.name
        try:
            shutil.copy2(ann_path, dest_ann)
        except Exception as e:
            logger.warning(f"Could not copy annotation {ann_path.name}: {e}")

        total_d40_count += count
        copied_images.append(img_path.name)
        processed += 1

        if processed % 100 == 0:
            logger.info(f"  Copied {processed} D40 images so far ...")

    logger.info(f"Images with D40     : {len(copied_images)}")
    logger.info(f"Total D40 boxes     : {total_d40_count}")
    logger.info(f"Images w/o ann file : {no_annotation}")

    report = {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(rdd_path),
        "output_images_dir": str(out_images),
        "output_labels_dir": str(out_labels),
        "total_images_scanned": len(all_images),
        "images_without_annotation": no_annotation,
        "images_with_d40": len(copied_images),
        "d40_annotation_count": total_d40_count,
        "copied_images": copied_images,
    }

    return report


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    """
    Entry point: load config, run D40 filter, save report, print summary.
    """
    logger = setup_logging("filter_d40")
    logger.info("=" * 60)
    logger.info("  STEP 2: FILTER D40 (POTHOLE) IMAGES")
    logger.info("=" * 60)

    try:
        config = load_config()
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    report = filter_d40(config, logger)

    output_path = resolve_path(config["dataset"]["outputs_path"]) / "filter_d40_report.json"
    save_report(report, output_path)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  FILTER D40 REPORT SUMMARY")
    print("=" * 50)
    print(f"  Status              : {report['status']}")
    print(f"  Total images scanned: {report.get('total_images_scanned', 0)}")
    print(f"  Images with D40     : {report.get('images_with_d40', 0)}")
    print(f"  D40 annotation boxes: {report.get('d40_annotation_count', 0)}")
    print(f"  Report saved to     : {output_path}")
    print("=" * 50 + "\n")

    return report


if __name__ == "__main__":
    main()
