"""
scripts/prepare_dataset.py
==========================
MASTER PIPELINE SCRIPT — Runs all preprocessing steps in order.

This is the single script to run if you want to go from raw RDD2022
dataset all the way to a fully prepared train/val/test split.

Steps executed:
  1. clean_dataset      — Validate images, detect duplicates
  2. filter_d40         — Keep only pothole (D40) images
  3. convert_annotations — Convert XML → YOLO format
  4. resize_images      — Letterbox resize to 416×416
  5. augment_dataset    — Flip, brightness, scale, crop augmentations
  6. split_dataset      — Train / Val / Test split (70/20/10)
  7. Generate final preparation report → data/outputs/preparation_report.json
  8. Print a summary table to the console

Usage:
  python scripts/prepare_dataset.py

If the RDD2022 dataset is missing, the script prints clear instructions
and exits gracefully without crashing.

Expected dataset placement:
  project_pathhole/
    dataset/
      rdd2022/          ← Place RDD2022 contents here
        <country>/
          train/
            images/
            annotations/  (or Annotations/)
"""

import sys
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
    save_report,
    setup_logging,
)

# Import each step's main processing function
# (not their main() — we want just the logic, not sys.exit)
from scripts.clean_dataset       import clean_dataset
from scripts.filter_d40          import filter_d40
from scripts.convert_annotations import convert_annotations
from scripts.resize_images       import resize_images
from scripts.augment_dataset     import augment_dataset
from scripts.split_dataset       import split_dataset


# ──────────────────────────────────────────────────────────────
# CONSOLE TABLE HELPER
# ──────────────────────────────────────────────────────────────

def print_summary_table(reports: dict) -> None:
    """
    Print a formatted summary table of all pipeline steps to the console.

    Parameters
    ----------
    reports : dict
        Dictionary keyed by step name, values are the step report dicts.
    """
    width = 62

    print("\n")
    print("╔" + "═" * width + "╗")
    print("║" + "  🚀  POTHOLE DETECTION — DATASET PREPARATION REPORT  ".center(width) + "║")
    print("╠" + "═" * width + "╣")

    def row(label, value, indent=2):
        line = " " * indent + f"{label:<32} {str(value):>20}"
        print("║" + line.ljust(width) + "║")

    # Step 1: Cleaning
    clean = reports.get("clean", {})
    print("║" + "  STEP 1 — Dataset Cleaning".ljust(width) + "║")
    row("Total files scanned:",        clean.get("total_files", "N/A"))
    row("Valid images:",                clean.get("valid", "N/A"))
    row("Invalid / corrupt:",          clean.get("invalid", "N/A"))
    row("Duplicates found:",           clean.get("duplicates", "N/A"))
    print("╠" + "═" * width + "╣")

    # Step 2: Filtering
    filt = reports.get("filter", {})
    print("║" + "  STEP 2 — D40 (Pothole) Filter".ljust(width) + "║")
    row("Images scanned:",             filt.get("total_images_scanned", "N/A"))
    row("Images with D40 potholes:",   filt.get("images_with_d40", "N/A"))
    row("D40 annotation boxes:",       filt.get("d40_annotation_count", "N/A"))
    print("╠" + "═" * width + "╣")

    # Step 3: Conversion
    conv = reports.get("convert", {})
    print("║" + "  STEP 3 — Annotation Conversion (VOC → YOLO)".ljust(width) + "║")
    row("XML files converted:",        conv.get("converted", "N/A"))
    row("Skipped (no D40 / error):",   conv.get("skipped", "N/A"))
    row("Total D40 boxes written:",    conv.get("total_d40_boxes", "N/A"))
    print("╠" + "═" * width + "╣")

    # Step 4: Resizing
    rsz = reports.get("resize", {})
    print("║" + "  STEP 4 — Letterbox Resize (416×416)".ljust(width) + "║")
    row("Images resized:",             rsz.get("resized", "N/A"))
    row("Resize failures:",            rsz.get("failed", "N/A"))
    print("╠" + "═" * width + "╣")

    # Step 5: Augmentation
    aug = reports.get("augment", {})
    print("║" + "  STEP 5 — Data Augmentation".ljust(width) + "║")
    row("Original images:",            aug.get("original_count", "N/A"))
    row("Augmented variants created:", aug.get("augmented_count", "N/A"))
    row("Total after augmentation:",   aug.get("total_count", "N/A"))
    print("╠" + "═" * width + "╣")

    # Step 6: Split
    spl = reports.get("split", {})
    print("║" + "  STEP 6 — Train / Val / Test Split".ljust(width) + "║")
    row("Train images:",               spl.get("train", "N/A"))
    row("Validation images:",          spl.get("val", "N/A"))
    row("Test images:",                spl.get("test", "N/A"))
    row("Total split images:",         spl.get("total", "N/A"))
    print("╠" + "═" * width + "╣")

    # Dataset.yaml location
    dsy = spl.get("dataset_yaml", "")
    if dsy:
        print("║" + f"  dataset.yaml → {dsy}".ljust(width) + "║")
        print("╠" + "═" * width + "╣")

    print("║" + f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".ljust(width) + "║")
    print("╚" + "═" * width + "╝")
    print()


# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ──────────────────────────────────────────────────────────────

def run_pipeline(config: dict, logger: logging.Logger) -> dict:
    """
    Execute all preprocessing steps in order.

    Each step is wrapped in a try/except so a failure in one step
    does not stop the rest (though later steps may have less data).

    Parameters
    ----------
    config : dict
        Loaded config.yaml.
    logger : logging.Logger

    Returns
    -------
    dict
        Combined report from all steps.
    """
    reports = {}

    # ── STEP 1: Clean ─────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 1/6: Cleaning dataset ...")
    try:
        reports["clean"] = clean_dataset(config, logger)
    except Exception as e:
        logger.error(f"Step 1 (clean) failed: {e}")
        reports["clean"] = {"status": f"ERROR: {e}"}

    # ── Check if dataset was found before continuing ──────────
    if reports["clean"].get("status") == "DATASET_NOT_FOUND":
        rdd_path = resolve_path(config["dataset"]["rdd2022_path"])
        print(
            f"\n{'='*65}\n"
            f"  ❌  RDD2022 DATASET NOT FOUND\n"
            f"{'='*65}\n"
            f"  Expected location: {rdd_path}\n\n"
            "  How to fix:\n"
            "  1. Download RDD2022 from:\n"
            "       https://github.com/sekilab/RoadDamageDetector\n"
            "  2. Extract and place the dataset folder so the structure is:\n"
            f"       {rdd_path}/\n"
            "         <country>/train/images/\n"
            "         <country>/train/annotations/\n"
            f"{'='*65}\n"
        )
        # Still generate a partial report
        return {
            "status": "DATASET_NOT_FOUND",
            "source_images": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "d40_annotations": 0,
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
            "timestamp": datetime.now().isoformat(),
        }

    # ── STEP 2: Filter D40 ────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 2/6: Filtering D40 images ...")
    try:
        reports["filter"] = filter_d40(config, logger)
    except Exception as e:
        logger.error(f"Step 2 (filter) failed: {e}")
        reports["filter"] = {"status": f"ERROR: {e}"}

    # ── STEP 3: Convert annotations ───────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 3/6: Converting annotations (VOC XML → YOLO) ...")
    try:
        reports["convert"] = convert_annotations(config, logger)
    except Exception as e:
        logger.error(f"Step 3 (convert) failed: {e}")
        reports["convert"] = {"status": f"ERROR: {e}"}

    # ── STEP 4: Resize images ─────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 4/6: Resizing images to 416×416 ...")
    try:
        reports["resize"] = resize_images(config, logger)
    except Exception as e:
        logger.error(f"Step 4 (resize) failed: {e}")
        reports["resize"] = {"status": f"ERROR: {e}"}

    # ── STEP 5: Augment ───────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 5/6: Augmenting dataset ...")
    try:
        reports["augment"] = augment_dataset(config, logger)
    except Exception as e:
        logger.error(f"Step 5 (augment) failed: {e}")
        reports["augment"] = {"status": f"ERROR: {e}"}

    # ── STEP 6: Split ─────────────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("STEP 6/6: Splitting into train/val/test ...")
    try:
        reports["split"] = split_dataset(config, logger)
    except Exception as e:
        logger.error(f"Step 6 (split) failed: {e}")
        reports["split"] = {"status": f"ERROR: {e}"}

    return reports


# ──────────────────────────────────────────────────────────────
# FINAL REPORT BUILDER
# ──────────────────────────────────────────────────────────────

def build_final_report(reports: dict) -> dict:
    """
    Combine all step reports into a single final preparation report.

    Parameters
    ----------
    reports : dict
        Step reports keyed by step name.

    Returns
    -------
    dict
        Final consolidated report.
    """
    clean   = reports.get("clean",   {})
    filt    = reports.get("filter",  {})
    conv    = reports.get("convert", {})
    rsz     = reports.get("resize",  {})
    aug     = reports.get("augment", {})
    spl     = reports.get("split",   {})

    return {
        "timestamp": datetime.now().isoformat(),
        "pipeline_status": "COMPLETE",
        # Step 1
        "source_images":            clean.get("total_image_files", 0),
        "valid_images":             clean.get("valid", 0),
        "invalid_images":           clean.get("invalid", 0),
        "duplicate_images":         clean.get("duplicates", 0),
        # Step 2
        "d40_images":               filt.get("images_with_d40", 0),
        "d40_annotations":          filt.get("d40_annotation_count", 0),
        # Step 3
        "annotations_converted":    conv.get("converted", 0),
        "total_yolo_boxes":         conv.get("total_d40_boxes", 0),
        # Step 4
        "images_resized":           rsz.get("resized", 0),
        # Step 5
        "original_count":           aug.get("original_count", 0),
        "augmented_count":          aug.get("augmented_count", 0),
        "total_after_augmentation": aug.get("total_count", 0),
        # Step 6
        "train_count":              spl.get("train", 0),
        "val_count":                spl.get("val", 0),
        "test_count":               spl.get("test", 0),
        "total_split":              spl.get("total", 0),
        "dataset_yaml":             spl.get("dataset_yaml", ""),
        # Step reports (full detail)
        "step_reports": reports,
    }


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    """
    Master entry point: run all pipeline steps, save final report,
    print summary table.
    """
    logger = setup_logging("prepare_dataset")

    print()
    print("=" * 65)
    print("  POTHOLE DETECTION — FULL DATASET PREPARATION PIPELINE")
    print("=" * 65)
    print()

    # ── Load config ───────────────────────────────────────────
    try:
        config = load_config()
        logger.info("Configuration loaded.")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Run pipeline ──────────────────────────────────────────
    reports = run_pipeline(config, logger)

    # ── Build and save final report ───────────────────────────
    final_report = build_final_report(reports)
    output_path  = resolve_path(config["dataset"]["outputs_path"]) / "preparation_report.json"
    save_report(final_report, output_path)
    logger.info(f"Final preparation report saved → {output_path}")

    # ── Print console summary ─────────────────────────────────
    print_summary_table(reports)
    print(f"  Full report saved to: {output_path}")
    print()

    return final_report


if __name__ == "__main__":
    main()
