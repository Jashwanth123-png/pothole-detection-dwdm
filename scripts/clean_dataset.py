"""
scripts/clean_dataset.py
========================
Step 1 of the pipeline: Scan and validate the RDD2022 dataset.

What this script does:
  1. Reads the dataset directory path from config.yaml
  2. Recursively scans all image files in the dataset
  3. For each image, checks:
       a. Is it a supported image format?
       b. Is it a valid, non-corrupt image?
       c. Is it a duplicate of another image (using MD5 hash)?
  4. Produces a detailed cleaning report saved to:
       data/outputs/cleaning_report.json

Run this script from the project root:
  python scripts/clean_dataset.py

Output:
  - Console log with progress
  - data/outputs/cleaning_report.json with full audit details
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# ── Make sure the project root is on sys.path so we can import utils ──
# When running as "python scripts/clean_dataset.py", the CWD should be
# the project root. This block handles edge cases.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import (
    load_config,
    resolve_path,
    ensure_dir,
    compute_md5,
    is_valid_image,
    save_report,
    setup_logging,
    list_images_recursive,
    IMAGE_EXTENSIONS,
)


# ──────────────────────────────────────────────────────────────
# MAIN CLEANING FUNCTION
# ──────────────────────────────────────────────────────────────

def clean_dataset(config: dict, logger: logging.Logger) -> dict:
    """
    Scan the RDD2022 dataset and classify every image file.

    Parameters
    ----------
    config : dict
        Loaded config.yaml dictionary.
    logger : logging.Logger
        Logger for console output.

    Returns
    -------
    dict
        Cleaning report containing counts and per-file details.
    """
    # ── Resolve dataset path ──────────────────────────────────
    rdd_path = resolve_path(config["dataset"]["rdd2022_path"])
    logger.info(f"Dataset path: {rdd_path}")

    # ── Check dataset exists ──────────────────────────────────
    if not rdd_path.exists():
        logger.error(
            f"\n{'='*60}\n"
            f"  DATASET NOT FOUND: {rdd_path}\n"
            f"{'='*60}\n"
            "  Please place the RDD2022 dataset at the path above.\n"
            "  You can download it from:\n"
            "    https://github.com/sekilab/RoadDamageDetector\n"
            f"{'='*60}"
        )
        # Return an empty report — do NOT crash the pipeline
        return {
            "status": "DATASET_NOT_FOUND",
            "dataset_path": str(rdd_path),
            "total_files": 0,
            "valid": 0,
            "invalid": 0,
            "duplicates": 0,
            "valid_files": [],
            "invalid_files": [],
            "duplicate_files": [],
            "timestamp": datetime.now().isoformat(),
        }

    # ── Collect all files (not just images) ──────────────────
    all_files = sorted(p for p in rdd_path.rglob("*") if p.is_file())
    logger.info(f"Total files found in dataset directory: {len(all_files)}")

    # ── Separate by extension ─────────────────────────────────
    image_files = [f for f in all_files if f.suffix.lower() in IMAGE_EXTENSIONS]
    non_image_files = [f for f in all_files if f.suffix.lower() not in IMAGE_EXTENSIONS]

    logger.info(f"  Image files  : {len(image_files)}")
    logger.info(f"  Other files  : {len(non_image_files)}")

    # ── Classify each image ───────────────────────────────────
    valid_files = []       # (path, md5)
    invalid_files = []     # (path, reason)
    duplicate_files = []   # (path, duplicate_of)

    seen_hashes = {}       # md5 → first_seen_path  (for duplicate tracking)

    for i, img_path in enumerate(image_files, start=1):
        if i % 500 == 0 or i == len(image_files):
            logger.info(f"  Checking {i}/{len(image_files)} ...")

        # --- Step A: Validity check ---
        valid, reason = is_valid_image(img_path)
        if not valid:
            invalid_files.append({
                "file": str(img_path.relative_to(rdd_path)),
                "reason": reason,
            })
            continue  # Skip hash check for corrupt files

        # --- Step B: Duplicate check (MD5) ---
        md5 = compute_md5(img_path)
        if not md5:
            # Could not hash — treat as invalid
            invalid_files.append({
                "file": str(img_path.relative_to(rdd_path)),
                "reason": "Could not compute MD5 hash",
            })
            continue

        if md5 in seen_hashes:
            # This file is a byte-for-byte duplicate of an earlier file
            duplicate_files.append({
                "file": str(img_path.relative_to(rdd_path)),
                "duplicate_of": str(seen_hashes[md5]),
                "md5": md5,
            })
        else:
            seen_hashes[md5] = img_path.relative_to(rdd_path)
            valid_files.append({
                "file": str(img_path.relative_to(rdd_path)),
                "md5": md5,
            })

    # ── Build report ──────────────────────────────────────────
    report = {
        "status": "OK",
        "dataset_path": str(rdd_path),
        "timestamp": datetime.now().isoformat(),
        "total_files": len(all_files),
        "total_image_files": len(image_files),
        "non_image_files": len(non_image_files),
        "valid": len(valid_files),
        "invalid": len(invalid_files),
        "duplicates": len(duplicate_files),
        "valid_files": valid_files,
        "invalid_files": invalid_files,
        "duplicate_files": duplicate_files,
    }

    return report


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    """
    Entry point: load config, run cleaning, save report, print summary.
    """
    logger = setup_logging("clean_dataset")
    logger.info("=" * 60)
    logger.info("  STEP 1: DATASET CLEANING")
    logger.info("=" * 60)

    # Load configuration
    try:
        config = load_config()
        logger.info("Configuration loaded successfully.")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Run the cleaning scan
    report = clean_dataset(config, logger)

    # Save JSON report to data/outputs/cleaning_report.json
    output_path = resolve_path(config["dataset"]["outputs_path"]) / "cleaning_report.json"
    save_report(report, output_path)

    # ── Print human-readable summary ──────────────────────────
    print("\n" + "=" * 50)
    print("  CLEANING REPORT SUMMARY")
    print("=" * 50)
    print(f"  Status          : {report['status']}")
    print(f"  Total files     : {report.get('total_files', 0)}")
    print(f"  Image files     : {report.get('total_image_files', 0)}")
    print(f"  Valid images    : {report.get('valid', 0)}")
    print(f"  Invalid images  : {report.get('invalid', 0)}")
    print(f"  Duplicates      : {report.get('duplicates', 0)}")
    print(f"  Report saved to : {output_path}")
    print("=" * 50 + "\n")

    return report


if __name__ == "__main__":
    main()
