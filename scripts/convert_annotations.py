"""
scripts/convert_annotations.py
================================
Step 3 of the pipeline: Convert Pascal VOC XML annotations → YOLO format.

Why convert?
  YOLO models require annotations in a specific normalised text format.
  Pascal VOC uses pixel coordinates in XML; YOLO uses a simpler, normalised
  format that is resolution-independent.

YOLO format (one line per bounding box):
  class_id  x_center  y_center  width  height
  All values are normalised to [0, 1] relative to image dimensions.

Formulae:
  x_center = (xmin + xmax) / 2 / image_width
  y_center = (ymin + ymax) / 2 / image_height
  width    = (xmax - xmin) / image_width
  height   = (ymax - ymin) / image_height

Since we are only detecting potholes (D40), class_id = 0.

Run from project root:
  python scripts/convert_annotations.py

Input  : data/processed/d40/labels/*.xml  (Pascal VOC)
Output : data/processed/d40/labels/*.txt  (YOLO format)
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
    ensure_dir,
    save_report,
    setup_logging,
    parse_voc_xml,
)

# Only class we care about
D40_CLASS_NAME = "D40"
D40_CLASS_ID   = 0    # YOLO class index (0-indexed, only one class here)


# ──────────────────────────────────────────────────────────────
# CONVERSION HELPERS
# ──────────────────────────────────────────────────────────────

def convert_voc_to_yolo(
    xmin: int, ymin: int, xmax: int, ymax: int,
    img_width: int, img_height: int
) -> tuple:
    """
    Convert a single Pascal VOC bounding box to YOLO normalised format.

    Parameters
    ----------
    xmin, ymin, xmax, ymax : int
        Pixel coordinates of the bounding box (from XML).
    img_width, img_height : int
        Dimensions of the full image.

    Returns
    -------
    tuple[float, float, float, float]
        (x_center, y_center, width, height) all in [0, 1].
    """
    # Guard: clamp to image bounds to avoid values outside [0, 1]
    xmin = max(0, min(xmin, img_width))
    ymin = max(0, min(ymin, img_height))
    xmax = max(0, min(xmax, img_width))
    ymax = max(0, min(ymax, img_height))

    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width    = (xmax - xmin) / img_width
    height   = (ymax - ymin) / img_height

    return x_center, y_center, width, height


def convert_xml_to_yolo_txt(xml_path: Path, output_txt_path: Path, logger: logging.Logger) -> int:
    """
    Convert a single Pascal VOC XML file to a YOLO .txt label file.

    Only D40 objects are included. Other damage classes are ignored.

    Parameters
    ----------
    xml_path : Path
        Path to the source .xml annotation.
    output_txt_path : Path
        Destination .txt file path.
    logger : logging.Logger
        Logger for warnings.

    Returns
    -------
    int
        Number of D40 bounding boxes written (0 means file not written).
    """
    data = parse_voc_xml(xml_path)

    if not data:
        logger.warning(f"Could not parse: {xml_path.name} — skipped.")
        return 0

    img_width  = data.get("width", 0)
    img_height = data.get("height", 0)

    if img_width <= 0 or img_height <= 0:
        logger.warning(
            f"Invalid image dimensions ({img_width}×{img_height}) "
            f"in {xml_path.name} — skipped."
        )
        return 0

    d40_lines = []
    for obj in data.get("objects", []):
        if obj["name"] != D40_CLASS_NAME:
            continue  # Ignore non-pothole annotations

        xmin, ymin = obj["xmin"], obj["ymin"]
        xmax, ymax = obj["xmax"], obj["ymax"]

        # Skip degenerate boxes (zero area)
        if xmax <= xmin or ymax <= ymin:
            logger.warning(
                f"Degenerate bounding box in {xml_path.name}: "
                f"({xmin},{ymin})-({xmax},{ymax}) — skipped."
            )
            continue

        x_c, y_c, w, h = convert_voc_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height)

        # YOLO line: class_id x_center y_center width height (6 decimal places)
        d40_lines.append(f"{D40_CLASS_ID} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    if not d40_lines:
        return 0  # No D40 boxes → don't create an empty label file

    ensure_dir(output_txt_path.parent)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(d40_lines) + "\n")

    return len(d40_lines)


# ──────────────────────────────────────────────────────────────
# MAIN CONVERSION FUNCTION
# ──────────────────────────────────────────────────────────────

def convert_annotations(config: dict, logger: logging.Logger) -> dict:
    """
    Convert all D40 XML annotations to YOLO format.

    Parameters
    ----------
    config : dict
        Loaded config.yaml.
    logger : logging.Logger

    Returns
    -------
    dict
        Conversion report.
    """
    processed_path = resolve_path(config["dataset"]["processed_path"])
    labels_dir = processed_path / "d40" / "labels"

    if not labels_dir.exists():
        logger.warning(
            f"Labels directory not found: {labels_dir}\n"
            "  Run filter_d40.py first to populate this directory."
        )
        return {
            "status": "NO_LABELS_DIR",
            "total_xml": 0,
            "converted": 0,
            "skipped": 0,
            "total_d40_boxes": 0,
            "timestamp": datetime.now().isoformat(),
        }

    # ── Find XML and TXT files in the labels directory ────────
    xml_files = sorted(labels_dir.glob("*.xml"))
    txt_files = sorted(labels_dir.glob("*.txt"))
    logger.info(f"XML annotation files found: {len(xml_files)}")
    logger.info(f"TXT annotation files found: {len(txt_files)}")

    if not xml_files and not txt_files:
        logger.warning(
            "No XML or TXT files found in labels directory.\n"
            "  Run filter_d40.py first to populate this directory."
        )
        return {
            "status": "NO_ANNOTATION_FILES",
            "total_xml": 0,
            "total_txt": 0,
            "converted": 0,
            "skipped": 0,
            "total_d40_boxes": 0,
            "timestamp": datetime.now().isoformat(),
        }

    converted = 0
    skipped   = 0
    total_d40_boxes = 0
    converted_files = []

    # ── Convert XML files if present ──────────────────────────
    for i, xml_path in enumerate(xml_files, start=1):
        out_txt = labels_dir / (xml_path.stem + ".txt")
        boxes_written = convert_xml_to_yolo_txt(xml_path, out_txt, logger)
        if boxes_written > 0:
            converted += 1
            total_d40_boxes += boxes_written
            converted_files.append({
                "source": xml_path.name,
                "txt": out_txt.name,
                "d40_boxes": boxes_written,
            })
        else:
            skipped += 1

    # ── Normalize TXT files if present ────────────────────────
    for i, txt_path in enumerate(txt_files, start=1):
        if xml_files and any(x.stem == txt_path.stem for x in xml_files):
            continue
        try:
            valid_lines = []
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_tok = parts[0].upper()
                    if cls_tok == D40_CLASS_NAME or cls_tok in {"0", "3"}:
                        valid_line = f"{D40_CLASS_ID} " + " ".join(parts[1:])
                        valid_lines.append(valid_line)

            if valid_lines:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(valid_lines) + "\n")
                converted += 1
                total_d40_boxes += len(valid_lines)
                converted_files.append({
                    "source": txt_path.name,
                    "txt": txt_path.name,
                    "d40_boxes": len(valid_lines),
                })
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"Error normalizing {txt_path.name}: {e}")
            skipped += 1

    logger.info(f"Converted / Verified: {converted} files")
    logger.info(f"Skipped             : {skipped} files (no D40 or parse error)")
    logger.info(f"D40 boxes           : {total_d40_boxes} total")

    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "labels_directory": str(labels_dir),
        "total_xml": len(xml_files),
        "total_txt": len(txt_files),
        "converted": converted,
        "skipped": skipped,
        "total_d40_boxes": total_d40_boxes,
        "converted_files": converted_files,
    }


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    """
    Entry point: load config, run conversion, save report, print summary.
    """
    logger = setup_logging("convert_annotations")
    logger.info("=" * 60)
    logger.info("  STEP 3: CONVERT ANNOTATIONS (VOC XML → YOLO TXT)")
    logger.info("=" * 60)

    try:
        config = load_config()
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    report = convert_annotations(config, logger)

    output_path = resolve_path(config["dataset"]["outputs_path"]) / "convert_report.json"
    save_report(report, output_path)

    print("\n" + "=" * 50)
    print("  ANNOTATION CONVERSION SUMMARY")
    print("=" * 50)
    print(f"  Status         : {report['status']}")
    print(f"  XML files found: {report.get('total_xml', 0)}")
    print(f"  Converted      : {report.get('converted', 0)}")
    print(f"  Skipped        : {report.get('skipped', 0)}")
    print(f"  Total D40 boxes: {report.get('total_d40_boxes', 0)}")
    print(f"  Report saved to: {output_path}")
    print("=" * 50 + "\n")

    return report


if __name__ == "__main__":
    main()
