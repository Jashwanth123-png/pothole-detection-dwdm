"""
ml/evaluate_yolo.py
===================
Evaluation script for the trained YOLO pothole-detection model.

This script:
  1. Loads the trained model from the path specified in config.yaml
  2. Runs YOLO validation on the test split
  3. Reports mAP@0.5, precision, and recall
  4. Saves a detailed evaluation report to data/outputs/evaluation_report.json

Prerequisites:
  1. A trained model at models/pothole_yolo.pt
     (run python ml/train_yolo.py first)
  2. The test split at data/processed/split/test/
     (run python scripts/prepare_dataset.py first)
  3. Install ultralytics: pip install ultralytics

Usage:
  python ml/evaluate_yolo.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import our own modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, get_project_root, setup_logger

logger = setup_logger(__name__)

# Message shown when no model is found
NO_MODEL_MSG = (
    "\nNOTE: Real detection requires a trained YOLO model. See models/README.md\n"
    "  Train the model first:  python ml/train_yolo.py\n"
)


def evaluate_yolo(config: dict = None) -> dict:
    """
    Evaluate the trained YOLO model on the test dataset split.

    Runs YOLO's built-in validation routine which computes:
      - mAP@0.5   (mean Average Precision at IoU threshold 0.5)
      - Precision  (fraction of detections that are correct)
      - Recall     (fraction of ground-truth potholes detected)

    Results are saved to data/outputs/evaluation_report.json.

    Args:
        config (dict, optional): Project configuration. Loads from
                                  config.yaml if None.

    Returns:
        dict: A dictionary with keys 'map50', 'precision', 'recall',
              'model_path', 'evaluated_at', and 'status'.
              Returns a dict with status='failed' on error.
    """
def evaluate_yolo(
    config: dict = None,
    model_override: str = None,
    split: str = None,
    batch_size: int = None,
    device: str = None,
    image_size: int = None,
) -> dict:
    """
    Evaluate the trained YOLO model on the test or validation dataset split.

    Runs YOLO's built-in validation routine which computes:
      - mAP@0.5   (mean Average Precision at IoU threshold 0.5)
      - Precision  (fraction of detections that are correct)
      - Recall     (fraction of ground-truth potholes detected)

    Results are saved to data/outputs/evaluation_report.json.

    Args:
        config (dict, optional): Project configuration. Loads from
                                  config.yaml if None.
        model_override (str, optional): Custom path to model weights.
        split (str, optional): Split to evaluate on ('test' or 'val').
        batch_size (int, optional): Batch size for validation.
        device (str, optional): Computation device ('cpu' or '0').
        image_size (int, optional): Input image size (default 416).

    Returns:
        dict: A dictionary with keys 'map50', 'precision', 'recall',
              'model_path', 'evaluated_at', and 'status'.
              Returns a dict with status='failed' on error.
    """
    if config is None:
        config = load_config()

    project_root = get_project_root()

    # ------------------------------------------------------------------
    # 1. Resolve model path
    # ------------------------------------------------------------------
    model_cfg = config.get("model", {})
    if model_override:
        model_path = Path(model_override)
        if not model_path.is_absolute():
            model_path = project_root / model_path
    else:
        cfg_model = model_cfg.get("yolo_model_path", "models/pothole_yolo.pt")
        model_path = project_root / cfg_model

        if not model_path.exists():
            # Check models/best.pt fallback
            best_candidate = project_root / "models" / "best.pt"
            if best_candidate.exists():
                model_path = best_candidate

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        print(NO_MODEL_MSG)
        print(f"Expected model at: {model_path}")
        return {"status": "failed", "reason": "model_not_found", "model_path": str(model_path)}

    # ------------------------------------------------------------------
    # 2. Resolve dataset YAML
    # ------------------------------------------------------------------
    dataset_yaml = project_root / "data" / "processed" / "split" / "dataset.yaml"
    if not dataset_yaml.exists():
        logger.error(f"Dataset YAML not found: {dataset_yaml}")
        print("ERROR: Dataset not prepared. Run: python scripts/prepare_dataset.py")
        return {"status": "failed", "reason": "dataset_not_found"}

    # ------------------------------------------------------------------
    # 3. Detect split to evaluate on
    # ------------------------------------------------------------------
    test_img_dir = project_root / "data" / "processed" / "split" / "test" / "images"
    val_img_dir = project_root / "data" / "processed" / "split" / "val" / "images"

    def count_images(d: Path) -> int:
        if not d.exists():
            return 0
        return sum(1 for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    test_count = count_images(test_img_dir)
    val_count = count_images(val_img_dir)

    target_split = split
    if not target_split:
        if test_count > 0:
            target_split = "test"
        elif val_count > 0:
            target_split = "val"
        else:
            print("ERROR: No images found in test or val splits.")
            return {"status": "failed", "reason": "no_images_in_splits"}
    else:
        # Validate requested split has images
        chosen_count = test_count if target_split == "test" else val_count
        if chosen_count == 0:
            alt_split = "val" if target_split == "test" else "test"
            alt_count = val_count if target_split == "test" else test_count
            if alt_count > 0:
                print(f"Notice: Split '{target_split}' is empty. Falling back to '{alt_split}' ({alt_count} images).")
                target_split = alt_split
            else:
                print(f"ERROR: Split '{target_split}' has no images.")
                return {"status": "failed", "reason": f"no_images_in_{target_split}"}

    # ------------------------------------------------------------------
    # 4. Import ultralytics
    # ------------------------------------------------------------------
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        print("ERROR: Install ultralytics: pip install ultralytics")
        return {"status": "failed", "reason": "ultralytics_not_installed"}

    # ------------------------------------------------------------------
    # 5. Load model and run validation
    # ------------------------------------------------------------------
    if image_size is None:
        image_size = model_cfg.get("image_size", 416)
    if device is None:
        device = model_cfg.get("device", "cpu")
    if batch_size is None:
        batch_size = config.get("training", {}).get("batch_size", 8)

    print("\n" + "=" * 60)
    print("YOLO POTHOLE DETECTION - EVALUATION")
    print("=" * 60)
    print(f"Model:    {model_path} ({model_path.stat().st_size / 1024**2:.1f} MB)")
    print(f"Dataset:  {dataset_yaml}")
    print(f"Split:    {target_split} ({test_count if target_split == 'test' else val_count} images)")
    print(f"Device:   {device}")
    print(f"Batch:    {batch_size}")
    print(f"ImgSize:  {image_size}")
    print("=" * 60 + "\n")

    try:
        model = YOLO(str(model_path))

        metrics = model.val(
            data=str(dataset_yaml),
            imgsz=image_size,
            batch=batch_size,
            device=device,
            split=target_split,
            verbose=True,
        )

        # ------------------------------------------------------------------
        # 6. Extract key metrics
        # ------------------------------------------------------------------
        map50     = float(metrics.box.map50)
        map50_95  = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall    = float(metrics.box.mr)

        report = {
            "status":       "success",
            "model_path":   str(model_path),
            "dataset":      str(dataset_yaml),
            "split":        target_split,
            "image_count":  test_count if target_split == "test" else val_count,
            "evaluated_at": datetime.now().isoformat(),
            "map50":        round(map50, 4),
            "map50_95":     round(map50_95, 4),
            "precision":    round(precision, 4),
            "recall":       round(recall, 4),
            # Percentages for readability
            "map50_pct":    round(map50 * 100, 2),
            "precision_pct": round(precision * 100, 2),
            "recall_pct":   round(recall * 100, 2),
        }

        # ------------------------------------------------------------------
        # 7. Save report to JSON
        # ------------------------------------------------------------------
        output_dir = project_root / "data" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "evaluation_report.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        logger.info(f"Evaluation report saved: {report_path}")

        # Update model_card.json if it exists
        card_path = project_root / "models" / "model_card.json"
        if card_path.exists():
            try:
                with open(card_path, "r", encoding="utf-8") as cf:
                    card_data = json.load(cf)
                card_data.setdefault("metrics", {})
                card_data["metrics"].update({
                    "mAP50": round(map50, 4),
                    "mAP50_95": round(map50_95, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "evaluated_split": target_split,
                    "last_evaluated": datetime.now().isoformat(),
                })
                with open(card_path, "w", encoding="utf-8") as cf:
                    json.dump(card_data, cf, indent=2)
            except Exception as ce:
                logger.warning(f"Could not update model_card.json: {ce}")

        # ------------------------------------------------------------------
        # 8. Print human-readable summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Split     : {target_split}")
        print(f"  mAP@0.5   : {map50:.4f}  ({map50*100:.2f}%)")
        print(f"  mAP@0.5:95: {map50_95:.4f}  ({map50_95*100:.2f}%)")
        print(f"  Precision : {precision:.4f}  ({precision*100:.2f}%)")
        print(f"  Recall    : {recall:.4f}  ({recall*100:.2f}%)")
        print("=" * 60)
        print(f"\nFull report saved to: {report_path}")

        return report

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\nERROR during evaluation: {e}")
        return {"status": "failed", "reason": str(e)}


def main():
    """Entry point for the evaluation script with CLI flags."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate YOLO Pothole Detection Model")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default=None, choices=["test", "val"], help="Dataset split to evaluate")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or '0')")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size")
    args = parser.parse_args()

    report = evaluate_yolo(
        model_override=args.model,
        split=args.split,
        batch_size=args.batch,
        device=args.device,
        image_size=args.imgsz,
    )

    if report.get("status") == "success":
        print("\nEvaluation complete.")
    else:
        reason = report.get("reason", "unknown error")
        print(f"\nEvaluation failed: {reason}")
        sys.exit(1)


# ==========================================================================
# Entry point
# ==========================================================================
if __name__ == "__main__":
    main()

