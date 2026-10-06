"""
ml/train_yolo.py
================
YOLO model training script for pothole detection.

This script trains a YOLOv8 model on the prepared RDD2022 D40 dataset.

Prerequisites:
  1. Run scripts/prepare_dataset.py to prepare the dataset
  2. Ensure data/processed/split/dataset.yaml exists
  3. Install ultralytics: pip install ultralytics

Usage:
  python ml/train_yolo.py

Output:
  Trained model saved to models/pothole_yolo.pt
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path so we can import our own modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, get_project_root, setup_logger

logger = setup_logger(__name__)


def detect_device(preferred: str = None) -> str:
    """Detect available computation device (GPU or CPU)."""
    if preferred and preferred.lower() not in {"auto", "none"}:
        return preferred
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}")
            return "0"
    except Exception:
        pass
    return "cpu"


def train_yolo(
    config: dict = None,
    epochs_override: int = None,
    batch_override: int = None,
    device_override: str = None,
    model_override: str = None,
    imgsz_override: int = None,
):
    """
    Train the YOLOv8 model on the prepared pothole dataset.

    This function:
      1. Checks that the dataset YAML exists (prepared by prepare_dataset.py)
      2. Loads YOLOv8 base weights
      3. Runs training with parameters from config.yaml or overrides
      4. Copies the best weights to models/pothole_yolo.pt and models/best.pt
      5. Updates models/model_card.json

    Args:
        config (dict, optional): Project configuration dict.
        epochs_override (int, optional): Epoch count override.
        batch_override (int, optional): Batch size override.
        device_override (str, optional): Device override ('cpu' or '0').
        model_override (str, optional): Base model weights path.
        imgsz_override (int, optional): Image size override.

    Returns:
        str | None: Absolute path to the saved best model weights,
                    or None if training fails.
    """
    if config is None:
        config = load_config()

    project_root = get_project_root()

    # ------------------------------------------------------------------
    # 1. Verify the dataset YAML is present
    # ------------------------------------------------------------------
    dataset_yaml = project_root / "data" / "processed" / "split" / "dataset.yaml"
    if not dataset_yaml.exists():
        logger.error(
            f"Dataset YAML not found: {dataset_yaml}\n"
            f"Please run: python scripts/prepare_dataset.py"
        )
        print("\n" + "=" * 60)
        print("ERROR: Dataset not prepared.")
        print("Run this first:")
        print("  python scripts/prepare_dataset.py")
        print("=" * 60 + "\n")
        return None

    # ------------------------------------------------------------------
    # 2. Import ultralytics
    # ------------------------------------------------------------------
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        print("ERROR: Install ultralytics: pip install ultralytics")
        return None

    # ------------------------------------------------------------------
    # 3. Read training hyper-parameters from config or overrides
    # ------------------------------------------------------------------
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    base_model   = model_override or model_cfg.get("yolo_base_model", "yolov8n.pt")
    image_size   = imgsz_override or model_cfg.get("image_size", 416)
    configured_dev = device_override or model_cfg.get("device", "cpu")
    device       = detect_device(configured_dev)
    epochs       = epochs_override if epochs_override is not None else train_cfg.get("epochs", 50)

    # Pick safe batch size if CPU
    if batch_override is not None:
        batch_size = batch_override
    else:
        cfg_batch = train_cfg.get("batch_size", 16)
        if device == "cpu" and cfg_batch > 8:
            batch_size = 4
        else:
            batch_size = cfg_batch

    workers      = train_cfg.get("workers", 2)
    project_name = train_cfg.get("project_name", "pothole_detection")

    output_dir = project_root / "data" / "outputs" / "training_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Log / print training summary
    # ------------------------------------------------------------------
    logger.info("Starting YOLO training")
    logger.info(f"  Base model:  {base_model}")
    logger.info(f"  Dataset:     {dataset_yaml}")
    logger.info(f"  Image size:  {image_size}")
    logger.info(f"  Epochs:      {epochs}")
    logger.info(f"  Batch size:  {batch_size}")
    logger.info(f"  Device:      {device}")

    print("\n" + "=" * 60)
    print("YOLO POTHOLE DETECTION - TRAINING")
    print("=" * 60)
    print(f"Base model:  {base_model}")
    print(f"Dataset:     {dataset_yaml}")
    print(f"Epochs:      {epochs}")
    print(f"Batch size:  {batch_size}")
    print(f"Device:      {device}")
    print(f"Image size:  {image_size}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 5. Run training
    # ------------------------------------------------------------------
    try:
        model = YOLO(base_model)

        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            workers=workers,
            project=str(output_dir),
            name=project_name,
            device=device,
            exist_ok=True,
        )

        # ------------------------------------------------------------------
        # 6. Copy best weights to canonical paths
        # ------------------------------------------------------------------
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)

        best_weights = output_dir / project_name / "weights" / "best.pt"
        last_weights = output_dir / project_name / "weights" / "last.pt"

        target_yolo = models_dir / "pothole_yolo.pt"
        target_best = models_dir / "best.pt"

        if best_weights.exists():
            shutil.copy2(best_weights, target_yolo)
            shutil.copy2(best_weights, target_best)
            logger.info(f"Model saved to: {target_yolo} and {target_best}")

            # Update model_card.json
            import json
            card_path = models_dir / "model_card.json"
            card_data = {
                "model_name": "YOLOv8n-RDD2022-D40-Real",
                "architecture": "YOLOv8n",
                "type": "REAL_TRAINED_MODEL",
                "note": "Trained on RDD2022 D40 (Pothole) dataset -- real weights, NOT demo",
                "classes": {"0": "D40_Pothole"},
                "class_mapping": "0 = D40 Pothole (RDD2022)",
                "dataset": str(dataset_yaml),
                "input_size": [image_size, image_size, 3],
                "epochs_completed": epochs,
                "batch_size": batch_size,
                "device": device,
                "confidence_default": 0.25,
                "iou_default": 0.45,
                "created_at": datetime.now().isoformat(),
            }
            with open(card_path, "w", encoding="utf-8") as cf:
                json.dump(card_data, cf, indent=2)

            print(f"\nTraining complete! Models saved to:")
            print(f"  - {target_yolo}")
            print(f"  - {target_best}")
            return str(target_yolo)
        else:
            logger.warning("best.pt not found after training.")
            print("\nWARNING: best.pt not found — check the training run output.")
            return None

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\nERROR during training: {e}")
        return None


# ==========================================================================
# Entry point
# ==========================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLOv8 Pothole Detection Model")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or '0')")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size (default: 416)")
    parser.add_argument("--model", type=str, default=None, help="Base model weights")
    args = parser.parse_args()

    trained_model = train_yolo(
        epochs_override=args.epochs,
        batch_override=args.batch,
        device_override=args.device,
        model_override=args.model,
        imgsz_override=args.imgsz,
    )
    if trained_model:
        print(f"\nSuccess! Real YOLO model ready.")
    else:
        print("\nTraining failed or dataset not ready.")
        sys.exit(1)
