"""
ml/train_rdd2022.py
====================
Train YOLOv8n on the prepared RDD2022 D40 (Pothole) dataset.

Usage:
    python ml/train_rdd2022.py --epochs 5     # test run
    python ml/train_rdd2022.py --epochs 50    # full run
    python ml/train_rdd2022.py --epochs 50 --resume  # resume
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU detected: {gpu_name}")
            return "0"
        else:
            print("  No CUDA GPU found -- using CPU")
            return "cpu"
    except ImportError:
        print("  torch not available for GPU check -- defaulting to CPU")
        return "cpu"


def get_safe_batch_size(device):
    if device != "cpu":
        return 16
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / 1024 ** 3
        print(f"  Available RAM: {free_gb:.1f} GB")
        if free_gb >= 6:
            batch = 8
        elif free_gb >= 3:
            batch = 4
        else:
            batch = 2
    except ImportError:
        batch = 4
    print(f"  Selected batch size: {batch} (CPU-safe)")
    return batch


def estimate_training_time(n_images, epochs, batch_size, device):
    if device == "cpu":
        secs_per_epoch = (n_images / 5000) * (4 / batch_size) * 8 * 60
    else:
        secs_per_epoch = (n_images / 5000) * (16 / batch_size) * 30
    total_mins = (secs_per_epoch * epochs) / 60
    if total_mins < 60:
        return f"~{total_mins:.0f} minutes"
    return f"~{total_mins/60:.1f} hours"


def save_model_card(models_dir, epochs, device, batch_size, dataset_yaml, results_csv):
    metrics = {}
    if results_csv and results_csv.exists():
        import csv
        rows = list(csv.DictReader(open(results_csv)))
        if rows:
            last = rows[-1]
            metrics = {
                "mAP50": round(float(last.get("metrics/mAP50(B)", 0)), 4),
                "mAP50_95": round(float(last.get("metrics/mAP50-95(B)", 0)), 4),
                "precision": round(float(last.get("metrics/precision(B)", 0)), 4),
                "recall": round(float(last.get("metrics/recall(B)", 0)), 4),
                "epochs_completed": len(rows),
            }

    card = {
        "model_name": "YOLOv8n-RDD2022-D40-Real",
        "architecture": "YOLOv8n",
        "type": "REAL_TRAINED_MODEL",
        "note": "Trained on RDD2022 D40 (Pothole) dataset -- real weights, not demo",
        "classes": {"0": "D40_Pothole"},
        "class_mapping": "0 = D40 Pothole (RDD2022)",
        "dataset": str(dataset_yaml),
        "input_size": [416, 416, 3],
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "device": device,
        "confidence_default": 0.25,
        "iou_default": 0.45,
        "created_at": datetime.now().isoformat(),
        "metrics": metrics,
    }
    card_path = models_dir / "model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"  Model card saved -> {card_path}")


def train(epochs=5, resume=False):
    print("\n" + "=" * 65)
    print("  RDD2022 POTHOLE DETECTION -- YOLO TRAINING")
    print("=" * 65)

    dataset_yaml = PROJECT_ROOT / "data" / "processed" / "split" / "dataset.yaml"
    output_dir   = PROJECT_ROOT / "data" / "outputs" / "training_runs"
    models_dir   = PROJECT_ROOT / "models"
    base_weights = PROJECT_ROOT / "yolov8n.pt"
    project_name = "pothole_detection"

    models_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_yaml.exists():
        print(f"\nERROR: dataset.yaml not found: {dataset_yaml}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("\n[1/4] Detecting hardware...")
    device     = detect_device()
    batch_size = get_safe_batch_size(device)

    train_imgs = list((PROJECT_ROOT / "data/processed/split/train/images").glob("*"))
    val_imgs   = list((PROJECT_ROOT / "data/processed/split/val/images").glob("*"))
    n_train = len(train_imgs)
    n_val   = len(val_imgs)

    print("\n[2/4] Training configuration:")
    print(f"  Base model   : yolov8n.pt (YOLOv8 Nano)")
    print(f"  Dataset YAML : {dataset_yaml}")
    print(f"  Train images : {n_train:,}")
    print(f"  Val images   : {n_val:,}")
    print(f"  Classes      : 0 = D40 Pothole (RDD2022)")
    print(f"  Image size   : 416x416")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Device       : {device}")
    est = estimate_training_time(n_train, epochs, batch_size, device)
    print(f"  Est. time    : {est}")
    print()

    print("[3/4] Loading model weights...")
    existing_last = output_dir / project_name / "weights" / "last.pt"
    if resume and existing_last.exists():
        print(f"  Resuming from: {existing_last}")
        model = YOLO(str(existing_last))
    else:
        if base_weights.exists():
            print(f"  Loading pretrained: {base_weights}")
            model = YOLO(str(base_weights))
        else:
            print("  Downloading yolov8n.pt...")
            model = YOLO("yolov8n.pt")

    print(f"\n[4/4] Starting training ({epochs} epochs)...")
    print("-" * 65)

    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=416,
            batch=batch_size,
            workers=2,
            project=str(output_dir),
            name=project_name,
            device=device,
            exist_ok=True,
            verbose=True,
            resume=resume,
        )
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE -- SAVING MODELS")
    print("=" * 65)

    best_weights = output_dir / project_name / "weights" / "best.pt"
    last_weights = output_dir / project_name / "weights" / "last.pt"
    dest_best = None

    if best_weights.exists():
        dest_best = models_dir / "best.pt"
        shutil.copy2(best_weights, dest_best)
        print(f"  best.pt  -> {dest_best}")

        dest_yolo = models_dir / "pothole_yolo.pt"
        shutil.copy2(best_weights, dest_yolo)
        print(f"  pothole_yolo.pt -> {dest_yolo}")
    else:
        print("  WARNING: best.pt not found in training output!")

    if last_weights.exists():
        print(f"  last.pt  -> {last_weights}")

    results_csv = output_dir / project_name / "results.csv"
    save_model_card(models_dir, epochs, device, batch_size, dataset_yaml, results_csv)

    print(f"\n  Model ready at: models/best.pt")
    print(f"  Model ready at: models/pothole_yolo.pt")
    print(f"  Start dashboard: streamlit run app.py")
    print("=" * 65)

    return str(dest_best) if dest_best else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n on RDD2022 D40 Pothole dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt checkpoint")
    args = parser.parse_args()
    train(epochs=args.epochs, resume=args.resume)
