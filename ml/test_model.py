"""
ml/test_model.py
================
Test models/best.pt on several RDD2022 images to verify detections.

Usage:
    python ml/test_model.py
    python ml/test_model.py --model models/best.pt --images dataset/rdd2022/India/train/images
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_model(model_path="models/best.pt", image_dir=None, n_images=10, conf=0.25):
    print("\n" + "=" * 65)
    print("  RDD2022 MODEL INFERENCE TEST")
    print("=" * 65)

    model_file = PROJECT_ROOT / model_path
    if not model_file.exists():
        print(f"ERROR: Model not found: {model_file}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)

    print(f"\nLoading model: {model_file} ({model_file.stat().st_size/1024**2:.1f} MB)")
    model = YOLO(str(model_file))
    print(f"Model classes: {model.names}")

    # Find test images
    if image_dir:
        img_dir = PROJECT_ROOT / image_dir
    else:
        img_dir = PROJECT_ROOT / "dataset" / "rdd2022" / "India" / "train" / "images"
        if not img_dir.exists():
            img_dir = PROJECT_ROOT / "data" / "processed" / "split" / "val" / "images"

    images = list(img_dir.glob("*.jpg"))[:n_images]
    if not images:
        images = list(img_dir.glob("*.png"))[:n_images]

    if not images:
        print(f"No images found in: {img_dir}")
        sys.exit(1)

    print(f"\nTesting on {len(images)} images from: {img_dir}")
    print(f"Confidence threshold: {conf}")
    print("-" * 65)

    total_det = 0
    images_with_det = 0
    class_counts = {}

    for img_path in images:
        results = model(str(img_path), conf=conf, verbose=False)
        boxes = results[0].boxes
        n_det = len(boxes) if boxes is not None else 0
        total_det += n_det
        if n_det > 0:
            images_with_det += 1

        confs = []
        cls_ids = []
        if boxes is not None and n_det > 0:
            confs = [round(float(b.conf[0]), 3) for b in boxes]
            cls_ids = [int(b.cls[0]) for b in boxes]
            for c in cls_ids:
                class_counts[c] = class_counts.get(c, 0) + 1

        status = "DETECTIONS FOUND" if n_det > 0 else "no detections"
        print(f"  {img_path.name:<40} | {n_det:>2} dets | {status}")
        if n_det > 0:
            for conf_v, cls_id in zip(confs, cls_ids):
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                print(f"    -> class={cls_id} ({cls_name}), conf={conf_v:.3f}")

    print("-" * 65)
    print(f"Total detections  : {total_det}")
    print(f"Images with dets  : {images_with_det}/{len(images)}")
    print(f"Class distribution: {class_counts}")
    print(f"Model type        : {'REAL_TRAINED' if model_file.stat().st_size > 5_000_000 else 'DEMO'}")
    print("=" * 65)

    if total_det == 0:
        print("\nWARNING: No detections! Consider lowering confidence threshold.")
        print("Try: python ml/test_model.py --conf 0.1")
    else:
        print(f"\nModel is working correctly - {total_det} potholes detected across {len(images)} images")

    return total_det


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLO model on RDD2022 images")
    parser.add_argument("--model", default="models/best.pt", help="Path to model .pt file")
    parser.add_argument("--images", default=None, help="Image directory to test on")
    parser.add_argument("--n", type=int, default=10, help="Number of images to test")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    test_model(args.model, args.images, args.n, args.conf)
