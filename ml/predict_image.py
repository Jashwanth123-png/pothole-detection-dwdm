"""
ml/predict_image.py
===================
Single-image inference module for pothole detection.

This module provides:
  - predict_image()   : Run the YOLO model on one image and return detections.
  - detect_demo()     : Generate fake detections for demo/testing (no model needed).
  - main()            : CLI entry point — python ml/predict_image.py --image <path>

Prerequisites:
  - Trained model at models/pothole_yolo.pt
    (run python ml/train_yolo.py first)
  - Install: pip install ultralytics opencv-python

Usage (CLI):
  python ml/predict_image.py --image path/to/pothole.jpg
  python ml/predict_image.py --image path/to/pothole.jpg --demo   # fake detections
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path

# Make sure we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from utils.helpers     import load_config, get_project_root, setup_logger
from utils.severity    import calculate_severity
from utils.preprocessing import load_image, resize_image

logger = setup_logger(__name__)

# Displayed when the model file is missing
NO_MODEL_MSG = (
    "\nNOTE: Real detection requires a trained YOLO model. See models/README.md\n"
    "  Train the model first:  python ml/train_yolo.py\n"
    "  Use --demo flag to run with fake detections for testing.\n"
)


# ==========================================================================
# Core prediction function
# ==========================================================================

def predict_image(image_path: str, config: dict = None):
    """
    Run YOLO inference on a single image and return structured detections.

    Each detection dictionary contains:
      - x1, y1, x2, y2  : Bounding box corners (pixel coordinates)
      - confidence       : Model confidence score (0.0 – 1.0)
      - severity         : 'low' | 'medium' | 'high'  (from utils/severity.py)
      - area_ratio       : Fraction of image area covered by the bounding box

    Args:
        image_path (str | Path): Path to the input image file.
        config     (dict, optional): Project config. Loads from config.yaml if None.

    Returns:
        tuple:
          - detections (list[dict]): List of detection dictionaries (may be empty).
          - annotated_image (np.ndarray | None): BGR image with bounding boxes drawn,
            or None if the image could not be loaded.
    """
    if config is None:
        config = load_config()

    project_root = get_project_root()

    # ------------------------------------------------------------------
    # 1. Load and validate the input image
    # ------------------------------------------------------------------
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        print(f"ERROR: Image file not found: {image_path}")
        return [], None

    # Use our preprocessing helper to load the image as a BGR numpy array
    image = load_image(str(image_path))
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return [], None

    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w  # total pixel area (used for area_ratio)

    # ------------------------------------------------------------------
    # 2. Resolve model path from config
    # ------------------------------------------------------------------
    model_cfg  = config.get("model", {})
    model_path = project_root / model_cfg.get("yolo_model_path", "models/pothole_yolo.pt")

    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        print(NO_MODEL_MSG)
        return [], image   # return original image so callers can still display it

    # ------------------------------------------------------------------
    # 3. Import ultralytics and load model
    # ------------------------------------------------------------------
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        print("ERROR: pip install ultralytics")
        return [], image

    image_size        = model_cfg.get("image_size", 416)
    confidence_thresh = model_cfg.get("confidence_threshold", 0.25)
    device            = model_cfg.get("device", "cpu")

    try:
        model = YOLO(str(model_path))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print(f"ERROR loading model: {e}")
        return [], image

    # ------------------------------------------------------------------
    # 4. Run inference
    # ------------------------------------------------------------------
    logger.info(f"Running inference on: {image_path.name}")
    print(f"Running inference on: {image_path.name} ...")

    try:
        results = model.predict(
            source=str(image_path),
            imgsz=image_size,
            conf=confidence_thresh,
            device=device,
            verbose=False,          # suppress per-image YOLO console spam
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        print(f"ERROR during inference: {e}")
        return [], image

    # ------------------------------------------------------------------
    # 5. Parse results into structured dicts
    # ------------------------------------------------------------------
    detections     = []
    annotated_image = image.copy()

    # results is a list with one element per input image
    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            # xyxy gives absolute pixel coordinates: [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            confidence = float(box.conf[0])

            # Compute the fraction of the image covered by this box
            box_area   = (x2 - x1) * (y2 - y1)
            area_ratio = box_area / img_area if img_area > 0 else 0.0

            # Use our severity module to label the pothole
            severity = calculate_severity(confidence, area_ratio)

            detections.append({
                "x1":         x1,
                "y1":         y1,
                "x2":         x2,
                "y2":         y2,
                "confidence": round(confidence, 4),
                "severity":   severity,
                "area_ratio": round(area_ratio, 6),
            })

        # Draw bounding boxes on a copy of the image
        annotated_image = _draw_boxes(image.copy(), detections)

    logger.info(f"Found {len(detections)} pothole(s) in {image_path.name}")
    print(f"Detected {len(detections)} pothole(s).")

    return detections, annotated_image


# ==========================================================================
# Demo / fake detection (no model needed)
# ==========================================================================

def detect_demo(image_path: str):
    """
    Generate plausible fake detections for testing without a trained model.

    This is useful for:
      - UI / dashboard development
      - Integration testing of downstream components
      - Demos when no GPU / trained model is available

    Args:
        image_path (str | Path): Path to the input image (must exist).

    Returns:
        tuple:
          - detections (list[dict]): 1–3 randomly placed fake detections.
          - annotated_image (np.ndarray): Image with fake bounding boxes drawn.
    """
    image_path = Path(image_path)
    image      = load_image(str(image_path))

    if image is None:
        logger.error(f"Could not load image for demo: {image_path}")
        # Return a blank 416×416 placeholder so the caller doesn't crash
        image = np.zeros((416, 416, 3), dtype=np.uint8)

    img_h, img_w = image.shape[:2]
    img_area     = img_h * img_w

    print("\n[DEMO MODE] Generating fake detections (no real model used).")
    print(NO_MODEL_MSG)

    # Build 1–3 random bounding boxes spread across the image
    num_fake = random.randint(1, 3)
    detections = []

    for _ in range(num_fake):
        # Random box that fits inside the image
        x1 = random.randint(0, img_w // 2)
        y1 = random.randint(0, img_h // 2)
        x2 = random.randint(x1 + 30, min(x1 + img_w // 3, img_w - 1))
        y2 = random.randint(y1 + 30, min(y1 + img_h // 3, img_h - 1))

        confidence = round(random.uniform(0.40, 0.90), 4)
        box_area   = (x2 - x1) * (y2 - y1)
        area_ratio = round(box_area / img_area, 6) if img_area > 0 else 0.0
        severity   = calculate_severity(confidence, area_ratio)

        detections.append({
            "x1":         x1,
            "y1":         y1,
            "x2":         x2,
            "y2":         y2,
            "confidence": confidence,
            "severity":   severity,
            "area_ratio": area_ratio,
        })

    annotated = _draw_boxes(image.copy(), detections, color=(0, 165, 255))  # orange for demo
    return detections, annotated


# ==========================================================================
# Helper: draw bounding boxes on an image
# ==========================================================================

def _draw_boxes(image: np.ndarray, detections: list, color: tuple = (0, 0, 255)) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of the image.

    Args:
        image      (np.ndarray): BGR image to annotate (modified in place).
        detections (list[dict]): Detections from predict_image() or detect_demo().
        color      (tuple):     BGR color for boxes. Default is red.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    import cv2  # OpenCV — used only here so the module can be imported without cv2

    # Choose label text color based on box color brightness
    severity_colors = {
        "low":    (0, 255, 0),    # green
        "medium": (0, 165, 255),  # orange
        "high":   (0, 0, 255),    # red
    }

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf     = det["confidence"]
        severity = det.get("severity", "unknown")

        # Use severity-based color if not overridden by caller
        box_color = severity_colors.get(severity, color)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

        # Build label: e.g. "Pothole 0.82 [high]"
        label = f"Pothole {conf:.2f} [{severity}]"

        # Draw filled background for the label text
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - label_h - baseline - 4),
            (x1 + label_w, y1),
            box_color,
            -1,     # filled
        )
        # Draw white label text
        cv2.putText(
            image, label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return image


# ==========================================================================
# CLI entry point
# ==========================================================================

def main():
    """
    Command-line interface for single-image pothole detection.

    Examples:
      python ml/predict_image.py --image road.jpg
      python ml/predict_image.py --image road.jpg --demo
      python ml/predict_image.py --image road.jpg --save output.jpg
    """
    parser = argparse.ArgumentParser(
        description="Pothole Detection — Single Image Inference"
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input image file (jpg/png)."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode with fake detections (no model needed)."
    )
    parser.add_argument(
        "--save", default=None,
        help="Optional path to save the annotated output image."
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Run detection or demo
    # ------------------------------------------------------------------
    if args.demo:
        detections, annotated = detect_demo(args.image)
    else:
        detections, annotated = predict_image(args.image)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    if not detections:
        print("No potholes detected (or model is missing — use --demo to test).")
    else:
        print(f"\n{'='*50}")
        print(f"DETECTIONS  ({len(detections)} pothole(s) found)")
        print(f"{'='*50}")
        for i, d in enumerate(detections, 1):
            print(
                f"  [{i}] Box=({d['x1']},{d['y1']})-({d['x2']},{d['y2']})  "
                f"Conf={d['confidence']:.4f}  "
                f"Severity={d['severity']}  "
                f"AreaRatio={d['area_ratio']:.4f}"
            )

    # ------------------------------------------------------------------
    # Optionally save / show the annotated image
    # ------------------------------------------------------------------
    if annotated is not None:
        import cv2
        if args.save:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), annotated)
            print(f"\nAnnotated image saved to: {save_path}")
        else:
            # Show in a window (press any key to close)
            cv2.imshow("Pothole Detection", annotated)
            print("\nPress any key in the image window to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
