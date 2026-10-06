"""
ml/predict_video.py
===================
Video inference module for pothole detection.

This module provides:
  - predict_video()                : Process a video file and return frame-level detections.
  - process_video_for_warehouse()  : Process video and return records ready for DB insertion.
  - main()                         : CLI entry point.

Prerequisites:
  - Trained model at models/pothole_yolo.pt
    (run python ml/train_yolo.py first)
  - Install: pip install ultralytics opencv-python

Usage (CLI):
  python ml/predict_video.py --video path/to/road.mp4
  python ml/predict_video.py --video path/to/road.mp4 --output annotated.mp4
  python ml/predict_video.py --video path/to/road.mp4 --demo
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root so we can import our own modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from utils.helpers     import load_config, get_project_root, setup_logger
from utils.severity    import calculate_severity
from utils.preprocessing import resize_image

logger = setup_logger(__name__)

# Message shown when the model file is missing
NO_MODEL_MSG = (
    "\nNOTE: Real detection requires a trained YOLO model. See models/README.md\n"
    "  Train the model first:  python ml/train_yolo.py\n"
    "  Use --demo flag to run with fake detections for testing.\n"
)


# ==========================================================================
# Core video prediction
# ==========================================================================

def predict_video(
    video_path: str,
    output_path: str = None,
    config: dict = None,
    demo_mode: bool = False,
):
    """
    Process a video file for pothole detection frame by frame.

    Behaviour:
      - Only every `frame_skip`-th frame is sent through the model
        (configurable in config.yaml → model.frame_skip). This keeps
        processing fast without a GPU.
      - If `output_path` is provided, an annotated video is written to disk.
      - Frames where the model was skipped are still included in the output
        video, but without new bounding boxes (the last known boxes are NOT
        carried over — they are cleared so the video is honest).

    Args:
        video_path  (str | Path):         Path to the input video file.
        output_path (str | Path, optional): Path for the annotated output video.
                                            Pass None to skip saving.
        config      (dict, optional):      Project config. Loads from config.yaml if None.
        demo_mode   (bool):               If True, generate fake detections (no model needed).

    Returns:
        tuple:
          - frame_detections (list[dict]):  One dict per *processed* frame:
              {frame_idx, timestamp_sec, detections (list), pothole_count}
          - total_pothole_count (int):      Sum of detections across all processed frames.
          - processed_frame_count (int):    Number of frames actually sent to the model.
    """
    if config is None:
        config = load_config()

    project_root = get_project_root()

    # ------------------------------------------------------------------
    # 1. Validate video path
    # ------------------------------------------------------------------
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        print(f"ERROR: Video file not found: {video_path}")
        return [], 0, 0

    # ------------------------------------------------------------------
    # 2. Load config values
    # ------------------------------------------------------------------
    model_cfg  = config.get("model", {})
    frame_skip = model_cfg.get("frame_skip", 5)        # process every N-th frame
    image_size = model_cfg.get("image_size", 416)
    conf_thresh = model_cfg.get("confidence_threshold", 0.25)
    device     = model_cfg.get("device", "cpu")
    model_path = project_root / model_cfg.get("yolo_model_path", "models/pothole_yolo.pt")

    # ------------------------------------------------------------------
    # 3. Load YOLO model (unless demo mode)
    # ------------------------------------------------------------------
    model = None

    if not demo_mode:
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            print(NO_MODEL_MSG)
            print(f"Expected model at: {model_path}")
            print("Falling back to DEMO mode (fake detections).")
            demo_mode = True
        else:
            try:
                from ultralytics import YOLO
                model = YOLO(str(model_path))
                logger.info(f"Loaded YOLO model from {model_path}")
            except ImportError:
                logger.error("ultralytics not installed.")
                print("ERROR: pip install ultralytics")
                return [], 0, 0
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                print(f"ERROR loading model: {e}")
                return [], 0, 0

    # ------------------------------------------------------------------
    # 4. Open video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        print(f"ERROR: Could not open video: {video_path}")
        return [], 0, 0

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {video_path.name}")
    print(f"  Resolution : {frame_w}x{frame_h}")
    print(f"  FPS        : {fps:.1f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Frame skip : every {frame_skip} frames")
    if demo_mode:
        print("  Mode       : DEMO (fake detections)")
    print()

    # ------------------------------------------------------------------
    # 5. Set up video writer (if output_path given)
    # ------------------------------------------------------------------
    writer = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    # ------------------------------------------------------------------
    # 6. Frame processing loop
    # ------------------------------------------------------------------
    frame_detections   = []   # results for each processed frame
    total_pothole_count = 0
    processed_frame_count = 0
    frame_idx = 0

    logger.info(f"Processing video: {video_path.name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        timestamp_sec = frame_idx / fps

        # Only run the model on every `frame_skip`-th frame
        if frame_idx % frame_skip == 0:
            detections = _run_detection_on_frame(
                frame, model, image_size, conf_thresh, device, demo_mode
            )

            pothole_count = len(detections)
            total_pothole_count += pothole_count
            processed_frame_count += 1

            # Annotate this frame if we have detections
            annotated_frame = frame.copy()
            if detections:
                annotated_frame = _draw_boxes_on_frame(annotated_frame, detections)

            frame_detections.append({
                "frame_idx":     frame_idx,
                "timestamp_sec": round(timestamp_sec, 3),
                "detections":    detections,
                "pothole_count": pothole_count,
            })

            # Progress indicator every 50 processed frames
            if processed_frame_count % 50 == 0:
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                logger.info(f"  Frame {frame_idx}/{total_frames} ({pct:.1f}%)  "
                            f"potholes so far: {total_pothole_count}")
                print(f"  Progress: {pct:.1f}%  |  Potholes detected: {total_pothole_count}")
        else:
            annotated_frame = frame  # pass through un-annotated frames unchanged

        if writer:
            writer.write(annotated_frame)

        frame_idx += 1

    # ------------------------------------------------------------------
    # 7. Cleanup
    # ------------------------------------------------------------------
    cap.release()
    if writer:
        writer.release()
        logger.info(f"Annotated video saved: {output_path}")
        print(f"\nAnnotated video saved to: {output_path}")

    logger.info(
        f"Video processing complete. "
        f"Processed {processed_frame_count} frames, "
        f"found {total_pothole_count} pothole detections."
    )
    print(f"\n{'='*50}")
    print(f"DONE  |  Processed frames : {processed_frame_count}")
    print(f"      |  Total detections : {total_pothole_count}")
    print(f"{'='*50}")

    return frame_detections, total_pothole_count, processed_frame_count


# ==========================================================================
# Warehouse-ready record builder
# ==========================================================================

def process_video_for_warehouse(
    video_path: str,
    location_info: dict,
    config: dict = None,
    demo_mode: bool = False,
):
    """
    Process a video and return detection records ready for data warehouse insertion.

    Instead of one database record per frame (which would be thousands), this
    function groups detections into *time windows* (default: 5-second buckets)
    and creates one aggregated record per window.  This is much more useful for
    analytical queries (e.g. "How many potholes were seen per minute?").

    Args:
        video_path    (str | Path): Path to the input video file.
        location_info (dict):       Metadata about the recording location, e.g.:
                                    {
                                      "city": "Mumbai",
                                      "road": "NH-48",
                                      "latitude": 19.076,
                                      "longitude": 72.877,
                                    }
        config        (dict, optional): Project config. Loads from config.yaml if None.
        demo_mode     (bool):       If True, use fake detections.

    Returns:
        list[dict]: Records suitable for inserting into the fact_detections table.
                    Each record has these keys:
                      video_path, city, road, latitude, longitude,
                      window_start_sec, window_end_sec,
                      pothole_count, avg_confidence, max_severity,
                      processed_at
    """
    if config is None:
        config = load_config()

    # Default time window: 5 seconds
    window_sec = config.get("warehouse", {}).get("video_time_window_sec", 5)

    # Run the video through the detector
    frame_detections, total_count, proc_count = predict_video(
        video_path, output_path=None, config=config, demo_mode=demo_mode
    )

    if not frame_detections:
        logger.warning("No frame detections returned; returning empty record list.")
        return []

    # ------------------------------------------------------------------
    # Group frames into time windows
    # ------------------------------------------------------------------
    # bucket_key = integer(timestamp_sec / window_sec)
    buckets = {}    # bucket_key → list of detection dicts

    for frame_info in frame_detections:
        if frame_info["pothole_count"] == 0:
            continue   # skip frames with no detections

        bucket_key = int(frame_info["timestamp_sec"] / window_sec)
        buckets.setdefault(bucket_key, [])
        buckets[bucket_key].extend(frame_info["detections"])

    # ------------------------------------------------------------------
    # Build one warehouse record per bucket
    # ------------------------------------------------------------------
    severity_rank = {"low": 1, "medium": 2, "high": 3}   # for finding max
    records = []
    now_str = datetime.now().isoformat()

    for bucket_key in sorted(buckets.keys()):
        dets = buckets[bucket_key]   # all detections in this time window

        confidences = [d["confidence"] for d in dets]
        severities  = [d.get("severity", "low") for d in dets]
        max_severity = max(severities, key=lambda s: severity_rank.get(s, 0))

        record = {
            # --- Location info (passed in by the caller) ---
            "video_path":       str(video_path),
            "city":             location_info.get("city",      "Unknown"),
            "road":             location_info.get("road",      "Unknown"),
            "latitude":         location_info.get("latitude",  None),
            "longitude":        location_info.get("longitude", None),

            # --- Time window ---
            "window_start_sec": round(bucket_key * window_sec, 3),
            "window_end_sec":   round((bucket_key + 1) * window_sec, 3),

            # --- Aggregate detection metrics ---
            "pothole_count":    len(dets),
            "avg_confidence":   round(sum(confidences) / len(confidences), 4),
            "max_severity":     max_severity,

            # --- Metadata ---
            "processed_at":     now_str,
        }
        records.append(record)

    logger.info(f"Generated {len(records)} warehouse records from video.")
    print(f"\nGenerated {len(records)} warehouse records (grouped into {window_sec}s windows).")
    return records


# ==========================================================================
# Internal helpers
# ==========================================================================

def _run_detection_on_frame(
    frame: np.ndarray,
    model,
    image_size: int,
    conf_thresh: float,
    device: str,
    demo_mode: bool,
) -> list:
    """
    Run detection on a single BGR numpy frame.

    Args:
        frame      : BGR image as numpy array.
        model      : Loaded YOLO model (None in demo mode).
        image_size : YOLO input image size.
        conf_thresh: Minimum confidence threshold.
        device     : 'cpu' or CUDA device id.
        demo_mode  : If True generate random fake detections.

    Returns:
        list[dict]: Detections for this frame.
    """
    import random  # used in demo mode

    img_h, img_w = frame.shape[:2]
    img_area     = img_h * img_w

    if demo_mode:
        # 30% chance of a pothole appearing in any given processed frame
        if random.random() > 0.30:
            return []

        num_fake = random.randint(1, 2)
        detections = []
        for _ in range(num_fake):
            x1 = random.randint(0, img_w // 2)
            y1 = random.randint(0, img_h // 2)
            x2 = random.randint(x1 + 20, min(x1 + img_w // 4, img_w - 1))
            y2 = random.randint(y1 + 20, min(y1 + img_h // 4, img_h - 1))
            conf = round(random.uniform(0.4, 0.9), 4)
            box_area   = (x2 - x1) * (y2 - y1)
            area_ratio = round(box_area / img_area, 6) if img_area > 0 else 0.0
            severity   = calculate_severity(conf, area_ratio)
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": conf, "severity": severity, "area_ratio": area_ratio,
            })
        return detections

    # Real YOLO inference on the frame (passed as numpy array)
    try:
        results = model.predict(
            source=frame,
            imgsz=image_size,
            conf=conf_thresh,
            device=device,
            verbose=False,
        )
    except Exception as e:
        logger.warning(f"Frame inference failed: {e}")
        return []

    detections = []
    result = results[0]

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf       = float(box.conf[0])
            box_area   = (x2 - x1) * (y2 - y1)
            area_ratio = round(box_area / img_area, 6) if img_area > 0 else 0.0
            severity   = calculate_severity(conf, area_ratio)
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": round(conf, 4),
                "severity":   severity,
                "area_ratio": area_ratio,
            })

    return detections


def _draw_boxes_on_frame(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes with severity-based colors on a video frame."""
    severity_colors = {
        "low":    (0, 255, 0),    # green
        "medium": (0, 165, 255),  # orange
        "high":   (0, 0, 255),    # red
    }

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        severity  = det.get("severity", "low")
        conf      = det.get("confidence", 0)
        color     = severity_colors.get(severity, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{conf:.2f} [{severity}]"
        cv2.putText(
            frame, label, (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )

    return frame


# ==========================================================================
# CLI entry point
# ==========================================================================

def main():
    """
    Command-line interface for video pothole detection.

    Examples:
      python ml/predict_video.py --video road.mp4
      python ml/predict_video.py --video road.mp4 --output annotated.mp4
      python ml/predict_video.py --video road.mp4 --demo
    """
    parser = argparse.ArgumentParser(
        description="Pothole Detection — Video Inference"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the input video file (mp4, avi, etc.)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save the annotated output video."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode with fake detections (no model needed)."
    )
    parser.add_argument(
        "--warehouse", action="store_true",
        help="Also print warehouse-ready grouped records."
    )
    args = parser.parse_args()

    frame_dets, total, proc = predict_video(
        args.video,
        output_path=args.output,
        demo_mode=args.demo,
    )

    if args.warehouse:
        print("\n--- Warehouse Records (sample) ---")
        location = {"city": "Demo City", "road": "Demo Road"}
        records = process_video_for_warehouse(
            args.video, location, demo_mode=args.demo
        )
        for r in records[:5]:   # show first 5 records
            print(r)
        if len(records) > 5:
            print(f"  ... and {len(records) - 5} more records.")


if __name__ == "__main__":
    main()
