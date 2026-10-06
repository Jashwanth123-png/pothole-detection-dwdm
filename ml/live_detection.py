"""
ml/live_detection.py
====================
Real-time pothole detection using a webcam (or any connected camera).

This module provides:
  - LiveDetector class : Object-oriented wrapper around the detection loop.
  - run_live_detection(): Convenience function for standalone usage.
  - main()              : CLI entry point.

Prerequisites:
  - Trained model at models/pothole_yolo.pt
    (run python ml/train_yolo.py first)
  - A connected camera (webcam or USB camera)
  - Install: pip install ultralytics opencv-python

Usage (CLI):
  python ml/live_detection.py
  python ml/live_detection.py --camera 0        # use default camera
  python ml/live_detection.py --demo            # fake detections, no model needed
  python ml/live_detection.py --save output.mp4 # save the session to a video file

Controls during detection:
  Press 'q' in the video window to stop.
"""

import os
import sys
import argparse
import logging
import time
import random
from pathlib import Path
from datetime import datetime

# Add project root so we can import our own modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from utils.helpers     import load_config, get_project_root, setup_logger
from utils.severity    import calculate_severity
from utils.preprocessing import resize_image

logger = setup_logger(__name__)

# Message shown when the model is missing
NO_MODEL_MSG = (
    "\nNOTE: Real detection requires a trained YOLO model. See models/README.md\n"
    "  Train the model first:  python ml/train_yolo.py\n"
    "  Use --demo flag to run with fake detections.\n"
)


# ==========================================================================
# LiveDetector class
# ==========================================================================

class LiveDetector:
    """
    Real-time pothole detector using a YOLO model and an OpenCV video capture.

    Typical usage:
        detector = LiveDetector()
        detector.start(camera_index=0)   # blocks until 'q' is pressed

    Or, for manual frame-by-frame control:
        detector = LiveDetector()
        detector._open_camera(0)
        while True:
            ret, frame = detector.cap.read()
            detections = detector.process_frame(frame)
            ...
        detector.stop()
    """

    def __init__(self, config: dict = None, demo_mode: bool = False):
        """
        Initialise the LiveDetector.

        Args:
            config    (dict, optional): Project config. Loads from config.yaml if None.
            demo_mode (bool):           If True, use fake detections (no model needed).
        """
        self.config    = config if config is not None else load_config()
        self.demo_mode = demo_mode

        project_root   = get_project_root()
        model_cfg      = self.config.get("model", {})

        self.image_size   = model_cfg.get("image_size", 416)
        self.conf_thresh  = model_cfg.get("confidence_threshold", 0.25)
        self.device       = model_cfg.get("device", "cpu")
        self.frame_skip   = model_cfg.get("frame_skip", 5)

        self.model = None       # loaded lazily in start()
        self.cap   = None       # OpenCV VideoCapture
        self.writer = None      # VideoWriter for saving sessions

        self._frame_count     = 0    # total frames read from camera
        self._detection_count = 0    # total potholes detected in this session

        # Try loading the model immediately so we can warn the user early
        model_path = project_root / model_cfg.get("yolo_model_path", "models/pothole_yolo.pt")
        self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: Path):
        """
        Load the YOLO model from disk.

        If the model file does not exist, or if ultralytics is not installed,
        the detector falls back to demo mode automatically.

        Args:
            model_path (Path): Absolute path to the .pt weights file.
        """
        if self.demo_mode:
            logger.info("Demo mode: skipping model load.")
            return

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}. Switching to demo mode.")
            print(NO_MODEL_MSG)
            self.demo_mode = True
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
            logger.info(f"YOLO model loaded from: {model_path}")
            print(f"Model loaded: {model_path}")
        except ImportError:
            logger.error("ultralytics not installed. pip install ultralytics")
            print("ERROR: pip install ultralytics  — switching to demo mode.")
            self.demo_mode = True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            print(f"ERROR loading model: {e}  — switching to demo mode.")
            self.demo_mode = True

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def _open_camera(self, camera_index: int = 0) -> bool:
        """
        Open the camera at the given device index.

        Args:
            camera_index (int): OpenCV camera index (0 = default webcam).

        Returns:
            bool: True if the camera opened successfully, False otherwise.
        """
        # On Windows, try DirectShow (cv2.CAP_DSHOW) first for instant hardware connection
        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_index)
        else:
            self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            logger.error(f"Cannot open camera index {camera_index}.")
            print(f"\nERROR: Could not open camera {camera_index}.")
            print("  • Make sure a webcam is connected.")
            print("  • Try a different --camera index (e.g. --camera 1).")
            return False

        # Set 640x480 resolution for fast, responsive real-time inference
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Log camera resolution
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        logger.info(f"Camera {camera_index} opened: {w}x{h} @ {fps:.1f}fps")
        print(f"Camera opened: {w}x{h}  |  frame_skip={self.frame_skip}")
        return True

    def stop(self):
        """
        Release the camera and close any video writer.

        Call this when you are done with the detector, even if an exception
        occurred — it is safe to call multiple times.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        cv2.destroyAllWindows()
        logger.info(
            f"Session ended. Frames: {self._frame_count}, "
            f"Detections: {self._detection_count}"
        )

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> list:
        """
        Run pothole detection on a single BGR frame.

        This is the core inference method. It is called by the detection
        loop for every `frame_skip`-th frame.

        Args:
            frame (np.ndarray): A BGR image as a numpy array (from cv2.read()).

        Returns:
            list[dict]: Detections, each with keys:
                        x1, y1, x2, y2, confidence, severity, area_ratio.
                        Empty list if no potholes detected.
        """
        img_h, img_w = frame.shape[:2]
        img_area     = img_h * img_w
        detections   = []

        if self.demo_mode:
            # Randomly generate 0–2 fake detections
            if random.random() < 0.25:    # 25% chance of a detection
                for _ in range(random.randint(1, 2)):
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
                        "confidence": conf, "severity": severity,
                        "area_ratio": area_ratio,
                    })
            return detections

        # Real YOLO inference
        if self.model is None:
            return []

        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.image_size,
                conf=self.conf_thresh,
                device=self.device,
                verbose=False,
            )
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
        except Exception as e:
            logger.warning(f"Frame inference error: {e}")

        return detections

    # ------------------------------------------------------------------
    # Warehouse record builder
    # ------------------------------------------------------------------

    def get_detection_record(
        self,
        frame: np.ndarray,
        detections: list,
        location_info: dict,
    ) -> dict:
        """
        Build a warehouse-ready detection record from a frame's detections.

        This is only called when `detections` is non-empty, so we never
        insert empty records into the database.

        Args:
            frame         (np.ndarray): The frame in which detections were found.
            detections    (list[dict]): Detections from process_frame().
            location_info (dict):       E.g. {"city": "Delhi", "road": "Ring Road"}.

        Returns:
            dict: A flat record suitable for inserting into fact_detections.
        """
        if not detections:
            return {}

        confidences  = [d["confidence"] for d in detections]
        severity_rank = {"low": 1, "medium": 2, "high": 3}
        severities   = [d.get("severity", "low") for d in detections]
        max_severity = max(severities, key=lambda s: severity_rank.get(s, 0))

        return {
            "detected_at":      datetime.now().isoformat(),
            "frame_index":      self._frame_count,
            "pothole_count":    len(detections),
            "avg_confidence":   round(sum(confidences) / len(confidences), 4),
            "max_severity":     max_severity,
            "city":             location_info.get("city",  "Unknown"),
            "road":             location_info.get("road",  "Unknown"),
            "latitude":         location_info.get("latitude",  None),
            "longitude":        location_info.get("longitude", None),
            "source":           "live_camera",
        }

    # ------------------------------------------------------------------
    # Main detection loop
    # ------------------------------------------------------------------

    def start(
        self,
        camera_index: int = 0,
        location_info: dict = None,
        save_path: str = None,
    ) -> list:
        """
        Start the webcam capture and detection loop.

        This method blocks until the user presses 'q' or the camera stream ends.

        Args:
            camera_index  (int):  OpenCV camera index (0 = default webcam).
            location_info (dict): Optional location metadata for warehouse records.
            save_path     (str):  Optional path to save the session as an .mp4 video.

        Returns:
            list[dict]: All warehouse records created during the session
                        (one per frame that had at least one detection).
        """
        if location_info is None:
            location_info = {"city": "Unknown", "road": "Unknown"}

        # Open camera
        if not self._open_camera(camera_index):
            return []

        # Set up video writer if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))
            print(f"Recording to: {save_path}")

        mode_label = "DEMO" if self.demo_mode else "LIVE"
        print(f"\n{'='*55}")
        print(f"  Pothole Detection — {mode_label} MODE")
        print(f"  Press 'q' in the video window to stop.")
        print(f"{'='*55}\n")

        warehouse_records = []
        severity_colors   = {
            "low":    (0, 255, 0),
            "medium": (0, 165, 255),
            "high":   (0, 0, 255),
        }

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to grab frame — camera may have disconnected.")
                print("WARNING: Lost camera feed.")
                break

            self._frame_count += 1
            detections = []

            # Only run inference on every frame_skip-th frame
            if self._frame_count % self.frame_skip == 0:
                detections = self.process_frame(frame)
                self._detection_count += len(detections)

                # Build warehouse record if potholes were found
                if detections:
                    record = self.get_detection_record(frame, detections, location_info)
                    warehouse_records.append(record)

            # ----------------------------------------------------------
            # Draw detections on the frame
            # ----------------------------------------------------------
            display = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                sev   = det.get("severity", "low")
                conf  = det.get("confidence", 0)
                color = severity_colors.get(sev, (255, 255, 255))

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                label = f"{conf:.2f} [{sev}]"
                cv2.putText(
                    display, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

            # Overlay HUD (status bar at the top)
            hud = (
                f"Frame:{self._frame_count}  "
                f"Detections:{self._detection_count}  "
                f"Mode:{mode_label}  "
                f"[Press 'q' to quit]"
            )
            cv2.putText(
                display, hud, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )

            # Write to disk if recording
            if self.writer:
                self.writer.write(display)

            # Show in window
            cv2.imshow("Pothole Detection — Live", display)

            # 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nUser pressed 'q' — stopping.")
                break

        self.stop()

        # Final summary
        print(f"\n{'='*50}")
        print(f"Session summary")
        print(f"  Total frames  : {self._frame_count}")
        print(f"  Potholes found: {self._detection_count}")
        print(f"  DB records    : {len(warehouse_records)}")
        print(f"{'='*50}")

        return warehouse_records


# ==========================================================================
# Convenience function
# ==========================================================================

def run_live_detection(config: dict = None, demo_mode: bool = False) -> list:
    """
    Convenience wrapper to start live detection with default settings.

    Args:
        config    (dict, optional): Project config. Loads from config.yaml if None.
        demo_mode (bool):           If True, use fake detections.

    Returns:
        list[dict]: Warehouse records from the session.
    """
    if config is None:
        config = load_config()

    detector = LiveDetector(config=config, demo_mode=demo_mode)
    records  = detector.start(camera_index=0)
    return records


# ==========================================================================
# CLI entry point
# ==========================================================================

def main():
    """
    Command-line interface for live pothole detection.

    Examples:
      python ml/live_detection.py
      python ml/live_detection.py --camera 1
      python ml/live_detection.py --demo
      python ml/live_detection.py --demo --save session.mp4
    """
    parser = argparse.ArgumentParser(
        description="Pothole Detection — Live Camera Inference"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0 = built-in webcam)."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Demo mode: generate fake detections (no model needed)."
    )
    parser.add_argument(
        "--save", default=None,
        help="Optional path to save the recorded session (e.g. session.mp4)."
    )
    parser.add_argument(
        "--city", default="Unknown",
        help="City name for warehouse records."
    )
    parser.add_argument(
        "--road", default="Unknown",
        help="Road name for warehouse records."
    )
    args = parser.parse_args()

    location_info = {
        "city": args.city,
        "road": args.road,
    }

    print("\n" + "=" * 55)
    print("  Pothole Live Detection")
    print(f"  Camera index : {args.camera}")
    print(f"  Demo mode    : {args.demo}")
    if args.save:
        print(f"  Recording to : {args.save}")
    print("=" * 55)

    detector = LiveDetector(demo_mode=args.demo)
    records  = detector.start(
        camera_index=args.camera,
        location_info=location_info,
        save_path=args.save,
    )

    if records:
        print(f"\n{len(records)} warehouse records were generated during this session.")
    else:
        print("\nNo potholes detected during the session.")


if __name__ == "__main__":
    main()
