"""
tests/test_detection.py
=======================
Automated test suite verifying the real-time detection pipeline:
- YOLO model loading and class validation
- Single-image detection pipeline (predict_image)
- Video detection pipeline (predict_video)
- LiveDetector processing pipeline
- Severity calculation thresholds
"""

import os
from pathlib import Path
import pytest
import numpy as np
import cv2

from utils.helpers import load_config, get_project_root
from utils.severity import calculate_severity, SeverityCalculator
from ml.predict_image import predict_image
from ml.predict_video import predict_video
from ml.live_detection import LiveDetector


@pytest.fixture(scope="module")
def root_dir():
    return get_project_root()


@pytest.fixture(scope="module")
def test_image_path(root_dir):
    """Path to a verified test image with potholes."""
    p = root_dir / "data" / "processed" / "split" / "test" / "images" / "China_Drone_000071.jpg"
    if not p.exists():
        # Fallback to any test image
        test_imgs = list((root_dir / "data" / "processed" / "split" / "test" / "images").glob("*.jpg"))
        assert len(test_imgs) > 0, "No test images found"
        p = test_imgs[0]
    return str(p)


@pytest.fixture(scope="module")
def test_video_path(root_dir):
    """Path to sample test road video."""
    p = root_dir / "data" / "demo" / "test_road.mp4"
    assert p.exists(), f"Sample road video not found at {p}"
    return str(p)


# --------------------------------------------------------------------------
# 1. Model Loading & Verification Tests
# --------------------------------------------------------------------------

def test_model_files_exist(root_dir):
    """Verify trained model files exist and have non-trivial size."""
    best_pt = root_dir / "models" / "best.pt"
    yolo_pt = root_dir / "models" / "pothole_yolo.pt"
    assert best_pt.exists(), "models/best.pt does not exist"
    assert best_pt.stat().st_size > 5_000_000, f"models/best.pt is unexpectedly small: {best_pt.stat().st_size} bytes"
    assert yolo_pt.exists(), "models/pothole_yolo.pt does not exist"
    assert yolo_pt.stat().st_size > 5_000_000, f"models/pothole_yolo.pt is unexpectedly small: {yolo_pt.stat().st_size} bytes"


def test_ultralytics_yolo_loads(root_dir):
    """Verify Ultralytics YOLO loads the model successfully and exposes expected task and classes."""
    from ultralytics import YOLO
    model_path = root_dir / "models" / "best.pt"
    model = YOLO(str(model_path))
    assert model is not None
    assert model.task == "detect"
    assert hasattr(model, "names")
    # Verify model class mapping (trained on RDD2022 D40 pothole)
    assert 0 in model.names
    assert "pothole" in model.names[0].lower() or "d40" in model.names[0].lower()


# --------------------------------------------------------------------------
# 2. Image Detection Pipeline Tests
# --------------------------------------------------------------------------

def test_predict_image_real_inference(test_image_path):
    """Test single-image real inference with YOLO model."""
    config = load_config()
    # Ensure confidence threshold allows detections
    config["model"]["confidence_threshold"] = 0.25

    detections, annotated = predict_image(test_image_path, config=config)

    assert isinstance(detections, list)
    assert annotated is not None
    assert isinstance(annotated, np.ndarray)
    assert annotated.ndim == 3  # BGR image

    # China_Drone_000071 has a clear pothole detection at conf > 0.7
    if "China_Drone_000071" in test_image_path:
        assert len(detections) >= 1
        d = detections[0]
        assert "x1" in d and "y1" in d and "x2" in d and "y2" in d
        assert "confidence" in d and d["confidence"] > 0.5
        assert "severity" in d and d["severity"] in ["Low", "Medium", "High", "low", "medium", "high"]
        assert "area_ratio" in d and d["area_ratio"] > 0


def test_predict_image_nonexistent_file():
    """Verify predict_image handles nonexistent files gracefully."""
    detections, annotated = predict_image("nonexistent_image_12345.jpg")
    assert detections == []
    assert annotated is None


# --------------------------------------------------------------------------
# 3. Video Detection Pipeline Tests
# --------------------------------------------------------------------------

def test_predict_video_real_inference(test_video_path, root_dir, tmp_path):
    """Test video file inference using real YOLO model."""
    out_video = tmp_path / "out_annotated.mp4"
    config = load_config()
    config["model"]["frame_skip"] = 10  # fast test

    frame_dets, total_potholes, processed_count = predict_video(
        video_path=test_video_path,
        output_path=str(out_video),
        config=config,
        demo_mode=False
    )

    assert processed_count > 0
    assert isinstance(total_potholes, int)
    assert isinstance(frame_dets, list)
    assert out_video.exists()
    assert out_video.stat().st_size > 0


def test_predict_video_nonexistent_file():
    """Verify predict_video handles nonexistent video gracefully."""
    frame_dets, total, processed = predict_video("nonexistent_video.mp4")
    assert frame_dets == []
    assert total == 0
    assert processed == 0


# --------------------------------------------------------------------------
# 4. Live Detection Component Tests
# --------------------------------------------------------------------------

def test_live_detector_init_and_frame_process(test_image_path):
    """Test LiveDetector initialization and single-frame inference."""
    detector = LiveDetector(demo_mode=False)
    assert detector.model is not None
    assert detector.demo_mode is False

    frame = cv2.imread(test_image_path)
    assert frame is not None

    detections = detector.process_frame(frame)
    assert isinstance(detections, list)

    # Check warehouse record builder
    if detections:
        record = detector.get_detection_record(
            frame=frame,
            detections=detections,
            location_info={"city": "Delhi", "road": "Ring Road"}
        )
        assert record["city"] == "Delhi"
        assert record["pothole_count"] == len(detections)
        assert record["source"] == "live_camera"

    # Stop detector safely
    detector.stop()


# --------------------------------------------------------------------------
# 5. Severity Calculation Tests
# --------------------------------------------------------------------------

def test_severity_calculator_logic():
    """Verify severity calculation against project thresholds."""
    calc = SeverityCalculator()

    # Small area (< 0.01) and low conf (< 0.40) -> Low
    assert calc.calculate(area_ratio=0.005, confidence=0.30) == "Low"

    # Medium area (0.02) and medium conf (0.55) -> Medium
    assert calc.calculate(area_ratio=0.02, confidence=0.55) == "Medium"

    # Large area (>= 0.05) and high conf (>= 0.70) -> High
    assert calc.calculate(area_ratio=0.06, confidence=0.85) == "High"
