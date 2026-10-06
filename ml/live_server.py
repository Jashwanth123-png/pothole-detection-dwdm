"""
ml/live_server.py
=================
Lightweight HTTP API server for real-time live webcam frame inference.
Connects the browser's WebRTC webcam feed to the real YOLO model (models/best.pt).

Endpoints:
  - GET  /api/health       -> Server status and model info
  - POST /api/detect       -> Real YOLO inference on webcam frame
  - POST /api/save         -> Save detection record to data warehouse
"""

import os
import sys
import json
import base64
import time
import logging
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from datetime import date, datetime

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from utils.severity import calculate_severity, SeverityCalculator
from utils.helpers import setup_logger

logger = setup_logger(__name__)

# Global singleton model cache
_MODEL = None
_MODEL_LOCK = threading.Lock()
_SERVER_INSTANCE = None
_SERVER_THREAD = None
_SERVER_PORT = 8502


def get_yolo_model(model_path: str = None):
    """Load and cache the real YOLO model from models/best.pt."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        if model_path is None:
            best_pt = PROJECT_ROOT / "models" / "best.pt"
            pothole_pt = PROJECT_ROOT / "models" / "pothole_yolo.pt"
            if not best_pt.exists() or best_pt.stat().st_size <= 5_000_000:
                try:
                    from setup_cloud import ensure_model
                    ensure_model()
                except Exception:
                    pass

            if best_pt.exists() and best_pt.stat().st_size > 5_000_000:
                model_path = str(best_pt)
            elif pothole_pt.exists():
                model_path = str(pothole_pt)
            else:
                model_path = str(best_pt)

        try:
            from ultralytics import YOLO
            logger.info(f"Loading real YOLO model from: {model_path}")
            _MODEL = YOLO(model_path)
            logger.info(f"YOLO model loaded successfully. Classes: {_MODEL.names}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {model_path}: {e}")
            raise e

    return _MODEL


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads for low latency."""
    daemon_threads = True
    allow_reuse_address = True


class LiveDetectionHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for live webcam frame detection."""

    def log_message(self, format, *args):
        # Suppress noisy standard HTTP access logs
        return

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200, "OK")
        self._set_cors_headers()
        self.end_headers()

    def _send_json(self, status_code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/api/health", "/api/status", "/health"):
            model = get_yolo_model()
            names = getattr(model, "names", {0: "D40 Pothole"})
            response = {
                "status": "healthy",
                "model_loaded": True,
                "model_path": "models/best.pt",
                "classes": names,
                "rdd2022_class": "D40 Pothole",
                "server_time": datetime.now().isoformat(),
            }
            self._send_json(200, response)
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/api/detect":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))

                # 1. Parse image data (base64)
                img_data = data.get("image", "")
                if "," in img_data:
                    img_data = img_data.split(",", 1)[1]
                img_bytes = base64.b64decode(img_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    raise ValueError("Could not decode frame image")

                conf_thresh = float(data.get("conf", 0.20))
                img_h, img_w = frame.shape[:2]
                img_area = max(img_h * img_w, 1)

                # 2. Run real YOLO inference
                model = get_yolo_model()
                t0 = time.perf_counter()
                results = model.predict(frame, conf=conf_thresh, verbose=False)
                inference_ms = round((time.perf_counter() - t0) * 1000, 1)

                # 3. Process detections
                detections = []
                sev_counts = {"High": 0, "Medium": 0, "Low": 0}

                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        conf = round(float(box.conf[0]), 3)
                        cls_id = int(box.cls[0])
                        # Map to RDD2022 label
                        cls_name = "D40 Pothole" if cls_id == 0 else f"Class_{cls_id}"

                        box_area = max(0, (x2 - x1) * (y2 - y1))
                        area_ratio = round(box_area / img_area, 5)
                        severity = calculate_severity(conf, area_ratio)
                        sev_counts[severity] = sev_counts.get(severity, 0) + 1

                        detections.append({
                            "box": [x1, y1, x2, y2],
                            "class": cls_name,
                            "class_id": cls_id,
                            "confidence": conf,
                            "severity": severity,
                            "area_ratio": area_ratio,
                        })

                n_det = len(detections)
                top_sev = "High" if sev_counts["High"] > 0 else (
                    "Medium" if sev_counts["Medium"] > 0 else ("Low" if n_det > 0 else "None")
                )
                avg_conf = round(sum(d["confidence"] for d in detections) / max(n_det, 1), 3) if n_det > 0 else 0.0

                resp = {
                    "success": True,
                    "pothole_count": n_det,
                    "avg_confidence": avg_conf,
                    "top_severity": top_sev,
                    "detections": detections,
                    "inference_time_ms": inference_ms,
                    "frame_dims": {"width": img_w, "height": img_h},
                }

                self._send_json(200, resp)

            except Exception as e:
                logger.error(f"Inference error in /api/detect: {e}")
                self._send_json(500, {"success": False, "error": str(e)})

        elif self.path == "/api/save":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))

                from warehouse.database import insert_detection

                CITY_COORDS = {
                    "Bangalore": (12.9716, 77.5946), "Mumbai": (19.0760, 72.8777),
                    "Delhi": (28.6139, 77.2090), "Chennai": (13.0827, 80.2707),
                    "Hyderabad": (17.3850, 78.4867), "Pune": (18.5204, 73.8567),
                    "Kolkata": (22.5726, 88.3639),
                }

                city = data.get("city", "Bangalore")
                lat, lon = CITY_COORDS.get(city, (12.9716, 77.5946))

                rec = {
                    "pothole_count": int(data.get("pothole_count", 1)),
                    "confidence": float(data.get("confidence", 0.75)),
                    "severity": data.get("severity", "Medium"),
                    "source_type": "webcam",
                    "detection_date": str(date.today()),
                    "city": city,
                    "area": data.get("area", "Indiranagar"),
                    "road_name": data.get("road_name", "100 Feet Road"),
                    "road_type": data.get("road_type", "Urban"),
                    "latitude": lat,
                    "longitude": lon,
                }

                det_id = insert_detection(rec)
                self._send_json(200, {"success": True, "det_id": det_id})

            except Exception as e:
                logger.error(f"Save error in /api/save: {e}")
                self._send_json(500, {"success": False, "error": str(e)})
        else:
            self._send_json(404, {"error": "Not found"})


def start_live_server(port: int = _SERVER_PORT, host: str = "127.0.0.1") -> bool:
    """Start the live detection HTTP server in a daemon thread."""
    global _SERVER_INSTANCE, _SERVER_THREAD

    # Check if already running on the port
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        s.close()
    except OSError:
        # Port already in use, verify it's our server
        try:
            import urllib.request
            req = urllib.request.Request(f"http://{host}:{port}/api/health")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    logger.info(f"Live detection server already running on http://{host}:{port}")
                    return True
        except Exception:
            logger.warning(f"Port {port} is occupied by another process.")
            return False

    try:
        _SERVER_INSTANCE = ThreadedHTTPServer((host, port), LiveDetectionHandler)
        _SERVER_THREAD = threading.Thread(target=_SERVER_INSTANCE.serve_forever, daemon=True)
        _SERVER_THREAD.start()
        logger.info(f"Started live detection server on http://{host}:{port}")
        return True
    except Exception as e:
        logger.error(f"Could not start live detection server: {e}")
        return False


def ensure_live_server(port: int = _SERVER_PORT) -> bool:
    """Ensure the live detection API server is running."""
    # Preload YOLO model in background thread
    try:
        threading.Thread(target=get_yolo_model, daemon=True).start()
    except Exception:
        pass
    return start_live_server(port=port)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", _SERVER_PORT))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Standalone Live YOLO Detection Server on http://{host}:{port}...")
    server = ThreadedHTTPServer((host, port), LiveDetectionHandler)
    # Warm up model
    get_yolo_model()
    print(f"Ready. Listening for frames on http://{host}:{port}/api/detect")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.shutdown()
