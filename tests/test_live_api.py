"""Test live detection server API."""
import time
import base64
import json
import urllib.request
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.live_server import ensure_live_server

def test_api():
    assert ensure_live_server(8502), "Server failed to start"
    time.sleep(1.0)
    
    # 1. Health check
    req = urllib.request.Request("http://127.0.0.1:8502/api/health")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        res = json.loads(resp.read().decode())
    print("Health check:", res)
    assert res["status"] == "healthy"
    assert res["rdd2022_class"] == "D40 Pothole"
    assert res["model_path"] == "models/best.pt"

    # 2. Detect endpoint
    img_path = Path("data/processed/augmented/images/China_Drone_000025.jpg")
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    
    post_data = json.dumps({"image": b64, "conf": 0.1}).encode()
    req2 = urllib.request.Request(
        "http://127.0.0.1:8502/api/detect",
        data=post_data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req2, timeout=10.0) as resp2:
        res2 = json.loads(resp2.read().decode())
    
    print("Detect result:")
    print("  Success:", res2["success"])
    print("  Pothole count:", res2["pothole_count"])
    print("  Severity:", res2["top_severity"])
    print("  Inference time ms:", res2["inference_time_ms"])
    if res2["detections"]:
        print("  First detection:", res2["detections"][0])
        d0 = res2["detections"][0]
        assert d0["class"] == "D40 Pothole"
        assert len(d0["box"]) == 4
        assert 0.0 <= d0["confidence"] <= 1.0
        assert d0["severity"] in ("High", "Medium", "Low")
    print("ALL API TESTS PASSED!")

if __name__ == "__main__":
    test_api()
