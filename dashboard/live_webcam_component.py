"""
dashboard/live_webcam_component.py
==================================
Interactive WebRTC Live Webcam Detection Component for Streamlit.
Provides direct browser webcam capture via navigator.mediaDevices.getUserMedia(),
real-time YOLO inference via the local Python API, live bounding box overlays,
RDD2022 D40 class labels, confidence scores, and severity calculation.
"""

def render_live_webcam_html(
    api_url: str = "http://127.0.0.1:8502",
    default_conf: float = 0.20,
    city: str = "Bangalore",
    area: str = "Indiranagar",
    road_name: str = "100 Feet Road",
    road_type: str = "Urban",
) -> str:
    """Generate the self-contained HTML/JS/CSS WebRTC live detection component."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  :root {{
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --border-color: #334155;
    --primary: #3b82f6;
    --primary-hover: #2563eb;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  body {{
    background-color: var(--bg-dark);
    color: var(--text-main);
    padding: 12px;
  }}
  .hud-card {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
  }}
  .status-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 13px;
  }}
  .badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 12px;
  }}
  .badge-idle {{ background: #334155; color: #cbd5e1; }}
  .badge-active {{ background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }}
  .badge-warn {{ background: #78350f; color: #fde68a; border: 1px solid #d97706; }}
  .badge-error {{ background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }}
  .badge-high {{ background: #991b1b; color: #fee2e2; }}
  .badge-med {{ background: #b45309; color: #fef3c7; }}
  .badge-low {{ background: #0e7490; color: #cffafe; }}

  .video-container {{
    position: relative;
    width: 100%;
    max-width: 640px;
    margin: 0 auto;
    border-radius: 10px;
    overflow: hidden;
    background: #000;
    border: 2px solid var(--border-color);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    aspect-ratio: 4 / 3;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  video#webcamVideo {{
    display: none;
  }}
  canvas#displayCanvas {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }}
  .hud-overlay {{
    position: absolute;
    top: 10px;
    left: 10px;
    right: 10px;
    display: flex;
    justify-content: space-between;
    pointer-events: none;
  }}
  .hud-pill {{
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    color: #fff;
  }}

  .controls {{
    display: flex;
    gap: 8px;
    margin-top: 12px;
    flex-wrap: wrap;
    justify-content: center;
  }}
  button.btn {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.2s ease;
  }}
  button.btn-primary {{
    background: var(--primary);
    color: white;
  }}
  button.btn-primary:hover {{ background: var(--primary-hover); }}
  button.btn-danger {{
    background: var(--danger);
    color: white;
  }}
  button.btn-danger:hover {{ background: #dc2626; }}
  button.btn-success {{
    background: var(--success);
    color: white;
  }}
  button.btn-success:hover {{ background: #059669; }}
  button.btn-secondary {{
    background: #475569;
    color: white;
  }}
  button.btn-secondary:hover {{ background: #334155; }}
  button:disabled {{
    opacity: 0.5;
    cursor: not-allowed;
  }}

  .slider-row {{
    display: flex;
    gap: 16px;
    align-items: center;
    justify-content: center;
    margin-top: 10px;
    font-size: 13px;
    color: var(--text-muted);
    flex-wrap: wrap;
  }}
  .slider-group {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  input[type=range] {{
    accent-color: var(--primary);
    cursor: pointer;
  }}

  .diag-box {{
    margin-top: 12px;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 12px;
    line-height: 1.4;
    display: none;
  }}
  .diag-warn {{
    background: #451a03;
    border: 1px solid #b45309;
    color: #fef3c7;
  }}
  .diag-error {{
    background: #450a0a;
    border: 1px solid #b91c1c;
    color: #fee2e2;
  }}
  .diag-info {{
    background: #172554;
    border: 1px solid #1d4ed8;
    color: #dbeafe;
  }}

  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin-top: 12px;
  }}
  .stat-card {{
    background: #0f172a;
    border: 1px solid var(--border-color);
    padding: 8px 12px;
    border-radius: 6px;
    text-align: center;
  }}
  .stat-label {{
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .stat-val {{
    font-size: 18px;
    font-weight: 700;
    margin-top: 2px;
    color: #fff;
  }}
</style>
</head>
<body>

<div class="hud-card">
  <div class="status-bar">
    <div style="display:flex; align-items:center; gap:8px;">
      <span id="camBadge" class="badge badge-idle">● Camera: Idle</span>
      <span id="serverBadge" class="badge badge-active">● YOLO API: models/best.pt</span>
    </div>
    <div style="font-size: 12px; color: var(--text-muted);">
      RDD2022 Class: <strong style="color:#60a5fa;">D40 Pothole</strong>
    </div>
  </div>

  <div id="diagBox" class="diag-box"></div>
</div>

<div class="video-container">
  <video id="webcamVideo" autoplay playsinline muted></video>
  <canvas id="displayCanvas" width="640" height="480"></canvas>
  <div class="hud-overlay">
    <div class="hud-pill" id="hudFps">FPS: 0</div>
    <div class="hud-pill" id="hudLatency">Latency: -- ms</div>
    <div class="hud-pill" id="hudDetections">Potholes: 0</div>
    <div class="hud-pill" id="hudSeverity">Severity: None</div>
  </div>
</div>

<div class="controls">
  <button id="btnStart" class="btn btn-primary" onclick="startCamera()">▶️ Start Live Webcam</button>
  <button id="btnStop" class="btn btn-danger" onclick="stopCamera()" disabled>⏹️ Stop Camera</button>
  <button id="btnFlip" class="btn btn-secondary" onclick="toggleFacingMode()">🔄 Switch Camera</button>
  <button id="btnSave" class="btn btn-success" onclick="saveCurrentDetection()" disabled>💾 Save Detection</button>
</div>

<div class="slider-row">
  <div class="slider-group">
    <label for="confRange">Confidence Threshold:</label>
    <input type="range" id="confRange" min="0.05" max="0.90" step="0.05" value="{default_conf}" oninput="updateConf(this.value)">
    <span id="confVal" style="font-weight:600; color:#fff;">{default_conf:.2f}</span>
  </div>
  <div class="slider-group">
    <label for="fpsSelect">Inference Rate:</label>
    <select id="fpsSelect" style="background:#1e293b; color:#fff; border:1px solid #334155; border-radius:4px; padding:2px 6px;">
      <option value="5">5 FPS (Low CPU)</option>
      <option value="10" selected>10 FPS (Balanced)</option>
      <option value="15">15 FPS (Fast)</option>
    </select>
  </div>
</div>

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-label">Potholes Detected</div>
    <div id="statCount" class="stat-val" style="color:#60a5fa;">0</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Top Severity</div>
    <div id="statSev" class="stat-val">None</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Avg Confidence</div>
    <div id="statConf" class="stat-val">0.0%</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Location Tag</div>
    <div class="stat-val" style="font-size:13px; color:#cbd5e1; margin-top:5px;">{city} ({area})</div>
  </div>
</div>

<!-- Hidden canvas used for JPEG encoding -->
<canvas id="captureCanvas" width="640" height="480" style="display:none;"></canvas>

<script>
  const API_URL = "{api_url}";
  const CITY = "{city}";
  const AREA = "{area}";
  const ROAD_NAME = "{road_name}";
  const ROAD_TYPE = "{road_type}";

  const video = document.getElementById("webcamVideo");
  const displayCanvas = document.getElementById("displayCanvas");
  const captureCanvas = document.getElementById("captureCanvas");
  const dCtx = displayCanvas.getContext("2d");
  const cCtx = captureCanvas.getContext("2d");

  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const btnSave = document.getElementById("btnSave");
  const camBadge = document.getElementById("camBadge");
  const diagBox = document.getElementById("diagBox");
  const confRange = document.getElementById("confRange");
  const confVal = document.getElementById("confVal");
  const fpsSelect = document.getElementById("fpsSelect");

  const hudFps = document.getElementById("hudFps");
  const hudLatency = document.getElementById("hudLatency");
  const hudDetections = document.getElementById("hudDetections");
  const hudSeverity = document.getElementById("hudSeverity");

  const statCount = document.getElementById("statCount");
  const statSev = document.getElementById("statSev");
  const statConf = document.getElementById("statConf");

  let currentStream = null;
  let isStreaming = false;
  let isDetecting = false;
  let facingMode = "user";
  let lastInferenceTime = 0;
  let frameCounter = 0;
  let fpsTimer = Date.now();
  let latestDetections = [];
  let latestPotholeCount = 0;
  let latestTopSeverity = "None";
  let latestAvgConf = 0.0;

  // Initialize placeholder in canvas
  function drawPlaceholder(text = "Camera is off. Click [Start Live Webcam]") {{
    dCtx.fillStyle = "#0f172a";
    dCtx.fillRect(0, 0, 640, 480);
    dCtx.fillStyle = "#94a3b8";
    dCtx.font = "15px sans-serif";
    dCtx.textAlign = "center";
    dCtx.fillText(text, 320, 240);
  }}
  drawPlaceholder();

  function showDiag(msg, type = "warn") {{
    diagBox.style.display = "block";
    diagBox.className = "diag-box diag-" + type;
    diagBox.innerHTML = msg;
  }}

  function hideDiag() {{
    diagBox.style.display = "none";
  }}

  // 1. Secure context and permission check
  function checkSecureContext() {{
    const isLocalhost = window.location.hostname === "localhost" ||
                          window.location.hostname === "127.0.0.1";
    if (!window.isSecureContext && !isLocalhost) {{
      showDiag(
        "⚠️ <strong>Insecure Context Detected:</strong> WebRTC camera access requires HTTPS or localhost.<br>" +
        "You are accessing from <code>" + window.location.origin + "</code>. " +
        "Please open the website via HTTPS to enable webcam streaming.",
        "warn"
      );
      return false;
    }}
    if (window.location.protocol === "https:" && API_URL.startsWith("http://") && !API_URL.includes("localhost") && !API_URL.includes("127.0.0.1")) {{
      showDiag(
        "⚠️ <strong>Mixed Content Warning:</strong> The site is running on HTTPS, but the YOLO API is HTTP (<code>" + API_URL + "</code>).<br>" +
        "Browsers block HTTP requests from HTTPS sites. Please configure an HTTPS backend URL (e.g. https://...)",
        "error"
      );
      return false;
    }}
    return true;
  }}
  checkSecureContext();

  function updateConf(val) {{
    confVal.innerText = parseFloat(val).toFixed(2);
  }}

  function toggleFacingMode() {{
    facingMode = (facingMode === "user") ? "environment" : "user";
    if (isStreaming) {{
      stopCamera();
      startCamera();
    }}
  }}

  // 2. Start Camera function with proper constraints and permission handling
  async function startCamera() {{
    hideDiag();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
      showDiag(
        "❌ <strong>WebRTC Not Supported:</strong> Your browser does not support <code>navigator.mediaDevices.getUserMedia</code>. " +
        "Please use modern Chrome, Edge, or Firefox.",
        "error"
      );
      return;
    }}

    camBadge.className = "badge badge-warn";
    camBadge.innerText = "● Camera: Requesting...";

    const constraints = {{
      video: {{
        width: {{ ideal: 640 }},
        height: {{ ideal: 480 }},
        facingMode: facingMode
      }},
      audio: false
    }};

    try {{
      console.log("[WebRTC] Requesting camera with constraints:", constraints);
      currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = currentStream;
      await video.play();

      isStreaming = true;
      btnStart.disabled = true;
      btnStop.disabled = false;
      btnSave.disabled = false;

      camBadge.className = "badge badge-active";
      camBadge.innerText = "● Camera: Active";

      console.log("[WebRTC] Camera stream started successfully.");
      requestAnimationFrame(processLoop);

    }} catch (err) {{
      console.error("[WebRTC] getUserMedia error:", err);
      camBadge.className = "badge badge-error";
      camBadge.innerText = "● Camera: Blocked / Error";

      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {{
        showDiag(
          "🔒 <strong>Camera Permission Blocked:</strong><br>" +
          "1. Look at your browser address bar (top of screen).<br>" +
          "2. Click the <strong>camera / padlock icon</strong> next to the URL.<br>" +
          "3. Change <strong>Camera</strong> from 'Block' to <strong>'Allow'</strong>.<br>" +
          "4. Click <strong>Start Live Webcam</strong> again.",
          "error"
        );
      }} else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {{
        showDiag("⚠️ <strong>No Camera Found:</strong> No webcam device was detected on your machine.", "warn");
      }} else if (err.name === "NotReadableError" || err.name === "TrackStartError") {{
        showDiag("⚠️ <strong>Camera Busy:</strong> The webcam is in use by another app (Zoom, Teams, etc.). Please close other camera apps.", "warn");
      }} else {{
        showDiag("⚠️ <strong>Camera Error:</strong> " + err.message, "error");
      }}
    }}
  }}

  // 3. Stop Camera function & track cleanup
  function stopCamera() {{
    isStreaming = false;
    isDetecting = false;
    if (currentStream) {{
      currentStream.getTracks().forEach(track => {{
        track.stop();
        console.log("[WebRTC] Track stopped:", track.kind);
      }});
      currentStream = null;
    }}
    if (video) {{
      video.srcObject = null;
    }}
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnSave.disabled = true;

    camBadge.className = "badge badge-idle";
    camBadge.innerText = "● Camera: Idle";
    hudFps.innerText = "FPS: 0";
    hudLatency.innerText = "Latency: -- ms";
    drawPlaceholder("Camera stopped. Click [Start Live Webcam]");
  }}

  window.addEventListener("beforeunload", stopCamera);
  window.addEventListener("pagehide", stopCamera);

  // 4. Real-time Inference & Render Loop
  async function processLoop() {{
    if (!isStreaming) return;

    const now = Date.now();
    frameCounter++;
    if (now - fpsTimer >= 1000) {{
      hudFps.innerText = "FPS: " + frameCounter;
      frameCounter = 0;
      fpsTimer = now;
    }}

    // Draw the latest camera frame
    if (video.videoWidth > 0 && video.videoHeight > 0) {{
      dCtx.drawImage(video, 0, 0, 640, 480);
      drawDetections(latestDetections);
    }}

    // Check target FPS
    const targetFps = parseInt(fpsSelect.value) || 10;
    const intervalMs = 1000 / targetFps;

    if (!isDetecting && (now - lastInferenceTime >= intervalMs)) {{
      lastInferenceTime = now;
      sendFrameForInference();
    }}

    requestAnimationFrame(processLoop);
  }}

  // 5. Capture frame and POST to Python YOLO backend
  async function sendFrameForInference() {{
    if (!isStreaming || isDetecting) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    isDetecting = true;
    try {{
      // Capture frame to offscreen canvas
      cCtx.drawImage(video, 0, 0, 640, 480);
      const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.65);
      const conf = parseFloat(confRange.value);

      const tStart = performance.now();
      const baseUrl = (API_URL === "/" || !API_URL) ? "" : (API_URL.endsWith('/') ? API_URL.slice(0, -1) : API_URL);
      const res = await fetch(baseUrl + "/api/detect", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ image: dataUrl, conf: conf }})
      }});

      if (res.ok) {{
        const data = await res.json();
        const latency = Math.round(performance.now() - tStart);
        hudLatency.innerText = "Latency: " + latency + " ms";

        if (data.success) {{
          latestDetections = data.detections || [];
          latestPotholeCount = data.pothole_count || 0;
          latestTopSeverity = data.top_severity || "None";
          latestAvgConf = data.avg_confidence || 0.0;

          updateHudStats(latestPotholeCount, latestTopSeverity, latestAvgConf);
        }}
      }}
    }} catch (e) {{
      console.warn("[YOLO] Inference network error:", e);
    }} finally {{
      isDetecting = false;
    }}
  }}

  // 6. Draw Bounding Boxes with RDD2022 labels and Severity
  function drawDetections(dets) {{
    if (!dets || dets.length === 0) return;

    dets.forEach(det => {{
      const [x1, y1, x2, y2] = det.box;
      const w = x2 - x1;
      const h = y2 - y1;
      const sev = det.severity || "Low";
      const confPercent = Math.round((det.confidence || 0) * 100);

      // Color coding based on severity
      let color = "#06b6d4"; // Low: cyan
      if (sev === "High") color = "#ef4444"; // High: red
      else if (sev === "Medium") color = "#f59e0b"; // Medium: yellow

      // Bounding box with glow
      dCtx.save();
      dCtx.strokeStyle = color;
      dCtx.lineWidth = 3;
      dCtx.shadowColor = color;
      dCtx.shadowBlur = 8;
      dCtx.strokeRect(x1, y1, w, h);
      dCtx.restore();

      // Label badge
      const labelText = "D40 Pothole · " + confPercent + "% · " + sev;
      dCtx.font = "bold 12px sans-serif";
      const textWidth = dCtx.measureText(labelText).width;

      const badgeY = Math.max(18, y1 - 6);
      dCtx.fillStyle = color;
      dCtx.fillRect(x1, badgeY - 14, textWidth + 10, 18);

      dCtx.fillStyle = "#ffffff";
      dCtx.fillText(labelText, x1 + 5, badgeY);
    }});
  }}

  function updateHudStats(count, sev, conf) {{
    hudDetections.innerText = "Potholes: " + count;
    hudSeverity.innerText = "Severity: " + sev;
    statCount.innerText = count;
    statSev.innerText = sev;
    statConf.innerText = Math.round(conf * 100) + "%";

    if (sev === "High") {{
      statSev.style.color = "#ef4444";
      hudSeverity.style.borderColor = "#ef4444";
    }} else if (sev === "Medium") {{
      statSev.style.color = "#f59e0b";
      hudSeverity.style.borderColor = "#f59e0b";
    }} else if (sev === "Low") {{
      statSev.style.color = "#06b6d4";
      hudSeverity.style.borderColor = "#06b6d4";
    }} else {{
      statSev.style.color = "#94a3b8";
      hudSeverity.style.borderColor = "rgba(255,255,255,0.15)";
    }}
  }}

  // 7. Save current detection to data warehouse
  async function saveCurrentDetection() {{
    if (latestPotholeCount === 0) {{
      alert("No potholes detected in current frame to save.");
      return;
    }}
    btnSave.disabled = true;
    btnSave.innerText = "Saving...";

    try {{
      const baseUrl = (API_URL === "/" || !API_URL) ? "" : (API_URL.endsWith('/') ? API_URL.slice(0, -1) : API_URL);
      const res = await fetch(baseUrl + "/api/save", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{
          pothole_count: latestPotholeCount,
          confidence: latestAvgConf,
          severity: latestTopSeverity,
          city: CITY,
          area: AREA,
          road_name: ROAD_NAME,
          road_type: ROAD_TYPE
        }})
      }});
      const data = await res.json();
      if (data.success) {{
        alert("✅ Pothole detection saved to Warehouse! (ID: #" + data.det_id + ")");
      }} else {{
        alert("Error saving detection: " + (data.error || "Unknown"));
      }}
    }} catch (e) {{
      alert("Network error saving detection: " + e.message);
    }} finally {{
      btnSave.disabled = false;
      btnSave.innerText = "💾 Save Detection";
    }}
  }}
</script>
</body>
</html>"""
