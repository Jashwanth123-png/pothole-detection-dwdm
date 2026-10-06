#!/bin/bash
set -e

# Port configuration: default 7860 (Hugging Face Spaces) or $PORT (Render / Cloud Run)
LISTEN_PORT=${PORT:-7860}
echo "Configuring Nginx Gateway to listen on port ${LISTEN_PORT}..."
sed -i "s/listen 7860;/listen ${LISTEN_PORT};/g" /etc/nginx/nginx.conf

# 1. Start the YOLO live inference server on localhost:8502
echo "Starting YOLO Live Inference Microservice (models/best.pt) on 127.0.0.1:8502..."
python ml/live_server.py &

# 2. Start the Streamlit application on localhost:8501
echo "Starting Streamlit UI on 127.0.0.1:8501..."
# Tell Streamlit that LIVE_API_URL is relative to same-origin reverse proxy gateway
export LIVE_API_URL="/"
streamlit run app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --browser.gatherUsageStats false &

# 3. Wait for services to be ready
echo "Waiting for services to initialize..."
sleep 4

# 4. Start Nginx in foreground to serve external requests
echo "Nginx Gateway active on port ${LISTEN_PORT}. Ready for traffic!"
exec nginx -g "daemon off;"
