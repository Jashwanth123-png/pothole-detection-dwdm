FROM python:3.10-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies & Nginx reverse proxy
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Setup Nginx configuration and entrypoint
COPY nginx.conf /etc/nginx/nginx.conf
RUN chmod +x entrypoint.sh

# Default port (7860 for Hugging Face Spaces, or overridden via $PORT on Render)
EXPOSE 7860

ENTRYPOINT ["./entrypoint.sh"]
