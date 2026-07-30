# ================================================================
# Dockerfile - Digital Meter Reading Pipeline
# ================================================================
# Base: PyTorch with CUDA 12.1 support for GPU inference (RTX 3090)
# Also supports CPU fallback if no GPU available.
# ================================================================

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL maintainer="AI Research Team"
LABEL description="Enterprise Water Meter Digital Reading Pipeline"
LABEL version="1.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OPENCV_IO_MAX_IMAGE_PIXELS=1000000000

# Install system dependencies (required by OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Create working directory
WORKDIR /app

# ============================================================
# Copy all source code
# ============================================================
COPY pipeline.py /app/pipeline.py
COPY predict.py /app/predict.py
COPY config/config.yaml /app/config/config.yaml

# ============================================================
# Copy detection models (YOLOv8)
# ============================================================
COPY models/detection/delivery_best.pt /app/models/detection/delivery_best.pt
COPY models/detection/yolov8n_obb_wordwheel.pt /app/models/detection/yolov8n_obb_wordwheel.pt

# ============================================================
# OpenOCR recognition model (SVTRv2, 118MB)
# Will be auto-downloaded from HuggingFace/ModelScope on first use.
# No internet? Pre-download manually:
#   python3 -c "from openocr import OpenOCR; OpenOCR(task='rec', mode='server', backend='torch', use_gpu='auto')"
# ============================================================
RUN mkdir -p /root/.cache/openocr

# ============================================================
# Entry point scripts
# ============================================================
COPY scripts/run_pipeline.py /app/scripts/run_pipeline.py
COPY scripts/api_server.py /app/scripts/api_server.py

# ============================================================
# Runtime configuration
# ============================================================
EXPOSE 5000

# Mount points for user data
RUN mkdir -p /data /results
VOLUME ["/data", "/results"]

# ============================================================
# Default command: production inference
#   docker run --gpus all -v /path/to/images:/data digital-meter-reading:1.0.0
#   -> Reads all images from /data, outputs results to /results
# ============================================================
ENTRYPOINT ["python3", "/app/predict.py"]
CMD ["--input", "/data", "--output", "/results"]
