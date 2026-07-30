#!/usr/bin/env python3
"""
Flask API server for Digital Meter Reading Pipeline.
Provides REST API endpoints for single-image inference.

Usage:
    python3 scripts/api_server.py [--port 5000] [--host 0.0.0.0]

Endpoints:
    POST /predict - Upload image, return reading prediction
    GET  /health  - Health check
"""
import sys
import os
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import MeterReadingPipeline

from flask import Flask, request, jsonify

app = Flask(__name__)
pipeline = None


def init_pipeline():
    global pipeline
    if pipeline is None:
        config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
        print(f"Initializing pipeline with config: {config_path}")
        pipeline = MeterReadingPipeline(config_path)
        print("Pipeline ready!")
    return pipeline


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "meter-reading-pipeline"})


@app.route("/predict", methods=["POST"])
def predict():
    """Upload an image file, receive reading prediction."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    pipe = init_pipeline()
    prediction, det_type = pipe.run_on_image(img)

    return jsonify({
        "prediction": prediction,
        "detection_type": det_type,
        "status": "success",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meter Reading API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    args = parser.parse_args()

    # Pre-load pipeline
    init_pipeline()

    app.run(host=args.host, port=args.port)
