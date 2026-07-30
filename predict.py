#!/usr/bin/env python3
"""
Production Inference Script
============================
企业直接使用入口：指定图片文件夹，自动识别所有水表图片，输出读数CSV。

用法:
    # 识别单张图片
    python3 predict.py --image test.jpg

    # 识别整个文件夹
    python3 predict.py --input ./images --output ./results

    # 保存可视化结果
    python3 predict.py --input ./images --output ./results --visualize

输出:
    results/results.csv - 所有图片的识别结果
    results/viz/         - 可视化结果（可选）
"""

import os
import sys
import cv2
import csv
import json
import time
import argparse
import re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import MeterReadingPipeline


def predict_single(pipe: MeterReadingPipeline, img_path: str) -> dict:
    """Run prediction on a single image file."""
    img = cv2.imread(img_path)
    if img is None:
        return {
            "file": os.path.basename(img_path),
            "prediction": "ERROR: Cannot read image",
            "detection_type": "NONE",
            "success": False,
        }

    prediction, det_type = pipe.run_on_image(img)

    return {
        "file": os.path.basename(img_path),
        "prediction": prediction if prediction else "NONE",
        "detection_type": det_type,
        "success": True,
    }


def visualize_result(img_path: str, prediction: str, det_type: str,
                     output_path: str):
    """Create visualization image with detection result."""
    img = cv2.imread(img_path)
    if img is None:
        return

    h, w = img.shape[:2]
    scale = min(800 / w, 600 / h, 1.0)
    if scale < 1.0:
        vis = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        vis = img.copy()

    vh, vw = vis.shape[:2]

    # Add info bar at top
    bar = np.full((50, vw, 3), 240, dtype=np.uint8)
    cv2.putText(bar, f"Prediction: {prediction}  |  Detection: {det_type}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0) if prediction else (0, 0, 200), 2)

    result = np.vstack([bar, vis])

    # Try to save
    try:
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception as e:
        print(f"  Warning: Could not save visualization: {e}")


def main():
    parser = argparse.ArgumentParser(description="Meter Reading - Production Inference")
    parser.add_argument("--input", "-i", default=None,
                        help="Input directory containing images, or single image file")
    parser.add_argument("--image", default=None,
                        help="Single image path (alternative to --input)")
    parser.add_argument("--output", "-o", default="./results",
                        help="Output directory for results CSV (default: ./results)")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Pipeline config path (default: config/config.yaml)")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Generate visualization images")
    parser.add_argument("--ext", default=".jpg,.jpeg,.png,.bmp",
                        help="Comma-separated list of image extensions (default: .jpg,.jpeg,.png,.bmp)")

    args = parser.parse_args()

    # Determine input path
    input_path = args.image or args.input
    if not input_path:
        parser.print_help()
        print("\nError: Please specify --input or --image")
        sys.exit(1)

    # Collect image files
    image_extensions = [ext.strip().lower() for ext in args.ext.split(",")]
    image_files = []

    if os.path.isfile(input_path):
        image_files.append(input_path)
    elif os.path.isdir(input_path):
        for fname in sorted(os.listdir(input_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(input_path, fname))
    else:
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)

    if len(image_files) == 0:
        print(f"Error: No image files found in {input_path}")
        sys.exit(1)

    print(f"Found {len(image_files)} image(s)")
    print(f"Output directory: {args.output}")
    os.makedirs(args.output, exist_ok=True)

    # Initialize pipeline
    print("Initializing pipeline...")
    pipe = MeterReadingPipeline(args.config)
    print("Pipeline ready!\n")

    # Run predictions
    results = []
    t0 = time.time()

    for idx, img_path in enumerate(image_files):
        basename = os.path.basename(img_path)
        print(f"  [{idx+1}/{len(image_files)}] {basename}...", end=" ")

        result = predict_single(pipe, img_path)
        results.append(result)

        if result["success"]:
            print(f"{result['prediction']} ({result['detection_type']})")
        else:
            print(f"FAILED")

        # Generate visualization
        if args.visualize and result["success"]:
            viz_dir = os.path.join(args.output, "viz")
            os.makedirs(viz_dir, exist_ok=True)
            viz_path = os.path.join(viz_dir, f"{os.path.splitext(basename)[0]}_result.jpg")
            visualize_result(img_path, result["prediction"],
                             result["detection_type"], viz_path)

    elapsed = time.time() - t0

    # Save results as CSV
    csv_path = os.path.join(args.output, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "prediction", "detection_type", "success"])
        writer.writeheader()
        writer.writerows(results)

    # Also save as JSON
    json_path = os.path.join(args.output, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    success_count = sum(1 for r in results if r["success"])
    detected_count = sum(1 for r in results if r["success"] and r["prediction"] not in ("NONE", ""))
    print(f"\n=== Done ({elapsed:.1f}s) ===")
    print(f"  Total: {len(results)}")
    print(f"  Succeeded: {success_count}")
    print(f"  Got reading: {detected_count}")
    print(f"  Results CSV: {csv_path}")
    print(f"  Results JSON: {json_path}")
    if args.visualize:
        print(f"  Visualizations: {os.path.join(args.output, 'viz')}")


if __name__ == "__main__":
    main()
