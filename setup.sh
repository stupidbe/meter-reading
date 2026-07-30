#!/usr/bin/env bash
# ============================================================
# 企业字轮式水表读数识别系统 - 一键安装脚本 (Linux)
# ============================================================
# 使用方法:
#   chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

echo "========================================"
echo "  企业字轮式水表读数识别系统 - 环境安装"
echo "========================================"
echo ""

# Check Python
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [ $? -ne 0 ]; then
    echo "❌ 未检测到 Python，请先安装 Python 3.10 (推荐通过 Miniconda)"
    echo "   下载: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "✅ Python: $(python3 --version)"

# Check conda
if command -v conda &> /dev/null; then
    echo "✅ Conda: $(conda --version)"
    USE_CONDA=true
else
    echo "⚠️ 未检测到 Conda，将使用 pip 安装"
    USE_CONDA=false
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
    echo "⚠️ 未检测到 NVIDIA GPU (将使用 CPU，速度较慢)"
fi
echo ""

# Install with conda (recommended)
if [ "$USE_CONDA" = true ]; then
    echo "📦 使用 Conda 创建环境..."
    # Check if environment already exists
    if conda env list | grep -q "meter-reading"; then
        echo "   环境 meter-reading 已存在，跳过创建"
    else
        conda env create -f environment.yml
    fi

    echo ""
    echo "📦 安装模型文件..."
    # Models are already included in the package

    echo ""
    echo "========================================"
    echo "✅ 安装完成！"
    echo "========================================"
    echo ""
    echo "使用方式:"
    echo ""
    echo "  # 激活环境"
    echo "  conda activate meter-reading"
    echo ""
    echo "  # 识别单张图片"
    echo "  python3 predict.py --image test.jpg"
    echo ""
    echo "  # 识别整个文件夹"
    echo "  python3 predict.py --input ./照片文件夹 --output ./结果"
    echo ""
    echo "  # 保存可视化结果"
    echo "  python3 predict.py --input ./照片 --output ./结果 --visualize"
    echo ""

# Install with pip (fallback)
else
    echo "📦 使用 pip 安装依赖..."
    pip install -r requirements.txt

    echo ""
    echo "========================================"
    echo "✅ 安装完成！"
    echo "========================================"
    echo ""
    echo "使用方式:"
    echo ""
    echo "  python3 predict.py --input ./照片文件夹 --output ./结果"
    echo ""
fi

# Check OpenOCR model
MODEL_PATH="models/recognition/openocr_svtrv2_ch.pth"
if [ -f "$MODEL_PATH" ]; then
    SIZE_MB=$(du -h "$MODEL_PATH" | cut -f1)
    echo "  模型文件: $MODEL_PATH ($SIZE_MB)"
else
    echo "  ⚠️ 模型文件未找到，首次运行时会自动下载"
fi

echo ""
echo "按 Enter 键退出..."
read
