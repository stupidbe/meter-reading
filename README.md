# 企业字轮式水表读数识别系统

基于 **YOLOv8 OBB/REG 检测** 与 **OpenOCR SVTRv2 识别**，自动识别水表照片中的读数。

> 🎯 测试集精度：**65.2%**（46张中30张完全正确）
> ⏱ 单张识别：**1~2秒**（RTX 3090）

---

## 📦 交付内容

**全部文件已打包，共 147MB，包含三个模型，无需联网下载。**

```
digital_meter_docker/
│
├── 🚀 run.bat                  # Windows一键运行（双击即可）
├── 🚀 setup.bat                # Windows一键安装环境（双击即可）
├── 🚀 setup.sh                 # Linux一键安装环境
│
├── pipeline.py                 # 核心识别算法
├── predict.py                  # 主入口脚本（识别照片）
├── Dockerfile                  # Docker 构建文件
├── docker-compose.yml          # Docker 部署配置
├── environment.yml             # Conda 环境配置
├── requirements.txt            # Python 依赖列表
├── config/config.yaml          # 算法参数配置
│
├── models/
│   ├── detection/
│   │   ├── delivery_best.pt          # YOLOv8s 水平框检测 (22MB)
│   │   └── yolov8n_obb_wordwheel.pt  # YOLOv8n 旋转框检测 (6.4MB)
│   └── recognition/
│       └── openocr_svtrv2_ch.pth     # SVTRv2 中文OCR模型 (118MB) ✓
│
└── scripts/
    ├── run_pipeline.py         # 批量测试（需标签文件）
    └── api_server.py           # HTTP API 服务
```

---

## 🚀 快速开始（三种方式）

### 方式一：Windows 一键运行 ⭐推荐

```bash
# 1. 双击 setup.bat    → 自动安装 Python 依赖（只需一次）
# 2. 把照片放到 data/ 文件夹
# 3. 双击 run.bat      → 自动识别，结果保存到 results/
```

### 方式二：Docker 部署（推荐给有 GPU 服务器的企业）

```bash
# 1. 构建镜像（只需一次）
docker build -t meter-reading:1.0.0 .

# 2. 运行识别（Linux）
docker run --gpus all \
  -v /path/to/photos:/data \
  -v /path/to/results:/results \
  meter-reading:1.0.0

# Windows PowerShell
docker run --gpus all `
  -v D:\照片:/data `
  -v D:\结果:/results `
  meter-reading:1.0.0
```

### 方式三：Linux 直接运行

```bash
# 1. 安装环境
chmod +x setup.sh && ./setup.sh

# 2. 激活环境
conda activate meter-reading

# 3. 识别照片
python3 predict.py --input ./照片文件夹 --output ./结果 --visualize
```

---

## 📋 硬件要求

| 组件 | 要求 |
|------|------|
| GPU | **推荐** NVIDIA GPU，8GB+ 显存（如 RTX 3090） |
| CPU | 无 GPU 也可运行，但每张图需 5~10 秒 |
| 内存 | 16GB+ |
| 硬盘 | 5GB 可用空间 |

---

## 📊 结果说明

运行后自动生成两个文件：

**`results/results.csv`** — 可用 Excel 直接打开：
```csv
file,prediction,detection_type
IMG_001.jpg,123.45,REG
IMG_002.jpg,67.89,OBB
IMG_003.jpg,0.020,OBBa
```

**`results/viz/`** — 可视化结果文件夹（带 --visualize 参数时生成）

---

## ❓ 常见问题

**Q: 没有 GPU 能用吗？**
A: 可以。系统自动检测 GPU，无 GPU 时使用 CPU（每张图约 5~10 秒）。

**Q: 需要自己下载模型吗？**
A: **不需要。** 所有模型已内置在 `models/` 目录中，共 147MB。

**Q: 支持哪些图片格式？**
A: JPG、JPEG、PNG、BMP。

**Q: Docker 和直接运行哪个好？**
A: 有 GPU 服务器推荐 Docker（环境隔离、即开即用）；Windows 测试推荐直接运行。
