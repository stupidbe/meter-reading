@echo off
chcp 65001 >nul
title 企业字轮式水表读数识别系统 - 环境安装

echo ========================================
echo   企业字轮式水表读数识别系统 - 环境安装
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未检测到 Python，请先安装 Python 3.10
    echo    下载: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
echo ✅ Python: %PY_VER%

REM Check pip
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未检测到 pip
    pause
    exit /b 1
)
echo ✅ pip 已安装

REM Check GPU (nvidia-smi)
nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=name --format^=csv,noheader 2^>^&1') do set GPU_NAME=%%i
    echo ✅ GPU: %GPU_NAME%
) else (
    echo ⚠️ 未检测到 NVIDIA GPU ^(将使用 CPU，速度较慢^)
)
echo.

echo 📦 正在安装依赖（约 10-30 分钟，取决于网络）...
echo.
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ 安装失败！请检查网络连接后重试
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 安装完成！
echo ========================================
echo.
echo 使用方式:
echo.
echo   识别单张图片:
echo     python predict.py --image test.jpg
echo.
echo   识别整个文件夹:
echo     python predict.py --input ./照片文件夹 --output ./结果
echo.
echo   保存可视化结果:
echo     python predict.py --input ./照片 --output ./结果 --visualize
echo.

pause
