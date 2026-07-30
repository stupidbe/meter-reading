@echo off
chcp 65001 >nul
title 水表读数识别系统

echo ========================================
echo   企业字轮式水表读数识别系统
echo ========================================
echo.

REM Check if photos folder exists, prompt user
set /p INPUT_FOLDER=请输入照片文件夹路径（直接回车默认使用 ./data）:
if "%INPUT_FOLDER%"=="" set INPUT_FOLDER=./data

set /p OUTPUT_FOLDER=请输入结果输出路径（直接回车默认使用 ./results）:
if "%OUTPUT_FOLDER%"=="" set OUTPUT_FOLDER=./results

echo.
echo 📂 输入: %INPUT_FOLDER%
echo 📂 输出: %OUTPUT_FOLDER%
echo.

echo 🔍 开始识别...
echo.
python predict.py --input "%INPUT_FOLDER%" --output "%OUTPUT_FOLDER%" --visualize
if %errorlevel% neq 0 (
    echo ❌ 运行失败！请先运行 setup.bat 安装依赖
    pause
    exit /b 1
)

echo.
echo ✅ 识别完成！
echo 📁 结果文件: %OUTPUT_FOLDER%\results.csv
echo 📁 可视化图片: %OUTPUT_FOLDER%\viz\
echo.
pause
