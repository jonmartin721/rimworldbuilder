@echo off
echo ============================================
echo RimWorld Base Generator - ML Training
echo Using RTX 5090 for accelerated training
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Check for CUDA
echo Checking for CUDA support...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>nul
if errorlevel 1 (
    echo.
    echo PyTorch not installed. Installing ML dependencies...
    echo This may take a few minutes...
    pip install -r requirements-ml.txt
)

echo.
echo Starting ML Training GUI...
echo ============================================
python ml_training_gui.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start ML Training GUI
    echo Check the error messages above
    pause
)