@echo off
echo ============================================
echo RimWorld Base Generator - ML Training
echo Using RTX 5090 for accelerated training
echo ============================================
echo.

REM Use Poetry environment
echo Setting up Poetry environment...
call poetry install

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
call poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
call poetry run pip install gputil nvidia-ml-py scikit-learn

echo.
echo Checking GPU...
poetry run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Mode\"}')"

echo.
echo Starting ML Training GUI...
echo ============================================
poetry run python ml_training_gui.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start ML Training GUI
    echo Check the error messages above
    pause
)