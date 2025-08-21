@echo off
echo ============================================
echo RimWorld ML Training - DEBUG MODE
echo RTX 5090 with CUDA Error Debugging
echo ============================================
echo.

REM Set CUDA debugging environment variables
set CUDA_LAUNCH_BLOCKING=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set TORCH_CUDA_ARCH_LIST=8.9;9.0

echo CUDA Debug Mode Enabled:
echo - CUDA_LAUNCH_BLOCKING = 1 (synchronous execution)
echo - PYTORCH_CUDA_ALLOC_CONF = max_split_size_mb:512
echo - TORCH_CUDA_ARCH_LIST = 8.9;9.0
echo.

REM Run with Poetry
echo Starting ML Training GUI in debug mode...
poetry run python ml_training_gui.py

if errorlevel 1 (
    echo.
    echo ============================================
    echo ERROR DETECTED - Check messages above
    echo ============================================
    pause
)