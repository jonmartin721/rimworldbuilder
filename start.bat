@echo off
echo ============================================
echo      RimWorld Base Assistant Launcher
echo ============================================
echo.
echo Choose an option:
echo 1. Professional GUI - Modern Interface (Recommended)
echo 2. Command-Line Interface
echo 3. Enhanced GUI (Previous Version)
echo 4. Classic GUI
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting Professional GUI with Modern Interface...
    poetry run python rimworld_assistant_gui_v3.py
) else if "%choice%"=="2" (
    echo.
    echo Starting CLI...
    poetry run python rimworld_assistant.py
) else if "%choice%"=="3" (
    echo.
    echo Starting Enhanced GUI...
    poetry run python rimworld_assistant_gui_v2.py
) else if "%choice%"=="4" (
    echo.
    echo Starting Classic GUI...
    poetry run python rimworld_assistant_gui.py
) else if "%choice%"=="5" (
    echo Goodbye!
    exit
) else (
    echo Invalid choice. Please run start.bat again.
    pause
)