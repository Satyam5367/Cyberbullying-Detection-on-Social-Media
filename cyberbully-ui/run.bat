@echo off
echo ============================================================
echo   CyberGuard AI - Setup and Launch
echo ============================================================

echo [1/3] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install from python.org
    pause
    exit /b 1
)

echo [2/3] Installing Flask...
pip install flask --quiet

echo [3/3] Starting server...
echo.
echo  Open your browser at: http://127.0.0.1:5000
echo  Press Ctrl+C to stop
echo.
python app.py
pause
