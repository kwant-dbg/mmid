@echo off
REM Setup script for Multi-Modal Icon Vision System

echo ==========================================
echo Multi-Modal Icon Vision System - Setup
echo ==========================================
echo.

echo Step 1: Installing core dependencies...
pip install pyyaml numpy opencv-python pillow flask flask-cors

echo.
echo Step 2: Installing deep learning frameworks...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo Step 3: Installing YOLOv8...
pip install ultralytics

echo.
echo Step 4: Installing additional packages...
pip install pandas matplotlib seaborn scikit-learn tqdm requests python-dotenv pytest

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo   1. Run tests: python tests\test_all.py
echo   2. Start backend: cd backend ^&^& python app.py
echo   3. Start frontend: cd frontend ^&^& python -m http.server 8000
echo.
pause
