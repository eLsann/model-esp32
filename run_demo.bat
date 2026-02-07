@echo off
echo ==========================================
echo Starting YOLOv5n Realtime Detection
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python tidak ditemukan!
    echo Silakan install Python 3.10+ dari https://python.org
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Memeriksa dependencies...
pip show opencv-python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing dependencies...
    pip install -r "%~dp0requirements.txt"
    echo.
)

echo Pastikan webcam terhubung...
echo Tekan 'Q' di jendela kamera untuk keluar.
echo Tekan 'D' untuk toggle deteksi dinding.
echo.

python "%~dp0testing_app\test_yolo_realtime.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==========================================
    echo [ERROR] Terjadi kesalahan!
    echo ==========================================
    echo.
    echo Kemungkinan penyebab:
    echo 1. Dependencies belum terinstall
    echo    Jalankan: pip install -r requirements.txt
    echo.
    echo 2. Model belum tersedia
    echo    Jalankan training terlebih dahulu.
    echo.
    echo 3. Webcam tidak terhubung
    echo    Pastikan kamera tersambung dengan benar.
    echo.
    pause
)
