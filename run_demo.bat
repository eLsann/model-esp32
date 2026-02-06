@echo off
echo ==========================================
echo Starting YOLO-Fastest Realtime Detection
echo ==========================================
echo.
echo Pastikan webcam terhubung...
echo Tekan 'Q' di jendela kamera untuk keluar.
echo.

python "%~dp0\testing_app\test_yolo_realtime.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Terjadi error! Pastikan environment Python aktif.
    echo Coba jalankan manual: python testing_app\test_yolo_realtime.py
    pause
)
