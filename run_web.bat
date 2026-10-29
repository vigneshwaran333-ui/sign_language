@echo off
REM ==============================================================
REM Z-Sign Real-Time Recognition Web UI Launcher
REM Launches the localhost web interface on http://localhost:5000
REM ==============================================================
cd /d "%~dp0"

echo =============================================================
echo Starting Z-Sign Web UI on http://localhost:5000 ...
echo =============================================================

set "PYTHON_EXE="

if exist ".\env\Scripts\python.exe" (
    set "PYTHON_EXE=.\env\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Loading AI model and starting Flask server...
echo The web browser will automatically open in a few seconds.
echo (Keep this window open. Press Ctrl+C to stop)
echo =============================================================

REM Open browser after 6 seconds delay once model finishes loading
start "" cmd /c "timeout /t 6 /nobreak >nul && start http://localhost:5000"

"%PYTHON_EXE%" app.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server stopped with error code %ERRORLEVEL%.
    pause
)

