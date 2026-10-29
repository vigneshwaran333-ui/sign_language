@echo off
REM ==============================================================
REM Real-Time YOLOv5 Sign Language Detection (Command Prompt)
REM ==============================================================
cd /d "%~dp0"

echo ==========================================================
echo Starting YOLOv5 Sign Language Real-Time Detection...
echo Model:   best.pt
echo Device:  Webcam (source 0)
echo Press 'q' on the OpenCV camera window to stop.
echo ==========================================================

".\env\Scripts\python.exe" run.py %*
