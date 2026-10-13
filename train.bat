@echo off
REM ==============================================================
REM YOLOv5 Sign Language Fine-Tuning Script
REM Merged Dataset: 16 classes (244 Train images, 157 Val images)
REM ==============================================================

cd /d "C:\zsign-detection"

echo ==========================================================
echo Starting YOLOv5 Fine-Tuning...
echo Old weights: best.pt
echo Dataset:    dataset_merged.yaml (16 classes)
echo Architecture: yolov5/models/custom_yolov5s_16cls.yaml
echo Hyperparams:  yolov5/data/hyps/hyp.sign_language.yaml
echo ==========================================================

".\env\Scripts\python.exe" yolov5/train.py ^
    --img 416 ^
    --batch 16 ^
    --epochs 50 ^
    --data dataset_merged.yaml ^
    --cfg yolov5/models/custom_yolov5s_16cls.yaml ^
    --weights best.pt ^
    --hyp yolov5/data/hyps/hyp.sign_language.yaml ^
    --project runs/train ^
    --name yolov5s_finetuned ^
    --exist-ok ^
    --workers 2 ^
    --cache

echo.
echo ==========================================================
echo Training completed!
echo Trained weights are located at:
echo   runs\train\yolov5s_finetuned\weights\best.pt
echo ==========================================================
pause
