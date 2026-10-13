# YOLOv5 Sign Language Fine-Tuning PowerShell Script
# Merged Dataset: 16 classes (244 Train images, 157 Val images)

Set-Location "C:\zsign-detection"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting YOLOv5 Fine-Tuning..." -ForegroundColor Green
Write-Host "Old weights:    best.pt"
Write-Host "Dataset:        dataset_merged.yaml (16 classes)"
Write-Host "Architecture:   yolov5/models/custom_yolov5s_16cls.yaml"
Write-Host "Hyperparams:    yolov5/data/hyps/hyp.sign_language.yaml"
Write-Host "==========================================================" -ForegroundColor Cyan

& ".\env\Scripts\python.exe" yolov5/train.py `
    --img 416 `
    --batch 16 `
    --epochs 50 `
    --data dataset_merged.yaml `
    --cfg yolov5/models/custom_yolov5s_16cls.yaml `
    --weights best.pt `
    --hyp yolov5/data/hyps/hyp.sign_language.yaml `
    --project runs/train `
    --name yolov5s_finetuned `
    --exist-ok `
    --workers 2 `
    --cache

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "Training finished!" -ForegroundColor Green
Write-Host "Best weights saved at: runs/train/yolov5s_finetuned/weights/best.pt"
Write-Host "==========================================================" -ForegroundColor Cyan
