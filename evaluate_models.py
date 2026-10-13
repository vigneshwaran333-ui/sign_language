import os
import sys
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "env" / "Scripts" / "python.exe"
YOLOV5_VAL = ROOT / "yolov5" / "val.py"
BASELINE_METRICS = ROOT / "baseline_backup" / "baseline_metrics.json"

def run_val(weights_path, data_yaml, name):
    print(f"\n==========================================")
    print(f"Running evaluation: {name}")
    print(f"Weights: {weights_path}")
    print(f"Data: {data_yaml}")
    print(f"==========================================")
    
    cmd = [
        str(PYTHON),
        str(YOLOV5_VAL),
        "--weights", str(weights_path),
        "--data", str(data_yaml),
        "--img", "416",
        "--project", str(ROOT / "runs" / "val"),
        "--name", name,
        "--exist-ok"
    ]
    return subprocess.run(cmd).returncode

def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "runs" / "train" / "yolov5s_finetuned" / "weights" / "best.pt")
    
    if not os.path.exists(weights):
        print(f"Error: Weights file '{weights}' not found.")
        print("Please specify the path to your trained weights file, e.g.:")
        print(f"  python evaluate_models.py runs/train/yolov5s_finetuned/weights/best.pt")
        sys.exit(1)
        
    print(f"Starting post-training evaluation for: {weights}")
    
    # 1. Evaluate on frozen baseline test set
    run_val(weights, str(ROOT / "dataset_baseline_test.yaml"), "eval_baseline_test")
    
    # 2. Evaluate on new-data validation slice
    run_val(weights, str(ROOT / "dataset_newdata_val.yaml"), "eval_newdata_val")
    
    # 3. Evaluate on full merged validation set
    run_val(weights, str(ROOT / "dataset_merged.yaml"), "eval_merged_val")
    
    print("\nEvaluations complete. Results and confusion matrices saved to runs/val/")
    if os.path.exists(BASELINE_METRICS):
        with open(BASELINE_METRICS) as f:
            base = json.load(f)
        print(f"\n--- Baseline Reference (Before Training) ---")
        print(f"Baseline Overall mAP50:    {base['overall']['mAP50']}")
        print(f"Baseline Overall mAP50-95: {base['overall']['mAP50_95']}")
        print(f"Baseline Precision:        {base['overall']['precision']}")
        print(f"Baseline Recall:           {base['overall']['recall']}")

if __name__ == "__main__":
    main()
