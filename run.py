import os
import sys
import subprocess
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parent
YOLOV5_DIR = ROOT / "yolov5" if (ROOT / "yolov5").exists() else ROOT
WEIGHTS = ROOT / "best.pt" if (ROOT / "best.pt").exists() else YOLOV5_DIR / "best.pt"

# Prefer virtual environment python if available
VENV_PYTHON = ROOT / "env" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = ROOT.parent / "env" / "Scripts" / "python.exe"
py_exec = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# Parse and normalize CLI arguments
args = list(sys.argv[1:])

# Convert relative file paths to absolute before changing directory
new_args = []
i = 0
while i < len(args):
    arg = args[i]
    if arg in ("--source", "-s", "--weights", "-w", "--data", "--project") and i + 1 < len(args):
        val = args[i + 1]
        p = Path(val)
        if not val.isnumeric() and not val.startswith(("http://", "https://", "rtsp://", "rtmp://", "screen")):
            abs_p = (Path.cwd() / p).resolve()
            if abs_p.exists():
                val = str(abs_p)
        new_args.extend([arg, val])
        i += 2
    else:
        new_args.append(arg)
        i += 1
args = new_args

os.chdir(YOLOV5_DIR)

if not any(arg in args for arg in ("--weights", "-w")):
    args = ["--weights", str(WEIGHTS.resolve())] + args

if not any(arg in args for arg in ("--img", "--img-size", "--imgsz")):
    args += ["--img", "416"]

# Use the original high-accuracy confidence threshold (0.80) to suppress all background noise
# Genuine gestures typically score 0.90+
if not any(arg in args for arg in ("--conf", "--conf-thres")):
    args += ["--conf", "0.80"]

# Specific threshold for IloveYou to eliminate false positives on face
if not any(arg in args for arg in ("--iloveyou-conf",)):
    args += ["--iloveyou-conf", "0.88"]

# Reduced font size and bounding box thickness for clean display
if not any(arg in args for arg in ("--font-scale",)):
    args += ["--font-scale", "0.55"]

if not any(arg in args for arg in ("--line-thickness",)):
    args += ["--line-thickness", "2"]

# Limit to 1 detection per frame to prevent screen clutter from background noise
if not any(arg in args for arg in ("--max-det",)):
    args += ["--max-det", "1"]

# Class-agnostic NMS to eliminate overlapping conflicting classes (e.g. Hello + IloveYou)
if not any(arg in args for arg in ("--agnostic-nms",)):
    args += ["--agnostic-nms"]

if not any(arg in args for arg in ("--source", "-s")):
    args += ["--source", "0"]

cmd = [py_exec, "detect.py"] + args
print(f"Running detection command: {' '.join(cmd)}")
sys.exit(subprocess.run(cmd).returncode)
