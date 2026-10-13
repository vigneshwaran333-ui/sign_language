import os
import sys
import subprocess
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parent
WEIGHTS = ROOT / "best.pt" if (ROOT / "best.pt").exists() else ROOT.parent / "best.pt"

# Prefer virtual environment python if available
VENV_PYTHON = ROOT / "env" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = ROOT.parent / "env" / "Scripts" / "python.exe"
py_exec = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

os.chdir(ROOT)

args = list(sys.argv[1:])
if not any(arg in args for arg in ("--weights", "-w")):
    args = ["--weights", str(WEIGHTS)] + args

if not any(arg in args for arg in ("--img", "--img-size", "--imgsz")):
    args += ["--img", "416"]

# Use a high confidence threshold (0.75) by default to suppress face / background noise
# Genuine gestures typically score 0.90+
if not any(arg in args for arg in ("--conf", "--conf-thres")):
    args += ["--conf", "0.75"]

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