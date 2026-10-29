"""
Inference Wrapper for z-sign YOLOv5 Detection Model.
Acts as a pure black-box wrapper around the existing z-sign project
located at C:/zsign-detection without modifying any files inside it.
"""

import os
import sys
import time
import logging
from pathlib import Path
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ZSignWrapper")

# Determine root directory dynamically (relative to this file)
DEFAULT_ZSIGN_ROOT = Path(__file__).resolve().parent
if not (DEFAULT_ZSIGN_ROOT / "yolov5").exists() and (DEFAULT_ZSIGN_ROOT.parent / "yolov5").exists():
    DEFAULT_ZSIGN_ROOT = DEFAULT_ZSIGN_ROOT.parent

YOLOV5_DIR = DEFAULT_ZSIGN_ROOT / "yolov5"
DEFAULT_WEIGHTS = DEFAULT_ZSIGN_ROOT / "best.pt"

# Add YOLOv5 directory to sys.path if not present (read-only import)
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

try:
    import torch
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes, check_img_size
    from utils.augmentations import letterbox
    from utils.torch_utils import select_device
    IMPORT_SUCCESS = True
except Exception as e:
    logger.error(f"Failed to import YOLOv5 modules from {YOLOV5_DIR}: {e}")
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


class ZSignDetector:
    """
    Black-box inference wrapper for the trained z-sign YOLOv5 model.
    Loads best.pt and exposes a clean interface for single-frame inference.
    """

    def __init__(
        self,
        weights_path: str = None,
        device_str: str = "",
        imgsz: int = 416,
        conf_thres: float = 0.80,
        iou_thres: float = 0.45,
        max_det: int = 1,
        agnostic_nms: bool = True,
        iloveyou_conf: float = 0.88,
    ):
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        self.device_str = device_str
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.iloveyou_conf = iloveyou_conf

        self.model = None
        self.device = None
        self.names = {}
        self.stride = 32
        self.pt = True
        self.is_loaded = False
        self.load_error = None

        self._load_model()

    def _load_model(self):
        """Loads weights and warms up the model."""
        if not IMPORT_SUCCESS:
            self.load_error = f"YOLOv5 dependencies could not be imported: {IMPORT_ERROR}"
            logger.error(self.load_error)
            return

        if not self.weights_path.exists():
            self.load_error = f"Weights file not found at: {self.weights_path}"
            logger.error(self.load_error)
            return

        try:
            logger.info(f"Loading z-sign model from: {self.weights_path}")
            self.device = select_device(self.device_str)
            
            # Use dataset yaml if present, else fallback
            data_yaml = DEFAULT_ZSIGN_ROOT / "dataset_merged.yaml"
            if not data_yaml.exists():
                data_yaml = YOLOV5_DIR / "data" / "coco128.yaml"

            self.model = DetectMultiBackend(
                self.weights_path,
                device=self.device,
                dnn=False,
                data=data_yaml,
                fp16=False,
            )
            self.stride = self.model.stride
            self.names = self.model.names
            self.pt = self.model.pt
            self.imgsz = check_img_size(self.imgsz, s=self.stride)

            # Warmup model
            logger.info("Warming up model...")
            self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
            self.is_loaded = True
            self.load_error = None
            logger.info(f"Model successfully loaded. {len(self.names)} classes detected: {list(self.names.values())}")
        except Exception as e:
            self.load_error = str(e)
            self.is_loaded = False
            logger.exception(f"Error loading model: {e}")

    def get_status(self) -> dict:
        """Returns model loading and configuration status."""
        return {
            "loaded": self.is_loaded,
            "weights_path": str(self.weights_path),
            "device": str(self.device) if self.device else "unknown",
            "classes_count": len(self.names),
            "classes": self.names,
            "error": self.load_error,
            "imgsz": self.imgsz,
            "default_conf": self.conf_thres,
        }

    def detect(
        self,
        frame: np.ndarray,
        conf_thres: float = None,
        iou_thres: float = None,
        max_det: int = None,
        agnostic_nms: bool = None,
        iloveyou_conf: float = None,
    ) -> dict:
        """
        Runs inference on a single BGR OpenCV frame.
        
        Args:
            frame: np.ndarray in BGR format (height, width, 3)
            conf_thres: Confidence threshold (default: 0.80)
            iou_thres: NMS IoU threshold (default: 0.45)
            max_det: Maximum detections (default: 1)
            agnostic_nms: Class-agnostic NMS (default: True)
            iloveyou_conf: Confidence threshold for 'IloveYou' class (default: 0.88)
            
        Returns:
            dict containing:
                - detections: list of detection dicts (label, confidence, box, box_norm)
                - inference_ms: latency in milliseconds
                - frame_size: {"width": W, "height": H}
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model is not loaded: {self.load_error}")

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {"detections": [], "inference_ms": 0.0, "frame_size": {"width": 0, "height": 0}}

        conf_thres = self.conf_thres if conf_thres is None else float(conf_thres)
        iou_thres = self.iou_thres if iou_thres is None else float(iou_thres)
        max_det = self.max_det if max_det is None else int(max_det)
        agnostic_nms = self.agnostic_nms if agnostic_nms is None else bool(agnostic_nms)
        iloveyou_conf = self.iloveyou_conf if iloveyou_conf is None else float(iloveyou_conf)

        h0, w0 = frame.shape[:2]
        t0 = time.perf_counter()

        # 1. Letterbox resizing (maintains aspect ratio with padding)
        im = letterbox(frame, self.imgsz, stride=self.stride, auto=self.pt)[0]
        
        # 2. BGR to RGB, HWC to CHW
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        # 3. Torch tensor conversion & normalization [0.0 - 1.0]
        im_t = torch.from_numpy(im).to(self.device)
        im_t = im_t.half() if self.model.fp16 else im_t.float()
        im_t /= 255.0
        if len(im_t.shape) == 3:
            im_t = im_t[None]  # Expand batch dimension [1, 3, H, W]

        # 4. Inference
        with torch.no_grad():
            pred = self.model(im_t, augment=False)
            
            # 5. Non-Max Suppression
            pred = non_max_suppression(
                pred,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                classes=None,
                agnostic=agnostic_nms,
                max_det=max_det,
            )

        # 6. Parse and rescale detections
        detections = []
        if len(pred) and len(pred[0]):
            det = pred[0].clone()
            det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], frame.shape).round()

            # 7. Apply heuristic filtering (specifically for IloveYou as defined in z-sign)
            for *box, conf, cls in det:
                c = int(cls)
                c_name = self.names.get(c, str(c))
                c_name_lower = c_name.lower()
                conf_val = float(conf)

                is_valid = True
                if "iloveyou" in c_name_lower or "i love you" in c_name_lower:
                    if conf_val < iloveyou_conf:
                        is_valid = False
                    elif conf_val < 0.90:
                        x1, y1, x2, y2 = [float(coord) for coord in box]
                        bw_norm = (x2 - x1) / max(w0, 1)
                        bh_norm = (y2 - y1) / max(h0, 1)
                        norm_ratio = bh_norm / max(bw_norm, 1e-5)
                        if norm_ratio < 1.15:
                            is_valid = False

                if is_valid:
                    x1, y1, x2, y2 = [int(v.item()) for v in box]
                    # Clamp bounding box coordinates
                    bx1 = max(0, min(w0 - 1, x1))
                    by1 = max(0, min(h0 - 1, y1))
                    bx2 = max(bx1 + 1, min(w0, x2))
                    by2 = max(by1 + 1, min(h0, y2))

                    detections.append({
                        "class_id": c,
                        "label": c_name,
                        "confidence": round(conf_val, 4),
                        "confidence_pct": round(conf_val * 100, 1),
                        "box": {
                            "x1": bx1,
                            "y1": by1,
                            "x2": bx2,
                            "y2": by2,
                            "width": bx2 - bx1,
                            "height": by2 - by1,
                        },
                        "box_norm": {
                            "x1": round(bx1 / max(w0, 1), 4),
                            "y1": round(by1 / max(h0, 1), 4),
                            "x2": round(bx2 / max(w0, 1), 4),
                            "y2": round(by2 / max(h0, 1), 4),
                            "width": round((bx2 - bx1) / max(w0, 1), 4),
                            "height": round((by2 - by1) / max(h0, 1), 4),
                        },
                    })

        inference_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "detections": detections,
            "inference_ms": inference_ms,
            "frame_size": {"width": w0, "height": h0},
        }

    def annotate_frame(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draws bounding boxes and labels onto a copy of the frame.
        Useful for server-side OpenCV stream rendering.
        """
        annotated = frame.copy()
        for det in detections:
            box = det["box"]
            label = det["label"]
            conf_pct = det.get("confidence_pct", 0.0)
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            # Main bounding box
            color = (0, 230, 115)  # Vibrant green/emerald in BGR
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Reticle corners
            corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            c_color = (0, 255, 255)  # Cyan
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), c_color, 3)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), c_color, 3)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), c_color, 3)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), c_color, 3)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), c_color, 3)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), c_color, 3)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), c_color, 3)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), c_color, 3)

            # Label banner
            text = f"{label} {conf_pct:.1f}%"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = max(y1, th + 8)
            cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 10, ty + baseline - 2), (20, 20, 20), -1)
            cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 10, ty + baseline - 2), color, 1)
            cv2.putText(annotated, text, (x1 + 5, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated
