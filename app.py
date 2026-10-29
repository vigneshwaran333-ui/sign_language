"""
Flask Web Application for z-sign Standalone UI.
Connects to the z-sign YOLOv5 detection model as a pure black box.
Provides REST and streaming endpoints for live camera detection.
"""

import os
import sys
import time
import base64
import logging
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response

# Add current folder to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from inference_wrapper import ZSignDetector, DEFAULT_ZSIGN_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ZSignUI")

app = Flask(
    __name__,
    static_folder=str(CURRENT_DIR / "static"),
    template_folder=str(CURRENT_DIR / "templates"),
)

# Initialize detector
logger.info("Initializing ZSignDetector...")
detector = ZSignDetector()
logger.info(f"ZSignDetector status: {detector.get_status()}")

# Active OpenCV VideoCapture for server-side streaming (optional)
server_camera = None
server_camera_id = None


def decode_base64_image(data_uri: str) -> np.ndarray:
    """Decodes a base64 DataURL or raw base64 string to an OpenCV BGR image."""
    try:
        if "," in data_uri:
            data_uri = data_uri.split(",", 1)[1]
        img_bytes = base64.b64decode(data_uri)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


@app.route("/")
def index():
    """Renders the main dashboard page."""
    status = detector.get_status()
    return render_template("index.html", status=status)


@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns the model loading status, classes, and environment info."""
    status = detector.get_status()
    return jsonify({
        "status": "ok" if status["loaded"] else "error",
        "model_loaded": status["loaded"],
        "device": status["device"],
        "classes": status["classes"],
        "classes_count": status["classes_count"],
        "default_conf": status["default_conf"],
        "imgsz": status["imgsz"],
        "weights_path": status["weights_path"],
        "zsign_root": str(DEFAULT_ZSIGN_ROOT),
        "error": status["error"],
    })


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Inference endpoint: accepts base64 image or multipart file,
    runs black-box inference, and returns detections JSON.
    """
    if not detector.is_loaded:
        return jsonify({
            "error": f"Model is not loaded: {detector.load_error}",
            "detections": [],
            "timestamp": time.strftime("%H:%M:%S"),
        }), 503

    frame = None
    conf_thres = None
    
    # Check JSON base64 payload
    if request.is_json:
        payload = request.get_json()
        image_data = payload.get("image")
        conf_thres = payload.get("conf_thres")
        if image_data:
            frame = decode_base64_image(image_data)
    # Check multipart form data
    elif "file" in request.files:
        file = request.files["file"]
        conf_thres = request.form.get("conf_thres")
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "No valid image provided"}), 400

    try:
        if conf_thres is not None:
            conf_thres = float(conf_thres)
    except (ValueError, TypeError):
        conf_thres = None

    try:
        result = detector.detect(frame, conf_thres=conf_thres)
        result["timestamp"] = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
        return jsonify(result)
    except Exception as e:
        logger.exception("Error during inference")
        return jsonify({"error": str(e), "detections": []}), 500


@app.route("/api/sample", methods=["GET"])
def api_sample():
    """
    Runs inference on an existing sample image from the z-sign dataset,
    allowing the user to verify the model immediately without a physical camera.
    """
    sample_dir = DEFAULT_ZSIGN_ROOT / "Data" / "final_data" / "test" / "images"
    sample_img_path = None

    if sample_dir.exists():
        for candidate in sample_dir.glob("*.jpg"):
            sample_img_path = candidate
            break

    if not sample_img_path or not sample_img_path.exists():
        # Create a synthetic test pattern if no sample images found
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_img, "Z-Sign Test Pattern", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        test_img = cv2.imread(str(sample_img_path))

    result = detector.detect(test_img, conf_thres=0.50)
    result["sample_file"] = sample_img_path.name if sample_img_path else "synthetic"
    result["timestamp"] = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"

    # Encode image with annotations to base64 for display
    annotated = detector.annotate_frame(test_img, result["detections"])
    _, buf = cv2.imencode(".jpg", annotated)
    result["annotated_image"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    return jsonify(result)


@app.route("/api/cameras", methods=["GET"])
def api_cameras():
    """
    Scans for connected OpenCV-accessible cameras.
    """
    available_cameras = []
    # Test up to index 3
    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available_cameras.append({"id": idx, "name": f"Camera {idx} ({w}x{h})"})
            cap.release()

    return jsonify({"cameras": available_cameras})


def generate_mjpeg_stream(camera_id: int):
    """Generates MJPEG multipart stream for backend OpenCV camera mode."""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        logger.error(f"Could not open camera {camera_id}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Run inference
            res = detector.detect(frame)
            annotated = detector.annotate_frame(frame, res["detections"])

            # Encode as JPEG
            _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.03)  # Limit to ~30 FPS
    finally:
        cap.release()


@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint for server OpenCV camera mode."""
    camera_id = request.args.get("camera_id", 0, type=int)
    return Response(
        generate_mjpeg_stream(camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 65)
    print("  Z-Sign Real-Time Recognition Web UI")
    print(f"  Connected to z-sign at: {DEFAULT_ZSIGN_ROOT}")
    print(f"  Model loaded: {detector.is_loaded}")
    print(f"  Open in browser: http://localhost:{port}")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
