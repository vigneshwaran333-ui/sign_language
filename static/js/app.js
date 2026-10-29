/**
 * Z-Sign Live Sign Language Detection - Client Application
 * Handles camera streaming, high-DPI canvas annotation, detection polling,
 * detection history, speech output, and status monitoring.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Navigation & Theme
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const ttsToggleBtn = document.getElementById('ttsToggleBtn');
    const alertBanner = document.getElementById('alertBanner');
    const alertTitle = document.getElementById('alertTitle');
    const alertMessage = document.getElementById('alertMessage');
    const alertCloseBtn = document.getElementById('alertCloseBtn');

    // DOM Elements - Video & Overlays
    const viewportContainer = document.getElementById('viewportContainer');
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('overlayCanvas');
    const serverStreamImg = document.getElementById('serverStreamImg');
    const standbyOverlay = document.getElementById('standbyOverlay');
    const detectionIndicator = document.getElementById('detectionIndicator');
    const captureCanvas = document.getElementById('captureCanvas');
    const sourceBadge = document.getElementById('sourceBadge');

    // DOM Elements - Controls & Sliders
    const cameraSelect = document.getElementById('cameraSelect');
    const captureModeSelect = document.getElementById('captureModeSelect');
    const startCamBtn = document.getElementById('startCamBtn');
    const stopCamBtn = document.getElementById('stopCamBtn');
    const sampleTestBtn = document.getElementById('sampleTestBtn');
    const confSlider = document.getElementById('confSlider');
    const confValue = document.getElementById('confValue');

    // DOM Elements - Status Indicators
    const camDot = document.getElementById('camDot');
    const camStatusText = document.getElementById('camStatusText');
    const modelDot = document.getElementById('modelDot');
    const modelStatusText = document.getElementById('modelStatusText');
    const fpsMetric = document.getElementById('fpsMetric');
    const latencyMetric = document.getElementById('latencyMetric');

    // DOM Elements - Current Detection Panel
    const emptyDetectionState = document.getElementById('emptyDetectionState');
    const activeDetectionView = document.getElementById('activeDetectionView');
    const activeCountBadge = document.getElementById('activeCountBadge');
    const currentSignLabel = document.getElementById('currentSignLabel');
    const currentSignClassId = document.getElementById('currentSignClassId');
    const currentConfPct = document.getElementById('currentConfPct');
    const confBarFill = document.getElementById('confBarFill');
    const currentTimestamp = document.getElementById('currentTimestamp');
    const currentBBox = document.getElementById('currentBBox');

    // DOM Elements - History & Reference
    const historyList = document.getElementById('historyList');
    const emptyHistoryState = document.getElementById('emptyHistoryState');
    const historyCounter = document.getElementById('historyCounter');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const classTagsGrid = document.getElementById('classTagsGrid');

    // Application State
    let currentStream = null;
    let isDetecting = false;
    let isProcessingFrame = false;
    let captureMode = 'browser'; // 'browser' or 'server'
    let voiceOutputEnabled = false;
    let lastSpokenSign = '';
    let lastSpokenTime = 0;
    let lastHistorySign = '';
    let lastHistoryTime = 0;
    let historyItems = [];
    const MAX_HISTORY = 50;

    // FPS Calculation State
    let frameCount = 0;
    let lastFpsUpdate = performance.now();

    // -------------------------------------------------------------
    // 1. Theme Management (Dark / Light)
    // -------------------------------------------------------------
    const savedTheme = localStorage.getItem('zsign_theme') || 'dark';
    setTheme(savedTheme);

    themeToggleBtn.addEventListener('click', () => {
        const activeTheme = document.documentElement.getAttribute('data-theme');
        const nextTheme = activeTheme === 'dark' ? 'light' : 'dark';
        setTheme(nextTheme);
    });

    function setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('zsign_theme', theme);
        const icon = themeToggleBtn.querySelector('.theme-icon');
        const text = themeToggleBtn.querySelector('.btn-text');
        if (theme === 'dark') {
            icon.textContent = '🌙';
            text.textContent = 'Dark';
        } else {
            icon.textContent = '☀️';
            text.textContent = 'Light';
        }
    }

    // -------------------------------------------------------------
    // 2. Alert Notification System
    // -------------------------------------------------------------
    function showAlert(title, message, type = 'danger') {
        alertTitle.textContent = title;
        alertMessage.textContent = message;
        alertBanner.className = `alert-banner ${type}`;
        alertBanner.classList.remove('hidden');
    }

    function hideAlert() {
        alertBanner.classList.add('hidden');
    }

    alertCloseBtn.addEventListener('click', hideAlert);

    // -------------------------------------------------------------
    // 3. Model Status Check
    // -------------------------------------------------------------
    async function checkModelStatus() {
        try {
            modelDot.className = 'status-dot yellow';
            modelStatusText.textContent = 'Model: Checking...';

            const resp = await fetch('/api/status');
            const data = await resp.json();

            if (data.model_loaded) {
                modelDot.className = 'status-dot green';
                modelStatusText.textContent = `Model: Ready (16 Signs • ${data.device.toUpperCase()})`;
                populateClassReference(data.classes);
            } else {
                modelDot.className = 'status-dot red';
                modelStatusText.textContent = 'Model: Load Failed';
                showAlert('Model Loading Error', data.error || 'Failed to load best.pt weights', 'danger');
            }
        } catch (err) {
            modelDot.className = 'status-dot red';
            modelStatusText.textContent = 'Backend: Offline';
            showAlert('Backend Connection Error', 'Could not connect to the Z-Sign backend server.', 'danger');
        }
    }

    function populateClassReference(classesDict) {
        if (!classesDict || !classTagsGrid) return;
        classTagsGrid.innerHTML = '';
        const names = Object.values(classesDict);
        names.forEach(name => {
            const pill = document.createElement('span');
            pill.className = 'class-pill';
            pill.id = `pill-${name.toLowerCase()}`;
            pill.textContent = name;
            classTagsGrid.appendChild(pill);
        });
    }

    // -------------------------------------------------------------
    // 4. Voice Output (Text-to-Speech)
    // -------------------------------------------------------------
    ttsToggleBtn.addEventListener('click', () => {
        voiceOutputEnabled = !voiceOutputEnabled;
        const textSpan = ttsToggleBtn.querySelector('.btn-text');
        if (voiceOutputEnabled) {
            ttsToggleBtn.style.borderColor = 'var(--accent-emerald)';
            textSpan.textContent = 'Voice: ON';
            speakText('Voice output enabled');
        } else {
            ttsToggleBtn.style.borderColor = '';
            textSpan.textContent = 'Voice: OFF';
        }
    });

    function speakText(text) {
        if (!voiceOutputEnabled || !('speechSynthesis' in window)) return;
        try {
            window.speechSynthesis.cancel(); // Stop prior speech
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        } catch (e) {
            console.warn('TTS error:', e);
        }
    }

    // -------------------------------------------------------------
    // 5. Camera Device Enumeration
    // -------------------------------------------------------------
    async function enumerateCameras() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
            cameraSelect.innerHTML = '<option value="">Default Camera</option>';
            return;
        }

        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === 'videoinput');

            cameraSelect.innerHTML = '';
            if (videoDevices.length === 0) {
                cameraSelect.innerHTML = '<option value="">No Camera Found</option>';
                return;
            }

            videoDevices.forEach((device, index) => {
                const opt = document.createElement('option');
                opt.value = device.deviceId;
                opt.textContent = device.label || `Camera ${index + 1}`;
                cameraSelect.appendChild(opt);
            });
        } catch (err) {
            console.error('Error enumerating cameras:', err);
            cameraSelect.innerHTML = '<option value="">Default Camera</option>';
        }
    }

    if (navigator.mediaDevices && navigator.mediaDevices.ondevicechange !== undefined) {
        navigator.mediaDevices.ondevicechange = enumerateCameras;
    }

    // -------------------------------------------------------------
    // 6. Camera Controls (Start / Stop)
    // -------------------------------------------------------------
    startCamBtn.addEventListener('click', startCamera);
    stopCamBtn.addEventListener('click', stopCamera);

    captureModeSelect.addEventListener('change', () => {
        captureMode = captureModeSelect.value;
        sourceBadge.textContent = captureMode === 'browser' ? 'Browser WebRTC' : 'Server OpenCV';
        if (isDetecting) {
            stopCamera();
            startCamera();
        }
    });

    cameraSelect.addEventListener('change', () => {
        if (isDetecting && captureMode === 'browser') {
            startCamera(); // Restart with new selected camera
        }
    });

    async function startCamera() {
        hideAlert();

        if (captureMode === 'server') {
            // Server OpenCV Mode
            startServerMode();
            return;
        }

        // Browser WebRTC Mode
        try {
            if (currentStream) {
                currentStream.getTracks().forEach(t => t.stop());
            }

            const deviceId = (cameraSelect.value && cameraSelect.value !== '') ? { ideal: cameraSelect.value } : undefined;
            const constraints = {
                video: deviceId ? {
                    deviceId: deviceId,
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                } : {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            currentStream = stream;
            video.srcObject = stream;
            await video.play();

            // Refresh camera labels if permission was just granted
            await enumerateCameras();

            // Setup UI states
            standbyOverlay.classList.add('hidden');
            serverStreamImg.classList.add('hidden');
            video.classList.remove('hidden');
            canvas.classList.remove('hidden');
            detectionIndicator.classList.remove('hidden');

            startCamBtn.classList.add('hidden');
            stopCamBtn.classList.remove('hidden');

            camDot.className = 'status-dot green';
            camStatusText.textContent = 'Camera: Active';

            isDetecting = true;
            lastFpsUpdate = performance.now();
            frameCount = 0;

            // Start detection loop
            requestAnimationFrame(detectionLoop);
        } catch (err) {
            console.warn('Browser WebRTC camera failed, falling back to Server OpenCV mode:', err);
            // Auto fallback to Server OpenCV mode
            try {
                startServerMode();
            } catch (fallbackErr) {
                console.error('Server fallback also failed:', fallbackErr);
                let msg = 'Could not access camera.';
                if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                    msg = 'Camera permission was denied. Please allow camera access in your browser address bar.';
                } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                    msg = 'No camera device found on this system.';
                } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
                    msg = 'Camera is currently in use by another application. Please close other camera apps.';
                }
                showAlert('Camera Error', msg, 'danger');
                camDot.className = 'status-dot red';
                camStatusText.textContent = 'Camera: Blocked';
            }
        }
    }

    function startServerMode() {
        standbyOverlay.classList.add('hidden');
        video.classList.add('hidden');
        canvas.classList.add('hidden');
        serverStreamImg.classList.remove('hidden');
        serverStreamImg.src = `/video_feed?camera_id=0&t=${Date.now()}`;

        startCamBtn.classList.add('hidden');
        stopCamBtn.classList.remove('hidden');

        camDot.className = 'status-dot green';
        camStatusText.textContent = 'Camera: OpenCV Active';
        sourceBadge.textContent = 'Server OpenCV';
        captureModeSelect.value = 'server';
        captureMode = 'server';
        isDetecting = true;
    }

    function stopCamera() {
        isDetecting = false;

        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }

        video.srcObject = null;
        serverStreamImg.src = '';
        serverStreamImg.classList.add('hidden');

        // Clear canvas overlay
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        standbyOverlay.classList.remove('hidden');
        detectionIndicator.classList.add('hidden');

        startCamBtn.classList.remove('hidden');
        stopCamBtn.classList.add('hidden');

        camDot.className = 'status-dot red';
        camStatusText.textContent = 'Camera: OFF';
        fpsMetric.textContent = '0.0';
        latencyMetric.textContent = '0 ms';

        resetCurrentDetectionUI();
    }

    // -------------------------------------------------------------
    // 7. Real-Time Detection Inference Loop
    // -------------------------------------------------------------
    async function detectionLoop() {
        if (!isDetecting || captureMode !== 'browser') return;

        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !isProcessingFrame) {
            isProcessingFrame = true;
            await processAndDetectFrame();
            isProcessingFrame = false;

            // Calculate FPS
            frameCount++;
            const now = performance.now();
            const elapsed = now - lastFpsUpdate;
            if (elapsed >= 1000) {
                const fps = (frameCount * 1000) / elapsed;
                fpsMetric.textContent = fps.toFixed(1);
                frameCount = 0;
                lastFpsUpdate = now;
            }
        }

        if (isDetecting) {
            requestAnimationFrame(detectionLoop);
        }
    }

    async function processAndDetectFrame() {
        // Sync canvas dimensions to video
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            captureCanvas.width = video.videoWidth;
            captureCanvas.height = video.videoHeight;
        }

        if (!video.videoWidth || !video.videoHeight) return;

        // Draw video frame to off-screen capture canvas
        const capCtx = captureCanvas.getContext('2d');
        capCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.80);

        const confThres = parseFloat(confSlider.value) / 100.0;

        try {
            const resp = await fetch('/api/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: dataUrl,
                    conf_thres: confThres
                })
            });

            if (!resp.ok) {
                const errData = await resp.json();
                console.warn('Inference response not OK:', errData);
                return;
            }

            const data = await resp.json();
            latencyMetric.textContent = `${data.inference_ms || 0} ms`;

            // Draw bounding boxes on transparent canvas overlay
            drawDetectionsOnCanvas(data.detections || [], canvas.width, canvas.height);

            // Update right sidebar detection cards
            handleDetectionResults(data.detections || [], data.timestamp);

        } catch (err) {
            console.error('Detection request failed:', err);
        }
    }

    // -------------------------------------------------------------
    // 8. Canvas Bounding Box & Label Rendering
    // -------------------------------------------------------------
    function drawDetectionsOnCanvas(detections, width, height) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);

        if (!detections || detections.length === 0) return;

        detections.forEach(det => {
            const box = det.box;
            const label = det.label;
            const confPct = det.confidence_pct;
            const x = box.x1;
            const y = box.y1;
            const w = box.width;
            const h = box.height;

            // 1. Box Glow Effect
            ctx.save();
            ctx.strokeStyle = 'rgba(16, 185, 129, 0.9)';
            ctx.lineWidth = 2.5;
            ctx.shadowColor = 'rgba(16, 185, 129, 0.6)';
            ctx.shadowBlur = 12;

            // Draw rounded rectangle
            drawRoundedRect(ctx, x, y, w, h, 6);
            ctx.stroke();
            ctx.restore();

            // 2. Corner reticle accents (assistive tech aesthetic)
            const cornerLen = Math.min(18, Math.floor(w / 4), Math.floor(h / 4));
            ctx.strokeStyle = '#06b6d4'; // Cyan
            ctx.lineWidth = 3.5;
            ctx.lineCap = 'round';

            // Top-Left
            ctx.beginPath();
            ctx.moveTo(x, y + cornerLen);
            ctx.lineTo(x, y);
            ctx.lineTo(x + cornerLen, y);
            ctx.stroke();

            // Top-Right
            ctx.beginPath();
            ctx.moveTo(x + w - cornerLen, y);
            ctx.lineTo(x + w, y);
            ctx.lineTo(x + w, y + cornerLen);
            ctx.stroke();

            // Bottom-Left
            ctx.beginPath();
            ctx.moveTo(x, y + h - cornerLen);
            ctx.lineTo(x, y + h);
            ctx.lineTo(x + cornerLen, y + h);
            ctx.stroke();

            // Bottom-Right
            ctx.beginPath();
            ctx.moveTo(x + w - cornerLen, y + h);
            ctx.lineTo(x + w, y + h);
            ctx.lineTo(x + w, y + h - cornerLen);
            ctx.stroke();

            // 3. Label Pill Badge above box
            const text = `${label}  ${confPct.toFixed(1)}%`;
            ctx.font = '600 14px "Inter", sans-serif';
            const textMetrics = ctx.measureText(text);
            const textWidth = textMetrics.width;
            const padX = 10;
            const padY = 6;
            const pillH = 26;
            const pillW = textWidth + padX * 2;
            const pillY = y - pillH - 4 > 4 ? y - pillH - 4 : y + 6;

            // Badge Background
            ctx.fillStyle = 'rgba(11, 15, 25, 0.88)';
            drawRoundedRect(ctx, x, pillY, pillW, pillH, 5);
            ctx.fill();

            // Badge Border
            ctx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
            ctx.lineWidth = 1;
            drawRoundedRect(ctx, x, pillY, pillW, pillH, 5);
            ctx.stroke();

            // Badge Text
            ctx.fillStyle = '#ffffff';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, x + padX, pillY + pillH / 2);
        });
    }

    function drawRoundedRect(ctx, x, y, width, height, radius) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
    }

    // -------------------------------------------------------------
    // 9. Detections Panel & History Management
    // -------------------------------------------------------------
    function handleDetectionResults(detections, timestamp) {
        if (!detections || detections.length === 0) {
            resetCurrentDetectionUI();
            return;
        }

        const topDet = detections[0];
        activeCountBadge.textContent = `${detections.length} Detected`;

        // Update Current Detection UI
        emptyDetectionState.classList.add('hidden');
        activeDetectionView.classList.remove('hidden');

        currentSignLabel.textContent = topDet.label;
        currentSignClassId.textContent = `Class #${topDet.class_id}`;
        currentConfPct.textContent = `${topDet.confidence_pct.toFixed(1)}%`;
        confBarFill.style.width = `${Math.min(100, topDet.confidence_pct)}%`;
        currentTimestamp.textContent = timestamp || new Date().toLocaleTimeString();
        currentBBox.textContent = `[${topDet.box.x1}, ${topDet.box.y1}, ${topDet.box.x2}, ${topDet.box.y2}]`;

        // Highlight class pill in reference drawer
        highlightClassPill(topDet.label);

        // Voice output debounce (speak once per distinct gesture, min 2s between repeats)
        const now = Date.now();
        if (topDet.label !== lastSpokenSign || (now - lastSpokenTime > 2500)) {
            speakText(topDet.label);
            lastSpokenSign = topDet.label;
            lastSpokenTime = now;
        }

        // Add to history log (debounce identical sign within 1.2s)
        if (topDet.label !== lastHistorySign || (now - lastHistoryTime > 1200)) {
            addHistoryItem({
                label: topDet.label,
                confidence_pct: topDet.confidence_pct,
                timestamp: timestamp || new Date().toLocaleTimeString(),
                class_id: topDet.class_id
            });
            lastHistorySign = topDet.label;
            lastHistoryTime = now;
        }
    }

    function resetCurrentDetectionUI() {
        activeCountBadge.textContent = '0 Detected';
        emptyDetectionState.classList.remove('hidden');
        activeDetectionView.classList.add('hidden');
        clearClassPillHighlight();
    }

    function highlightClassPill(label) {
        clearClassPillHighlight();
        const pill = document.getElementById(`pill-${label.toLowerCase()}`);
        if (pill) pill.classList.add('highlight');
    }

    function clearClassPillHighlight() {
        document.querySelectorAll('.class-pill.highlight').forEach(p => p.classList.remove('highlight'));
    }

    function addHistoryItem(item) {
        historyItems.unshift(item);
        if (historyItems.length > MAX_HISTORY) {
            historyItems.pop();
        }
        renderHistory();
    }

    function renderHistory() {
        historyCounter.textContent = `(${historyItems.length} items)`;
        if (historyItems.length === 0) {
            emptyHistoryState.classList.remove('hidden');
            historyList.innerHTML = '';
            return;
        }

        emptyHistoryState.classList.add('hidden');
        historyList.innerHTML = '';

        historyItems.forEach(item => {
            const li = document.createElement('li');
            li.className = 'history-item';
            li.innerHTML = `
                <div class="history-item-left">
                    <span class="history-tag">${escapeHtml(item.label)}</span>
                    <span class="history-conf">${item.confidence_pct.toFixed(1)}%</span>
                </div>
                <span class="history-time">${escapeHtml(item.timestamp)}</span>
            `;
            historyList.appendChild(li);
        });
    }

    clearHistoryBtn.addEventListener('click', () => {
        historyItems = [];
        lastHistorySign = '';
        renderHistory();
    });

    function escapeHtml(str) {
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // -------------------------------------------------------------
    // 10. Sample Sign Verification Button
    // -------------------------------------------------------------
    sampleTestBtn.addEventListener('click', async () => {
        sampleTestBtn.disabled = true;
        sampleTestBtn.textContent = '⏳ Testing...';

        try {
            const resp = await fetch('/api/sample');
            const data = await resp.json();

            if (data.detections && data.detections.length > 0) {
                showAlert(
                    'Sample Sign Verified!',
                    `Successfully detected sign "${data.detections[0].label}" with ${data.detections[0].confidence_pct}% confidence in sample (${data.sample_file}).`,
                    'success'
                );
                handleDetectionResults(data.detections, data.timestamp);

                // If annotated image is returned, show in standby view temporarily
                if (data.annotated_image) {
                    serverStreamImg.src = data.annotated_image;
                    serverStreamImg.classList.remove('hidden');
                    standbyOverlay.classList.add('hidden');
                }
            } else {
                showAlert('Sample Test Completed', 'No detections found in the sample image.', 'warning');
            }
        } catch (err) {
            console.error('Sample test error:', err);
            showAlert('Sample Test Error', 'Could not run inference on sample image.', 'danger');
        } finally {
            sampleTestBtn.disabled = false;
            sampleTestBtn.textContent = '🧪 Test Sample Sign';
        }
    });

    // -------------------------------------------------------------
    // 11. Confidence Slider Listener
    // -------------------------------------------------------------
    confSlider.addEventListener('input', () => {
        confValue.textContent = `${confSlider.value}%`;
    });

    // Click on standby viewport to start camera immediately
    if (standbyOverlay) {
        standbyOverlay.style.cursor = 'pointer';
        standbyOverlay.addEventListener('click', () => {
            if (!isDetecting) {
                startCamera();
            }
        });
    }

    // Initializations
    checkModelStatus();
    enumerateCameras().then(() => {
        // Auto-start camera after cameras are ready
        setTimeout(() => {
            console.log('Auto-starting camera detection...');
            startCamera();
        }, 500);
    }).catch(() => {
        setTimeout(startCamera, 500);
    });
});
