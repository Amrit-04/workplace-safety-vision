# Workplace Safety Vision 🦺

### A Real-Time Computer Vision Framework for Personal Protective Equipment (PPE) Detection & Hazardous Area Intrusion Monitoring

[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-11518869-00629B.svg?style=flat&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/11518869)
[![Conference](https://img.shields.io/badge/IEEE-RMKMATE%202026-blue.svg)](https://ieeexplore.ieee.org/document/11518869)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-v11-00FFFF.svg?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg?logo=flask&logoColor=white)](https://palletsprojects.com/p/flask/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

**Workplace Safety Vision** is an edge-optimized, real-time computer vision system engineered for industrial facilities, manufacturing floors, and construction environments. It simultaneously addresses the two most critical aspects of occupational safety:

1. **Personal Protective Equipment (PPE) Compliance**: Real-time multi-class object detection identifying required safety gear (`Hardhat`, `Mask`, `Safety Vest`, `Safety Cone`, `Machinery`, `Vehicle`) and proactively flagging non-compliance (`NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`).
2. **Hazardous Zone Intrusion Monitoring**: Dynamic polygon-based geofencing of high-risk or hazardous machinery operational boundaries, paired with centroid tracking (`SimpleTracker`) and multi-point ground-plane probing to detect, track, and alert unauthorized personnel entries.

This repository contains the complete implementation presented and published in **IEEE RMKMATE 2026** under the title:  
> **"Workplace Safety Using Computer Vision: A Framework for Personal Protective Equipment Detection and Intrusion Monitoring"**  
> *IEEE Xplore Record:* [11518869](https://ieeexplore.ieee.org/document/11518869)

---

## 🎯 Key Features

- **Asynchronous Dual-Worker Pipeline**: PPE detection and intrusion monitoring execute concurrently in isolated worker threads with queue-based decoupling (`Queue(maxsize=1)`), preventing inferential latency from blocking video ingest or rendering.
- **Resilient Stream Ingestion**: Dedicated `FrameGrabber` running with single-frame hardware buffers, supporting RTSP, MJPEG, webcams, and IP camera feeds with automatic reconnect logic on stream dropouts.
- **Interactive In-Browser Geofencing**: Safety supervisors can draw, edit, test, and persist arbitrary multi-vertex boundary polygons (`zones.json`) directly over the live stream via an interactive HTML5 canvas. Coordinates are normalized to support dynamic video scaling.
- **Multi-Point Anatomical Probing**: Intrusion detection evaluates both worker foot-level ground contact points `(cx, y2)` and torso/centroid coordinates against polygon boundaries, eliminating false alarms caused by tall bounding boxes extending outside danger perimeters.
- **Temporal Worker Tracking**: Integrated centroid tracker (`SimpleTracker`) assigns persistent IDs across frames, logging precise entry, active dwell time, and exit events for every personnel in a restricted zone.
- **Cooldown-Throttled Incident Auditing**: Alerts and infractions are throttled with configurable cooldown windows to prevent log flooding during persistent violations.
- **Web Dashboard & Real-Time Telemetry**: Responsive Flask control dashboard featuring live MJPEG streaming, instant mode toggling (`PPE` vs. `Intrusion`), real-time FPS telemetry (Render FPS, PPE Infer FPS, Intrusion Infer FPS), and one-click CSV incident export.
- **Hardware Acceleration Ready**: Auto-detects NVIDIA CUDA GPUs with TensorFloat-32 (`TF32`) and cuDNN benchmark optimizations enabled; gracefully falls back to multi-threaded CPU inference.

---

## 🏛️ System Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │         IP Camera / RTSP Stream / Video Feed           │
               └───────────────────────────┬────────────────────────────┘
                                           │ (cv2.CAP_FFMPEG, buffer=1)
                                           ▼
                                 ┌───────────────────┐
                                 │   FrameGrabber    │ (Auto-reconnect & Grabber Thread)
                                 └─────────┬─────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼ (Queue Decoupled)                           ▼ (Queue Decoupled)
        ┌───────────────────────┐                     ┌─────────────────────────┐
        │       PPE Worker      │                     │    Intrusion Worker     │
        │ ───────────────────── │                     │ ─────────────────────── │
        │  Fine-Tuned YOLO      │                     │  YOLO11 Person Detector │
        │  10 Safety Classes    │                     │  SimpleTracker (ID Map) │
        │  Violation Flagging   │                     │  Multi-Point Polygon    │
        │  (NO-Hardhat/Vest...) │                     │  Point-in-Zone Testing  │
        └───────────┬───────────┘                     └────────────┬────────────┘
                    │                                              │
                    └──────────────────────┬───────────────────────┘
                                           │ (Thread-Safe Synchronization)
                                           ▼
                                ┌─────────────────────┐
                                │    Render Worker    │ ◄─── Runtime Settings & Mode
                                └──────────┬──────────┘
                                           │ (MJPEG Stream @ Target FPS)
                                           ▼
                ┌─────────────────────────────────────────────────────┐
                │             Web UI & Analytics Dashboard            │
                │  - Live Video Stream & SVG Polygon Overlay          │
                │  - Interactive Zone Boundary Editor                 │
                │  - Telemetry: Pipeline & Inference FPS              │
                │  - Real-Time Incident Log & CSV Audit Download      │
                └─────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
workplace-safety-vision/
├── app.py                     # Main application: Grabber, Detectors, Web Server & API
├── requirements.txt           # Core Python dependencies
├── zones.json                 # Persisted polygon boundaries for restricted zones
├── best.pt                    # Fine-tuned YOLO PPE compliance detection weights
├── yolo11n.pt                 # Ultralytics YOLO11 person detection weights
├── static/
│   ├── app.js                 # Interactive zone drawing, telemetry updates, API clients
│   └── styles.css             # Dashboard UI styling and responsive layouts
├── templates/
│   └── index.html             # Safety dashboard layout and control panels
├── .gitignore                 # Standard Python/runtime ignore rules
├── LICENSE                    # MIT License
└── README.md                  # Comprehensive documentation and citation
```

---

## 🏷️ Detection Classes

The fine-tuned PPE model (`best.pt`) evaluates workplace gear across 10 distinct classes:

| Class ID | Label | Category | Description |
|---|---|---|---|
| `0` | **Hardhat** | Compliant | Safety helmet detected |
| `1` | **Mask** | Compliant | Protective face mask detected |
| `2` | **NO-Hardhat** | ⚠️ Violation | Worker missing required hardhat |
| `3` | **NO-Mask** | ⚠️ Violation | Worker missing required face mask |
| `4` | **NO-Safety Vest** | ⚠️ Violation | Worker missing high-visibility safety vest |
| `5` | **Person** | General | Worker / individual detected |
| `6` | **Safety Cone** | Environmental | Traffic / hazard cone perimeter marker |
| `7` | **Safety Vest** | Compliant | High-visibility reflective vest detected |
| `8` | **Machinery** | Equipment | Heavy industrial or construction machinery |
| `9` | **Vehicle** | Equipment | Site transport / industrial vehicle |

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- (Recommended) NVIDIA GPU with CUDA 11.8+ / 12.1+ for real-time edge/server inference

### 2. Clone the Repository
```bash
git clone https://github.com/Amrit-04/workplace-safety-vision.git
cd workplace-safety-vision
```

### 3. Create a Virtual Environment
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Dashboard
```bash
python app.py
```
By default, the server starts on `http://localhost:5000`. Open this address in your browser to view the live dashboard.

---

## ⚙️ Configuration

All pipeline parameters can be configured via environment variables or modified directly in `app.py`:

| Variable | Default Value | Description |
|---|---|---|
| `STREAM_URL` | `rtsp://192.168.68.109:8554/live` | Video input stream URL (RTSP, HTTP MJPEG, video file, or `0` for local webcam) |
| `PPE_MODEL_PATH` | `best.pt` | Path to the custom fine-tuned YOLO PPE weights |
| `INTRUSION_MODEL_PATH` | `yolo11n.pt` | Path to the YOLO person detection model |
| `PPE_CONF` | `0.65` | Confidence threshold for PPE violation detection |
| `INTRUSION_CONF` | `0.45` | Confidence threshold for person intrusion detection |
| `IMG_SIZE` | `416` | Square image resolution for model inference |
| `TARGET_FPS` | `15` | Target output stream and rendering frame rate |
| `JPEG_QUALITY` | `70` | MJPEG frame compression quality (1-100) |
| `PPE_EVERY_N` | `1` | PPE inference frame cadence (1 = evaluate every frame) |
| `INTRUSION_EVERY_N` | `2` | Intrusion inference frame cadence (2 = evaluate every 2nd frame) |
| `LOG_COOLDOWN_SEC` | `2.0` | Minimum interval (in seconds) between repeated violation alerts |
| `ZONE_FILE` | `zones.json` | Path to the JSON file where drawn polygon coordinates are stored |
| `HOST` | `0.0.0.0` | Flask server binding host |
| `PORT` | `5000` | Flask server port |

---

## 🌐 REST API Reference

The dashboard exposes an asynchronous REST API for monitoring, telemetry, and integration with site-wide security systems:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web dashboard UI |
| `GET` | `/video_feed` | Multipart MJPEG continuous video stream (`multipart/x-mixed-replace`) |
| `GET` | `/api/status` | JSON telemetry: pipeline FPS, individual inference FPS, GPU/CPU device, active intrusions, stream status |
| `POST` | `/api/mode` | Toggle active overlay visualization (`{"mode": "ppe"}` or `{"mode": "intrusion"}`) |
| `GET` | `/api/zones` | Retrieve defined restricted zones (normalized `[0.0 - 1.0]` coordinates) |
| `POST` | `/api/zones` | Save updated restricted zones (`{"zones": [...], "normalized": true}`) |
| `GET` | `/api/logs` | Fetch recent security and compliance incident logs (supports `?limit=100&mode=ppe`) |
| `GET` | `/api/logs/download`| Export incident logs as a formatted `.csv` audit file |

---

## 📝 Citation

If you use this framework, dataset concepts, or models in your research or commercial deployments, please cite our IEEE publication:

```bibtex
@inproceedings{yadav2026workplace,
  title={Workplace Safety Using Computer Vision: A Framework for Personal Protective Equipment Detection and Intrusion Monitoring},
  author={Yadav, Amriteshkumar and Hariharan, B. and Chelladurai, S. Prince and Das, Ahaan Sekhar and Sagar, Prajwal},
  booktitle={2026 International Conference on Research Methodologies in Knowledge Management, Artificial Intelligence and Telecommunication Engineering (RMKMATE)},
  year={2026},
  doi={10.1109/RMKMATE.2026.11518869},
  note={IEEE Xplore: 11518869}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.

---

<p align="center">
  <b>Developed by Amriteshkumar Yadav</b><br>
  <i>Building reliable, observable, and production-grade AI systems.</i>
</p>
