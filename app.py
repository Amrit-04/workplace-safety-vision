import os
import cv2
import time
import json
import csv
import io
import torch
import logging
import threading
import numpy as np
from queue import Queue, Full, Empty
from collections import deque
from datetime import datetime
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

STREAM_URL = os.getenv("STREAM_URL", "rtsp://192.168.68.109:8554/live")
PPE_MODEL_PATH = os.getenv("PPE_MODEL_PATH", "best.pt")
INTRUSION_MODEL_PATH = os.getenv("INTRUSION_MODEL_PATH", "yolo11n.pt")
PPE_CONF = float(os.getenv("PPE_CONF", "0.65"))
INTRUSION_CONF = float(os.getenv("INTRUSION_CONF", "0.45"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "416"))
TARGET_FPS = float(os.getenv("TARGET_FPS", "15"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))
PPE_EVERY_N = int(os.getenv("PPE_EVERY_N", "1"))
INTRUSION_EVERY_N = int(os.getenv("INTRUSION_EVERY_N", "2"))
LOG_COOLDOWN_SEC = float(os.getenv("LOG_COOLDOWN_SEC", "2.0"))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ZONE_FILE = os.getenv("ZONE_FILE", "zones.json")
APP_VERSION = "mode-logs-zones-20260507-3"


def _setup_logging():
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"unified_web_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
    )
    return path


LOG_FILE = _setup_logging()


class EventLog:
    def __init__(self, maxlen=600):
        self._items = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    @staticmethod
    def _category(message):
        text = message.upper()
        if text.startswith("INTRUSION "):
            return "intrusion"
        if text.startswith("PPE VIOLATION"):
            return "ppe"
        return "system"

    def add(self, level, message):
        row = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level.upper(),
            "message": message,
            "category": self._category(message),
        }
        with self._lock:
            self._items.appendleft(row)
        if level.lower() == "error":
            logging.error(message)
        elif level.lower() == "warning":
            logging.warning(message)
        else:
            logging.info(message)

    def list(self, limit=100, mode=None):
        with self._lock:
            rows = list(self._items)
        if mode in ("ppe", "intrusion"):
            rows = [row for row in rows if row.get("category") == mode]
        return rows[:limit]

    def all(self, mode=None):
        with self._lock:
            rows = list(self._items)
        if mode in ("ppe", "intrusion"):
            rows = [row for row in rows if row.get("category") == mode]
        return rows


EVENTS = EventLog()
EVENTS.add("info", f"Logging to {LOG_FILE}")


def optimize_runtime():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        EVENTS.add("info", "CUDA enabled")
    else:
        EVENTS.add("warning", "CUDA unavailable, running on CPU")


class FrameGrabber:
    def __init__(self, source):
        self.source = source
        self._cap = None
        self._frame = None
        self._ok = False
        self._shape = None
        self._connected = False
        self._last_error = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _open(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while True:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open()
                if not self._cap.isOpened():
                    msg = f"Unable to open stream: {self.source}"
                    with self._lock:
                        self._connected = False
                        self._last_error = msg
                    EVENTS.add("error", msg)
                    time.sleep(1.0)
                    continue
                EVENTS.add("info", f"Connected to stream: {self.source}")
                with self._lock:
                    self._connected = True
                    self._last_error = None

            ok, frame = self._cap.read()
            if not ok or frame is None:
                msg = "Stream read failed, reconnecting"
                with self._lock:
                    self._connected = False
                    self._last_error = msg
                EVENTS.add("warning", msg)
                self._cap.release()
                self._cap = None
                time.sleep(0.3)
                continue

            with self._lock:
                self._ok = True
                self._frame = frame
                h, w = frame.shape[:2]
                self._shape = (w, h)
            time.sleep(0.001)

    def latest(self):
        with self._lock:
            if not self._ok or self._frame is None:
                return False, None
            return True, self._frame.copy()

    def shape(self):
        with self._lock:
            return self._shape

    def status(self):
        with self._lock:
            return {"connected": self._connected, "last_error": self._last_error, "shape": self._shape}


class RuntimeSettings:
    def __init__(self, img_size, target_fps, jpeg_quality):
        self.img_size = img_size
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()

    def snapshot(self):
        with self._lock:
            return {"img_size": int(self.img_size), "target_fps": float(self.target_fps), "jpeg_quality": int(self.jpeg_quality)}

    def apply(self, img_size=None, target_fps=None, jpeg_quality=None):
        with self._lock:
            if img_size is not None:
                self.img_size = int(img_size)
            if target_fps is not None:
                self.target_fps = float(target_fps)
            if jpeg_quality is not None:
                self.jpeg_quality = int(jpeg_quality)


class SimpleTracker:
    def __init__(self, max_age=20, max_dist=80):
        self.max_age = max_age
        self.max_dist = max_dist
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, boxes):
        assigned, used_tracks = {}, set()
        centers = [self._center(b) for b in boxes]
        for i, c in enumerate(centers):
            best_tid, best_d = None, 1e9
            for tid, data in self.tracks.items():
                if tid in used_tracks:
                    continue
                px, py = data["center"]
                d = ((c[0] - px) ** 2 + (c[1] - py) ** 2) ** 0.5
                if d < best_d and d <= self.max_dist:
                    best_d, best_tid = d, tid
            if best_tid is None:
                best_tid = self.next_id
                self.next_id += 1
                self.tracks[best_tid] = {"center": c, "age": 0}
            else:
                self.tracks[best_tid]["center"] = c
                self.tracks[best_tid]["age"] = 0
            used_tracks.add(best_tid)
            assigned[i] = best_tid
        for tid in list(self.tracks):
            if tid not in used_tracks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        return assigned


class DetectorEngine:
    PPE_LABELS = {
        0: "Hardhat",
        1: "Mask",
        2: "NO-Hardhat",
        3: "NO-Mask",
        4: "NO-Safety Vest",
        5: "Person",
        6: "Safety Cone",
        7: "Safety Vest",
        8: "Machinery",
        9: "Vehicle",
    }

    def __init__(self):
        EVENTS.add("info", f"Loading PPE model: {PPE_MODEL_PATH}")
        self.ppe = YOLO(PPE_MODEL_PATH)
        EVENTS.add("info", f"Loading intrusion model: {INTRUSION_MODEL_PATH}")
        self.intrusion = YOLO(INTRUSION_MODEL_PATH)
        for m in [self.ppe, self.intrusion]:
            try:
                m.fuse()
            except Exception:
                pass

        self.settings = RuntimeSettings(IMG_SIZE, TARGET_FPS, JPEG_QUALITY)
        self.mode = "ppe"
        self.mode_lock = threading.Lock()
        self.out_lock = threading.Lock()
        self.last_jpeg = None
        self.last_fps = 0.0
        self.ppe_infer_fps = 0.0
        self.intrusion_infer_fps = 0.0
        self._render_t0 = time.time()
        self._render_count = 0
        self._ppe_t0 = time.time()
        self._ppe_count = 0
        self._intr_t0 = time.time()
        self._intr_count = 0

        self.zones = []
        self.zones_lock = threading.Lock()
        self.tracker = SimpleTracker()
        self.seen_zone_pairs = set()
        self.active_intrusions = set()
        self.log_last = {}

        self.ppe_q = Queue(maxsize=1)
        self.intrusion_q = Queue(maxsize=1)
        self.ppe_result = []
        self.intrusion_result = []
        self.result_lock = threading.Lock()

        self._load_zones()
        self._warmup()
        self._start_workers()

    def _start_workers(self):
        threading.Thread(target=self._ppe_worker, daemon=True).start()
        threading.Thread(target=self._intrusion_worker, daemon=True).start()
        threading.Thread(target=self._render_worker, daemon=True).start()

    def _warmup(self):
        cfg = self.settings.snapshot()
        dummy = np.zeros((cfg["img_size"], cfg["img_size"], 3), dtype=np.uint8)
        with torch.inference_mode():
            for model in [self.ppe, self.intrusion]:
                try:
                    model.predict(dummy, imgsz=cfg["img_size"], conf=0.25, device=DEVICE, verbose=False)
                except Exception as ex:
                    EVENTS.add("warning", f"Model warmup failed: {ex}")

    def _load_zones(self):
        if not os.path.exists(ZONE_FILE):
            return
        try:
            with open(ZONE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self.zones_lock:
                self.zones = [np.array(z, dtype=np.int32) for z in data if len(z) >= 3]
            EVENTS.add("info", f"Loaded {len(self.zones)} zone(s) from {ZONE_FILE}")
        except Exception as ex:
            EVENTS.add("error", f"Failed to load zones: {ex}")

    def _log_throttled(self, key, level, message):
        now = time.time()
        prev = self.log_last.get(key, 0.0)
        if now - prev >= LOG_COOLDOWN_SEC:
            self.log_last[key] = now
            EVENTS.add(level, message)

    def submit_frame(self, frame):
        for q in [self.ppe_q, self.intrusion_q]:
            try:
                q.put_nowait(frame)
            except Full:
                try:
                    _ = q.get_nowait()
                except Empty:
                    pass
                try:
                    q.put_nowait(frame)
                except Full:
                    pass

    def set_mode(self, mode):
        if mode not in ("ppe", "intrusion"):
            return False
        with self.mode_lock:
            self.mode = mode
        EVENTS.add("info", f"Mode switched to: {mode}")
        return True

    def get_mode(self):
        with self.mode_lock:
            return self.mode

    def get_zones_pixels(self):
        with self.zones_lock:
            return [z.tolist() for z in self.zones]

    def get_zones_normalized(self, width, height):
        if not width or not height:
            return []
        with self.zones_lock:
            out = []
            for z in self.zones:
                pts = [[max(0.0, min(1.0, float(x) / width)), max(0.0, min(1.0, float(y) / height))] for x, y in z]
                out.append(pts)
            return out

    def save_zones(self, zones_pixels):
        arr = [np.array(z, dtype=np.int32) for z in zones_pixels if len(z) >= 3]
        with self.zones_lock:
            self.zones = arr
        with open(ZONE_FILE, "w", encoding="utf-8") as f:
            json.dump([z.tolist() for z in arr], f)
        EVENTS.add("info", f"Saved {len(arr)} zone(s)")

    def save_normalized_zones(self, zones_norm, width, height):
        zones_px = []
        for zone in zones_norm:
            if len(zone) < 3:
                continue
            pts = []
            for p in zone:
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    continue
                x = int(max(0, min(width - 1, round(float(p[0]) * width))))
                y = int(max(0, min(height - 1, round(float(p[1]) * height))))
                pts.append([x, y])
            if len(pts) >= 3:
                zones_px.append(pts)
        self.save_zones(zones_px)

    def _point_in_zone(self, box):
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        probe_points = [
            (cx, int(y2)),  # foot point
            (cx, cy),       # body center
            (int(x1), int(y2)),
            (int(x2), int(y2)),
            (int(x1), cy),
            (int(x2), cy),
        ]
        with self.zones_lock:
            zones = list(self.zones)
        hits = []
        for idx, zone in enumerate(zones):
            point_hit = any(cv2.pointPolygonTest(zone, p, False) >= 0 for p in probe_points)
            zone_vertex_hit = any(x1 <= int(px) <= x2 and y1 <= int(py) <= y2 for px, py in zone)
            if point_hit or zone_vertex_hit:
                hits.append(idx)
        return hits

    def _ppe_worker(self):
        frame_idx = 0
        while True:
            try:
                frame = self.ppe_q.get(timeout=0.2)
            except Empty:
                continue
            frame_idx += 1
            if frame_idx % max(1, PPE_EVERY_N) != 0:
                continue

            cfg = self.settings.snapshot()
            dets = []
            try:
                with torch.inference_mode():
                    result = self.ppe.predict(
                        frame,
                        imgsz=cfg["img_size"],
                        conf=PPE_CONF,
                        device=DEVICE,
                        half=(DEVICE.startswith("cuda")),
                        verbose=False,
                    )[0]
                boxes = result.boxes
                if boxes is not None:
                    for b in boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        conf = float(b.conf[0])
                        cls = int(b.cls[0])
                        label = self.PPE_LABELS.get(cls, f"Class_{cls}")
                        bad = label.startswith("NO-")
                        dets.append({"box": [x1, y1, x2, y2], "label": label, "conf": conf, "bad": bad})
                        if bad:
                            self._log_throttled(f"ppe:{label}", "warning", f"PPE violation: {label} ({conf:.2f})")
                with self.result_lock:
                    self.ppe_result = dets
            except Exception as ex:
                EVENTS.add("error", f"PPE worker error: {ex}")
                continue

            self._ppe_count += 1
            if self._ppe_count >= 10:
                now = time.time()
                dt = now - self._ppe_t0
                self.ppe_infer_fps = self._ppe_count / dt if dt > 0 else 0.0
                self._ppe_t0 = now
                self._ppe_count = 0

    def _intrusion_worker(self):
        frame_idx = 0
        while True:
            try:
                frame = self.intrusion_q.get(timeout=0.2)
            except Empty:
                continue
            frame_idx += 1
            if frame_idx % max(1, INTRUSION_EVERY_N) != 0:
                continue

            cfg = self.settings.snapshot()
            dets = []
            try:
                with torch.inference_mode():
                    result = self.intrusion.predict(
                        frame,
                        imgsz=cfg["img_size"],
                        conf=INTRUSION_CONF,
                        classes=[0],
                        device=DEVICE,
                        half=(DEVICE.startswith("cuda")),
                        verbose=False,
                    )[0]
                boxes_obj = result.boxes
                if boxes_obj is not None and hasattr(boxes_obj, "xyxy"):
                    boxes = [tuple(map(int, xy.tolist())) for xy in boxes_obj.xyxy]
                    id_map = self.tracker.update(boxes)
                    current_intrusions = set()
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box
                        tid = id_map.get(i, -1)
                        hit = self._point_in_zone(box)
                        dets.append({"box": [x1, y1, x2, y2], "id": tid, "hit": hit})
                        for z in hit:
                            pair = (tid, z)
                            current_intrusions.add(pair)
                            if pair not in self.active_intrusions:
                                EVENTS.add("warning", f"INTRUSION ENTER: Person ID {tid} entered Zone {z + 1}")
                            else:
                                self._log_throttled(f"intrusion-active:{tid}:{z}", "warning", f"INTRUSION ACTIVE: Person ID {tid} in Zone {z + 1}")
                    for tid, z in self.active_intrusions - current_intrusions:
                        EVENTS.add("info", f"INTRUSION EXIT: Person ID {tid} left Zone {z + 1}")
                    self.active_intrusions = current_intrusions
                elif self.active_intrusions:
                    for tid, z in self.active_intrusions:
                        EVENTS.add("info", f"INTRUSION EXIT: Person ID {tid} left Zone {z + 1}")
                    self.active_intrusions = set()
                with self.result_lock:
                    self.intrusion_result = dets
            except Exception as ex:
                EVENTS.add("error", f"Intrusion worker error: {ex}")
                continue

            self._intr_count += 1
            if self._intr_count >= 10:
                now = time.time()
                dt = now - self._intr_t0
                self.intrusion_infer_fps = self._intr_count / dt if dt > 0 else 0.0
                self._intr_t0 = now
                self._intr_count = 0

    def _draw_ppe_overlay(self, frame, dets):
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            color = (0, 0, 255) if d["bad"] else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{d['label']} {d['conf']:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _draw_intrusion_overlay(self, frame, dets):
        with self.zones_lock:
            zones = list(self.zones)
        for i, zone in enumerate(zones):
            zone_active = any(i in d["hit"] for d in dets)
            zone_color = (0, 0, 255) if zone_active else (0, 255, 0)
            cv2.polylines(frame, [zone], True, zone_color, 3 if zone_active else 2)
            c = np.mean(zone, axis=0).astype(int)
            cv2.putText(frame, f"Zone {i + 1}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            breached = len(d["hit"]) > 0
            color = (0, 0, 255) if breached else (255, 120, 0)
            thickness = 4 if breached else 2
            label = f"INTRUDER ID {d['id']} Z{','.join(str(z + 1) for z in d['hit'])}" if breached else f"ID {d['id']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            if breached:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
                foot = (int((x1 + x2) / 2), int(y2))
                cv2.circle(frame, foot, 8, (0, 0, 255), -1)
                cv2.circle(frame, foot, 13, (255, 255, 255), 2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65 if breached else 0.5, 2)[0]
            label_y = max(24, y1 - 10)
            cv2.rectangle(frame, (x1, label_y - text_size[1] - 8), (x1 + text_size[0] + 10, label_y + 5), color, -1)
            cv2.putText(frame, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65 if breached else 0.5, (255, 255, 255), 2)

    def _render_worker(self):
        while True:
            cfg = self.settings.snapshot()
            budget = 1.0 / max(1.0, cfg["target_fps"])
            t0 = time.time()
            ok, frame = grabber.latest()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            with self.result_lock:
                ppe = list(self.ppe_result)
                intrusion = list(self.intrusion_result)

            mode = self.get_mode()
            out = frame.copy()
            if mode == "ppe":
                self._draw_ppe_overlay(out, ppe)
            else:
                self._draw_intrusion_overlay(out, intrusion)

            self._render_count += 1
            if self._render_count >= 10:
                now = time.time()
                dt = now - self._render_t0
                self.last_fps = self._render_count / dt if dt > 0 else 0.0
                self._render_t0 = now
                self._render_count = 0

            cv2.putText(out, f"Render FPS: {self.last_fps:.1f}", (12, out.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(
                out,
                f"PPE Infer: {self.ppe_infer_fps:.1f} | Intr Infer: {self.intrusion_infer_fps:.1f}",
                (12, out.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 220, 0),
                2,
            )
            cv2.putText(out, f"Mode: {mode.upper()} (both models active)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 220, 255), 2)

            ok_jpg, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), cfg["jpeg_quality"]])
            if ok_jpg:
                with self.out_lock:
                    self.last_jpeg = buf.tobytes()

            dt = time.time() - t0
            if dt < budget:
                time.sleep(budget - dt)

    def latest_jpeg(self):
        with self.out_lock:
            return self.last_jpeg

optimize_runtime()
grabber = FrameGrabber(STREAM_URL).start()
engine = DetectorEngine()
app = Flask(__name__)


def fanout_frames():
    while True:
        ok, frame = grabber.latest()
        if ok and frame is not None:
            engine.submit_frame(frame)
        time.sleep(0.001)


threading.Thread(target=fanout_frames, daemon=True).start()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    def gen():
        while True:
            jpg = engine.latest_jpeg()
            if jpg is None:
                time.sleep(0.03)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def status():
    cfg = engine.settings.snapshot()
    return jsonify(
        {
            "mode": engine.get_mode(),
            "fps": round(engine.last_fps, 2),
            "ppe_infer_fps": round(engine.ppe_infer_fps, 2),
            "intrusion_infer_fps": round(engine.intrusion_infer_fps, 2),
            "device": DEVICE,
            "target_fps": cfg["target_fps"],
            "img_size": cfg["img_size"],
            "jpeg_quality": cfg["jpeg_quality"],
            "zones": len(engine.get_zones_pixels()),
            "stream_url": STREAM_URL,
            "stream": grabber.status(),
            "both_models_active": True,
            "app_version": APP_VERSION,
            "active_intrusions": len(engine.active_intrusions),
        }
    )


@app.get("/api/version")
def version():
    return jsonify({"version": APP_VERSION})


@app.post("/api/mode")
def set_mode():
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode", "").strip().lower()
    if not engine.set_mode(mode):
        return jsonify({"ok": False, "error": "Mode must be 'ppe' or 'intrusion'"}), 400
    return jsonify({"ok": True, "mode": engine.get_mode(), "both_models_active": True})


@app.get("/api/logs")
def logs():
    limit = int(request.args.get("limit", "120"))
    mode = request.args.get("mode", engine.get_mode()).strip().lower()
    return jsonify({"items": EVENTS.list(limit=limit, mode=mode), "mode": mode})


@app.get("/api/logs/download")
def download_logs():
    mode = request.args.get("mode", engine.get_mode()).strip().lower()
    if mode not in ("ppe", "intrusion"):
        mode = None
    rows = list(reversed(EVENTS.all(mode=mode)))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["ts", "level", "category", "message"])
    writer.writeheader()
    writer.writerows(rows)
    label = mode or "all"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        stream.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=session_{label}_logs_{stamp}.csv"},
    )


@app.get("/api/zones")
def get_zones():
    shape = grabber.shape()
    if not shape:
        return jsonify({"ok": True, "zones": [], "normalized": True})
    w, h = shape
    return jsonify({"ok": True, "zones": engine.get_zones_normalized(w, h), "normalized": True})


@app.post("/api/zones")
def set_zones():
    payload = request.get_json(silent=True) or {}
    zones = payload.get("zones", [])
    normalized = bool(payload.get("normalized", False))
    if not isinstance(zones, list):
        return jsonify({"ok": False, "error": "zones must be a list"}), 400
    try:
        if not zones:
            engine.save_zones([])
        elif normalized:
            shape = grabber.shape()
            if not shape:
                return jsonify({"ok": False, "error": "No frame available yet"}), 400
            w, h = shape
            engine.save_normalized_zones(zones, w, h)
        else:
            engine.save_zones(zones)
        return jsonify({"ok": True, "zones": len(engine.get_zones_pixels())})
    except Exception as ex:
        return jsonify({"ok": False, "error": str(ex)}), 400


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    EVENTS.add("info", f"Server starting on http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)
