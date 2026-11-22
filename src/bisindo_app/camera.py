"""Capture + MediaPipe + Random Forest di thread terpisah dari GUI."""
from __future__ import annotations

from dataclasses import dataclass
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from bisindo_preproc.pipeline import HandExtractor, prepare_webcam_bgr
from .core import TemporalSmoother
from .model import load_rf_model, predict_one

# MediaPipe Hands 21-joint topology. Tidak memakai API drawings yang dapat berubah.
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4),
               (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


@dataclass
class CameraSnapshot:
    rgb: np.ndarray | None = None
    raw_letter: str | None = None
    score: float = 0.0
    stable_letter: str | None = None
    hands: int = 0
    status: str = "Kamera belum aktif"
    fps: float = 0.0
    index: int = 0


def draw_landmarks(rgb: np.ndarray, hands: list) -> np.ndarray:
    out = rgb.copy()
    height, width = out.shape[:2]
    for hand in hands:
        pts = [
            (min(max(round(pt.x * width), 0), width - 1),
             min(max(round(pt.y * height), 0), height - 1))
            for pt in hand
        ]
        for a, b in CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (75, 220, 150), 2, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(out, (x, y), 3, (250, 210, 75), -1, cv2.LINE_AA)
    return out


class CameraWorker:
    def __init__(self, model_path: str | Path, landmarker_path: str | Path, camera_index: int = 0):
        self.model_path = Path(model_path)
        self.landmarker_path = Path(landmarker_path)
        self.camera_index = camera_index
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._snapshot = CameraSnapshot()
        self._counter = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="bisindo-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def snapshot(self) -> CameraSnapshot:
        with self._lock:
            return self._snapshot

    def _publish(self, *, rgb=None, raw_letter=None, score=0.0,
                 stable_letter=None, hands=0, status="", fps=0.0):
        with self._lock:
            self._counter += 1
            self._snapshot = CameraSnapshot(rgb, raw_letter, score,
                                            stable_letter, hands, status, fps, self._counter)

    def _run(self):
        cap = None
        try:
            self._publish(status="Memuat model…")
            model, _meta = load_rf_model(self.model_path)
            if not self.landmarker_path.is_file():
                raise FileNotFoundError(f"Model MediaPipe tidak ditemukan: {self.landmarker_path}")
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.camera_index, backend)
            if not cap.isOpened():
                raise RuntimeError(f"Kamera index={self.camera_index} tidak dapat dibuka")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            smoother = TemporalSmoother()
            self._publish(status="Kamera aktif")
            with HandExtractor(self.landmarker_path) as extractor:
                last = time.monotonic()
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("Gagal membaca frame webcam")
                    rgb = prepare_webcam_bgr(frame)  # sama seperti training, tanpa mirror
                    vector, hands, landmarks = extractor.extract_with_landmarks(rgb)
                    if vector is None:
                        raw, score, stable = None, 0.0, smoother.update(None)
                        status = "Tidak ada tangan terdeteksi"
                    else:
                        raw, score = predict_one(model, vector)
                        stable = smoother.update(raw, score)
                        status = (
                            f"Terdeteksi {hands} tangan — prediksi stabil"
                            if stable else f"Terdeteksi {hands} tangan — menstabilkan prediksi"
                        )
                    # Mirror hanya pada preview SESUDAH inference, bukan input model.
                    preview = np.ascontiguousarray(np.flip(draw_landmarks(rgb, landmarks), axis=1))
                    now = time.monotonic()
                    fps = 1.0 / max(now - last, 1e-5)
                    last = now
                    self._publish(rgb=preview, raw_letter=raw, score=score,
                                  stable_letter=stable, hands=hands, status=status, fps=fps)
        except Exception as exc:
            self._publish(status=f"ERROR: {type(exc).__name__}: {exc}")
        finally:
            if cap is not None:
                cap.release()
