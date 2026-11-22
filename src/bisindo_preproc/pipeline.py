"""Satu sumber kebenaran preprocessing training dan webcam.

Tidak memuat, melatih, atau memilih classifier. Versi fitur BERBEDA dari model lama
hingga kesetaraan numerik dibuktikan secara eksplisit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    # Beberapa foto UMN memakai konten HEIF walaupun berekstensi JPG.
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

FEATURE_VERSION = "BISINDO-GEOM128-v1.0"
MAX_SIDE = 640
NUM_HANDS = 2
NUM_LANDMARKS = 21
N_FEATURES = 128
MIN_DETECTION_CONFIDENCE = 0.3
MIN_PRESENCE_CONFIDENCE = 0.3

SPEC = {
    "feature_version": FEATURE_VERSION,
    "image": {"orientation": "EXIF transpose", "colorspace": "RGB8", "resize": "preserve_aspect_to_max_side", "max_side": MAX_SIDE,
              "crop": False, "mirror": False, "clahe": False, "smoothing": False, "augmentation": False},
    "mediapipe": {"running_mode": "IMAGE", "num_hands": NUM_HANDS,
                  "min_hand_detection_confidence": MIN_DETECTION_CONFIDENCE,
                  "min_hand_presence_confidence": MIN_PRESENCE_CONFIDENCE},
    "features": {"dimensions": N_FEATURES, "order": "wrist_x_ascending",
                 "coordinates": "x_px,y_px,z_scaled_from_wrist",
                 "center": "mean_of_global_xy_bbox_minmax",
                 "scale": "max(global_bbox_width,global_bbox_height,1px)",
                 "missing": "zero_fill_unoccupied_slot",
                 "presence_mask": "last_two_features",
                 "no_hand": "skip_classification_and_log"},
}


def spec_digest() -> str:
    return hashlib.sha256(json.dumps(SPEC, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_rgb(rgb: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """Input RGB uint8 -> RGB uint8 contiguous; hanya downscale, tidak crop/flip."""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError("Input harus HxWx3, uint8, format RGB")
    h, w = a.shape[:2]
    if h < 1 or w < 1 or max_side < 1:
        raise ValueError("Dimensi citra tidak valid")
    if max(h, w) > max_side:
        ratio = max_side / max(h, w)
        w2, h2 = max(1, round(w * ratio)), max(1, round(h * ratio))
        a = cv2.resize(a, (w2, h2), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(a)


def decode_rgb(path: str | Path) -> np.ndarray:
    """Membaca piksel sesuai format aktual, bukan berdasarkan ekstensi file."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return prepare_rgb(rgb)


def prepare_webcam_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Webcam OpenCV tanpa mirror; mirror khusus UI dilakukan SETELAH inference."""
    frame = np.asarray(frame_bgr)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame webcam harus BGR uint8 HxWx3")
    return prepare_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def landmarks_to_features(hands: Sequence[Any], image_shape: Sequence[int]) -> np.ndarray | None:
    """2 tangan x 21 titik x xyz + 2 masker; jika 0 tangan -> None.

    x/y dibawa ke koordinat piksel agar rasio aspek tidak mendistorsi bentuk.
    z MediaPipe adalah kedalaman relatif (bukan meter), dinolkan per-pergelangan.
    Seluruh tangan memakai SATU kotak gabungan untuk mempertahankan konfigurasi
    posisi antartangan; slot diurutkan menurut posisi x pergelangan.
    """
    if len(hands) == 0:
        return None
    if len(hands) > NUM_HANDS:
        raise ValueError("Maksimal dua tangan")
    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError("Ukuran citra tidak valid")

    points = []
    for hand in hands:
        if len(hand) != NUM_LANDMARKS:
            raise ValueError("Setiap tangan harus punya 21 landmark")
        pts = np.array([[float(p.x) * w, float(p.y) * h, float(p.z) * w] for p in hand], dtype=np.float64)
        if not np.all(np.isfinite(pts)):
            raise ValueError("Koordinat landmark tidak finite")
        points.append(pts)

    # Sort stabil untuk input IMAGE yang tidak memiliki identitas tracked-hand.
    points.sort(key=lambda p: (float(p[0, 0]), float(p[0, 1]), float(p[:, 0].mean()), float(p[:, 1].mean()), tuple(p[:, 0].tolist())))
    both_xy = np.concatenate([p[:, :2] for p in points], axis=0)
    low, high = both_xy.min(axis=0), both_xy.max(axis=0)
    center = (low + high) / 2
    scale = max(float((high - low).max()), 1.0)
    out = np.zeros((NUM_HANDS, NUM_LANDMARKS, 3), dtype=np.float32)
    mask = np.zeros(NUM_HANDS, dtype=np.float32)
    for i, p in enumerate(points):
        out[i, :, :2] = ((p[:, :2] - center) / scale).astype(np.float32)
        out[i, :, 2] = ((p[:, 2] - p[0, 2]) / scale).astype(np.float32)
        mask[i] = 1.0
    v = np.concatenate([out.reshape(-1), mask])
    assert v.shape == (N_FEATURES,)
    return v


class HandExtractor:
    """Pembungkus MediaPipe Tasks IMAGE, cocok untuk dataset dan webcam baseline."""
    def __init__(self, model_path: str | Path):
        import mediapipe as mp  # lazy: unit test geometri tanpa instalasi MediaPipe
        self.mp = mp
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(opts)

    def extract_with_landmarks(self, rgb: np.ndarray) -> tuple[np.ndarray | None, int, list]:
        """Satu kali MediaPipe detect; kembalikan titik untuk overlay UI.

        Perhitungan fitur tetap identik dengan ekstraksi dataset.
        """
        rgb = prepare_rgb(rgb)
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)
        count = len(result.hand_landmarks)
        return landmarks_to_features(result.hand_landmarks, rgb.shape), count, result.hand_landmarks

    def extract(self, rgb: np.ndarray) -> tuple[np.ndarray | None, int]:
        features, count, _landmarks = self.extract_with_landmarks(rgb)
        return features, count

    def close(self) -> None:
        self.detector.close()

    def __enter__(self) -> "HandExtractor":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
