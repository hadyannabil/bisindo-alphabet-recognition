"""Model RF tepercaya dan pemeriksaan kompatibilitas hasil training."""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np

from bisindo_preproc import pipeline

EXPECTED_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_rf_model(model_path: str | Path):
    """Jangan gunakan joblib tak tepercaya: pickle bisa menjalankan kode."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model RF tidak ditemukan: {path}")
    meta_path = path.parent / "metrics.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Metadata training tidak ditemukan: {meta_path}. "
            "Salin metrics.json bersama random_forest.joblib."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "max_side": pipeline.MAX_SIDE,
        "num_hands": pipeline.NUM_HANDS,
        "min_detection_confidence": pipeline.MIN_DETECTION_CONFIDENCE,
        "min_presence_confidence": pipeline.MIN_PRESENCE_CONFIDENCE,
    }
    if meta.get("feature_version") != pipeline.FEATURE_VERSION:
        raise ValueError("Feature version model tidak cocok dengan preprocessing")
    for key, value in expected.items():
        if meta.get("detector_config", {}).get(key) != value:
            raise ValueError(
                f"Konfigurasi {key} model={meta.get('detector_config', {}).get(key)} "
                f"tidak cocok pipeline={value}"
            )
    model = joblib.load(path)
    if getattr(model, "n_features_in_", None) != pipeline.N_FEATURES:
        raise ValueError("Jumlah fitur model tidak cocok (wajib 128)")
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model harus mendukung predict_proba")
    if set(map(str, model.classes_)) != EXPECTED_LABELS:
        raise ValueError("Model harus berisi kelas A–Z lengkap")
    return model, meta


def predict_one(model, features: np.ndarray) -> tuple[str, float]:
    x = np.asarray(features, dtype=np.float32).reshape(1, -1)
    if x.shape != (1, pipeline.N_FEATURES) or not np.isfinite(x).all():
        raise ValueError("Vektor webcam tidak valid")
    prob = model.predict_proba(x)[0]
    index = int(np.argmax(prob))
    return str(model.classes_[index]), float(prob[index])
