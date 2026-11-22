"""Melatih RF baseline dari fitur TRAIN dan mengevaluasi VALIDATION saja.

Hanya jalankan jika cache output_threshold03 sudah tersedia; tidak mengakses TEST.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

FEATURE_VERSION = "BISINDO-GEOM128-v1.0"
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(
    os.environ.get(
        "BISINDO_FEATURE_DIR",
        str(ROOT / "data" / "processed" / "umn" / "threshold03")
    )
).expanduser().resolve()
RESULT_DIR = DATA_DIR / "rf_baseline"


def load_split(split: str):
    path = DATA_DIR / f"features_{split}_{FEATURE_VERSION}.npz"
    with np.load(path, allow_pickle=False) as z:
        X, y = z["X"], z["y"]
    if X.ndim != 2 or X.shape[1] != 128 or len(y) != len(X) or not np.isfinite(X).all():
        raise ValueError(f"Fitur {split} invalid: {X.shape}")
    return X, y


def load_summary(split: str):
    return json.loads((DATA_DIR / f"summary_{split}_{FEATURE_VERSION}.json").read_text(encoding="utf-8"))


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    train_sum, val_sum = load_summary("train"), load_summary("validation")
    assert len(X_train) == train_sum["extracted"]
    assert len(X_val) == val_sum["extracted"]
    assert len(np.unique(y_train)) == 26
    assert set(np.unique(y_val)) == set(np.unique(y_train))
    from bisindo_preproc.pipeline import (MAX_SIDE, NUM_HANDS,
        MIN_DETECTION_CONFIDENCE, MIN_PRESENCE_CONFIDENCE)
    assert MAX_SIDE == 640 and NUM_HANDS == 2
    assert MIN_DETECTION_CONFIDENCE == MIN_PRESENCE_CONFIDENCE == 0.3
    model = RandomForestClassifier(n_estimators=300, random_state=42,
                                   n_jobs=-1, class_weight="balanced_subsample")
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    correct = int(np.count_nonzero(pred == y_val))
    total = int(val_sum["total"])
    labels = list(model.classes_)
    report = classification_report(y_val, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_val, pred, labels=labels)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULT_DIR / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, cm):
            writer.writerow([label, *row.tolist()])
    metrics = {
        "feature_version": FEATURE_VERSION,
        "model": "RandomForestClassifier",
        "n_estimators": 300,
        "random_state": 42,
        "detector_config": {
            "max_side": MAX_SIDE,
            "num_hands": NUM_HANDS,
            "min_detection_confidence": MIN_DETECTION_CONFIDENCE,
            "min_presence_confidence": MIN_PRESENCE_CONFIDENCE,
        },
        "train_total": int(train_sum["total"]),
        "train_extracted": len(X_train),
        "validation_total": total,
        "validation_extracted": len(X_val),
        "accuracy_detected": float(accuracy_score(y_val, pred)),
        "macro_precision": float(precision_score(y_val, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_val, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        "detection_rate": float(len(X_val) / total),
        "end_to_end_accuracy": float(correct / total),
        "correct_predictions": correct,
        "classes": [str(x) for x in labels],
    }
    joblib.dump(model, RESULT_DIR / "random_forest.joblib")
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("accuracy_detected", "macro_f1",
        "detection_rate", "end_to_end_accuracy", "correct_predictions")}, indent=2))
    print("Hasil di", RESULT_DIR)


if __name__ == "__main__":
    main()
