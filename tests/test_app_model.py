import json
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
import joblib
from bisindo_app.model import load_rf_model, predict_one
from bisindo_preproc.pipeline import FEATURE_VERSION


def _bundle(tmp_path, threshold=.3):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(52, 128)).astype(np.float32)
    y = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") * 2)
    rf = RandomForestClassifier(n_estimators=4, random_state=1).fit(X, y)
    path = tmp_path / "random_forest.joblib"
    joblib.dump(rf, path)
    metrics = {"feature_version": FEATURE_VERSION,
               "detector_config": {"max_side": 640, "num_hands": 2,
                                  "min_detection_confidence": threshold,
                                  "min_presence_confidence": threshold}}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    return path


def test_bundle_load_and_predict(tmp_path):
    path = _bundle(tmp_path)
    rf, _ = load_rf_model(path)
    label, score = predict_one(rf, np.zeros(128, dtype=np.float32))
    assert label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert 0 <= score <= 1


def test_bundle_rejects_wrong_threshold(tmp_path):
    path = _bundle(tmp_path, threshold=.5)
    with pytest.raises(ValueError, match="Konfigurasi"):
        load_rf_model(path)


def test_bundle_requires_metadata(tmp_path):
    path = tmp_path / "random_forest.joblib"
    path.write_bytes(b"fake")
    with pytest.raises(FileNotFoundError, match="metrics.json"):
        load_rf_model(path)


def test_predict_rejects_bad_features(tmp_path):
    rf, _ = load_rf_model(_bundle(tmp_path))
    with pytest.raises(ValueError):
        predict_one(rf, np.ones(127))
