"""Tes matematika fitur tanpa membutuhkan MediaPipe dan gambar UMN."""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from bisindo_preproc.pipeline import (
    FEATURE_VERSION, N_FEATURES, decode_rgb, landmarks_to_features,
    prepare_rgb, prepare_webcam_bgr, spec_digest
)
from bisindo_preproc.extract_dataset import read_manifest


def hand(wx: float, wy: float, w: float = 0.1, h: float = 0.1):
    return [SimpleNamespace(x=wx + (i % 5) * w / 4,
                            y=wy + (i // 5) * h / 4,
                            z=(i - 10) * 0.002) for i in range(21)]


def test_single_hand_mask_and_padding():
    a = landmarks_to_features([hand(0.1, 0.2)], (640, 640, 3))
    assert a.shape == (128,) and a.dtype == np.float32
    assert np.array_equal(a[-2:], [1, 0])
    assert np.count_nonzero(a[63:126]) == 0
    assert np.isfinite(a).all()


def test_no_hand_is_abstention():
    assert landmarks_to_features([], (640, 640, 3)) is None


def test_order_does_not_depend_on_mediapipe_array_order():
    a, b = hand(0.6, 0.2), hand(0.1, 0.25)
    x = landmarks_to_features([a, b], (480, 640, 3))
    y = landmarks_to_features([b, a], (480, 640, 3))
    np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(x[-2:], [1, 1])


def test_global_translation_and_uniform_scaling_invariance():
    base = [hand(0.1, 0.2), hand(0.45, 0.3)]
    translated = [[SimpleNamespace(x=p.x + 0.1, y=p.y + 0.07, z=p.z) for p in one] for one in base]
    scaled = [[SimpleNamespace(x=p.x * 0.7, y=p.y * 0.7, z=p.z * 0.7) for p in one] for one in base]
    x = landmarks_to_features(base, (640, 640, 3))
    np.testing.assert_allclose(x, landmarks_to_features(translated, (640, 640, 3)), atol=2e-6)
    np.testing.assert_allclose(x, landmarks_to_features(scaled, (640, 640, 3)), atol=2e-6)


def test_relative_two_hand_position_is_preserved():
    a = [hand(0.1, 0.3), hand(0.4, 0.3)]
    b = [hand(0.1, 0.3), hand(0.6, 0.3)]
    assert not np.allclose(landmarks_to_features(a, (480, 640)), landmarks_to_features(b, (480, 640)))


def test_invalid_landmarks_are_rejected():
    with pytest.raises(ValueError):
        landmarks_to_features([hand(0.1, 0.1)[:-1]], (640, 640))


def test_webcam_bgr_is_converted_without_flip():
    rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    np.testing.assert_array_equal(prepare_webcam_bgr(bgr), rgb)


def test_resize_does_not_stretch_or_upscale():
    a = np.zeros((800, 400, 3), dtype=np.uint8)
    assert prepare_rgb(a).shape == (640, 320, 3)
    assert prepare_rgb(np.zeros((40, 80, 3), dtype=np.uint8)).shape == (40, 80, 3)


def test_decode_source_png_to_rgb(tmp_path):
    path = tmp_path / "image.jpg"
    Image.fromarray(np.zeros((10, 20, 3), dtype=np.uint8)).save(path, format="PNG")
    assert decode_rgb(path).shape == (10, 20, 3)


def test_spec_digest_stable():
    assert len(spec_digest()) == 64 and FEATURE_VERSION == "BISINDO-GEOM128-v1.0"


def test_frozen_train_validation_test_manifest_counts():
    root = Path(__file__).resolve().parents[1] / "data" / "manifests" / "umn_baseline"
    for s, n in [("train", 3011), ("validation", 1001), ("test", 700)]:
        rows = read_manifest(root, s)
        assert len(rows) == n
        assert all(row["label"] == row["path"].split("/", 1)[0] for row in rows)


def test_exif_rotation_is_applied(tmp_path):
    raw = Image.new("RGB", (10, 20), (100, 120, 140))
    exif = Image.Exif()
    exif[274] = 6  # Rotation 90 degrees clockwise (EXIF)
    path = tmp_path / "rotated.jpg"
    raw.save(path, exif=exif)
    assert decode_rgb(path).shape == (10, 20, 3)


def test_test_set_is_locked_without_explicit_final_permission(tmp_path):
    from argparse import Namespace
    from bisindo_preproc.extract_dataset import extract
    with pytest.raises(SystemExit, match="TEST DIKUNCI"):
        extract(Namespace(split="test", enable_final_test=False, split_dir="split_frozen",
                          images_root=str(tmp_path), output=str(tmp_path), model="missing.task",
                          verify_sha256=False, progress_every=0))
