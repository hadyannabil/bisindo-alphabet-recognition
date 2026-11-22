"""Ekstraksi TRAIN/VALIDATION; TEST sengaja dikunci untuk evaluasi akhir.

Contoh:
  python -m bisindo_preproc.extract_dataset --split train --images-root /data/UMN_Originals \
    --split-dir split_frozen --model models/hand_landmarker.task --output output
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import __version__ as pillow_version

from .pipeline import FEATURE_VERSION, N_FEATURES, SPEC, HandExtractor, decode_rgb, spec_digest

SPLIT_FILES = {
    "train": "train_manifest.csv",
    "validation": "validation_manifest.csv",
    "test": "test_evaluation_eligible_700.csv",
}
EXPECTED = {"train": 3011, "validation": 1001, "test": 700}


def read_manifest(split_dir: Path, split: str) -> list[dict]:
    path = split_dir / SPLIT_FILES[split]
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED[split]:
        raise ValueError(f"Manifest {split}: {len(rows)} != {EXPECTED[split]}")
    paths = [r["path"] for r in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Path duplikat di manifest")
    if len(set(r["label"] for r in rows)) != 26:
        raise ValueError("Manifest harus mempunyai kelas A–Z lengkap")
    for r in rows:
        if len(r["label"]) != 1 or r["label"] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            raise ValueError("Label di luar A-Z")
        p = Path(r["path"])
        if p.is_absolute() or ".." in p.parts or p.parts[0] != r["label"]:
            raise ValueError(f"Path tidak aman/tidak cocok label: {r['path']}")
        if r.get("review_status", "KEEP") != "KEEP":
            raise ValueError("Ada data karantina di subset")
    return rows


def extract(args: argparse.Namespace) -> None:
    if args.split == "test" and not args.enable_final_test:
        raise SystemExit("TEST DIKUNCI: hanya dapat dibuka setelah model/pipeline final dengan --enable-final-test.")
    split_dir, image_root, output_dir = (Path(x) for x in (args.split_dir, args.images_root, args.output))
    if not image_root.is_dir():
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {image_root}")
    entries = read_manifest(split_dir, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{FEATURE_VERSION}"
    features: list[np.ndarray] = []
    labels: list[str] = []
    kept_paths: list[str] = []
    records: list[dict] = []
    stats = Counter()
    per_label = defaultdict(Counter)
    import mediapipe as mp
    with HandExtractor(args.model) as extractor:
        for index, item in enumerate(entries, 1):
            rel, label = item["path"], item["label"]
            path = image_root / rel
            status, count, shape, error = "unknown", 0, "", ""
            try:
                if not path.is_file():
                    raise FileNotFoundError(str(path))
                if args.verify_sha256:
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()
                    if actual.lower() != item["sha256"].lower():
                        raise ValueError("SHA-256 tidak cocok dengan manifest")
                rgb = decode_rgb(path)
                shape = f"{rgb.shape[1]}x{rgb.shape[0]}"
                vector, count = extractor.extract(rgb)
                if vector is None:
                    status = "no_hand"
                else:
                    if vector.shape != (N_FEATURES,) or not np.isfinite(vector).all():
                        raise ValueError("Vektor fitur rusak")
                    features.append(vector)
                    labels.append(label)
                    kept_paths.append(rel)
                    status = f"detected_{count}_hand"
            except Exception as exc:
                status, error = "error", f"{type(exc).__name__}: {exc}"[:240]
            stats[status] += 1
            per_label[label][status] += 1
            records.append({"path": rel, "label": label, "status": status, "hands_detected": count,
                            "processed_size": shape, "error": error})
            if args.progress_every and index % args.progress_every == 0:
                print(f"{args.split}: {index}/{len(entries)} | {dict(stats)}", flush=True)
        meta = {
            "version": FEATURE_VERSION,
            "spec_digest_sha256": spec_digest(),
            "spec": SPEC,
            "split": args.split,
            "split_file": str((split_dir / SPLIT_FILES[args.split]).resolve()),
            "split_file_sha256": hashlib.sha256((split_dir / SPLIT_FILES[args.split]).read_bytes()).hexdigest(),
            "landmarker_model_sha256": extractor.model_sha256,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "opencv": cv2.__version__, "mediapipe": mp.__version__, "pillow": pillow_version},
            "total": len(entries), "extracted": len(features), "status_counts": dict(stats),
            "per_label": {k: dict(v) for k, v in sorted(per_label.items())},
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "notice": "This is feature extraction, NOT classification accuracy or final testing",
        }
        if args.split == "test":
            meta["test_status"] = "FINAL_TEST_UNLOCKED_BY_OPERATOR"
        # Simpan hanya data dengan landmark; no_hand dan error tetap tercatat pada CSV.
        X = np.stack(features).astype(np.float32) if features else np.empty((0, N_FEATURES), dtype=np.float32)
        np.savez_compressed(output_dir / f"features_{stem}.npz", X=X, y=np.asarray(labels, dtype="<U1"),
                            paths=np.asarray(kept_paths, dtype=f"<U{max(1, max(map(len, kept_paths), default=1))}"))
        with (output_dir / f"records_{stem}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "status", "hands_detected", "processed_size", "error"])
            writer.writeheader()
            writer.writerows(records)
        (output_dir / f"summary_{stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"split": args.split, "total": len(entries), "features": len(features),
                          "status_counts": dict(stats), "output": str(output_dir)}, ensure_ascii=False, indent=2))
        if stats["error"]:
            raise SystemExit(f"ERROR: {stats['error']} gambar bermasalah. Periksa records; jangan training sebelum diperbaiki.")


def main() -> None:
    p = argparse.ArgumentParser(description="UMN preprocessing+landmark+128 fitur (tanpa training)")
    p.add_argument("--split", choices=list(SPLIT_FILES), required=True)
    p.add_argument("--images-root", required=True, help="Root yang berisi folder A/..Z/ gambar asli")
    p.add_argument("--split-dir", default="data/manifests/umn_baseline")
    p.add_argument("--model", default="models/hand_landmarker.task")
    p.add_argument("--output", default="output")
    p.add_argument("--verify-sha256", action="store_true", help="Baca & hash ulang bytes file pada run pertama")
    p.add_argument("--enable-final-test", action="store_true", help="Buka 700 TEST hanya untuk evaluasi final")
    p.add_argument("--progress-every", type=int, default=100)
    extract(p.parse_args())


if __name__ == "__main__":
    main()
