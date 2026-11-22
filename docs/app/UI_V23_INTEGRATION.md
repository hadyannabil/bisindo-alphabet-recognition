# Integrasi BISINDO UI v2.3

Sumber: arsip lokal `BISINDO_App.zip` yang diberikan untuk integrasi repository.

- `ui.py` dan `core.py` dipertahankan byte-for-byte dari arsip lokal.
- Prediksi Kata Cerdas hanya placeholder visual dan **bukan fitur aktif**.
- `camera.py`, `model.py`, dan `bisindo_preproc/pipeline.py` yang sudah disiapkan di Codespaces tidak ditimpa oleh paket ini.
- `smoke.py` dan `tests/test_app_core.py` diperbaiki karena versi lama masih memanggil `WordPredictor`, `distance`, `choose`, dan `current_token` yang sudah tidak ada.
- `run_demo_windows.bat` dari arsip lama tidak disertakan: UI v2.3 tidak menerima argumen `--demo`.
- Path manifest pada `tests/test_pipeline.py` dan default `--split-dir` di `extract_dataset.py` diperbarui ke `data/manifests/umn_baseline/`; logika ekstraksi fitur tidak diubah.
- `random_forest.joblib` 97 MB dan `hand_landmarker.task` sengaja tidak disertakan. Model RF harus diambil dari artefak baseline yang telah diverifikasi; file model jangan di-commit langsung.
- README asal arsip lokal tidak disertakan karena masih mengklaim fitur prediksi kata aktif dan mode demo.

## Menjalankan aplikasi di Windows

Di root repository, instal dependensi sesuai `requirements.txt`, pastikan `models/random_forest.joblib`, `models/metrics.json`, dan `models/hand_landmarker.task` tersedia, lalu jalankan `run_windows.bat` atau `PYTHONPATH=src python -m bisindo_app.ui`.

## Tes tanpa webcam

`PYTHONPATH=src python -m bisindo_app.smoke`

`PYTHONPATH=src python -m pytest -q tests/test_app_core.py tests/test_autoconfirm_gate.py`

Pengujian di Codespaces tidak membuktikan webcam Windows atau tampilan pada DPI/ukuran layar tertentu berfungsi: itu perlu tes langsung di Windows.
