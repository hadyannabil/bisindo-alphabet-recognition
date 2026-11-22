# BISINDO Alphabet Recognition

Aplikasi desktop untuk **mengenali alfabet Bahasa Isyarat Indonesia (BISINDO) A–Z melalui webcam** dan menyusunnya menjadi teks. Proyek ini menggunakan **MediaPipe Hand Landmarker** untuk mendeteksi titik tangan serta **Random Forest** untuk mengklasifikasikan huruf. Antarmuka dibangun dengan Python dan Tkinter.

Aplikasi dikembangkan sebagai fondasi pengenalan isyarat dan penyusunan teks. **Prediksi Kata Cerdas** merupakan tahap pengembangan berikutnya dan saat ini **belum aktif**.

## Fitur yang Tersedia

- Pengenalan 26 huruf (A–Z) dari kamera secara real-time, dengan dukungan deteksi satu atau dua tangan.
- Visualisasi titik landmark tangan dan informasi hasil prediksi.
- Stabilisasi prediksi sebelum huruf dikonfirmasi.
- Konfirmasi huruf secara manual melalui tombol **Konfirmasi Huruf** atau **Enter**.
- Opsi **Konfirmasi Otomatis** setelah gestur stabil selama sekitar 1,8 detik.
- **Teks Tersusun** dengan tombol **Hapus**, **Spasi**, dan **Bersihkan**.
- Empat kotak **Prediksi Kata Cerdas** sebagai *placeholder* UI (belum menghasilkan saran atau menerima klik).

## Struktur Proyek

```text
bisindo-alphabet-recognition/
├── assets/
│   └── camera_placeholder.jpg      # Ilustrasi saat kamera belum aktif
├── data/
│   ├── external/                   # Dataset asli (lokal, tidak masuk Git)
│   ├── manifests/umn_baseline/     # Manifest dataset
│   └── processed/                  # Cache fitur (lokal, tidak masuk Git)
├── docs/                           # Dokumentasi teknis
├── models/
│   ├── hand_landmarker.task        # Model deteksi landmark tangan
│   ├── metrics.json                # Metadata model aplikasi
│   └── random_forest.joblib        # Model klasifikasi (disediakan terpisah)
├── results/umn_baseline/           # Ringkasan evaluasi baseline
├── src/
│   ├── bisindo_app/
│   │   ├── ui.py                   # Antarmuka desktop
│   │   ├── camera.py               # Kamera dan inferensi real-time
│   │   ├── core.py                 # Stabilisasi, konfirmasi, penyusunan teks
│   │   ├── model.py                # Pemuatan model klasifikasi
│   │   └── smoke.py                # Pengujian fungsi dasar
│   └── bisindo_preproc/
│       ├── pipeline.py             # Preprocessing dan ekstraksi 128 fitur
│       └── extract_dataset.py      # Ekstraksi fitur dataset
├── tests/                          # Unit test
├── training/
│   └── train_rf_baseline.py        # Script training baseline
├── .gitignore
├── requirements.txt
├── run_windows.bat                  # Menjalankan aplikasi di Windows
└── README.md
```

> Dataset gambar asli, cache fitur, dan model `random_forest.joblib` tidak disertakan dalam Git. **Dataset tidak diperlukan untuk menjalankan aplikasi** jika model klasifikasi sudah tersedia.

## Menjalankan Aplikasi di Windows (Lokal)

### 1. Persiapan

Pastikan tersedia:

- **Python 3.12** (direkomendasikan) dan Git.
- Webcam yang dapat digunakan.
- File `models/random_forest.joblib` dari pemilik/pengelola proyek. File ini belum tersedia sebagai unduhan otomatis dari repository.
- File `models/hand_landmarker.task` dan `models/metrics.json` yang sudah ada dalam repository.

### 2. Unduh repository

Buka **PowerShell** atau terminal Windows:

```powershell
git clone https://github.com/hadyannabil/bisindo-alphabet-recognition.git
cd bisindo-alphabet-recognition
```

> Gunakan branch yang berisi versi aplikasi terbaru apabila perubahan belum digabungkan ke `main`.

### 3. Buat virtual environment dan instal dependensi

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Letakkan model klasifikasi

Simpan file yang diperoleh dari pengelola proyek di:

```text
models/random_forest.joblib
```

Pastikan nama file sesuai. Model ini harus cocok dengan pipeline fitur `BISINDO-GEOM128-v1.0` (**128 fitur, 26 kelas**). Jangan menggunakan model RF versi eksperimen lain tanpa pemeriksaan kompatibilitas.

### 5. Jalankan aplikasi

Dari folder utama repository:

```powershell
.\run_windows.bat
```

Atau jalankan secara manual:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m bisindo_app.ui
```

Jika ingin mencoba indeks kamera lain (misalnya kamera kedua):

```powershell
.\run_windows.bat --camera 1
```

Di aplikasi, aktifkan kamera, peragakan isyarat, lalu konfirmasi huruf ketika prediksi sudah stabil. Gunakan **Spasi**, **Hapus**, atau **Bersihkan** untuk mengatur teks hasil pengenalan.

### Pengujian (opsional)

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe -m bisindo_app.smoke
```

Unit test tidak menggantikan pengujian kamera dan tampilan langsung pada komputer pengguna.

## Pengembangan Berikutnya: Prediksi Kata Cerdas

**Bagian ini disiapkan untuk anggota tim yang mengembangkan Smart Word Prediction.** Fitur tersebut belum diimplementasikan: empat kotak saran di UI saat ini hanya menampilkan tanda `—` dan tidak dapat diklik.

### Tujuan

Memberikan hingga **empat saran kata bahasa Indonesia** berdasarkan huruf/kata yang sudah masuk ke **Teks Tersusun**, agar pengguna lebih cepat menyelesaikan kata tanpa harus mengisyaratkan setiap huruf.

Contoh alur yang diharapkan:

```text
Teks Tersusun : SAYA MAK
Saran kata    : MAKAN | MAKANAN | MAKNA | MAKIN
Pengguna klik : MAKAN
Hasil teks    : SAYA MAKAN
```

*Contoh tersebut hanya ilustrasi perilaku fitur, bukan hasil prediksi dari implementasi yang sudah ada.*

### Pembagian Tanggung Jawab

| Komponen | Tugas |
|---|---|
| `src/bisindo_app/core.py` | Menyimpan teks hasil konfirmasi melalui `TextComposer`; jangan mengganggu proses ini. |
| `src/bisindo_app/ui.py` | Mengambil teks tersusun dan menampilkan hingga empat saran di bagian **Prediksi Kata Cerdas**. |
| **Modul baru** `src/bisindo_app/word_prediction.py` | Menangani logika pemrosesan teks, pemilihan kandidat, dan pemeringkatan saran kata. |
| `tests/` | Menambahkan pengujian untuk prediksi kata dan integrasi ke teks tersusun. |

### Kontrak Integrasi yang Disarankan

Buat fungsi terpisah dengan antarmuka sederhana, misalnya:

```python
def suggest_words(formed_text: str, limit: int = 4) -> list[str]:
    """Kembalikan maksimal `limit` kandidat kata tanpa mengubah teks input."""
```

Ketentuan integrasi:

1. **Input berasal dari `TextComposer.text`**, yaitu teks yang sudah dikonfirmasi, **bukan** prediksi huruf mentah atau prediksi stabil sementara dari webcam.
2. Untuk melengkapi kata, gunakan **token terakhir yang sedang diketik**. Contoh: `saya mak` menggunakan `mak` sebagai prefiks. Pertahankan kata-kata sebelumnya.
3. UI memperbarui saran setelah teks berubah: **Konfirmasi Huruf** (manual/otomatis), **Hapus**, **Spasi**, atau **Bersihkan**. Titik pembaruan yang dapat digunakan ada di `BisindoApp._refresh_text()` pada `ui.py`.
4. Di `BisindoApp._build_words()`, empat `PillButton` yang sekarang masih *placeholder* perlu disimpan sebagai referensi, diberi label dari hasil prediksi, dan baru diaktifkan jika kandidat tersedia.
5. Saat saran dipilih, **ganti hanya kata parsial terakhir** dalam `TextComposer.text` dengan kandidat terpilih; jangan menimpa seluruh kalimat.
6. Ketika teks kosong, token terakhir kosong (misalnya tepat setelah Spasi), atau tidak ada kandidat, tampilkan keadaan kosong/nonaktif. Penanganan prediksi kata berikutnya setelah spasi dapat ditambahkan sebagai pengembangan terpisah.
7. Pisahkan logika prediksi kata dari MediaPipe, Random Forest, `CameraWorker`, dan pipeline fitur 128 dimensi. **Tidak perlu retraining model pengenal huruf** untuk menambahkan fitur ini.

Metode pencarian/pemeringkatan kandidat (misalnya kosakata, frekuensi kata, *prefix matching*, atau model bahasa) dapat dipilih dan diuji oleh pengembang fitur. Pastikan sumber kosakata, aturan normalisasi teks, dan keterbatasannya didokumentasikan.

### Kriteria Selesai

- Empat kotak saran dapat menampilkan **0–4 kata relevan**, tanpa duplikat.
- Saran berubah ketika pengguna menambah, menghapus, atau membersihkan teks.
- Pemilihan saran memperbarui **Teks Tersusun** dengan benar, termasuk untuk kalimat yang memiliki spasi.
- Saat tidak ada saran, UI tetap stabil dan tidak menampilkan kata lama.
- Fungsi pengenal huruf, konfirmasi, dan kontrol teks tetap berjalan; unit test lama tidak rusak.
- Ada unit test khusus untuk teks kosong, satu kata, beberapa kata, saran tidak ditemukan, serta pemilihan saran.

## Status

**Sudah tersedia:** pengenalan alfabet, stabilisasi prediksi, konfirmasi huruf, Teks Tersusun, dan UI desktop.

**Pengembangan berikutnya:** Prediksi Kata Cerdas (logika, integrasi empat kotak saran, dan pengujiannya).
