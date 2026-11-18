# BISINDO Alphabet Recognition with Smart Word Prediction

> **Proyek akhir Computer Vision** — prototipe pengenalan alfabet Bahasa Isyarat Indonesia (BISINDO) A–Z melalui webcam, dengan target penyusunan huruf menjadi kata dan prediksi kata berbahasa Indonesia.

**Status proyek (November 2026): dalam pengembangan.** Deteksi landmark tangan, training model, evaluasi, dan pengenalan huruf melalui webcam lokal sudah berhasil dijalankan. **Word Formation, Text Correction, dan Smart Word Prediction masih menjadi target pengembangan**, belum fitur final yang berfungsi.

## 1. Tujuan dan ruang lingkup

Proyek ini bertujuan membuat aplikasi yang menerima gestur alfabet BISINDO melalui webcam dan mengubahnya menjadi huruf. Nantinya pengguna dapat mengonfirmasi huruf, merangkainya menjadi kata, mengoreksi teks, dan memperoleh saran kata bahasa Indonesia.

Fokus proyek adalah **alfabet/fingerspelling A–Z**, **bukan** penerjemahan bahasa isyarat secara menyeluruh atau pengenalan kalimat kontinu. Beberapa huruf melibatkan dua tangan; kesesuaian contoh gestur pada dataset dengan penggunaan BISINDO perlu diperiksa, termasuk kemungkinan variasi daerah.

## 2. Arsitektur sistem

### Pipeline training

```text
Gambar BISINDO A–Z (Kaggle)
    → Audit label & split train/validation
    → OpenCV: baca gambar dan konversi BGR → RGB
    → MediaPipe Hand Landmarker (maks. 2 tangan)
    → Normalisasi koordinat, ekstraksi 128 fitur
    → Random Forest (scikit-learn)
    → Evaluasi Accuracy, Macro-F1, per kelas, confusion matrix
    → Model tersimpan (.joblib)
```

### Pipeline inference saat ini

```text
Webcam laptop → OpenCV lokal → MediaPipe Hands
    → 128 fitur landmark → Random Forest
    → Prediksi huruf + skor + visualisasi landmark
```

### Rencana pipeline aplikasi akhir

```text
Prediksi huruf antarframe → Stabilisasi → Konfirmasi huruf
    → Letter Buffer / Word Formation → Smart Word Prediction
    → Koreksi, hapus, spasi, dan output teks
```

## 3. Teknologi

| Komponen | Teknologi | Fungsi |
|---|---|---|
| Bahasa | Python | Pengembangan pipeline |
| Pengolahan citra | OpenCV, NumPy | Membaca webcam, konversi warna, visualisasi |
| Ekstraksi gestur | MediaPipe Hand Landmarker | 21 landmark per tangan, hingga 2 tangan |
| Klasifikasi | scikit-learn Random Forest | Pengenalan alfabet A–Z |
| Model | joblib | Menyimpan dan memuat model |
| Analisis | Matplotlib, CSV/JSON | Evaluasi dan dokumentasi eksperimen |
| Demo saat ini | OpenCV di Windows | Inferensi webcam real-time dalam satu jendela |
| Eksperimen UI sebelumnya | Gradio, FastRTC | Streaming webcam melalui Codespaces |

Topik Computer Vision yang relevan: representasi citra, konversi ruang warna, transformasi intensitas, filtering spasial (*smoothing*), dan estimasi pose/landmark tangan. **Eksperimen intensitas dan smoothing belum diklaim sebagai peningkatan performa yang teruji.**

## 4. Dataset dan ekstraksi fitur

Sumber dataset: [Indonesian Sign Language (BISINDO) — Kaggle](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo).

Dataset memiliki pembagian `images/train/A-Z` dan `images/val/A-Z`. Anotasi YOLO juga tersedia, tetapi klasifikasi Random Forest menggunakan **nama folder sebagai label huruf**, bukan nomor kelas pada file anotasi.

| Statistik | Nilai |
|---|---:|
| Total gambar | 11.470 |
| Training / validation | 9.169 / 2.301 |
| Berhasil diekstraksi — training | 7.549 (82,33%) |
| Berhasil diekstraksi — validation | 1.910 (83,01%) |
| Sampel dengan landmark (total) | 9.459 |
| Sampel tanpa landmark | 2.011 |
| Dimensi fitur | 128 |

**Skema fitur v1:** dua slot tangan × 21 landmark × koordinat `(x, y, z)` = 126 fitur, ditambah dua penanda keberadaan tangan = **128 fitur**. Koordinat dinormalisasi terhadap ruang bersama tangan yang terdeteksi; tangan diurutkan berdasarkan posisi horizontal pergelangan. Bila hanya satu tangan berhasil terdeteksi, slot lainnya diisi nol.

Konfigurasi baseline MediaPipe: sisi gambar terpanjang maksimal 640 piksel, `num_hands=2`, `min_hand_detection_confidence=0.3`, dan `min_hand_presence_confidence=0.3`. Pengujian cropping menggunakan anotasi YOLO belum menunjukkan peningkatan tingkat deteksi dibandingkan penggunaan gambar penuh pada sampel audit.

Dataset mencakup gambar yang dirotasi, dibalik (*flip*), dan diaugmentasi. Hubungan antara gambar asli dan augmentasinya lintas split masih perlu pemeriksaan lebih lanjut.

## 5. Progres yang sudah dicapai

- [x] Menyiapkan lingkungan Python dan pustaka pendukung di GitHub Codespaces.
- [x] Membuat prototipe webcam Gradio dan mendeteksi landmark satu atau dua tangan.
- [x] Mengunduh dan mengaudit dataset BISINDO 26 huruf.
- [x] Memvisualisasikan sampel gambar asli dan hasil augmentasi.
- [x] Menguji parameter MediaPipe serta cropping YOLO pada sampel dataset.
- [x] Mengekstraksi landmark menjadi fitur numerik (128 dimensi).
- [x] Melatih Random Forest v1 dan mengevaluasi hasil klasifikasi.
- [x] Mengaudit fitur validation yang identik dengan training.
- [x] Melatih Random Forest v2 dengan percobaan pembobotan sampel A, M, N; hasilnya belum melampaui v1.
- [x] Menjalankan deteksi dan prediksi alfabet pada webcam **lokal via OpenCV**, yang lebih responsif daripada streaming Codespaces.
- [x] Membuat eksperimen pembanding referensi dataset vs webcam, beserta log prediksi manual.
- [x] Menyiapkan alat pengujian stabilitas prediksi antarframe (`test_stability.py`).

**Belum selesai:** analisis hasil uji stabilitas kontinu, validasi kinerja pada pengguna baru, pengembangan fitur teks/kata, dan finalisasi UI demo.

## 6. Hasil evaluasi model

| Metrik | Random Forest v1 | Random Forest v2 |
|---|---:|---:|
| Validation Accuracy (sampel berlandmark) | **97,12%** | 96,96% |
| Validation Macro-F1 | **97,08%** | 96,94% |
| Accuracy setelah mengeluarkan fitur identik | **96,51%** | 96,32% |
| Macro-F1 setelah mengeluarkan fitur identik | **96,41%** | 96,25% |

Pada v1, sebanyak **1.855 dari 1.910** sampel validation berlandmark diprediksi benar. Jika seluruh 2.301 gambar validation dijadikan penyebut dan kegagalan deteksi dihitung sebagai kegagalan pengenalan, keberhasilan pipeline adalah **80,62%**.

Audit menemukan **332 sampel validation** yang fitur numeriknya identik dengan fitur training menurut metode perbandingan yang digunakan. Setelah sampel tersebut dikeluarkan, akurasi masih **96,51%**. Meski begitu, hasil ini **belum membuktikan tidak adanya data leakage**, sebab foto hasil augmentasi dapat tetap berkaitan tanpa memiliki fitur yang sama persis.

Eksperimen Random Forest v2 menggunakan bobot lebih tinggi untuk sampel dua tangan pada huruf A, M, dan N. Hasilnya tidak meningkatkan validation; karena itu **Random Forest v1 tetap digunakan sebagai baseline**.

Dalam pengujian webcam manual, sejumlah prediksi Top-1 sudah sesuai dengan huruf yang diuji, tetapi skor yang rendah dapat membuatnya ditolak oleh threshold. Frame yang disimpan secara manual **tidak cukup** untuk menyimpulkan akurasi webcam secara keseluruhan.

## 7. Permasalahan yang sedang ditangani

1. **Perbedaan dataset dan webcam (*domain shift*):** model cukup baik di validation tetapi belum konsisten pada gestur pengguna baru.
2. **Deteksi tangan tidak lengkap:** beberapa foto gestur dua tangan, termasuk contoh huruf N, hanya menghasilkan landmark satu tangan.
3. **Skor vs keputusan:** prediksi Top-1 benar bisa tersembunyi jika skor berada di bawah ambang; skor Random Forest bukan jaminan probabilitas terkalibrasi.
4. **Stabilitas real-time:** prediksi bisa berubah antarfame dan perlu mekanisme konfirmasi huruf.
5. **Kualitas evaluasi:** data augmentasi berpotensi menimbulkan kemiripan lintas split; diperlukan pengujian yang benar-benar independen.
6. **Validasi gestur:** label dan contoh dataset perlu ditinjau, tidak otomatis menjadi standar tunggal BISINDO.

## 8. Roadmap pengembangan

### Prioritas berikutnya — kualitas pengenalan

- [ ] Jalankan **pengujian stabilitas prediksi per frame** pada webcam lokal dan analisis `frame_log.csv` serta `session_summary.csv`.
- [ ] Petakan kesalahan per huruf, termasuk apakah satu atau dua tangan yang terdeteksi.
- [ ] Periksa kualitas data dan kemungkinan *leakage* akibat augmentasi.
- [ ] Uji penyesuaian fitur, preprocessing, atau model alternatif **hanya jika hasil evaluasi mendukung**; bandingkan dengan RF v1 pada data yang sama.

### Target fitur produk

- [ ] **Temporal smoothing** agar hasil tidak mudah berganti setiap frame.
- [ ] **Konfirmasi huruf** dan pencegahan huruf berulang ketika gestur ditahan.
- [ ] **Word Formation:** menyusun huruf yang dikonfirmasi menjadi kata.
- [ ] **Text Correction:** hapus huruf, spasi, koreksi, dan reset.
- [ ] **Smart Word Prediction:** rekomendasi kata bahasa Indonesia melalui kamus, prefix matching, dan/atau edit distance.
- [ ] Penyempurnaan antarmuka dan pengujian end-to-end untuk demo akhir.

## 9. Menjalankan prototipe webcam lokal

Contoh lingkungan: **Windows, Python 3.12**, model yang telah diekspor dari Codespaces, dan aplikasi OpenCV lokal.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install opencv-python mediapipe numpy scikit-learn joblib
.\.venv\Scripts\python.exe local_webcam.py
```

Contoh struktur workspace lokal:

```text
bisindo-alphabet-recognition/
├── local_webcam.py
├── models/
│   ├── bisindo_rf_v1.joblib
│   └── hand_landmarker.task
└── .venv/
```

> **Catatan sinkronisasi:** struktur di atas mencerminkan workspace lokal/Codespaces. File eksperimen dan artefak model yang dibahas dalam dokumentasi **belum tentu sudah di-commit ke branch `main` GitHub**. Periksa keberadaan file sebelum menjalankan aplikasi. Model `joblib` hanya boleh dimuat dari sumber tepercaya, dan versi dependensi perlu kompatibel dengan lingkungan training.

## 10. Batasan penggunaan

Proyek ini adalah **prototipe pembelajaran Computer Vision**, bukan penerjemah BISINDO tersertifikasi dan bukan pengganti penerjemah bahasa isyarat profesional. Performa dataset tidak boleh disamakan dengan akurasi pada semua pengguna, kondisi pencahayaan, atau variasi BISINDO.
