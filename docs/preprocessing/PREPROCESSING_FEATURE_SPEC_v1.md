# Spesifikasi Tetap — BISINDO Preprocessing & Feature Engineering v1.0

**Tanggal desain:** 22 November 2026  
**Feature ID:** `BISINDO-GEOM128-v1.0`  
**Fokus:** ekstraksi fitur alfabet BISINDO A–Z dari citra statis dengan 0, 1, atau 2 tangan yang terdeteksi.  
**Status:** implementasi dan pengujian unit selesai; ekstraksi dataset asli dan validasi MediaPipe end-to-end belum dieksekusi dalam sesi ini.

## 1. Keputusan sebelumnya yang tetap dipakai

- Dataset utama: **UMN Alphabet Image Data v1 — `01. Original Images`**, bukan dataset augmentasi, UM, atau BINUS.
- Pembagian yang sudah disetujui: 4.769 total, 4 gambar konflik dikarantina, **TRAIN 3.011, VALIDATION 1.001, RESERVED 753**.
- Dari RESERVED: 700 masuk daftar evaluasi yang disetujui dan 53 ditahan karena risiko alias grup metadata. Daftar tersebut telah disimpan; **TEST tidak diakses oleh tahap desain ini**.
- Ada satu grup alias lintas TRAIN–VALIDATION dan tidak ada bukti signer-independent; jangan menyebut hasil DEV/test sebagai evaluasi penutur baru.
- Eksperimen RF sebelumnya **hanya sumber pembelajaran masalah**, bukan pengikat pilihan konfigurasi atau bukti performa desain ini.

## 2. Keputusan preprocessing utama

| Langkah | Aturan final baseline | Mengapa |
|---|---|---|
| Inventaris | Ambil `path` dan `label` dari manifest yang sudah dibekukan | Mencegah split baru diam-diam |
| Decode | Pillow/Pillow-HEIF berdasarkan isi file aktual | Sebagian JPG UMN mengandung HEIF |
| Orientasi | `ImageOps.exif_transpose` | Menangani metadata rotasi kamera |
| Ruang warna | Konversi ke RGB 8-bit, 3 kanal | Format input MediaPipe |
| Resize | **Hanya downscale** dengan `max(width,height) <= 640`, rasio aspek sama | Seragam dan efisien tanpa distorsi |
| Crop / segmentasi | **Tidak** | Dapat memotong satu dari dua tangan; tambahan kompleksitas |
| Mirror | **Tidak** sebelum deteksi; bila diperlukan, mirror hanya output visualisasi | Hindari label/geometry yang tak konsisten |
| Gamma, CLAHE, Gaussian, sharpen | **Tidak** pada baseline | Jangan menerapkan peningkatan citra tanpa diagnosis |
| Augmentasi | **Tidak** pada baseline | Jaga identitas pose; terutama jangan horizontal flip otomatis |

Semua parameter eksplisit di `src/bisindo_preproc/pipeline.py` dan dicatat sebagai hash `spec_digest_sha256` dalam hasil ekstraksi.

## 3. Ekstraksi tangan

- API: MediaPipe **Tasks `HandLandmarker`**, bukan `mp.solutions.hands` untuk jalur ini.
- Mode: **IMAGE** baik untuk foto maupun webcam baseline; artinya webcam diproses frame demi frame tanpa pelacakan antar-frame. Pilihan ini memprioritaskan konsistensi model. Ukur FPS saat integrasi nanti.
- Maksimal dua tangan; 21 landmark per tangan; koordinat `x,y,z`.
- Ambang deteksi **0.50** dan presence **0.50** sebagai nilai tetap awal spesifikasi baru, **bukan angka yang sudah terbukti paling optimal**.
- Model `.task`: simpan SHA-256 aset yang digunakan; jangan mengubah aset di tengah pengembangan.
- Data RGB yang masuk ke MediaPipe harus **sama format/urutan/resize** antara dataset dan webcam.

## 4. Normalisasi fitur 128 dimensi (definisi eksplisit)

Input: 0, 1, atau 2 set landmark dari MediaPipe, tiap set berisi 21 titik.

**A. Urutan tangan:** untuk 2 tangan, urutkan menurut `x` pergelangan (`landmark 0`) dalam **koordinat gambar** (kiri ke kanan dalam gambar), tie-break `y` pergelangan. Jangan menggunakan posisi output API atau tag `Left/Right` sebagai dasar slot utama; tag handedness sensitif pada konvensi input mirror. Jika hanya 1 tangan, isi slot pertama.

**B. Pemetaan koordinat (citra yang sudah di-resize):**

- `X = x * width`
- `Y = y * height`
- `Z = z * width`

MediaPipe `z` bukan kedalaman fisik dan tidak mengukur jarak absolut antartangan secara dapat diandalkan. Karenanya tiap tangan memakai kedalaman relatif terhadap `z` pergelangan tangan itu sendiri.

**C. Bounding box global:** gabungkan seluruh koordinat `X,Y` dari 1–2 tangan yang terdeteksi. Hitung `x_min`, `x_max`, `y_min`, `y_max`, pusat `cx=(x_min+x_max)/2`, `cy=(y_min+y_max)/2`, dan skala `s=max(x_max-x_min, y_max-y_min, 1.0)`.

**D. Fitur setiap landmark:**

- `x' = (X - cx) / s`
- `y' = (Y - cy) / s`
- `z' = (Z - Z_wrist_of_same_hand) / s`

Satu kotak gabungan **mempertahankan posisi dan konfigurasi kedua tangan**. Normalisasi masing-masing tangan terpisah akan menghapus jarak dan posisi antartangan yang mungkin penting untuk label BISINDO. Penggunaan satu skala dan normalisasi bersama juga membuat fitur sensitif terhadap hilangnya tangan kedua; itulah sebabnya masker dan metrik deteksi harus ada.

**E. Vektor keluaran**

`[slot_0:(21 x 3), slot_1:(21 x 3), presence_0, presence_1]`

- 2 tangan: 126 angka landmark + `[1,1]`.
- 1 tangan: 63 angka landmark pertama, 63 nol, + `[1,0]`.
- 0 tangan: **`None / no_hand`**, tidak menghasilkan vektor nol dan tidak digunakan untuk klasifikasi.
- Landmark tidak finite, jumlah tidak tepat 21, atau lebih dari 2 tangan: **error** dengan jejak audit.

## 5. Keputusan tentang occlusion dan jumlah tangan

**Jangan menyimpulkan setiap 1 tangan yang terdeteksi sebagai pose satu tangan yang benar.** Gestur dua tangan dapat kehilangan satu tangan karena saling menutupi (occlusion), tangan keluar frame, atau detektor gagal.

- Simpan `hands_detected` setiap citra dan ringkas per huruf.
- Sampel dengan 1 tangan **tetap disimpan sebagai fitur baseline** dan ditandai `detected_1_hand`. Belum ada dasar linguistik kuat untuk otomatis memindahkannya ke karantina berdasarkan label huruf.
- Jangan mengisi landmark tangan yang hilang dengan tebakan atau interpolasi spasial.
- Untuk 0 tangan, tulis `no_hand`, log setiap kasus, dan anggap kegagalan ekstraksi dalam metrik end-to-end.
- Saat webcam, `no_hand` menghasilkan **abstention** (tidak menambahkan huruf). Penghalusan prediksi antarframe baru akan ditentukan pada tahap aplikasi.
- Urutan tangan berdasar posisi x bersifat deterministik untuk foto, **tetapi slot dapat tertukar saat tangan berpotongan**. Ini batas metode statis, bukan masalah yang diselesaikan smoothing.

## 6. Validasi tahap preprocessing/fitur tanpa eksperimen classifier

**Gate A — konsistensi data:** manifest TRAIN/VALIDATION cocok dengan split dan setiap file dapat dibaca; hash byte bisa diperiksa pada ekstraksi pertama dengan `--verify-sha256`.

**Gate B — kualitas data output:** fitur berdimensi `(N_detected,128)`, tipe `float32`, tanpa NaN/inf, label/urutan path cocok dengan record. Semua 0 tangan dan error dicatat sehingga jumlahnya tetap masuk penyebut kualitas detektor.

**Gate C — sifat geometris:** unit test membuktikan invarian translasi, invarian skala seragam, penanganan rasio aspek, stabilitas ketika urutan tangan dari detektor tertukar, masking satu tangan, dan `no_hand` yang abstain.

**Gate D — konsistensi train vs webcam:** jalankan fungsi `prepare_rgb` dan `landmarks_to_features` yang sama. Fungsi `prepare_webcam_bgr` hanya mengubah BGR OpenCV menjadi RGB, **tanpa mirror**.

**Gate E — reproduksibilitas:** catat feature ID, hash spesifikasi, hash file manifest, hash `.task`, dan versi pustaka dalam summary JSON.

## 7. Belum dilakukan dan bukan target tahap ini

- Belum menjalankan MediaPipe terhadap 4.765 citra asli; raw JPG/HEIF dan `.task` tidak tersedia bersama paket ini di lingkungan eksekusi saat penyusunan.
- Belum mengukur detection rate, keterbacaan HEIF aktual seluruh set, kinerja occlusion, FPS webcam, atau kemampuan klasifikasi versi ini.
- Belum melakukan training baru maupun menyentuh RESERVED (700 eligible / 53 ditahan).
- Belum memutuskan bentuk gestur dinamis (misalnya ketukan dan lintasan); classifier gambar statis **tidak boleh diklaim memahami urutan gerakan**. Itu batas penelitian yang harus ditulis.

## 8. Handoff ke tahap berikutnya

Setelah ekstraksi train dan validation berhasil tanpa error yang tidak dijelaskan, lanjut **training Random Forest dari awal** menggunakan fitur spesifikasi ini, lalu evaluasi pada VALIDATION. Jangan memuat RF v4 lama secara otomatis karena definisi 128 dimensi belum dibuktikan sama. Setelah arsitektur/model/threshold benar-benar dikunci, TEST 700 eligible baru boleh diekstraksi dan dinilai satu kali.

Teknik tambahan seperti CLAHE, crop, tracking, tangan baru saat occlusion, atau model temporal **tidak dibuka kecuali** ditemukan akar masalah spesifik yang perlu diselesaikan demi target proyek.
