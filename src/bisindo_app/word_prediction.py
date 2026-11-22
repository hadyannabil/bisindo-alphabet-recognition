"""Modul Prediksi Kata Cerdas (Smart Word Prediction) untuk BISINDO.

Sumber Kosakata:
    Kosakata frekuensi tinggi bahasa Indonesia (KBBI umum dan percakapan sehari-hari)
    yang mencakup kata ganti, kata kerja, kata benda, kata sifat, angka, serta istilah
    yang relevan dengan BISINDO dan komunikasi visual sehari-hari.

Aturan Normalisasi Teks:
    - Pemrosesan bersifat case-insensitive saat pencocokan awalan (prefix matching).
    - Menjaga bentuk huruf (casing) dari token terakhir pada kata saran:
        * Jika input uppercase (misal 'MAK'), saran yang dikembalikan berupa UPPERCASE ('MAKAN').
        * Jika input titlecase (misal 'Mak'), saran berupa Titlecase ('Makan').
        * Jika input lowercase (misal 'mak'), saran berupa lowercase ('makan').
    - Input dipisahkan berdasarkan spasi untuk mengambil kata/token parsial terakhir.
    - Kandidat disaring agar tidak ada duplikasi kata dan dibatasi maksimal `limit`.

Keterbatasan:
    - Fitur ini berfokus pada pelengkapan kata (word completion) berdasarkan prefiks
      token terakhir yang sedang diketik.
    - Ketika teks kosong atau berakhir dengan spasi (tepat setelah spasi), saran dikosongkan
      (prediksi kata berikutnya / next-word prediction setelah spasi disiapkan untuk tahap
      pengembangan terpisah sesuai spesifikasi).
"""
from __future__ import annotations

from typing import Sequence

# Kosakata frekuensi tinggi bahasa Indonesia, diurutkan berdasarkan prioritas/relevansi.
# Untuk prefiks 'mak', urutan kata diatur: makan, makanan, makna, makin sesuai spesifikasi alur.
DEFAULT_VOCABULARY: list[str] = [
    # Prioritas contoh & kata sangat umum
    "makan", "makanan", "makna", "makin", "maksud", "makmur", "maksimal", "makhluk",
    "saya", "kamu", "dia", "kita", "mereka", "kami", "anda", "kalian", "ia", "dirinya",
    "ini", "itu", "sini", "situ", "sana",
    "apa", "apakah", "siapa", "bagaimana", "mengapa", "kenapa", "kapan", "dimana", "berapa",
    "ada", "adalah", "akan", "bisa", "dapat", "sudah", "telah", "sedang", "lagi", "belum",
    "tidak", "bukan", "jangan", "hanya", "juga", "masih", "selalu", "sering", "kadang",
    "dan", "atau", "tetapi", "tapi", "namun", "karena", "sebab", "sehingga", "agar", "supaya",
    "jika", "kalau", "bila", "apabila", "meskipun", "walaupun", "setelah", "sebelum", "ketika", "saat",
    "di", "ke", "dari", "pada", "untuk", "dengan", "secara", "tentang", "seperti", "sebagai", "antara", "sampai", "hingga",
    
    # Kata kerja umum
    "baca", "membaca", "bacaan",
    "tulis", "menulis", "tulisan",
    "lihat", "melihat", "terlihat", "kelihatan",
    "dengar", "mendengar", "terdengar", "pendengaran",
    "bicara", "berbicara", "pembicara",
    "kata", "berkata", "mengatakan", "perkataan",
    "buat", "membuat", "buatan", "pembuat",
    "kerja", "bekerja", "pekerja", "pekerjaan",
    "belajar", "pembelajaran", "pelajar",
    "ajar", "mengajar", "pengajar", "ajaran",
    "jalan", "berjalan", "perjalanan",
    "lari", "berlari", "pelari",
    "duduk", "terduduk", "kedudukan",
    "berdiri", "mendirikan",
    "tidur", "tertidur",
    "bangun", "membangun", "bangunan",
    "mandi", "memandikan",
    "minum", "minuman", "peminum",
    "pergi", "bepergian",
    "pulang", "kepulangan",
    "datang", "kedatangan",
    "tiba", "ketibaan",
    "masuk", "memasuki", "pemasukan",
    "keluar", "mengeluarkan", "pengeluaran",
    "naik", "menaiki", "kenaikan",
    "turun", "menuruni", "penurunan",
    "ambil", "mengambil", "pengambilan",
    "bawa", "membawa", "pembawa",
    "beri", "memberi", "memberikan", "pemberian",
    "bantu", "membantu", "bantuan",
    "tolong", "menolong", "pertolongan",
    "buka", "membuka", "terbuka", "bukaan",
    "tutup", "menutup", "tertutup", "penutupan",
    "cari", "mencari", "pencarian",
    "tahu", "mengetahui", "pengetahuan",
    "kenal", "mengenal", "perkenalan",
    "paham", "memahami", "pemahaman",
    "mengerti", "pengertian",
    "rasa", "merasa", "merasakan", "perasaan",
    "pikir", "berpikir", "memikirkan", "pikiran",
    "ingat", "mengingat", "peringatan", "ingatan",
    "lupa", "melupakan",
    "harap", "berharap", "harapan",
    "tunggu", "menunggu",
    "mulai", "memulai", "permulaan",
    "selesai", "menyelesaikan", "penyelesaian",
    "suka", "menyukai", "kesukaan",
    "cinta", "mencintai", "pecinta",
    "ingin", "keinginan",
    "mau", "kemauan",
    "harus", "keharusan",
    "perlu", "memerlukan", "keperluan",
    "coba", "mencoba", "percobaan",
    "tanya", "bertanya", "pertanyaan",
    "jawab", "menjawab", "jawaban",
    "pilih", "memilih", "pilihan",
    "kirim", "mengirim", "pengiriman",
    "terima", "menerima", "penerimaan",
    "beli", "membeli", "pembelian", "pembeli",
    "jual", "menjual", "penjualan", "penjual",
    "bayar", "membayar", "pembayaran",
    "pesan", "memesan", "pesanan",
    "pakai", "memakai", "pakaian",
    "pasang", "memasang", "pemasangan",
    "lepas", "melepas", "melepaskan",
    "simpan", "menyimpan", "penyimpanan",
    "buang", "membuang",
    "cuci", "mencuci", "cucian",
    "masak", "memasak", "masakan",
    "potong", "memotong", "potongan",
    "tarik", "menarik", "tarikan",
    "dorong", "mendorong", "dorongan",
    "lempar", "melempar", "lemparan",
    "tangkap", "menangkap", "penangkapan",
    "pegang", "memegang", "pegangan",
    "sentuh", "menyentuh", "sentuhan",

    # BISINDO dan komunikasi
    "isyarat", "bisindo", "tuli", "tunarungu", "bahasa", "huruf", "kata", "kalimat",
    "tangan", "gerak", "gerakan", "isyaratkan", "isyaratnya", "komunikasi", "penerjemah",
    "jari", "jemari", "telapak", "ekspresi", "wajah",

    # Kata sifat umum
    "baik", "bagus", "indah", "cantik", "tampan",
    "buruk", "jelek", "rusak",
    "besar", "kecil", "luas", "sempit", "tinggi", "rendah", "panjang", "pendek",
    "jauh", "dekat", "tepat",
    "cepat", "lambat", "pelan",
    "panas", "dingin", "hangat", "sejuk",
    "bersih", "kotor",
    "mudah", "gampang", "sulit", "susah", "rumit",
    "berat", "ringan",
    "senang", "bahagia", "gembira", "sedih", "marah", "takut", "berani", "khawatir", "cemas",
    "sehat", "sakit", "kuat", "lemah",
    "benar", "betul", "salah", "keliru",
    "sama", "beda", "berbeda",
    "pintar", "cerdas", "pandai", "rajin", "malas",
    "ramai", "sepi", "sunyi", "tenang",
    "kenyang", "lapar", "haus",
    "jelas", "kabur", "nyata", "pasti",
    "terang", "gelap", "merah", "kuning", "hijau", "biru", "putih", "hitam", "cokelat",

    # Kata benda lingkungan, keluarga, waktu, tempat
    "orang", "manusia", "anak", "ibu", "bunda", "mama", "ayah", "bapak", "papa",
    "adik", "kakak", "kakek", "nenek", "saudara", "keluarga", "teman", "kawan", "sahabat",
    "guru", "dosen", "dokter", "polisi", "petugas",
    "kepala", "mata", "telinga", "hidung", "mulut", "gigi", "bibir", "lidah", "rambut", "leher",
    "badan", "tubuh", "dada", "perut", "kaki", "lutut", "kulit", "darah", "hati", "jiwa",
    "rumah", "kamar", "sekolah", "kampus", "kantor", "pasar", "toko", "rumah sakit",
    "jalan", "kota", "desa", "daerah", "negara", "indonesia", "dunia",
    "hari", "minggu", "bulan", "tahun", "waktu", "jam", "menit", "detik",
    "pagi", "siang", "sore", "malam", "fajar", "petang", "kemarin", "sekarang", "besok", "lusa",
    "air", "api", "angin", "tanah", "udara", "matahari", "bulan", "bintang", "hujan", "awan",
    "nasi", "roti", "lauk", "sayur", "buah", "apel", "pisang", "jeruk", "mangga",
    "kopi", "teh", "susu", "jus",
    "buku", "pensil", "pulpen", "kertas", "meja", "kursi", "papan", "pintu", "jendela",
    "baju", "celana", "sepatu", "kaos", "topi", "tas",
    "uang", "harga", "nomor", "nama", "angka", "gambar", "foto", "video", "pesan",
    "ponsel", "komputer", "laptop", "layar", "kamera", "aplikasi", "internet",

    # Alfabet awal beragam untuk kelengkapan A-Z
    "awal", "akhir", "arti", "angka", "aman", "alam", "acara", "alamat", "akal",
    "bintang", "bunga", "burung", "bebas", "biar", "biaya", "benda",
    "cahaya", "cukup", "catat", "catatan", "cerita", "contoh",
    "daya", "daun", "danau", "darurat", "derajat",
    "empat", "emas", "enak", "ekonomi",
    "faktor", "fakta", "fasilitas", "fungsi",
    "guna", "gunung", "gadis", "gigi", "gelas",
    "hasil", "hidup", "hukum", "harga", "hutan",
    "ikut", "izin", "ilmu", "isi", "ikan",
    "jarak", "janji", "jumlah", "judul", "jenis",
    "kabar", "kunci", "kapal", "kucing", "kuda",
    "lirik", "lantai", "langit", "lampu", "luas",
    "nama", "nilai", "nyaman", "nyata", "norma",
    "obat", "ombak", "olahraga", "organisasi",
    "pohon", "pintu", "peta", "pagi", "pusat",
    "rahasia", "rakyat", "ruang", "ruangan", "roda",
    "surat", "suara", "sepatu", "sepeda", "sungai",
    "tanah", "taman", "tamu", "tanda", "tujuan",
    "udara", "uang", "usia", "utama", "umum",
    "warga", "warna", "wilayah", "wujud",
    "yakin", "yaitu",
    "zaman", "zona"
]


def suggest_words(
    formed_text: str | None,
    limit: int = 4,
    vocabulary: Sequence[str] | None = None,
) -> list[str]:
    """Kembalikan maksimal `limit` kandidat kata tanpa mengubah teks input.

    Parameters:
        formed_text: Teks yang sedang tersusun (dari TextComposer.text atau input teks).
        limit: Jumlah maksimal saran kata yang dikembalikan (default: 4).
        vocabulary: Daftar kosakata opsional untuk pencarian (default: DEFAULT_VOCABULARY).

    Returns:
        List string saran kata tanpa duplikat. Mengembalikan list kosong jika teks kosong,
        token terakhir kosong (misalnya tepat setelah spasi), atau tidak ada saran cocok.
    """
    if not formed_text or limit <= 0:
        return []

    # Jika teks berakhir dengan spasi, pengguna belum mengetik huruf untuk kata berikutnya.
    if formed_text.endswith(" "):
        return []

    tokens = formed_text.split(" ")
    last_token = tokens[-1]
    if not last_token:
        return []

    # Ambil karakter alfabet dari token terakhir sebagai prefiks
    prefix = "".join(ch for ch in last_token if ch.isalpha())
    if not prefix:
        return []

    prefix_lower = prefix.lower()
    vocab = DEFAULT_VOCABULARY if vocabulary is None else vocabulary

    candidates: list[str] = []
    seen: set[str] = set()

    for word in vocab:
        w_lower = word.lower()
        if w_lower.startswith(prefix_lower):
            # Sesuaikan format casing dengan prefiks input
            if prefix.isupper():
                formatted = w_lower.upper()
            elif prefix.istitle():
                formatted = w_lower.capitalize()
            else:
                formatted = w_lower

            key = formatted.lower()
            if key not in seen:
                seen.add(key)
                candidates.append(formatted)
                if len(candidates) >= limit:
                    break

    return candidates


def apply_suggestion(formed_text: str, suggestion: str) -> str:
    """Ganti kata/token parsial terakhir dalam formed_text dengan saran terpilih.

    Mempertahankan kata-kata sebelumnya dan spasi yang sudah ada dalam kalimat.

    Contoh:
        apply_suggestion("SAYA MAK", "MAKAN") -> "SAYA MAKAN"
        apply_suggestion("saya mak", "makan") -> "saya makan"
        apply_suggestion("mak", "makan") -> "makan"
    """
    if not formed_text:
        return suggestion

    if formed_text.endswith(" "):
        return formed_text + suggestion

    idx = formed_text.rfind(" ")
    if idx == -1:
        return suggestion

    return formed_text[: idx + 1] + suggestion
