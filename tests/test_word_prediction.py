"""Unit test untuk fitur Prediksi Kata Cerdas (Smart Word Prediction)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bisindo_app.core import TextComposer
from bisindo_app.word_prediction import apply_suggestion, suggest_words

def test_suggest_words_empty_text():
    """Uji teks kosong dan None mengembalikan list kosong."""
    assert suggest_words("") == []
    assert suggest_words("   ") == []
    assert suggest_words(None) == []


def test_suggest_words_trailing_space():
    """Uji token terakhir kosong (setelah spasi) tidak menghasilkan saran."""
    assert suggest_words("saya ") == []
    assert suggest_words("saya mak ") == []
    assert suggest_words("kata ") == []


def test_suggest_words_single_word():
    """Uji satu kata parsial menghasilkan kandidat relevan."""
    lower_res = suggest_words("mak")
    assert lower_res == ["makan", "makanan", "makna", "makin"]

    upper_res = suggest_words("MAK")
    assert upper_res == ["MAKAN", "MAKANAN", "MAKNA", "MAKIN"]

    title_res = suggest_words("Mak")
    assert title_res == ["Makan", "Makanan", "Makna", "Makin"]


def test_suggest_words_multiple_words():
    """Uji kalimat dengan beberapa kata hanya menggunakan token terakhir sebagai prefiks."""
    res_lower = suggest_words("saya mak")
    assert res_lower == ["makan", "makanan", "makna", "makin"]

    res_upper = suggest_words("SAYA MAK")
    assert res_upper == ["MAKAN", "MAKANAN", "MAKNA", "MAKIN"]

    res_long = suggest_words("saya ingin mak")
    assert res_long == ["makan", "makanan", "makna", "makin"]

    res_house = suggest_words("di rum")
    assert "rumah" in res_house


def test_suggest_words_not_found():
    """Uji prefiks yang tidak ada dalam kosakata mengembalikan list kosong."""
    assert suggest_words("xyzq") == []
    assert suggest_words("saya zzzzz") == []
    assert suggest_words("qwerty") == []


def test_suggest_words_no_duplicates_and_limit():
    """Uji tidak ada duplikasi kata dan batasan limit berfungsi."""
    res = suggest_words("m", limit=4)
    assert len(res) == 4
    assert len(res) == len(set(res))

    res_limit_2 = suggest_words("m", limit=2)
    assert len(res_limit_2) == 2


def test_apply_suggestion():
    """Uji penggantian token parsial terakhir dengan kata saran terpilih."""
    assert apply_suggestion("SAYA MAK", "MAKAN") == "SAYA MAKAN"
    assert apply_suggestion("saya mak", "makan") == "saya makan"
    assert apply_suggestion("mak", "makan") == "makan"
    assert apply_suggestion("SAYA INGIN MAK", "MAKAN") == "SAYA INGIN MAKAN"
    assert apply_suggestion("", "SAYA") == "SAYA"


def test_text_composer_replace_last_word():
    """Uji integrasi penggantian kata pada TextComposer."""
    composer = TextComposer()
    composer.text = "saya mak"
    composer.replace_last_word("makan")
    assert composer.text == "saya makan"

    composer.text = "SAYA MAK"
    composer.replace_last_word("MAKAN")
    assert composer.text == "SAYA MAKAN"

    composer.text = "mak"
    composer.replace_last_word("makan")
    assert composer.text == "makan"


def test_full_flow_with_composer():
    """Uji alur lengkap: pengetikan huruf -> saran kata -> pemilihan saran."""
    composer = TextComposer(min_confirm_gap=0)
    for ch in "SAYA":
        composer.confirm(ch)
    assert composer.text == "saya"

    composer.add_space()
    assert composer.text == "saya "
    
    assert suggest_words(composer.text) == []

    for ch in "MAK":
        composer.confirm(ch)
    assert composer.text == "saya mak"

    suggestions = suggest_words(composer.text)
    assert suggestions == ["makan", "makanan", "makna", "makin"]

    chosen = suggestions[0]
    composer.replace_last_word(chosen)
    assert composer.text == "saya makan"
