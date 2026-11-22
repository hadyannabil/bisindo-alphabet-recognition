"""Tes core aplikasi tanpa fitur prediksi kata (masih placeholder UI)."""
import pytest
from bisindo_app.core import TextComposer, TemporalSmoother


def test_smoother_requires_six_votes():
    smoother = TemporalSmoother(window=9, min_votes=6)
    for _ in range(5):
        assert smoother.update("A", 0.8) is None
    assert smoother.update("A", 0.8) == "A"


def test_smoother_clears_no_hand():
    smoother = TemporalSmoother(window=3, min_votes=2)
    smoother.update("B", 0.9)
    assert smoother.update("B", 0.9) == "B"
    assert smoother.update(None) is None
    assert smoother.update("B", 0.9) is None


def test_smoother_clears_low_score():
    smoother = TemporalSmoother(window=3, min_votes=2, min_score=0.5)
    smoother.update("A", 0.6)
    assert smoother.update("A", 0.3) is None
    assert smoother.update("A", 0.6) is None


def test_smoother_rejects_bad_parameters():
    with pytest.raises(ValueError):
        TemporalSmoother(window=2, min_votes=3)
    with pytest.raises(ValueError):
        TemporalSmoother(min_score=1.2)


def test_text_confirm_cooldown_and_repeat():
    composer = TextComposer(min_confirm_gap=0.7)
    assert composer.confirm("M", now=10.0)
    assert not composer.confirm("M", now=10.2)
    assert composer.confirm("M", now=10.8)
    assert composer.text == "mm"


def test_text_ops():
    composer = TextComposer(min_confirm_gap=0)
    for ch in "SAYA":
        assert composer.confirm(ch)
    assert composer.text == "saya"
    composer.add_space()
    composer.add_space()
    assert composer.text == "saya "
    composer.backspace()
    assert composer.text == "saya"
    composer.clear()
    assert composer.text == ""


def test_text_rejects_invalid_letter():
    composer = TextComposer()
    assert not composer.confirm("AA", now=10)
    assert not composer.confirm("1", now=10)
    assert composer.text == ""
