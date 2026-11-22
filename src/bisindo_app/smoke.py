"""Pemeriksaan ringan stabilisasi gestur dan penyusunan teks (tanpa webcam)."""
from __future__ import annotations
from .core import AutoConfirmGate, TemporalSmoother, TextComposer


def main():
    smoother = TemporalSmoother(window=5, min_votes=3)
    assert [smoother.update("A", 0.8) for _ in range(3)] == [None, None, "A"]
    assert smoother.update(None) is None

    composer = TextComposer(min_confirm_gap=0)
    for ch in "MAKAN":
        assert composer.confirm(ch)
    assert composer.text == "makan"
    composer.add_space()
    assert composer.text == "makan "
    composer.backspace()
    assert composer.text == "makan"
    composer.clear()
    assert composer.text == ""

    gate = AutoConfirmGate(hold_seconds=1.8)
    assert not gate.update("A", now=10.0)
    assert gate.update("A", now=11.8)
    assert not gate.update("A", now=12.0)
    print("PASS: stabilisasi, penyusunan teks, dan konfirmasi otomatis")


if __name__ == "__main__":
    main()
