"""Modul stabilisasi, penyusunan teks, dan konfirmasi otomatis BISINDO."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import time

LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class TemporalSmoother:
    """Voting sliding window; no-hand menghapus prediksi agar tidak stale."""

    def __init__(self, window: int = 9, min_votes: int = 6, min_score: float = 0.42):
        if window < 1 or not (1 <= min_votes <= window):
            raise ValueError("Parameter voting tidak valid")
        if not (0 <= min_score <= 1):
            raise ValueError("min_score tidak valid")
        self.items: deque[tuple[str, float]] = deque(maxlen=window)
        self.min_votes = min_votes
        self.min_score = min_score

    def reset(self) -> None:
        self.items.clear()

    def update(self, label: str | None, score: float = 0.0) -> str | None:
        if label is None or label not in LETTERS or score < self.min_score:
            self.reset()
            return None
        self.items.append((label, score))
        counts = Counter(name for name, _ in self.items)
        candidate, votes = counts.most_common(1)[0]
        # Harus prediksi terbaru juga cocok, sehingga tidak menampilkan huruf lama.
        if votes >= self.min_votes and candidate == label:
            return candidate
        return None


@dataclass
class TextComposer:
    text: str = ""
    min_confirm_gap: float = 0.7
    _last_confirm_at: float = field(default=float("-inf"), repr=False)

    def confirm(self, letter: str | None, now: float | None = None) -> bool:
        if not letter or letter not in LETTERS:
            return False
        now = time.monotonic() if now is None else now
        if now - self._last_confirm_at < self.min_confirm_gap:
            return False
        self.text += letter.lower()
        self._last_confirm_at = now
        return True

    def backspace(self) -> None:
        self.text = self.text[:-1]

    def add_space(self) -> None:
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def clear(self) -> None:
        self.text = ""

    def replace_last_word(self, replacement: str) -> None:
        """Ganti kata parsial terakhir dengan kata saran terpilih."""
        word = replacement
        if self.text and self.text.islower():
            word = replacement.lower()
        elif self.text and self.text.isupper():
            word = replacement.upper()

        if not self.text:
            self.text = word
            return
        if self.text.endswith(" "):
            self.text += word
            return
        idx = self.text.rfind(" ")
        if idx == -1:
            self.text = word
        else:
            self.text = self.text[: idx + 1] + word



@dataclass
class AutoConfirmGate:
    """Konfirmasi sekali per gestur; bisa diulang setelah tangan dilepas.

    UI memasok jumlah tangan melalui `hands` untuk debounce pelepasan.
    Pemanggil lama tanpa argumen `hands` tetap didukung.
    """

    hold_seconds: float = 1.8
    candidate: str | None = None
    since: float | None = None
    consumed: bool = False
    release_seconds: float = 0.5
    locked_letter: str | None = None
    no_hand_since: float | None = None

    def reset(self) -> None:
        self.candidate = None
        self.since = None
        self.consumed = False
        self.locked_letter = None
        self.no_hand_since = None

    def update(
        self, letter: str | None, now: float | None = None, *, hands: int | None = None
    ) -> bool:
        now = time.monotonic() if now is None else now

        # Kompatibilitas untuk pemanggil lama yang belum memberi jumlah tangan.
        if hands is None and letter not in LETTERS:
            self.reset()
            return False

        # Hanya keadaan tangan benar-benar tidak terdeteksi yang melepas kunci.
        # Satu frame hilang tidak cukup untuk mengonfirmasi huruf yang sama lagi.
        if hands is not None and hands <= 0:
            if self.no_hand_since is None:
                self.no_hand_since = now
            elif now - self.no_hand_since >= self.release_seconds:
                self.reset()
            return False

        # Jika tangan muncul lagi sebelum batas release, kunci lama tetap aktif.
        self.no_hand_since = None

        # Prediksi tidak stabil tidak dianggap sebagai pelepasan gestur.
        # Timer calon huruf baru harus mulai lagi setelah prediksi stabil.
        if letter not in LETTERS:
            self.candidate = None
            self.since = None
            return False

        # Gestur yang sudah dikonfirmasi tidak boleh terkonfirmasi ulang
        # selama belum ada pelepasan tangan atau huruf lain yang sah.
        if letter == self.locked_letter:
            self.candidate = None
            self.since = None
            return False

        if self.candidate != letter:
            self.candidate = letter
            self.since = now
            self.consumed = False
            return False

        if self.since is not None and now - self.since >= self.hold_seconds:
            self.locked_letter = letter
            self.candidate = None
            self.since = None
            self.consumed = True
            return True
        return False

    def mark_manual_confirm(self, letter: str | None) -> None:
        if letter and letter in LETTERS:
            self.locked_letter = letter
            self.candidate = None
            self.since = None
            self.consumed = True
