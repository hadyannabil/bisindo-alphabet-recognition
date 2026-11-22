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

    def confirm(self, letter: str, now: float | None = None) -> bool:
        if letter not in LETTERS:
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
    """One auto-confirm per uninterrupted stable gesture. Off by default in UI."""
    hold_seconds: float = 1.8
    candidate: str | None = None
    since: float | None = None
    consumed: bool = False

    def reset(self) -> None:
        self.candidate = None
        self.since = None
        self.consumed = False

    def update(self, letter: str | None, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if letter not in LETTERS:
            self.reset()
            return False
        if self.candidate != letter:
            self.candidate = letter
            self.since = now
            self.consumed = False
            return False
        if not self.consumed and self.since is not None and now - self.since >= self.hold_seconds:
            self.consumed = True
            return True
        return False

    def mark_manual_confirm(self, letter: str) -> None:
        if letter in LETTERS:
            self.candidate = letter
            self.consumed = True
