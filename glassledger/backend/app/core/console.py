"""Console output that survives a cp1252 Windows terminal.

The rupee sign is not in cp1252, so ``print("₹1,200")`` raises
``UnicodeEncodeError`` on a default Windows console -- and it raises *after* the
work is done, which is the worst possible time to lose a report.

``init()`` flips stdout/stderr to UTF-8 where the stream supports it and records
whether that worked. ``money()`` then emits ``₹`` or falls back to ``Rs`` based on
what the terminal can actually render, so the same script produces readable output
in PowerShell, Git Bash, CI, and a redirect to a file.
"""

from __future__ import annotations

import sys

from .money import rupees

_UNICODE_OK = True


def init() -> bool:
    global _UNICODE_OK
    ok = True
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            ok = False
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            ok = False
    _UNICODE_OK = ok
    return ok


def money(paise: int) -> str:
    return ("₹" if _UNICODE_OK else "Rs ") + rupees(paise)


def sym(unicode_char: str, ascii_fallback: str) -> str:
    return unicode_char if _UNICODE_OK else ascii_fallback
