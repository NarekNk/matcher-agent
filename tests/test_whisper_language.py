"""Tests for Whisper → curator language label mapping."""

from __future__ import annotations

import pytest

from matcher_agent.audio.whisper_language import normalize_whisper_language_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("english", "english"),
        ("English", "english"),
        ("en", "english"),
        ("es", "spanish"),
        ("spanish", "spanish"),
        ("", None),
        ("  ", None),
        (None, None),
        ("fr", "french"),
        ("zh", "chinese"),
    ],
)
def test_normalize_whisper_language_label(raw: str | None, expected: str | None) -> None:
    assert normalize_whisper_language_label(raw) == expected
