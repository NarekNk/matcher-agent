"""Detect spoken language from a preview audio file using local OpenAI Whisper.

Uses the public ``openai-whisper`` package (``import whisper``) — no API key.
Used at prediction time when the caller did not supply ``--track-language``:
we run a short decode to obtain the detected language, then map ISO codes to
the same lowercase vocabulary as ``attribute_normalizer`` (e.g. ``english``).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VALID_WHISPER_MODELS: frozenset[str] = frozenset(
    {"tiny", "base", "small", "medium", "large"}
)

# Whisper / ISO-639-1 codes → curator-style labels (must match playlist JSON
# vocabulary after ``normalize_attribute_labels`` lowercasing).
_WHISPER_OR_ISO_TO_LABEL: dict[str, str] = {
    # Full names
    "english": "english",
    "spanish": "spanish",
    "french": "french",
    "german": "german",
    "italian": "italian",
    "portuguese": "portuguese",
    "dutch": "dutch",
    "polish": "polish",
    "russian": "russian",
    "ukrainian": "ukrainian",
    "japanese": "japanese",
    "korean": "korean",
    "chinese": "chinese",
    "arabic": "arabic",
    "hindi": "hindi",
    "turkish": "turkish",
    "swedish": "swedish",
    "danish": "danish",
    "norwegian": "norwegian",
    "finnish": "finnish",
    "greek": "greek",
    "hebrew": "hebrew",
    "czech": "czech",
    "romanian": "romanian",
    "hungarian": "hungarian",
    "vietnamese": "vietnamese",
    "thai": "thai",
    "indonesian": "indonesian",
    "malay": "malay",
    "tagalog": "tagalog",
    "filipino": "tagalog",
    "bengali": "bengali",
    "urdu": "urdu",
    "persian": "persian",
    "farsi": "persian",
    "tamil": "tamil",
    "telugu": "telugu",
    "punjabi": "punjabi",
    "swahili": "swahili",
    "afrikaans": "afrikaans",
    "catalan": "catalan",
    "galician": "galician",
    "slovak": "slovak",
    "croatian": "croatian",
    "serbian": "serbian",
    "bulgarian": "bulgarian",
    "lithuanian": "lithuanian",
    "latvian": "latvian",
    "estonian": "estonian",
    "slovenian": "slovenian",
    "macedonian": "macedonian",
    "albanian": "albanian",
    "icelandic": "icelandic",
    "welsh": "welsh",
    "irish": "irish",
    "basque": "basque",
    "maltese": "maltese",
    "latin": "latin",
    # ISO 639-1 (local ``whisper`` returns these on ``result["language"]``)
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "pl": "polish",
    "ru": "russian",
    "uk": "ukrainian",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
    "ar": "arabic",
    "hi": "hindi",
    "tr": "turkish",
    "sv": "swedish",
    "da": "danish",
    "no": "norwegian",
    "nb": "norwegian",
    "nn": "norwegian",
    "fi": "finnish",
    "el": "greek",
    "he": "hebrew",
    "cs": "czech",
    "ro": "romanian",
    "hu": "hungarian",
    "vi": "vietnamese",
    "th": "thai",
    "id": "indonesian",
    "ms": "malay",
    "tl": "tagalog",
    "fil": "tagalog",
    "bn": "bengali",
    "ur": "urdu",
    "fa": "persian",
    "ta": "tamil",
    "te": "telugu",
    "pa": "punjabi",
    "sw": "swahili",
    "af": "afrikaans",
    "ca": "catalan",
    "gl": "galician",
    "sk": "slovak",
    "hr": "croatian",
    "sr": "serbian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "lv": "latvian",
    "et": "estonian",
    "sl": "slovenian",
    "mk": "macedonian",
    "sq": "albanian",
    "is": "icelandic",
    "cy": "welsh",
    "ga": "irish",
    "eu": "basque",
    "mt": "maltese",
    "la": "latin",
}


def normalize_whisper_language_label(raw: str | None) -> str | None:
    """Map Whisper ``language`` string to a single lowercase label."""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if not key:
        return None
    return _WHISPER_OR_ISO_TO_LABEL.get(key, key)


def _resolve_torch_device(explicit: str | None) -> str:
    """Return ``cuda`` or ``cpu`` for ``whisper.load_model``."""
    if explicit is not None:
        e = explicit.strip().lower()
        if e in ("cuda", "cpu"):
            return e
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@lru_cache(maxsize=16)
def _load_local_whisper(model_name: str, device: str) -> Any:
    import whisper

    return whisper.load_model(model_name, device=device)


def detect_spoken_language_from_audio(
    audio_path: Path,
    *,
    model_name: str = "tiny",
    device: str | None = None,
) -> str | None:
    """Run local Whisper on ``audio_path`` and return a normalized language label.

    Requires the ``openai-whisper`` package (``pip install openai-whisper``).
    No API key. The ``tiny`` model is the default for speed on short previews.

    Returns ``None`` on missing file, missing dependency, or decode failure.
    """
    path = Path(audio_path)
    if not path.is_file():
        logger.warning("Whisper language: audio file missing: %s", path)
        return None

    try:
        import whisper  # noqa: F401
    except ImportError:
        logger.warning(
            "Whisper language: openai-whisper not installed; "
            "pip install openai-whisper to enable preview language detection."
        )
        return None

    m = (model_name or "tiny").strip().lower()
    if m not in _VALID_WHISPER_MODELS:
        logger.warning("Whisper language: invalid WHISPER_MODEL=%r; using tiny", model_name)
        m = "tiny"

    torch_device = _resolve_torch_device(device)
    try:
        model = _load_local_whisper(m, torch_device)
        result = model.transcribe(
            str(path),
            fp16=(torch_device == "cuda"),
            verbose=False,
            condition_on_previous_text=False,
        )
    except Exception as exc:
        logger.warning("Whisper language: transcribe failed: %s", exc)
        return None

    raw_lang = result.get("language") if isinstance(result, dict) else None
    label = normalize_whisper_language_label(raw_lang)
    if label:
        logger.info(
            "Whisper language: model=%s device=%s raw=%r -> %r",
            m,
            torch_device,
            raw_lang,
            label,
        )
    else:
        logger.warning("Whisper language: empty language in transcribe result")
    return label
