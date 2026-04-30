from __future__ import annotations

from matcher_agent.features.attribute_normalizer import (
    SOFT_ATTRIBUTE_NAMES,
    normalize_attribute_labels,
)


def test_lowercases_and_dedupes() -> None:
    assert normalize_attribute_labels(["Workout", "workout", "Party"]) == {
        "workout",
        "party",
    }


def test_drops_default_and_blank_values() -> None:
    raw = ["Any", "any", "OTHER", "  ", "None", "n/a", "Unknown", "ALL", None]
    assert normalize_attribute_labels(raw) == set()


def test_keeps_real_values_alongside_defaults() -> None:
    assert normalize_attribute_labels(
        ["Energetic", "any", "Uplifting", None, ""]
    ) == {"energetic", "uplifting"}


def test_handles_non_string_inputs_gracefully() -> None:
    # Numbers/objects should be coerced to strings; empty iterables yield empty set.
    assert normalize_attribute_labels(None) == set()
    assert normalize_attribute_labels([]) == set()
    assert normalize_attribute_labels([42, "Pop"]) == {"42", "pop"}


def test_soft_attribute_names_constant_is_complete() -> None:
    # Sanity: keeps in sync with PlaylistProfile fields and service code.
    assert set(SOFT_ATTRIBUTE_NAMES) == {
        "activities",
        "countries",
        "languages",
        "tempos",
        "moods",
    }
