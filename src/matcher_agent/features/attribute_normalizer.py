"""Normalize Xano "soft" playlist attributes (activity, countries, languages,
tempos, moods) and the matching optional track-side inputs.

These attributes differ from genres in two important ways:

1. **Sparse on the playlist side** — curators frequently leave them as
   ``"any"`` / ``"other"`` / null, which means "no preference". We treat
   those values as no-signal and drop them.
2. **Absent on the historical training data** — track-side moods, languages,
   activities, etc. are not available in our 60k historical match records.
   That's why these attributes are NOT model features. They flow into a
   small post-rerank penalty at inference time (see
   ``MatcherService._apply_soft_attribute_penalty``) so the model is never
   asked to learn weights for signals that don't exist at training time.

Normalization keeps the raw curator vocabulary (lowercase) — there's no
canonical taxonomy to map onto here, so simple string matching is the
right tool.
"""
from __future__ import annotations

from collections.abc import Iterable

# Values that mean "no preference / unknown". They are dropped during
# normalization so they don't trigger any soft-attribute penalty.
_DEFAULT_VALUES: frozenset[str] = frozenset(
    {
        "",
        "any",
        "other",
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
        "all",
    }
)

SOFT_ATTRIBUTE_NAMES: tuple[str, ...] = (
    "activities",
    "countries",
    "languages",
    "tempos",
    "moods",
)


def _norm(value: str) -> str:
    return value.strip().lower()


def normalize_attribute_labels(values: Iterable[str] | None) -> set[str]:
    """Normalize a free-form string array into a canonical lowercase set.

    Steps:
      * coerce each entry to a stripped, lowercase string
      * drop empty strings and known no-signal placeholders (``any``,
        ``other``, ``null``, ``unknown``, ``all``, ``n/a`` ...)
      * deduplicate

    The returned set is *empty* when the input carries no real signal.
    Downstream code uses an empty set as a sentinel meaning "no preference",
    which is also why the soft-penalty logic only fires when BOTH sides of a
    (track, playlist) pair have non-empty normalized sets.
    """
    if not values:
        return set()
    out: set[str] = set()
    for v in values:
        if v is None:
            continue
        text = _norm(str(v))
        if not text or text in _DEFAULT_VALUES:
            continue
        out.add(text)
    return out
