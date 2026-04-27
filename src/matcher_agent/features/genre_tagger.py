from __future__ import annotations

import re
from collections.abc import Iterable

# Mapping of canonical genre tag -> ordered list of recognition patterns.
# Patterns are matched against lowercased text using word-boundary regex.
# Order matters: more specific patterns first (e.g. "hip hop" before "pop").
_GENRE_PATTERNS: list[tuple[str, list[str]]] = [
    ("hip_hop", [r"hip\s*-?\s*hop", r"hiphop", r"\brap\b", r"\btrap\b", r"\bdrill\b", r"\burban\b"]),
    ("rnb", [r"\br&b\b", r"\brnb\b", r"\br\s*&\s*b\b", r"\bsoul\b", r"\bneo[\s-]*soul\b"]),
    ("country", [r"\bcountry\b", r"\bbluegrass\b", r"\bnashville\b", r"\bhonky[\s-]*tonk\b"]),
    ("rock", [r"\brock\b", r"\bmetal\b", r"\bpunk\b", r"\bhardcore\b", r"\bgrunge\b", r"\bemo\b"]),
    ("alt_indie", [r"\bindie\b", r"\balternative\b", r"\balt[\s-]*rock\b", r"\bshoegaze\b"]),
    ("pop", [r"\bpop\b", r"\btop\s*40\b", r"\bmainstream\b", r"\bchart\b"]),
    ("edm", [
        r"\bedm\b", r"\belectronic\b", r"\belectro\b", r"\bhouse\b", r"\btechno\b",
        r"\btrance\b", r"\bdubstep\b", r"\bdnb\b", r"drum\s*(&|and|n)?\s*bass", r"\bbass\s*music\b",
        r"\bfuture\s*bass\b", r"\bhardstyle\b", r"\bbig\s*room\b", r"\bambient\b",
    ]),
    ("latin", [
        r"\blatin\b", r"\breggaeton\b", r"\bsalsa\b", r"\bbachata\b", r"\bcumbia\b",
        r"\bmerengue\b", r"\bbanda\b", r"\bregional\s*mexicano\b",
    ]),
    ("afro", [
        r"\bafro\s*beats?\b", r"\bafrobeats?\b", r"\bnaija\b", r"\bamapiano\b",
        r"\bafro\s*pop\b", r"\bafro\s*house\b", r"\bafro\b",
    ]),
    ("reggae", [r"\breggae\b", r"\bdancehall\b", r"\bska\b", r"\bdub\b"]),
    ("jazz", [r"\bjazz\b", r"\bswing\b", r"\bbebop\b", r"\bbig\s*band\b"]),
    ("classical", [r"\bclassical\b", r"\borchestra\b", r"\bsymphon", r"\bopera\b", r"\bpiano\b"]),
    ("folk_acoustic", [
        r"\bfolk\b", r"\bacoustic\b", r"\bsinger[\s-]*songwriter\b", r"\bunplugged\b",
    ]),
    ("gospel_christian", [r"\bgospel\b", r"\bchristian\b", r"\bworship\b", r"\bccm\b"]),
    ("chill_lofi", [
        r"\bchill\b", r"\bchill\s*hop\b", r"\blo[\s-]*fi\b", r"\blofi\b", r"\brelax\b",
        r"\bstudy\b", r"\bsleep\b", r"\bmeditat", r"\bbeats\s*to\b",
    ]),
    ("workout_party", [
        r"\bworkout\b", r"\bgym\b", r"\bfitness\b", r"\brunning\b", r"\bcardio\b",
        r"\bparty\b", r"\bclub\b", r"\bdance\b",
    ]),
    ("kpop_jpop", [r"\bk[\s-]*pop\b", r"\bj[\s-]*pop\b", r"\bk[\s-]*drama\b"]),
    ("french_world", [r"\bfran[cç]ais\b", r"\bvariete\b", r"\bchanson\b"]),
]

# Rough semantic conflicts for the hard genre filter at inference.
_CONFLICT_GROUPS: list[set[str]] = [
    {"hip_hop", "country"},
    {"hip_hop", "classical"},
    {"hip_hop", "folk_acoustic"},
    {"country", "edm"},
    {"country", "rnb"},
    {"classical", "edm"},
    {"classical", "rock"},
    {"gospel_christian", "edm"},
]


def _compile_patterns() -> list[tuple[str, list[re.Pattern[str]]]]:
    return [(g, [re.compile(p) for p in pats]) for g, pats in _GENRE_PATTERNS]


_COMPILED = _compile_patterns()


def normalize_for_tagging(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def tag_text(text: str | None) -> set[str]:
    """Extract canonical genre tags from a free-form text string."""
    norm = normalize_for_tagging(text)
    if not norm:
        return set()
    tags: set[str] = set()
    for genre, patterns in _COMPILED:
        if any(p.search(norm) for p in patterns):
            tags.add(genre)
    return tags


def tag_texts(texts: Iterable[str | None]) -> list[set[str]]:
    return [tag_text(t) for t in texts]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def has_conflict(track_tags: set[str], playlist_tags: set[str]) -> bool:
    """Return True if a track and playlist tag set are semantically incompatible.

    Conflict requires both: (a) at least one cross-genre pair from
    _CONFLICT_GROUPS is present; and (b) they share zero overlapping tags.
    """
    if not track_tags or not playlist_tags:
        return False
    if track_tags & playlist_tags:
        return False
    for group in _CONFLICT_GROUPS:
        if (track_tags & group) and (playlist_tags & group) and not (track_tags & playlist_tags):
            if (track_tags & group) != (playlist_tags & group):
                return True
    return False


def all_known_tags() -> list[str]:
    return [g for g, _ in _GENRE_PATTERNS]
