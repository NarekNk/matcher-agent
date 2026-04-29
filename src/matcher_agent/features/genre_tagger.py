from __future__ import annotations

import re
from collections.abc import Iterable

# Mapping of canonical genre tag -> ordered list of recognition patterns.
# Patterns are matched against lowercased text using word-boundary regex.
# Order matters: more specific patterns first (e.g. "hip hop" before "pop").
#
# These tags are the SHARED VOCABULARY used across:
#   - rule-based tagger (this file, regex over titles/descriptions)
#   - genre_normalizer.py (explicit Xano genre/subgenre + Spotify label maps)
#   - playlist_profiles.PlaylistProfile.tags
#
# When you add a tag here you should also add it to genre_normalizer's
# external maps and decide whether it deserves a conflict-group entry below.
_GENRE_PATTERNS: list[tuple[str, list[str]]] = [
    ("hip_hop", [
        r"hip\s*-?\s*hop", r"hiphop", r"\brap\b", r"\btrap\b", r"\bdrill\b",
        r"\burban\b", r"\bboom\s*bap\b", r"\bgangsta\b", r"\bcloud\s*rap\b",
        r"\bconscious\s*rap\b", r"\beast\s*coast\b", r"\bwest\s*coast\b",
    ]),
    ("rnb", [r"\br&b\b", r"\brnb\b", r"\br\s*&\s*b\b", r"\bquiet\s*storm\b"]),
    ("soul", [r"\bsoul\b", r"\bneo[\s-]*soul\b", r"\bmotown\b"]),
    ("country", [r"\bcountry\b", r"\bbluegrass\b", r"\bnashville\b", r"\bhonky[\s-]*tonk\b", r"\bamericana\b", r"\boutlaw\b"]),
    ("metal", [
        r"\bmetal\b", r"\bmetalcore\b", r"\bthrash\b", r"\bdeath\s*metal\b",
        r"\bblack\s*metal\b", r"\bdoom\s*metal\b", r"\bpower\s*metal\b",
    ]),
    ("punk", [r"\bpunk\b", r"\bhardcore\b", r"\bgrunge\b", r"\bpost[\s-]*punk\b"]),
    ("rock", [r"\brock\b", r"\bemo\b", r"\bshoegaze\b", r"\bgarage\b", r"\bprog\b", r"\bpsychedelic\b"]),
    ("alt_indie", [r"\bindie\b", r"\balternative\b", r"\balt[\s-]*rock\b", r"\bbedroom\b", r"\bart\s*pop\b"]),
    ("pop", [r"\bpop\b", r"\btop\s*40\b", r"\bmainstream\b", r"\bchart\b"]),
    ("edm", [
        r"\bedm\b", r"\belectronic\b", r"\belectro\b", r"\bhouse\b", r"\btechno\b",
        r"\btrance\b", r"\bdubstep\b", r"\bdnb\b", r"drum\s*(&|and|n)?\s*bass", r"\bbass\s*music\b",
        r"\bfuture\s*bass\b", r"\bhardstyle\b", r"\bbig\s*room\b", r"\bsynthwave\b", r"\bbreakbeat\b",
    ]),
    ("ambient", [r"\bambient\b", r"\bdrone\b", r"\bnew\s*age\b", r"\bspace\s*music\b"]),
    ("latin", [
        r"\blatin\b", r"\breggaeton\b", r"\bsalsa\b", r"\bbachata\b", r"\bcumbia\b",
        r"\bmerengue\b", r"\bbanda\b", r"\bregional\s*mexicano\b", r"\bflamenco\b",
    ]),
    ("afro", [
        r"\bafro\s*beats?\b", r"\bafrobeats?\b", r"\bnaija\b", r"\bamapiano\b",
        r"\bafro\s*pop\b", r"\bafro\s*house\b", r"\bafro\b",
    ]),
    ("reggae", [r"\breggae\b", r"\bdancehall\b", r"\bska\b", r"\bdub\b", r"\blovers\s*rock\b"]),
    ("jazz", [r"\bjazz\b", r"\bswing\b", r"\bbebop\b", r"\bbig\s*band\b", r"\bbossa\b"]),
    ("classical", [r"\bclassical\b", r"\borchestra\b", r"\bsymphon", r"\bopera\b", r"\bbaroque\b", r"\bromantic\s*era\b", r"\bchoral\b", r"\bminimalism\b"]),
    ("folk_acoustic", [
        r"\bfolk\b", r"\bacoustic\b", r"\bsinger[\s-]*songwriter\b", r"\bunplugged\b",
    ]),
    ("blues", [r"\bblues\b", r"\bdelta\s*blues\b", r"\bchicago\s*blues\b"]),
    ("funk", [r"\bfunk\b", r"\bp[\s-]*funk\b", r"\bdisco\b"]),
    ("gospel_christian", [r"\bgospel\b", r"\bchristian\b", r"\bworship\b", r"\bccm\b", r"\bspiritual\b"]),
    ("chill_lofi", [
        r"\bchill\b", r"\bchill\s*hop\b", r"\blo[\s-]*fi\b", r"\blofi\b", r"\brelax\b",
        r"\bstudy\b", r"\bsleep\b", r"\bmeditat", r"\bbeats\s*to\b",
    ]),
    ("workout_party", [
        r"\bworkout\b", r"\bgym\b", r"\bfitness\b", r"\brunning\b", r"\bcardio\b",
        r"\bparty\b", r"\bclub\b", r"\bdance\s*pop\b",
    ]),
    ("kpop_jpop", [r"\bk[\s-]*pop\b", r"\bj[\s-]*pop\b", r"\bk[\s-]*drama\b", r"\bcity\s*pop\b"]),
    ("world", [r"\bworld\s*music\b", r"\bbhangra\b", r"\bcelt", r"\bafrican\s*music\b", r"\bbalkan\b"]),
    ("french_world", [r"\bfran[cç]ais\b", r"\bvariete\b", r"\bchanson\b"]),
]

# Rough semantic conflicts for the hard genre filter at inference. A conflict
# fires only when the track has tags from one side of a group AND the playlist
# has tags from the OTHER side AND they share no tags overall. The set is
# intentionally conservative — we'd rather miss a soft conflict than wrongly
# penalize a borderline match.
_CONFLICT_GROUPS: list[set[str]] = [
    {"hip_hop", "country"},
    {"hip_hop", "classical"},
    {"hip_hop", "folk_acoustic"},
    {"hip_hop", "metal"},
    {"hip_hop", "blues"},
    {"country", "edm"},
    {"country", "rnb"},
    {"country", "metal"},
    {"classical", "edm"},
    {"classical", "rock"},
    {"classical", "metal"},
    {"classical", "punk"},
    {"gospel_christian", "edm"},
    {"gospel_christian", "metal"},
    {"metal", "pop"},
    {"metal", "kpop_jpop"},
    {"metal", "chill_lofi"},
    {"metal", "ambient"},
    {"punk", "pop"},
    {"punk", "kpop_jpop"},
    {"punk", "chill_lofi"},
    {"reggae", "metal"},
    {"reggae", "classical"},
    {"latin", "metal"},
    {"latin", "country"},
    {"jazz", "metal"},
    {"jazz", "edm"},
    {"blues", "edm"},
    {"blues", "kpop_jpop"},
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
