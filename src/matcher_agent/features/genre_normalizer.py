"""Map authoritative free-form genre labels to canonical tag set.

Three sources feed our genre representation:

1. Xano playlist payloads, which now expose `genres` and `subgenres`
   string arrays. These are the most authoritative signal because they
   are curator-selected.
2. Spotify artist genres returned by `sp.artists()` — useful for tracks
   at inference time.
3. Free-form playlist text (name + description) and track titles —
   handled by the regex-based tagger in `genre_tagger.py`.

This module produces canonical tags (the same vocabulary as
`genre_tagger._GENRE_PATTERNS`) so the model and the conflict detector
see a unified label space regardless of source.
"""
from __future__ import annotations

from collections.abc import Iterable

from matcher_agent.features.genre_tagger import tag_text


def _key(label: str) -> str:
    return label.strip().lower()


# All canonical Xano top-level genres -> canonical tags. Wired directly
# to the closest tag(s) in `genre_tagger`. Empty set means "ignore"
# (e.g. "Other"). Multi-tag mappings are intentional — for example
# "Reggaeton" is both Latin and reggaeton-styled.
_XANO_GENRE_TO_TAGS: dict[str, set[str]] = {
    "pop": {"pop"},
    "hip-hop": {"hip_hop"},
    "hiphop": {"hip_hop"},
    "rap": {"hip_hop"},
    "r&b": {"rnb"},
    "rnb": {"rnb"},
    "rock": {"rock"},
    "electronic": {"edm"},
    "country": {"country"},
    "jazz": {"jazz"},
    "classical": {"classical"},
    "acoustic": {"folk_acoustic"},
    "latin": {"latin"},
    "world": {"world"},
    "reggae": {"reggae"},
    "metal": {"metal"},
    "blues": {"blues"},
    "indie": {"alt_indie"},
    "alt": {"alt_indie"},
    "alternative": {"alt_indie"},
    "funk": {"funk"},
    "gospel": {"gospel_christian"},
    "ambient": {"ambient"},
    "punk": {"punk"},
    "other": set(),
    "reggaeton": {"latin"},
    "soul": {"soul"},
    "afrobeat": {"afro"},
    "afrobeats": {"afro"},
    "folk": {"folk_acoustic"},
}


# Xano subgenres -> canonical tags. We also fold in common Spotify
# free-form genre strings here so that artist_genres like "trap",
# "drill", "dancehall", "neo soul" map to the same tags as the
# corresponding Xano subgenre.
_XANO_SUBGENRE_TO_TAGS: dict[str, set[str]] = {
    # Pop
    "electropop": {"pop", "edm"},
    "teen pop": {"pop"},
    "synthpop": {"pop", "edm"},
    "synth pop": {"pop", "edm"},
    "dance pop": {"pop", "workout_party"},
    "indie pop": {"pop", "alt_indie"},
    "pop rock": {"pop", "rock"},
    "k-pop": {"pop", "kpop_jpop"},
    "kpop": {"pop", "kpop_jpop"},
    "j-pop": {"pop", "kpop_jpop"},
    "jpop": {"pop", "kpop_jpop"},
    "power pop": {"pop", "rock"},
    "art pop": {"pop", "alt_indie"},
    "bedroom pop": {"pop", "alt_indie", "chill_lofi"},
    "alt-pop": {"pop", "alt_indie"},
    "alt pop": {"pop", "alt_indie"},
    # Hip-Hop
    "trap": {"hip_hop"},
    "boom bap": {"hip_hop"},
    "lo-fi hip-hop": {"hip_hop", "chill_lofi"},
    "lofi hip-hop": {"hip_hop", "chill_lofi"},
    "lofi hip hop": {"hip_hop", "chill_lofi"},
    "lo-fi rap": {"hip_hop", "chill_lofi"},
    "conscious rap": {"hip_hop"},
    "drill": {"hip_hop"},
    "gangsta rap": {"hip_hop"},
    "alternative hip-hop": {"hip_hop", "alt_indie"},
    "alt hip hop": {"hip_hop", "alt_indie"},
    "east coast": {"hip_hop"},
    "west coast": {"hip_hop"},
    "west coast rap": {"hip_hop"},
    "cloud rap": {"hip_hop"},
    "underground hip hop": {"hip_hop", "alt_indie"},
    # R&B / Soul
    "neo-soul": {"rnb", "soul"},
    "neo soul": {"rnb", "soul"},
    "contemporary r&b": {"rnb"},
    "contemporary rnb": {"rnb"},
    "quiet storm": {"rnb"},
    "alternative r&b": {"rnb", "alt_indie"},
    "alt r&b": {"rnb", "alt_indie"},
    "alt-r&b": {"rnb", "alt_indie"},
    "funk r&b": {"rnb", "funk"},
    # Rock
    "classic rock": {"rock"},
    "hard rock": {"rock"},
    "punk rock": {"rock", "punk"},
    "garage rock": {"rock"},
    "progressive rock": {"rock"},
    "indie rock": {"rock", "alt_indie"},
    "alternative rock": {"rock", "alt_indie"},
    "psychedelic rock": {"rock"},
    "blues rock": {"rock", "blues"},
    "folk rock": {"rock", "folk_acoustic"},
    "acoustic rock": {"rock", "folk_acoustic"},
    "grunge": {"rock", "alt_indie"},
    # Electronic
    "edm": {"edm"},
    "house": {"edm"},
    "deep house": {"edm"},
    "tech house": {"edm"},
    "techno": {"edm"},
    "trance": {"edm"},
    "drum & bass": {"edm"},
    "drum and bass": {"edm"},
    "dnb": {"edm"},
    "dubstep": {"edm"},
    "electro": {"edm"},
    "electro house": {"edm"},
    "synthwave": {"edm"},
    "ambient electronic": {"edm", "ambient"},
    "breakbeat": {"edm"},
    "indietronica": {"edm", "alt_indie"},
    "electro-funk": {"edm", "funk"},
    "future bass": {"edm"},
    "big room": {"edm"},
    # Country
    "classic country": {"country"},
    "country pop": {"country", "pop"},
    "alt-country": {"country", "alt_indie"},
    "alt country": {"country", "alt_indie"},
    "bluegrass": {"country", "folk_acoustic"},
    "americana": {"country", "folk_acoustic"},
    "outlaw country": {"country"},
    "modern country": {"country"},
    "country rock": {"country", "rock"},
    # Jazz
    "smooth jazz": {"jazz"},
    "bebop": {"jazz"},
    "swing": {"jazz"},
    "cool jazz": {"jazz"},
    "jazz fusion": {"jazz"},
    "vocal jazz": {"jazz"},
    "jazz-funk": {"jazz", "funk"},
    "jazz funk": {"jazz", "funk"},
    "bossa nova": {"jazz", "world"},
    # Classical
    "baroque": {"classical"},
    "romantic": {"classical"},
    "modern classical": {"classical"},
    "minimalism": {"classical", "ambient"},
    "choral": {"classical"},
    "orchestral": {"classical"},
    "soundtrack": {"classical"},
    # Folk / Acoustic
    "folk": {"folk_acoustic"},
    "singer-songwriter": {"folk_acoustic"},
    "indie folk": {"folk_acoustic", "alt_indie"},
    "alternative folk": {"folk_acoustic", "alt_indie"},
    # Latin
    "reggaeton": {"latin"},
    "salsa": {"latin"},
    "bachata": {"latin"},
    "cumbia": {"latin"},
    "latin pop": {"latin", "pop"},
    "merengue": {"latin"},
    "latin trap": {"latin", "hip_hop"},
    # World / regional
    "afrobeat": {"afro"},
    "afrobeats": {"afro"},
    "amapiano": {"afro"},
    "bhangra": {"world"},
    "celtic": {"world", "folk_acoustic"},
    "flamenco": {"world", "folk_acoustic"},
    "k-drama": {"kpop_jpop"},
    # Reggae
    "roots reggae": {"reggae"},
    "dub": {"reggae"},
    "dancehall": {"reggae"},
    "lovers rock": {"reggae"},
    # Metal
    "heavy metal": {"metal"},
    "thrash metal": {"metal"},
    "death metal": {"metal"},
    "black metal": {"metal"},
    "power metal": {"metal"},
    "metalcore": {"metal"},
    "doom metal": {"metal"},
    # Blues
    "delta blues": {"blues"},
    "chicago blues": {"blues"},
    "electric blues": {"blues"},
    "acoustic blues": {"blues", "folk_acoustic"},
    # Indie/Alt extras
    "lo-fi": {"chill_lofi"},
    "lofi": {"chill_lofi"},
    "shoegaze": {"alt_indie", "rock"},
    # Funk
    "p-funk": {"funk"},
    "p funk": {"funk"},
    "funk rock": {"funk", "rock"},
    "funk soul": {"funk", "soul"},
    # Gospel
    "contemporary gospel": {"gospel_christian"},
    "traditional gospel": {"gospel_christian"},
    "urban gospel": {"gospel_christian"},
    "gospel choir": {"gospel_christian"},
    "southern gospel": {"gospel_christian"},
    "christian rock": {"gospel_christian", "rock"},
    "worship": {"gospel_christian"},
    # Ambient / chill
    "dark ambient": {"ambient"},
    "drone": {"ambient"},
    "space ambient": {"ambient"},
    "new age": {"ambient", "chill_lofi"},
    "chillhop": {"chill_lofi", "hip_hop"},
    "chill hop": {"chill_lofi", "hip_hop"},
    # Punk
    "pop punk": {"punk", "pop"},
    "hardcore punk": {"punk"},
    "post-punk": {"punk", "alt_indie"},
    "post punk": {"punk", "alt_indie"},
    "skate punk": {"punk"},
    "crust punk": {"punk"},
    # Other (non-musical) — explicitly empty so they don't pollute tags
    "experimental": set(),
    "noise": set(),
    "spoken word": set(),
    "comedy": set(),
}


def _normalize_one(label: str | None) -> set[str]:
    """Map a single free-form label to canonical tags.

    Order: explicit subgenre map -> explicit genre map -> regex tagger.
    Each step short-circuits if it produced any tags.
    """
    if not label:
        return set()
    key = _key(label)
    if key in _XANO_SUBGENRE_TO_TAGS:
        return set(_XANO_SUBGENRE_TO_TAGS[key])
    if key in _XANO_GENRE_TO_TAGS:
        return set(_XANO_GENRE_TO_TAGS[key])
    return tag_text(label)


def normalize_xano_labels(
    genres: Iterable[str] | None,
    subgenres: Iterable[str] | None,
) -> set[str]:
    """Convert Xano-provided genres+subgenres arrays into canonical tags."""
    out: set[str] = set()
    for label in list(genres or []) + list(subgenres or []):
        out |= _normalize_one(label)
    return out


def normalize_external_labels(labels: Iterable[str] | None) -> set[str]:
    """Map a free-form label list (e.g. Spotify artist_genres) to canonical tags.

    Falls back to the regex tagger for unknown labels so that compound
    Spotify strings like "west coast hip hop" still get tagged.
    """
    out: set[str] = set()
    for label in labels or []:
        out |= _normalize_one(label)
    return out
