# Track → Playlist Matcher Agent

Given a Spotify track and an integer `n`, this agent returns the top-`n` playlists from your Xano catalog that are most likely to **accept** the track when pitched to a curator.

It is trained on your historical pitch outcomes (`accepted` / `closed` / `declined`) and ranks playlists by **track-playlist fit**, not by playlist popularity.

---

## Why the previous model was wrong (and what changed)

The first iteration learned that some playlists accept almost everything, so it kept recommending the same handful of "Made for You" / "TikTok Popular Songs" playlists for every track — pop, hip-hop, country, EDM all got the same answer.

We fixed that with a redesign focused on **high-precision genre matching**:

1. **Dropped popularity-bias features.** `playlist_acceptance_rate`, `playlist_accepted_track_count`, and `playlist_declined_track_count` are no longer model inputs. They were teaching the model to predict "this playlist accepts a lot" instead of "this playlist accepts *this kind of track*".
2. **Added pairwise similarity features.** Every (track, playlist) pair is now scored by:
   - **Semantic similarity** between the track text embedding and the playlist's *semantic centroid* (a blend of its title/description embedding and the mean embedding of tracks it has historically accepted).
   - **Title text similarity** between track and playlist text.
   - **Audio-centroid cosine + L2** between the track's audio features and the playlist's accepted-track audio centroid.
   - **Audio z-score** of the track relative to the playlist's accepted distribution.
   - **Genre overlap, Jaccard, and conflict flag** from canonical-tag sets built from (a) curator-tagged Xano `genres`/`subgenres`, (b) Spotify artist genres at inference, and (c) a regex pass over track/playlist text.
   - **Track-vs-playlist popularity fit**: `track_popularity_norm`, `popularity_diff_norm`, `popularity_zscore`, `popularity_available`. The model can learn that low-popularity playlists prefer low-popularity tracks without using raw playlist popularity as a confounder. Stats are computed across the playlist's *accepted* tracks only.
3. **Curator-tagged genres from Xano.** Every playlist row in Xano now exposes `genres` (e.g. `"Hip-Hop"`) and `subgenres` (e.g. `"Trap"`, `"Drill"`, `"Lo-fi Hip-Hop"`) string arrays. These are the most authoritative genre signal we have and are mapped to a canonical tag set by `features/genre_normalizer.py` (which also handles Spotify-style strings like `"west coast rap"`, `"drum and bass"`, `"neo soul"`).
4. **Spotify artist genres + popularity at inference.** When you call `recommend` with a Spotify track ID, we fetch the artists' genres AND the track's `popularity` (0-100). Genres are normalized to canonical tags; popularity flows into the popularity-fit features.
5. **Hard genre-conflict filter.** Predictions for playlists that conflict with the track's genre (e.g. country track → hip-hop playlist) are multiplied by `0.05`, pushing them out of the top-`n`.
6. **Leakage-free training.** Train/test split is grouped by `track_id` *before* building any playlist profile, so no test track ever influences the playlist features used to score it.
7. **Realistic evaluation.** Each held-out track is scored against the **full catalog** of 1,668 playlists (not just the few it was historically pitched to).

---

## Project layout

```
src/matcher_agent/
  cli/                # Command-line entrypoints
    sync_xano.py        # Pull playlists + match history from Xano → Parquet
    build_features.py   # Resolve previews, download audio, extract Essentia features
    build_embeddings.py # Pre-compute & cache sentence-transformer text embeddings
    train.py            # Train the calibrated GBM ranker
    recommend.py        # Top-N playlists for a Spotify track ID
    plot_results.py     # Charts of training metrics
  clients/
    xano_client.py      # Paginated Xano REST fetch + normalization
    spotify_client.py   # Spotify metadata + artist-genre lookup
  data/                 # Parquet IO & label mapping
  embeddings/
    text_embedder.py    # Sentence-Transformer with on-disk Parquet cache
  features/
    feature_builder.py    # Pairwise (track,playlist) feature matrix incl. popularity-fit
    playlist_profiles.py  # Per-playlist semantic/audio centroids, tags, popularity stats
    genre_tagger.py       # Rule-based genre regex + conflict groups
    genre_normalizer.py   # Xano genres/subgenres + Spotify labels → canonical tags
    audio_features.py     # Essentia BPM/MFCC/loudness/etc.
  inference/
    candidates.py     # Candidate playlist pool
    service.py        # MatcherService.recommend_playlists(track, n)
  training/
    dataset.py        # Build training bundle (train-only profiles)
    train_ranker.py   # GroupShuffleSplit, calibrated GBM, full-catalog eval
    metrics.py        # Hit@K / Precision@K / Recall@K / MRR
scripts/
  qualitative_demo.py   # Sanity check: pop/hip-hop/country/EDM/latin/folk
tests/
```

---

## Setup

```bash
python3 -m venv music-env
source music-env/bin/activate
pip install -r requirements.txt
```

Create a `.env`:

```bash
XANO_BASE_URL=https://...
XANO_API_KEY=...
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
PREVIEW_RESOLVER_URL=https://...
# Optional limits for fast iteration
MAX_PLAYLISTS=500
MAX_TRACKS_PER_PLAYLIST=30
XANO_HISTORICAL_MAX_PAGES=
# Optional build_features performance knobs
FEATURE_DOWNLOAD_CONCURRENCY=16
FEATURE_ANALYSIS_WORKERS=
FEATURE_PROGRESS_EVERY=25
# Genre & negative sampling knobs
HARD_GENRE_FILTER=true
NEGATIVE_SAMPLE_RATIO=0.0
SEMANTIC_BLEND=0.5
```

---

## Pipeline (run in this order)

```bash
# 1. Pull data from Xano (incremental; safe to re-run)
PYTHONPATH=src music-env/bin/python -m matcher_agent.cli.sync_xano

# 2. Resolve preview URLs, download audio, extract Essentia features
# (parallel network + process-based audio analysis)
PYTHONPATH=src music-env/bin/python -m matcher_agent.cli.build_features \
  --download-concurrency 20 --analysis-workers 8 --progress-every 25

# 3. Pre-compute & cache text embeddings (playlists + historical tracks)
PYTHONPATH=src music-env/bin/python -m matcher_agent.cli.build_embeddings

# 4. Train the ranker (writes artifacts/ + output/feature_importance.csv)
PYTHONPATH=src music-env/bin/python -m matcher_agent.cli.train

# 5. Get top-N playlists for any Spotify track ID
PYTHONPATH=src music-env/bin/python -m matcher_agent.cli.recommend \
  --spotify-track-id 0VjIjW4GlUZAMYd2vXMi3b --n 10
```

To bypass the genre conflict filter:

```bash
... -m matcher_agent.cli.recommend --spotify-track-id ... --n 10 --no-genre-filter
```

---

## Simple GET API

Start the API server:

```bash
PYTHONPATH=src music-env/bin/python -m matcher_agent.api.recommend_api --host 127.0.0.1 --port 8080
```

Call it:

```bash
curl "http://127.0.0.1:8080/recommend?spotify_track_id=0VjIjW4GlUZAMYd2vXMi3b&n=10"
```

Optional query params:
- `track_id` (alias for `spotify_track_id`)
- `tracks_csv` (default: `output/training_data.csv`)
- `no_genre_filter=1` (disables hard genre conflict filter)

Response fields per result now include:
- `playlist_id` (internal id)
- `spotify_playlist_id`
- `spotify_playlist_url`
- `playlist_name`
- `acceptance_probability`
- `rank`

Health endpoint:

```bash
curl "http://127.0.0.1:8080/health"
```

---

## Qualitative sanity check

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src music-env/bin/python scripts/qualitative_demo.py
```

This runs six archetype tracks (pop / hip-hop / country / EDM / latin / indie folk) through the live recommender. After the redesign, each archetype gets genre-appropriate playlists in its top-5 (e.g. hip-hop → "TRAP GOD", "Hip Hop & Rap vibes"; country → "country radio"; EDM → "Tech House", "EDM HOUSE").

---

## Notes on metrics

- **AUC-PR / AUC-ROC** are computed on the held-out historical pitches.
- **Hit@K / Precision@K / Recall@K / MRR** are computed against the **full 1,668-playlist catalog** per held-out track. They are intentionally low because each test track had only 1–3 historical pitches in the catalog — recommending other genre-correct playlists doesn't count even when it's the right answer. Treat them as a lower bound and use `qualitative_demo.py` as the deployment-relevance check.
- **Feature importances** are written to `output/feature_importance.csv`. After the fix, `semantic_similarity` and `title_text_similarity` dominate (~92%), with audio, genre, and popularity-fit features filling the rest. No `*_id` or `playlist_acceptance_rate` features — that's the point.

---

## Re-syncing after these changes

The Xano genre/subgenre arrays and the Spotify popularity field are NEW data fields. To pick them up on existing local snapshots:

```bash
# 1. Refresh the playlists table so each row has genres / subgenres.
#    Use --full-refresh because incremental sync only updates rows that
#    Xano marked as updated; old rows otherwise keep their stale schema.
PYTHONPATH=src music-env/bin/python -m matcher_agent.cli.sync_xano --full-refresh

# 2. Re-run build_features. Cached audio analysis is reused; only the
#    `popularity` column is enriched (one extra Spotify call per playlist
#    fetch, no audio re-download).
PYTHONPATH=src music-env/bin/python -m matcher_agent.cli.build_features

# 3. Re-train. The model contract now includes 4 popularity features so
#    old artifacts are incompatible.
PYTHONPATH=src music-env/bin/python -m matcher_agent.cli.train
```

If you skip step 1 the model still trains, but playlists fall back to regex-only tags (less precise). If you skip step 2 the popularity-fit features will be all zeros for every track.
