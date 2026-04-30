from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import pandas as pd

from matcher_agent.pipeline import build_track_feature_export_sync


class _DummyExecutor:
    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def map(self, fn, iterable):
        return map(fn, iterable)

    def submit(self, fn, *args, **kwargs):
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive in test double
            fut.set_exception(exc)
        return fut


def test_build_features_parallel_path_keeps_cache_and_dedupes(
    tmp_path: Path, monkeypatch
) -> None:
    output_csv = tmp_path / "training_data.csv"
    cached = pd.DataFrame(
        [
            {
                "track_id": "t_cached",
                "track_name": "Cached Song",
                "artist": "Cached Artist",
                "playlist_id": "p1",
                "playlist_name": "Playlist One",
                "preview_url": "cached-preview",
                "bpm": 100.0,
            }
        ]
    )
    cached.to_csv(output_csv, index=False)

    playlist_sources = [
        {
            "playlist_id": "p1",
            "playlist_name": "Playlist One",
            "spotify_playlist_id": "sp1",
        },
        {
            "playlist_id": "p2",
            "playlist_name": "Playlist Two",
            "spotify_playlist_id": "sp2",
        },
    ]

    def _fake_fetch_playlist_tracks(_sp, spotify_playlist_id: str, max_tracks=None):
        if spotify_playlist_id == "sp1":
            return [
                {"track_id": "t_cached", "track_name": "Cached Song", "artist": "Cached Artist"},
                {"track_id": "t_new_ok", "track_name": "New Song", "artist": "New Artist"},
            ]
        return [
            {"track_id": "t_new_ok", "track_name": "New Song", "artist": "New Artist"},
            {"track_id": "t_missing_preview", "track_name": "No Preview", "artist": "No Artist"},
            {"track_id": "t_download_fail", "track_name": "Bad Download", "artist": "Bad Artist"},
        ]

    async def _fake_enrich(tracks, preview_resolver_url, *, concurrency=12):
        out = []
        for t in tracks:
            t = dict(t)
            if t["track_id"] == "t_missing_preview":
                t["preview_url"] = ""
            else:
                t["preview_url"] = f"https://example.com/{t['track_id']}.mp3"
            out.append(t)
        return out

    def _fake_download(track_id: str, preview_url: str, audio_dir: Path):
        if track_id == "t_download_fail":
            return None
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{track_id}.mp3"
        audio_path.write_bytes(b"audio")
        return audio_path

    def _fake_analyze(audio_path: Path):
        return {"bpm": 128.0, "energy": 0.8, "mfcc_1": 1.0}

    monkeypatch.setattr("matcher_agent.pipeline.get_spotify_client", lambda *_: object())
    monkeypatch.setattr("matcher_agent.pipeline.parse_playlist_id", lambda x: x)
    monkeypatch.setattr("matcher_agent.pipeline.fetch_playlist_tracks", _fake_fetch_playlist_tracks)
    monkeypatch.setattr("matcher_agent.pipeline.enrich_tracks_preview_urls", _fake_enrich)
    monkeypatch.setattr("matcher_agent.pipeline.download_preview", _fake_download)
    monkeypatch.setattr("matcher_agent.pipeline._analyze_audio_worker", _fake_analyze)
    monkeypatch.setattr("matcher_agent.pipeline.ProcessPoolExecutor", _DummyExecutor)

    df = build_track_feature_export_sync(
        playlist_sources=playlist_sources,
        spotify_client_id="id",
        spotify_client_secret="secret",
        preview_resolver_url="https://resolver.example",
        audio_dir=tmp_path / "audio",
        output_csv=output_csv,
        max_tracks_per_playlist=30,
        download_concurrency=4,
        analysis_workers=2,
        progress_every=1,
    )

    # cached row + one newly analyzed row only
    assert sorted(df["track_id"].tolist()) == ["t_cached", "t_new_ok"]
    assert len(df["track_id"].unique()) == len(df)
    assert {"bpm", "energy", "mfcc_1", "audio_path"}.issubset(df.columns)
