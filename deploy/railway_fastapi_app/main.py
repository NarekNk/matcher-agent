from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from supabase import create_client

app = FastAPI(title="Matcher Recommend API", version="1.0.0")

TRACK_ATTR_QUERY_TO_FLAG: dict[str, str] = {
    "track_genre": "--track-genre",
    "track_subgenre": "--track-subgenre",
    "track_mood": "--track-mood",
    "track_activity": "--track-activity",
    "track_language": "--track-language",
    "track_country": "--track-country",
    "track_tempo": "--track-tempo",
}


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _local_root() -> Path:
    return Path(os.getenv("LOCAL_DATA_ROOT", "/app/runtime_data")).resolve()


def _bucket_prefix() -> str:
    return os.getenv("SUPABASE_PREFIX", "").strip().strip("/")


def _required_bucket_keys(include_embeddings: bool = True) -> list[str]:
    keys = [
        "artifacts/model.joblib",
        "artifacts/metadata.json",
        "data/playlists.parquet",
        "data/historical_matches.parquet",
        "output/training_data.csv",
    ]
    if include_embeddings:
        keys.append("data/embeddings/text_embeddings.parquet")
    return keys


def _join_prefix(prefix: str, key: str) -> str:
    return f"{prefix}/{key}" if prefix else key


def _spotify_playlist_id(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"open\.spotify\.com/playlist/([A-Za-z0-9]+)", url)
    if not m:
        return None
    return m.group(1)


def _extract_json_array(raw: str) -> list[dict[str, Any]]:
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "[":
            return json.loads("\n".join(lines[i:]))
    raise ValueError("Could not parse JSON payload from recommend command output.")


def _recommend_command_base() -> list[str]:
    raw = os.getenv("RECOMMEND_COMMAND", "python -m matcher_agent.cli.recommend")
    return shlex.split(raw)


def _download_required_files() -> None:
    supabase_url = _required_env("SUPABASE_URL")
    supabase_key = _required_env("SUPABASE_SERVICE_ROLE_KEY")
    bucket = _required_env("SUPABASE_BUCKET")
    include_embeddings = os.getenv("REQUIRE_EMBEDDINGS", "1").strip() not in {
        "0",
        "false",
        "False",
    }

    root = _local_root()
    prefix = _bucket_prefix()
    client = create_client(supabase_url, supabase_key)

    for key in _required_bucket_keys(include_embeddings=include_embeddings):
        remote_key = _join_prefix(prefix, key)
        local_path = root / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        payload = client.storage.from_(bucket).download(remote_key)
        local_path.write_bytes(payload)


@lru_cache(maxsize=1)
def _playlist_meta() -> dict[str, dict[str, Any]]:
    path = _local_root() / "data" / "playlists.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if df.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        playlist_id = str(row.get("playlist_id") or "").strip()
        if not playlist_id:
            continue
        playlist_url = str(row.get("playlist_url") or "").strip() or None
        out[playlist_id] = {
            "spotify_playlist_id": _spotify_playlist_id(playlist_url),
            "spotify_playlist_url": playlist_url,
            "tier": row.get("tier"),
        }
    return out


def _run_recommend_cli(
    *,
    spotify_track_id: str,
    n: int,
    tracks_csv: str,
    no_genre_filter: bool,
    track_tier: int | None,
    track_attributes: dict[str, list[str]],
) -> list[dict[str, Any]]:
    cmd = _recommend_command_base() + [
        "--spotify-track-id",
        spotify_track_id,
        "--n",
        str(n),
        "--tracks-csv",
        tracks_csv,
    ]
    if no_genre_filter:
        cmd.append("--no-genre-filter")
    if track_tier is not None:
        cmd.extend(["--track-tier", str(track_tier)])
    for query_name, values in track_attributes.items():
        flag = TRACK_ATTR_QUERY_TO_FLAG.get(query_name)
        if not flag:
            continue
        for value in values:
            clean = (value or "").strip()
            if clean:
                cmd.extend([flag, clean])

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    env.setdefault("DATA_DIR", str(_local_root() / "data"))
    env.setdefault("MODEL_DIR", str(_local_root() / "artifacts"))
    env.setdefault("EMBEDDINGS_DIR", str(_local_root() / "data" / "embeddings"))
    env.setdefault("OUTPUT_DIR", str(_local_root() / "output"))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if proc.stdout:
        for line in proc.stdout.splitlines():
            print(f"[recommend-cli] {line}", flush=True)
    if proc.stderr:
        for line in proc.stderr.splitlines():
            print(f"[recommend-cli:err] {line}", flush=True, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"recommend command failed with code {proc.returncode}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return _extract_json_array(proc.stdout)


def _enrich_recommendations(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    meta_by_id = _playlist_meta()
    enriched: list[dict[str, Any]] = []
    for rec in recs:
        playlist_id = str(rec.get("playlist_id") or "").strip()
        meta = meta_by_id.get(
            playlist_id,
            {"spotify_playlist_id": None, "spotify_playlist_url": None, "tier": None},
        )
        merged = dict(rec)
        merged.update(meta)
        enriched.append(merged)
    return enriched


@app.on_event("startup")
def _startup() -> None:
    _download_required_files()
    _playlist_meta.cache_clear()
    _playlist_meta()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/recommend")
def recommend(
    spotify_track_id: str | None = Query(default=None),
    track_id: str | None = Query(default=None),
    n: int = Query(default=5, gt=0),
    tracks_csv: str = Query(default="output/training_data.csv"),
    no_genre_filter: bool = Query(default=False),
    track_tier: int | None = Query(default=None),
    track_genre: list[str] = Query(default=[]),
    track_subgenre: list[str] = Query(default=[]),
    track_mood: list[str] = Query(default=[]),
    track_activity: list[str] = Query(default=[]),
    track_language: list[str] = Query(default=[]),
    track_country: list[str] = Query(default=[]),
    track_tempo: list[str] = Query(default=[]),
) -> dict[str, Any]:
    resolved_track_id = spotify_track_id or track_id
    if not resolved_track_id:
        raise HTTPException(
            status_code=400,
            detail="Missing required query param: spotify_track_id (or track_id).",
        )
    if track_tier is not None and track_tier not in {1, 2, 3, 4}:
        raise HTTPException(
            status_code=400,
            detail="Query param 'track_tier' must be an integer 1, 2, 3, or 4.",
        )

    track_attributes = {
        "track_genre": track_genre,
        "track_subgenre": track_subgenre,
        "track_mood": track_mood,
        "track_activity": track_activity,
        "track_language": track_language,
        "track_country": track_country,
        "track_tempo": track_tempo,
    }

    runtime_tracks_csv = str((_local_root() / tracks_csv).resolve())
    try:
        recs = _run_recommend_cli(
            spotify_track_id=resolved_track_id,
            n=n,
            tracks_csv=runtime_tracks_csv,
            no_genre_filter=no_genre_filter,
            track_tier=track_tier,
            track_attributes=track_attributes,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    recs = _enrich_recommendations(recs)

    return {
        "spotify_track_id": resolved_track_id,
        "n": n,
        "track_tier": track_tier,
        "count": len(recs),
        "track_attributes": {k: v for k, v in track_attributes.items() if v},
        "results": recs,
    }
