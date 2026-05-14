from __future__ import annotations

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from matcher_agent.audio.analyzer import analyze_audio
from matcher_agent.audio.downloader import download_preview
from matcher_agent.clients.preview_resolver_client import enrich_tracks_preview_urls
from matcher_agent.clients.spotify_client import fetch_playlist_tracks, get_spotify_client, parse_playlist_id
from matcher_agent.features.playlist_profiles import ensure_audio_columns


def _resolve_analysis_workers(analysis_workers: int | None) -> int:
    if analysis_workers is not None and analysis_workers > 0:
        return analysis_workers
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


async def _download_tracks_parallel(
    tracks: list[dict],
    *,
    audio_dir: Path,
    download_concurrency: int,
    progress_every: int = 25,
) -> list[tuple[dict, Path | None]]:
    semaphore = asyncio.Semaphore(max(1, download_concurrency))
    progress_every = max(1, progress_every)
    total = len(tracks)
    processed = 0
    succeeded = 0
    failed = 0
    skipped_no_preview = 0
    progress_lock = asyncio.Lock()

    async def _download_one(track: dict) -> tuple[dict, Path | None]:
        nonlocal processed, succeeded, failed, skipped_no_preview
        if not track.get("preview_url"):
            audio_path: Path | None = None
            status = "no_preview"
        else:
            async with semaphore:
                audio_path = await asyncio.to_thread(
                    download_preview, track["track_id"], track["preview_url"], audio_dir
                )
            status = "ok" if audio_path else "failed"

        async with progress_lock:
            processed += 1
            if status == "ok":
                succeeded += 1
            elif status == "failed":
                failed += 1
            else:
                skipped_no_preview += 1
            if processed % progress_every == 0 or processed == total:
                print(
                    f"[Features] Downloads {processed}/{total} "
                    f"succeeded={succeeded} failed={failed} no_preview={skipped_no_preview}"
                )
        return track, audio_path

    tasks = [_download_one(track) for track in tracks]
    return await asyncio.gather(*tasks)


def _analyze_audio_worker(audio_path: Path) -> dict | None:
    return analyze_audio(audio_path)


def _analyze_with_progress(
    ready_for_analysis: list[tuple[dict, Path]],
    *,
    workers: int,
    progress_every: int,
) -> list[tuple[dict, Path, dict | None]]:
    """Run process-pool audio analysis with true in-flight progress logs."""
    results: list[tuple[dict, Path, dict | None]] = []
    total = len(ready_for_analysis)
    if total == 0:
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(_analyze_audio_worker, audio_path): (track, audio_path)
            for track, audio_path in ready_for_analysis
        }
        processed = 0
        for future in as_completed(future_to_item):
            processed += 1
            track, audio_path = future_to_item[future]
            try:
                features = future.result()
            except Exception:
                features = None
            results.append((track, audio_path, features))
            if processed % progress_every == 0 or processed == total:
                print(
                    f"[Features] Analysis {processed}/{total} "
                    f"completed={len(results)}"
                )
    return results


async def build_track_feature_export(
    playlist_sources: list[dict[str, str]],
    *,
    spotify_client_id: str,
    spotify_client_secret: str,
    preview_resolver_url: str | None,
    audio_dir: Path,
    output_csv: Path,
    max_tracks_per_playlist: int | None = 100,
    download_concurrency: int = 16,
    analysis_workers: int | None = None,
    progress_every: int = 25,
) -> pd.DataFrame:
    progress_every = max(1, progress_every)
    print(
        f"[Features] Starting build for {len(playlist_sources)} playlists "
        f"(max_tracks_per_playlist={max_tracks_per_playlist})."
    )
    cached_by_track_id: dict[str, dict] = {}
    if output_csv.exists():
        cached_df = pd.read_csv(output_csv)
        cached_df = ensure_audio_columns(cached_df)
        if "track_id" in cached_df.columns:
            for _, row in cached_df.drop_duplicates(subset=["track_id"], keep="last").iterrows():
                cached_by_track_id[str(row["track_id"])] = row.to_dict()
    print(f"[Features] Cache entries available: {len(cached_by_track_id)}")

    sp = get_spotify_client(spotify_client_id, spotify_client_secret)
    rows: list[dict] = []
    n_skipped_playlists = 0
    for idx, source in enumerate(playlist_sources, 1):
        playlist_id = str(source["playlist_id"])
        playlist_name = source["playlist_name"]
        try:
            spotify_playlist_id = parse_playlist_id(source["spotify_playlist_id"])
            tracks = fetch_playlist_tracks(sp, spotify_playlist_id, max_tracks=max_tracks_per_playlist)
        except Exception as exc:
            n_skipped_playlists += 1
            print(
                f"[Features] Playlist {idx}/{len(playlist_sources)} '{playlist_name}' "
                f"SKIPPED (fetch failed: {exc})"
            )
            continue
        print(f"[Features] Playlist {idx}/{len(playlist_sources)} '{playlist_name}' -> {len(tracks)} tracks")
        for track in tracks:
            track["playlist_id"] = playlist_id
            track["playlist_name"] = playlist_name
            rows.append(track)
    if n_skipped_playlists:
        print(f"[Features] Skipped {n_skipped_playlists} playlists due to fetch errors.")
    print(f"[Features] Raw fetched rows: {len(rows)}")

    unique_tracks: list[dict] = []
    seen_ids: set[str] = set()
    for track in rows:
        tid = track.get("track_id")
        if tid and tid not in seen_ids:
            seen_ids.add(tid)
            unique_tracks.append(track)
    print(f"[Features] Unique tracks discovered: {len(unique_tracks)}")

    reused_rows: list[dict] = []
    tracks_to_analyze: list[dict] = []
    for track in unique_tracks:
        tid = track["track_id"]
        cached = cached_by_track_id.get(tid)
        if cached:
            merged = dict(cached)
            # Prefer freshly-fetched popularity (it's the most up-to-date
            # signal) but fall back to the cached value so previously
            # populated rows aren't wiped on a subsequent run that
            # happened to fetch a track without popularity.
            popularity = track.get("popularity")
            if popularity is None:
                popularity = cached.get("popularity")
            merged.update(
                {
                    "track_id": tid,
                    "track_name": track.get("track_name"),
                    "artist": track.get("artist"),
                    "album": track.get("album"),
                    "duration_ms": track.get("duration_ms"),
                    "playlist_id": track.get("playlist_id"),
                    "playlist_name": track.get("playlist_name"),
                    "preview_url": track.get("preview_url") or cached.get("preview_url"),
                    "popularity": popularity,
                }
            )
            reused_rows.append(merged)
        else:
            tracks_to_analyze.append(track)
    print(
        f"[Features] Reused from cache: {len(reused_rows)} | "
        f"To analyze now: {len(tracks_to_analyze)}"
    )

    tracks_to_analyze = await enrich_tracks_preview_urls(
        tracks_to_analyze,
        preview_resolver_url,
        concurrency=download_concurrency,
    )
    print("[Features] Starting parallel preview downloads.")
    downloaded = await _download_tracks_parallel(
        tracks_to_analyze,
        audio_dir=audio_dir,
        download_concurrency=download_concurrency,
        progress_every=progress_every,
    )

    analyzed_rows: list[dict] = []
    if reused_rows:
        analyzed_rows.extend(reused_rows)
    total_to_analyze = len(downloaded)
    ready_for_analysis: list[tuple[dict, Path]] = []
    for idx, (track, audio_path) in enumerate(downloaded, 1):
        if not track.get("preview_url"):
            if idx % progress_every == 0 or idx == total_to_analyze:
                print(f"[Features] Progress {idx}/{total_to_analyze} (no preview; skipped)")
            continue
        if not audio_path:
            if idx % progress_every == 0 or idx == total_to_analyze:
                print(f"[Features] Progress {idx}/{total_to_analyze} (download failed; skipped)")
            continue
        ready_for_analysis.append((track, audio_path))

    workers = _resolve_analysis_workers(analysis_workers)
    print(
        f"[Features] Starting parallel audio analysis for {len(ready_for_analysis)} tracks "
        f"(workers={workers})."
    )
    analyzed_items = await asyncio.to_thread(
        _analyze_with_progress,
        ready_for_analysis,
        workers=workers,
        progress_every=progress_every,
    )
    analyzed_count = 0
    for track, audio_path, features in analyzed_items:
        if not features:
            continue
        analyzed_rows.append({**track, **features, "audio_path": str(audio_path)})
        analyzed_count += 1
    print(
        f"[Features] Analysis finished total={len(ready_for_analysis)} "
        f"succeeded={analyzed_count} failed={len(ready_for_analysis) - analyzed_count}"
    )

    df = pd.DataFrame(analyzed_rows)
    if "track_id" in df.columns:
        df = df.sort_values(by="track_id", kind="stable").drop_duplicates(
            subset=["track_id"], keep="last"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[Features] Completed rows={len(df)} output_csv={output_csv}")
    return df


def build_track_feature_export_sync(*args, **kwargs) -> pd.DataFrame:
    return asyncio.run(build_track_feature_export(*args, **kwargs))


def playlist_sources_from_parquet(playlists_df: pd.DataFrame) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for _, row in playlists_df.iterrows():
        playlist_name = str(row.get("playlist_name") or "").strip()
        playlist_url = str(row.get("playlist_url") or "").strip()
        playlist_id = str(row.get("playlist_id") or "").strip()
        if not playlist_name or not playlist_id:
            continue
        if playlist_url:
            spotify_playlist_id = playlist_url
        else:
            continue
        out.append(
            {
                "playlist_id": playlist_id,
                "playlist_name": playlist_name,
                "spotify_playlist_id": spotify_playlist_id,
            }
        )
    return out


def limit_playlist_sources(
    playlist_sources: list[dict[str, str]],
    *,
    max_playlists: int | None,
) -> list[dict[str, str]]:
    if max_playlists is None or max_playlists <= 0:
        return playlist_sources
    return playlist_sources[:max_playlists]
