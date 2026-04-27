from __future__ import annotations

import asyncio
from typing import Optional

import httpx


async def enrich_with_preview_service(
    track: dict,
    *,
    preview_resolver_url: str | None,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict:
    if track.get("preview_url") or not preview_resolver_url:
        return track
    track_id = track.get("track_id")
    if not track_id:
        return track

    track_url = f"https://open.spotify.com/track/{track_id}"
    async with semaphore:
        try:
            response = await client.get(preview_resolver_url, params={"track_url": track_url}, timeout=15)
            response.raise_for_status()
            new_preview: Optional[str] = None
            content_type = (response.headers.get("content-type") or "").lower()
            if "application/json" in content_type:
                payload = response.json()
                if isinstance(payload, str):
                    new_preview = payload.strip()
                elif isinstance(payload, dict):
                    new_preview = (
                        (payload.get("preview_url") or "").strip()
                        or (payload.get("audio_url") or "").strip()
                        or (payload.get("url") or "").strip()
                    )
            else:
                new_preview = response.text.strip()
            if new_preview:
                track["preview_url"] = new_preview.strip().strip('"').strip("'")
        except Exception:
            return track
    return track


async def enrich_tracks_preview_urls(
    tracks: list[dict],
    preview_resolver_url: str | None,
    *,
    concurrency: int = 12,
) -> list[dict]:
    total = len(tracks)
    if total == 0:
        return tracks
    print(
        f"[PreviewResolver] Enrichment started total={total} "
        f"concurrency={max(1, concurrency)}"
    )

    # If resolver is disabled, keep behavior explicit in logs.
    if not preview_resolver_url:
        print("[PreviewResolver] Resolver URL is not set; skipping remote enrichment.")
        return tracks

    semaphore = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient() as client:
        async def _run_one(idx: int, track: dict) -> tuple[int, dict, bool]:
            had_preview = bool(track.get("preview_url"))
            enriched = await enrich_with_preview_service(
                track=track,
                preview_resolver_url=preview_resolver_url,
                client=client,
                semaphore=semaphore,
            )
            has_preview = bool(enriched.get("preview_url"))
            return idx, enriched, (not had_preview and has_preview)

        tasks = [asyncio.create_task(_run_one(i, t)) for i, t in enumerate(tracks)]
        out: list[dict | None] = [None] * total
        processed = 0
        resolved_new = 0
        already_had = sum(1 for t in tracks if t.get("preview_url"))
        missing_after = 0

        for fut in asyncio.as_completed(tasks):
            idx, track, got_new = await fut
            out[idx] = track
            processed += 1
            if got_new:
                resolved_new += 1
            if not track.get("preview_url"):
                missing_after += 1
            if processed % 25 == 0 or processed == total:
                print(
                    f"[PreviewResolver] Progress {processed}/{total} "
                    f"already_had={already_had} newly_resolved={resolved_new} "
                    f"missing_after={missing_after}"
                )

        print(
            f"[PreviewResolver] Enrichment completed total={total} "
            f"already_had={already_had} newly_resolved={resolved_new} "
            f"missing_after={missing_after}"
        )
        return [t for t in out if t is not None]
