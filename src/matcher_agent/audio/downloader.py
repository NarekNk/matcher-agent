from __future__ import annotations

from pathlib import Path

import requests


def download_preview(track_id: str, preview_url: str, audio_dir: Path) -> Path | None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    dest = audio_dir / f"{track_id}.mp3"
    if dest.exists():
        return dest
    try:
        response = requests.get(preview_url, timeout=30)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return dest
    except Exception:
        return None
