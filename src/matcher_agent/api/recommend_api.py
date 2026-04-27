from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from matcher_agent.clients.spotify_client import parse_playlist_id
from matcher_agent.config import get_settings
from matcher_agent.data.repository import DataRepository
from matcher_agent.storage.parquet_store import ParquetStore


def _extract_json_array(raw: str) -> list[dict]:
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "[":
            payload = "\n".join(lines[i:])
            return json.loads(payload)
    raise ValueError("Could not parse JSON payload from recommend command output.")


def _run_recommend_cli(
    *,
    spotify_track_id: str,
    n: int,
    tracks_csv: str,
    no_genre_filter: bool,
) -> list[dict]:
    cmd = [
        sys.executable,
        "-m",
        "matcher_agent.cli.recommend",
        "--spotify-track-id",
        spotify_track_id,
        "--n",
        str(n),
        "--tracks-csv",
        tracks_csv,
    ]
    if no_genre_filter:
        cmd.append("--no-genre-filter")

    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH") or "src"
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"recommend CLI failed with code {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return _extract_json_array(proc.stdout)


@lru_cache(maxsize=1)
def _playlist_spotify_meta() -> dict[str, dict[str, str | None]]:
    """Map internal playlist_id -> spotify identifiers/urls from local parquet."""
    settings = get_settings()
    repo = DataRepository(ParquetStore(settings.data_dir))
    playlists_df = repo.load_playlists()
    if playlists_df.empty:
        return {}

    out: dict[str, dict[str, str | None]] = {}
    for _, row in playlists_df.iterrows():
        pid = str(row.get("playlist_id") or "").strip()
        if not pid:
            continue
        playlist_url = str(row.get("playlist_url") or "").strip() or None
        spotify_id: str | None = None
        if playlist_url:
            try:
                spotify_id = parse_playlist_id(playlist_url)
            except Exception:
                spotify_id = None
        out[pid] = {
            "spotify_playlist_id": spotify_id,
            "spotify_playlist_url": playlist_url,
        }
    return out


def _enrich_recommendations_with_spotify_meta(recs: list[dict]) -> list[dict]:
    meta_by_id = _playlist_spotify_meta()
    enriched: list[dict] = []
    for rec in recs:
        pid = str(rec.get("playlist_id") or "")
        meta = meta_by_id.get(
            pid,
            {"spotify_playlist_id": None, "spotify_playlist_url": None},
        )
        merged = dict(rec)
        merged.update(meta)
        enriched.append(merged)
    return enriched


class RecommendHandler(BaseHTTPRequestHandler):
    server_version = "MatcherRecommendAPI/1.0"

    def _write_json(self, code: int, payload: dict | list) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        if parsed.path != "/recommend":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        query = parse_qs(parsed.query)
        spotify_track_id = (
            (query.get("spotify_track_id") or [None])[0]
            or (query.get("track_id") or [None])[0]
        )
        n_raw = (query.get("n") or ["5"])[0]
        no_genre_filter = (query.get("no_genre_filter") or ["0"])[0] in {
            "1",
            "true",
            "True",
        }
        tracks_csv = (query.get("tracks_csv") or ["output/training_data.csv"])[0]

        if not spotify_track_id:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Missing required query param: spotify_track_id (or track_id)."},
            )
            return
        try:
            n = int(n_raw)
            if n <= 0:
                raise ValueError
        except ValueError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Query param 'n' must be a positive integer."})
            return

        try:
            recs = _run_recommend_cli(
                spotify_track_id=spotify_track_id,
                n=n,
                tracks_csv=tracks_csv,
                no_genre_filter=no_genre_filter,
            )
            recs = _enrich_recommendations_with_spotify_meta(recs)
        except Exception as exc:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        self._write_json(
            HTTPStatus.OK,
            {
                "spotify_track_id": spotify_track_id,
                "n": n,
                "count": len(recs),
                "results": recs,
            },
        )

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Keep logs concise and readable in terminal.
        print(f"[RecommendAPI] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP API wrapper around matcher_agent.cli.recommend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RecommendHandler)
    print(f"[RecommendAPI] Listening on http://{args.host}:{args.port}")
    print("[RecommendAPI] GET /recommend?spotify_track_id=<id>&n=<int>")
    print("[RecommendAPI] GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("[RecommendAPI] Server stopped.")


if __name__ == "__main__":
    main()
