from __future__ import annotations

import argparse
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


@dataclass(frozen=True)
class UploadItem:
    local_path: Path
    bucket_key: str


def _required_items(root_dir: Path, include_embeddings: bool) -> list[UploadItem]:
    items = [
        UploadItem(root_dir / "artifacts" / "model.joblib", "artifacts/model.joblib"),
        UploadItem(root_dir / "artifacts" / "metadata.json", "artifacts/metadata.json"),
        UploadItem(root_dir / "data" / "playlists.parquet", "data/playlists.parquet"),
        UploadItem(
            root_dir / "data" / "historical_matches.parquet",
            "data/historical_matches.parquet",
        ),
        UploadItem(root_dir / "output" / "training_data.csv", "output/training_data.csv"),
    ]
    if include_embeddings:
        items.append(
            UploadItem(
                root_dir / "data" / "embeddings" / "text_embeddings.parquet",
                "data/embeddings/text_embeddings.parquet",
            )
        )
    return items


def _upload_file(*, supabase_url: str, supabase_key: str, bucket: str, item: UploadItem) -> None:
    if not item.local_path.exists():
        raise FileNotFoundError(f"Required file not found: {item.local_path}")

    guessed_type, _ = mimetypes.guess_type(str(item.local_path))
    content_type = guessed_type or "application/octet-stream"
    with item.local_path.open("rb") as fh:
        payload = fh.read()

    client = create_client(supabase_url, supabase_key)
    client.storage.from_(bucket).upload(
        item.bucket_key,
        payload,
        {"content-type": content_type, "upsert": "true"},
    )


def _manifest(items: Iterable[UploadItem]) -> list[dict[str, str]]:
    return [
        {"local_path": str(item.local_path), "bucket_key": item.bucket_key}
        for item in items
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload inference artifacts required by recommend API to Supabase Storage."
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Project root containing artifacts/, data/, and output/ directories.",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("SUPABASE_BUCKET"),
        help="Supabase bucket name (or set SUPABASE_BUCKET).",
    )
    parser.add_argument(
        "--supabase-url",
        default=os.getenv("SUPABASE_URL"),
        help="Supabase project URL (or set SUPABASE_URL).",
    )
    parser.add_argument(
        "--supabase-key",
        default=os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_API_KEY"),
        help="Supabase service role key/api key (or set SUPABASE_SERVICE_ROLE_KEY).",
    )
    parser.add_argument(
        "--exclude-embeddings",
        action="store_true",
        help="Skip uploading data/embeddings/text_embeddings.parquet.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print upload manifest and exit without uploading.",
    )
    args = parser.parse_args()

    if not args.bucket:
        raise ValueError("Missing bucket name. Set --bucket or SUPABASE_BUCKET.")
    if not args.supabase_url:
        raise ValueError("Missing Supabase URL. Set --supabase-url or SUPABASE_URL.")
    if not args.supabase_key:
        raise ValueError(
            "Missing Supabase key. Set --supabase-key or SUPABASE_SERVICE_ROLE_KEY."
        )

    root_dir = Path(args.root_dir).resolve()
    items = _required_items(root_dir, include_embeddings=not args.exclude_embeddings)

    print("[UploadSupabase] Required artifact manifest:")
    print(json.dumps(_manifest(items), indent=2))
    if args.dry_run:
        print("[UploadSupabase] Dry run complete. No files uploaded.")
        return

    for item in items:
        print(f"[UploadSupabase] Uploading {item.local_path} -> {args.bucket}/{item.bucket_key}")
        _upload_file(
            supabase_url=args.supabase_url,
            supabase_key=args.supabase_key,
            bucket=args.bucket,
            item=item,
        )
    print("[UploadSupabase] Completed successfully.")


if __name__ == "__main__":
    main()
