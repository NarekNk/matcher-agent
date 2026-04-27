from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency in tests
    SentenceTransformer = None  # type: ignore[assignment]


def _stable_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split()).strip().lower()


class TextEmbedder:
    """Sentence-transformer text embedder with parquet-backed caching.

    The cache is keyed by the SHA1 of the normalized text so identical strings
    are never re-embedded. Cache rows live in a single parquet file; embeddings
    are stored as a numpy float32 list per row.
    """

    def __init__(
        self,
        cache_path: Path,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.cache_path = cache_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: SentenceTransformer | None = None
        self._cache_df: pd.DataFrame | None = None

    @property
    def dim(self) -> int:
        # all-MiniLM-L6-v2 yields 384-d embeddings; resolved lazily once model loads
        if self._model is not None:
            return int(self._model.get_sentence_embedding_dimension())
        return 384

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is not installed; install it to compute embeddings."
                )
            print(f"[TextEmbedder] Loading model '{self.model_name}'.")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _load_cache(self) -> pd.DataFrame:
        if self._cache_df is not None:
            return self._cache_df
        if self.cache_path.exists():
            df = pd.read_parquet(self.cache_path)
            if "embedding" in df.columns:
                df["embedding"] = df["embedding"].map(
                    lambda v: np.asarray(v, dtype=np.float32)
                )
        else:
            df = pd.DataFrame(columns=["text_hash", "model_name", "embedding"])
        self._cache_df = df
        return df

    def _persist_cache(self) -> None:
        assert self._cache_df is not None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        out = self._cache_df.copy()
        if "embedding" in out.columns:
            out["embedding"] = out["embedding"].map(
                lambda v: np.asarray(v, dtype=np.float32).tolist()
            )
        tmp = self.cache_path.with_suffix(".parquet.tmp")
        out.to_parquet(tmp, index=False)
        tmp.replace(self.cache_path)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Return an (N, dim) array of embeddings for the given texts.

        Uses cache when possible; only encodes the missing strings, then
        upserts them to the cache file.
        """
        texts_list = [_normalize_text(t) for t in texts]
        if not texts_list:
            return np.zeros((0, self.dim), dtype=np.float32)

        cache_df = self._load_cache()
        cached_lookup: dict[str, np.ndarray] = {}
        if not cache_df.empty:
            relevant = cache_df[cache_df["model_name"] == self.model_name]
            for _, row in relevant.iterrows():
                cached_lookup[str(row["text_hash"])] = np.asarray(
                    row["embedding"], dtype=np.float32
                )

        hashes = [_stable_text_hash(t) for t in texts_list]
        missing_indices = [i for i, h in enumerate(hashes) if h not in cached_lookup]
        if missing_indices:
            print(
                f"[TextEmbedder] Cache hit "
                f"{len(texts_list) - len(missing_indices)}/{len(texts_list)}; "
                f"encoding {len(missing_indices)} new texts."
            )
            model = self._ensure_model()
            new_texts = [texts_list[i] for i in missing_indices]
            new_embeds = model.encode(
                new_texts,
                batch_size=self.batch_size,
                show_progress_bar=len(new_texts) > 256,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            new_rows = pd.DataFrame(
                {
                    "text_hash": [hashes[i] for i in missing_indices],
                    "model_name": [self.model_name] * len(missing_indices),
                    "embedding": [emb for emb in new_embeds],
                }
            )
            for h, emb in zip(new_rows["text_hash"], new_rows["embedding"]):
                cached_lookup[str(h)] = np.asarray(emb, dtype=np.float32)
            self._cache_df = pd.concat([cache_df, new_rows], ignore_index=True)
            self._cache_df = self._cache_df.drop_duplicates(
                subset=["text_hash", "model_name"], keep="last"
            )
            self._persist_cache()
        else:
            print(f"[TextEmbedder] Cache hit {len(texts_list)}/{len(texts_list)}; no new encoding.")

        out = np.vstack([cached_lookup[h] for h in hashes]).astype(np.float32)
        return out


def embed_dataframe(
    df: pd.DataFrame,
    text_col: str,
    *,
    embedder: TextEmbedder,
    out_prefix: str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with an `<out_prefix>_emb` array column added."""
    out_prefix = out_prefix or text_col
    embeddings = embedder.encode(df[text_col].fillna("").astype(str).tolist())
    out = df.copy()
    out[f"{out_prefix}_emb"] = list(embeddings)
    return out
