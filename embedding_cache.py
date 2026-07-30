"""Shared image/text embedding cache, keyed by model name, stored on Drive.

Layout under DATASET_ROOT/embeddings_cache/{model_name}/:
    image_embeddings.npy   float32 array, shape (N, D)
    image_paths.json       list[str], index-aligned with image_embeddings
    text_embeddings.npy    float32 array, shape (M, D)
    texts.json             list[str], index-aligned with text_embeddings

Encoding scripts should call `load_cache` / `save_cache` and only encode the
subset of images/texts missing from the cache -- avoids re-touching Drive
(slow FUSE reads) and re-running the GPU forward pass for anything already
encoded in a prior run. Keyed by exact image path / exact text string, so a
new scrape batch or a caption edit is picked up automatically (mismatched
entries just aren't found in the cache and get re-encoded).
"""

from pathlib import Path
import json

import numpy as np


def cache_dir(dataset_root: Path, model_name: str) -> Path:
    d = Path(dataset_root) / "embeddings_cache" / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_cache(dataset_root: Path, model_name: str, kind: str):
    """kind is 'image' or 'text'. Returns (embeddings: np.ndarray[N,D] or None, keys: list[str])."""

    d = cache_dir(dataset_root, model_name)
    emb_path = d / f"{kind}_embeddings.npy"
    keys_path = d / ("image_paths.json" if kind == "image" else "texts.json")

    if not emb_path.exists() or not keys_path.exists():
        return None, []

    embeddings = np.load(emb_path)
    keys = json.loads(keys_path.read_text(encoding="utf-8"))

    if len(keys) != embeddings.shape[0]:
        # Corrupt/partial cache from an interrupted run -- ignore it rather
        # than trusting misaligned data.
        return None, []

    return embeddings, keys


def save_cache(dataset_root: Path, model_name: str, kind: str, embeddings, keys: list):
    """Merges with whatever is already on disk (keyed by path/text), then writes back.

    Safe to call from multiple runs over time (not concurrent processes in the
    same instant -- same caveat as dataset_utils.save_records_safe).
    """

    d = cache_dir(dataset_root, model_name)
    emb_path = d / f"{kind}_embeddings.npy"
    keys_path = d / ("image_paths.json" if kind == "image" else "texts.json")

    existing_embeddings, existing_keys = load_cache(dataset_root, model_name, kind)

    merged = {}
    if existing_embeddings is not None:
        for key, vec in zip(existing_keys, existing_embeddings):
            merged[key] = vec

    new_embeddings = np.asarray(embeddings)
    for key, vec in zip(keys, new_embeddings):
        merged[key] = vec

    merged_keys = list(merged.keys())
    merged_embeddings = np.stack([merged[k] for k in merged_keys]).astype(np.float32)

    np.save(emb_path, merged_embeddings)
    keys_path.write_text(json.dumps(merged_keys), encoding="utf-8")

    return merged_embeddings, merged_keys


def split_cached(keys_wanted: list, cached_keys: list):
    """Returns (indices_of_keys_wanted_that_are_cached, cached_key_to_row, missing_keys)."""

    cached_row = {key: i for i, key in enumerate(cached_keys)}
    hit_positions = []
    hit_rows = []
    missing = []

    for i, key in enumerate(keys_wanted):
        if key in cached_row:
            hit_positions.append(i)
            hit_rows.append(cached_row[key])
        else:
            missing.append(key)

    return hit_positions, hit_rows, missing
