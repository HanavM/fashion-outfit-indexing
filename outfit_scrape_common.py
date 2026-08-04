"""Shared plumbing for every `outfit_dataset/` scraper.

Extracted from reddit_outfit_scraper.py so the browser-driven scraper and
any future source reuse the *same* dedup and validation rules rather than
re-implementing them slightly differently. The rules themselves are
documented in SCRAPING_PROCESS.md, "Real-outfit scraping".

Nothing here writes garment/category/brand fields. A scraper's job is
images plus provenance; labeling happens later, unsupervised, from this
project's own models, into separately namespaced keys.
"""

import hashlib
import io
import time
from pathlib import Path

import imagehash
import requests
from PIL import Image

# Perceptual-hash radius for "same photo". Reposts across these sources are
# endemic and reappear resized under new ids, so URL/id equality does not
# catch them.
PHASH_DISTANCE = 6

MIN_IMAGE_BYTES = 15_000
MIN_IMAGE_SIDE = 320
IMAGE_SLEEP = 0.35

USER_AGENT = (
    "fashion-tests-outfit-research/1.0 "
    "(non-commercial dataset research; contact hanavmw13@gmail.com)"
)
HEADERS = {"User-Agent": USER_AGENT}


def load_seen_hashes(records) -> list:
    """Every phash already in the dataset, for cross-run dedup."""
    hashes = []
    for record in records:
        for value in record.get("phash", []):
            try:
                hashes.append(imagehash.hex_to_hash(value))
            except Exception:
                continue
    return hashes


def is_duplicate(candidate, seen_hashes) -> bool:
    return any(candidate - other <= PHASH_DISTANCE for other in seen_hashes)


def stable_id(value: str, length: int = 16) -> str:
    """Deterministic short id for sources with no native post id.

    Must be stable across runs -- it is the dedup key for records, and a
    random id would re-download the same page forever.
    """
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _fetch_first_usable(candidates, headers, sleep, stats):
    """Try candidate URLs in order, return (url, PIL image) for the first
    that fetches AND passes the size floor.

    Sources that only ever declare a thumbnail in the DOM need a
    higher-resolution URL derived from it, and derived URLs sometimes 404
    (wrong extension, missing original). A ladder is the difference between
    collecting full-size photos and collecting nothing, so a miss on the
    first candidate must fall through rather than drop the image.
    """
    for url in candidates:
        try:
            resp = requests.get(url, headers=headers or HEADERS, timeout=45)
            resp.raise_for_status()
            blob = resp.content
        except Exception as error:
            if url is candidates[-1]:
                print(f"    image fetch failed ({error}): {url[:90]}")
                stats["fetch_failed"] += 1
            continue
        finally:
            time.sleep(sleep)

        if len(blob) < MIN_IMAGE_BYTES:
            if url is candidates[-1]:
                stats["too_small"] += 1
            continue
        try:
            image = Image.open(io.BytesIO(blob)).convert("RGB")
        except Exception:
            if url is candidates[-1]:
                stats["undecodable"] += 1
            continue
        if min(image.size) < MIN_IMAGE_SIDE:
            if url is candidates[-1]:
                stats["too_small"] += 1
            continue
        return url, image
    return None, None


def download_images(dest_dir, urls, seen_hashes, stats, *, max_images,
                    headers=None, sleep=IMAGE_SLEEP):
    """Download up to `max_images` images into `dest_dir`, skipping dupes.

    Each entry of `urls` is either a URL string, or a list of candidate URLs
    for the SAME image tried best-first (see _fetch_first_usable).

    Idempotent: a file already on disk is re-hashed rather than re-fetched,
    so re-running costs no bandwidth and creates no duplicates. Returns
    (paths, urls, phashes) in parallel order.
    """
    dest_dir = Path(dest_dir)
    saved_paths, saved_urls, saved_hashes = [], [], []

    for entry in urls[:max_images]:
        candidates = [entry] if isinstance(entry, str) else list(entry)
        if not candidates:
            continue

        dest = dest_dir / f"image_{len(saved_paths)}.jpg"
        if dest.is_file():
            try:
                phash = imagehash.phash(Image.open(dest).convert("RGB"))
            except Exception:
                dest.unlink(missing_ok=True)
                continue
            saved_paths.append(str(dest))
            saved_urls.append(candidates[0])
            saved_hashes.append(str(phash))
            seen_hashes.append(phash)
            continue

        url, image = _fetch_first_usable(candidates, headers, sleep, stats)
        if image is None:
            continue

        phash = imagehash.phash(image)
        already = [imagehash.hex_to_hash(h) for h in saved_hashes]
        if is_duplicate(phash, seen_hashes) or is_duplicate(phash, already):
            stats["dup"] += 1
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        image.save(dest, "JPEG", quality=92)
        saved_paths.append(str(dest))
        saved_urls.append(url)
        saved_hashes.append(str(phash))
        seen_hashes.append(phash)

    return saved_paths, saved_urls, saved_hashes


def new_stats() -> dict:
    return {"dup": 0, "too_small": 0, "undecodable": 0, "fetch_failed": 0}


def format_stats(stats: dict) -> str:
    return (f"{stats['dup']} perceptual dupes, {stats['too_small']} too small, "
            f"{stats['undecodable']} undecodable, {stats['fetch_failed']} fetch failures")
