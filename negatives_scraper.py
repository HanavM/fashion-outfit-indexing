"""Collect REAL non-clothing photos: the negative half of a garment-detection gate.

`hierarchical_retrieval_pipeline.py` will confidently name a product for any
image it is handed, including a chair (see docs/eval_log.md, the open-set
rejection row -- false-accept ~68% at any usable false-reject rate). The
mitigation is a SigLIP2 zero-shot garment gate in FRONT of the pipeline, but a
gate is only as honest as the negatives it was calibrated on. Calibrating
against synthetic negatives (bar charts, solid colours, text blocks) gives
AUROC 1.0000, which is a statement about how unlike a photograph those images
are, not about how well the gate works on the thing users actually do: point a
phone at a real object.

So this collects real photographs of real things -- furniture, cars, food,
landscapes, buildings, electronics, animals, plants, books, kitchenware, tools,
streets, screenshots -- from two keyless, crawl-friendly sources.

HARD RULE, and the reason several obvious queries are worded oddly below: a
negative must contain NO prominent clothing. A photo of a person in a jacket is
a POSITIVE. Getting that wrong does not just add noise, it drags the threshold
in the direction that makes the gate reject real garments. Queries therefore
prefer objects and empty scenes over anything person-centred, and
`review_negatives.py` triages what slips through.

Storage is a NEW tree (negatives_dataset/) and this script never touches
apparel_dataset/ or outfit_dataset/.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import requests

from outfit_scrape_common import (
    MAX_IMAGE_SIDE,
    _fetch_first_usable,
    format_stats,
    is_duplicate,
    load_seen_hashes,
    new_stats,
    stable_id,
)

DATASET_DIR = Path(__file__).resolve().parent / "negatives_dataset"
METADATA_PATH = DATASET_DIR / "metadata.json"

USER_AGENT = (
    "fashion-tests-negatives-research/1.0 "
    "(non-commercial dataset research; contact hanavmw13@gmail.com)"
)
HEADERS = {"User-Agent": USER_AGENT}

# Polite by default. Both sources invite crawling; neither invites hammering.
API_SLEEP = 1.0
IMAGE_SLEEP = 0.4

# What a person actually points a phone at. Deliberately NOT abstract images.
# Each theme is a list of source queries; the theme is what gets recorded so
# coverage can be reported and re-balanced per-theme rather than per-query.
THEMES = {
    "furniture": [
        "wooden chair furniture", "sofa living room furniture",
        "dining table furniture", "bookshelf furniture", "bed bedroom furniture",
    ],
    "vehicles": [
        "car parked street", "motorcycle parked", "bus public transport vehicle",
        "bicycle parked", "truck vehicle road",
    ],
    "food": [
        "plate of food meal", "pizza food", "breakfast table food",
        "fruit bowl", "cake dessert", "bowl of soup",
    ],
    "nature": [
        "mountain landscape", "forest trees landscape", "beach coast landscape",
        "lake reflection landscape", "desert landscape", "waterfall river",
    ],
    "buildings": [
        "building facade architecture", "church exterior architecture",
        "bridge architecture", "house exterior", "skyscraper city architecture",
    ],
    "interiors": [
        "empty room interior", "kitchen interior", "office interior empty",
        "library interior", "staircase interior",
    ],
    "electronics": [
        "laptop computer desk", "smartphone device", "camera equipment",
        "keyboard computer hardware", "television screen device",
        "circuit board electronics",
    ],
    "animals": [
        "cat pet animal", "dog pet animal", "bird perched", "horse field animal",
        "fish aquarium", "squirrel wildlife",
    ],
    "plants": [
        "houseplant pot plant", "flower garden plant", "cactus succulent plant",
        "tree trunk bark", "leaves close up plant",
    ],
    "documents": [
        "open book pages", "stack of books", "handwritten manuscript page",
        "newspaper page print", "printed map document",
    ],
    "kitchenware": [
        "coffee mug cup", "cooking pot pan", "cutlery fork knife spoon",
        "drinking glasses", "teapot kettle",
    ],
    "tools": [
        "hand tools workshop", "hammer wrench tool", "power drill tool",
        "toolbox tools", "gardening tools",
    ],
    # Streets and screenshots are the two themes most likely to smuggle a
    # person in clothing into the negative set, so they are worded to prefer
    # the empty/technical variants and are capped smaller than the rest.
    "street": [
        "empty street road", "road highway landscape", "traffic sign road",
        "railway track", "parking lot",
    ],
    "screens": [
        "software screenshot user interface", "website screenshot browser",
        "spreadsheet screenshot", "terminal console screenshot",
        "map application screenshot",
    ],
}


class _SingleWriter:
    """Refuse to start if another copy of this scraper is already running.

    Learned the hard way on the first bulk run, and it is the same incident
    SCRAPING_PROCESS.md records for apparel_dataset: two processes each hold
    the whole record list in memory and rewrite metadata.json wholesale, so the
    later save silently erases everything the other one added. It cost 127
    already-downloaded images -- files on disk with no record, invisible to
    every consumer and un-re-addable, since the scraper skips paths that
    already exist. Cheaper to refuse than to reconcile.

    A pid file rather than a real lock: this is one script on one machine, and
    the failure it must prevent is "the operator started it twice."
    """

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        if self.path.is_file():
            try:
                other = int(self.path.read_text().strip())
                os.kill(other, 0)
            except (ValueError, ProcessLookupError):
                pass  # stale pid file from a crash; ours now
            except PermissionError:
                raise SystemExit(f"another scraper (pid {other}) is running")
            else:
                raise SystemExit(
                    f"another negatives_scraper is running (pid {other}). "
                    f"Two writers WILL destroy metadata.json -- see _SingleWriter. "
                    f"If that pid is dead, delete {self.path}.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


def load_records() -> list:
    if METADATA_PATH.is_file():
        return json.loads(METADATA_PATH.read_text())
    return []


def save_records(records) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(records, indent=2))


def _get_json(url, params, tries=3):
    for attempt in range(tries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=45)
            resp.raise_for_status()
            return resp.json()
        except Exception as error:
            if attempt == tries - 1:
                print(f"    api failed ({error})")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def search_commons(query, limit):
    """Wikimedia Commons file search -> candidate image dicts.

    Uses `generator=search` in the File namespace rather than
    `list=categorymembers`: categories on Commons are curated for topic, not
    for "is this a usable photograph", and Category:Chairs really does return
    .ogg pronunciation files. Search with `filetype:bitmap` returns photos.

    `iiurlwidth` asks Commons for a pre-scaled thumbnail at our storage cap, so
    a 40 MB original is never transferred to produce a 1536px JPEG.
    """
    payload = _get_json(
        "https://commons.wikimedia.org/w/api.php",
        {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
            "gsrlimit": limit, "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata", "iiurlwidth": MAX_IMAGE_SIDE,
        },
    )
    time.sleep(API_SLEEP)
    if not payload:
        return []

    out = []
    for page in (payload.get("query", {}).get("pages") or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        if not info.get("url"):
            continue
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        meta = info.get("extmetadata") or {}
        out.append({
            "source": "wikimedia",
            "source_id": str(page["pageid"]),
            "title": page.get("title", "").removeprefix("File:"),
            "page_url": info.get("descriptionurl", ""),
            "licence": (meta.get("LicenseShortName") or {}).get("value", "unknown"),
            # thumburl first, original as fallback: derived thumbs occasionally
            # 404 for exotic source formats, and a miss must not drop the image.
            "image_urls": [u for u in (info.get("thumburl"), info.get("url")) if u],
        })
    return out


def search_openverse(query, limit):
    """Openverse (CC-licensed aggregator) -> candidate image dicts.

    Anonymous access is rate-limited, so this is the supplementary source:
    it widens provider diversity (Flickr, museums) beyond Commons' house
    style, but Commons carries the volume.
    """
    payload = _get_json(
        "https://api.openverse.org/v1/images/",
        {"q": query, "page_size": min(limit, 20), "mature": "false"},
    )
    time.sleep(API_SLEEP)
    if not payload:
        return []

    out = []
    for item in payload.get("results", []):
        urls = [u for u in (item.get("url"), item.get("thumbnail")) if u]
        if not urls:
            continue
        out.append({
            "source": "openverse",
            "source_id": item.get("id") or stable_id(urls[0]),
            "title": item.get("title") or "",
            "page_url": item.get("foreign_landing_url") or "",
            "licence": f"CC {item.get('license', '?')} {item.get('license_version', '')}".strip(),
            "image_urls": urls,
        })
    return out


SEARCHERS = {"wikimedia": search_commons, "openverse": search_openverse}


def collect(sources, per_query, target, themes, stats):
    records = load_records()
    seen_hashes = load_seen_hashes(records)
    seen_keys = {(r["source"], r["source_id"]) for r in records}
    kept = len(records)

    for theme in themes:
        queries = THEMES[theme]
        theme_count = sum(1 for r in records if r["theme"] == theme)
        theme_cap = max(1, target // len(THEMES))
        # street/screens are the two themes most at risk of containing people
        # in clothing, so they get a deliberately thinner slice.
        if theme in ("street", "screens"):
            theme_cap = max(1, theme_cap // 2)

        for query in queries:
            if theme_count >= theme_cap:
                break
            for source in sources:
                candidates = SEARCHERS[source](query, per_query)
                print(f"  [{theme}] {source}: '{query}' -> {len(candidates)} candidates")

                for cand in candidates:
                    if theme_count >= theme_cap:
                        break
                    key = (cand["source"], cand["source_id"])
                    if key in seen_keys:
                        continue

                    dest_dir = DATASET_DIR / cand["source"]
                    dest = dest_dir / f"{cand['source_id']}.jpg"
                    if dest.is_file():
                        seen_keys.add(key)
                        continue

                    url, image = _fetch_first_usable(
                        cand["image_urls"], HEADERS, IMAGE_SLEEP, stats)
                    if image is None:
                        continue

                    phash = imagehash.phash(image)
                    if is_duplicate(phash, seen_hashes):
                        stats["dup"] += 1
                        continue

                    dest_dir.mkdir(parents=True, exist_ok=True)
                    image.save(dest, "JPEG", quality=92)
                    seen_hashes.append(phash)
                    seen_keys.add(key)
                    records.append({
                        "source": cand["source"],
                        "source_id": cand["source_id"],
                        "page_url": cand["page_url"],
                        "licence": cand["licence"],
                        "title": cand["title"],
                        "theme": theme,
                        "query": query,
                        "image_url": url,
                        "path": str(dest.relative_to(DATASET_DIR.parent)),
                        "phash": [str(phash)],
                        "scraped_at": datetime.now(timezone.utc)
                                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
                    })
                    theme_count += 1
                    kept += 1

            save_records(records)
        print(f"[{theme}] {theme_count} images (total {kept})")

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=560,
                        help="approximate total images across all themes")
    parser.add_argument("--per-query", type=int, default=20)
    parser.add_argument("--sources", default="wikimedia,openverse")
    parser.add_argument("--themes", default="", help="comma-separated subset")
    args = parser.parse_args()

    sources = [s for s in args.sources.split(",") if s in SEARCHERS]
    themes = [t for t in (args.themes.split(",") if args.themes else THEMES)
              if t in THEMES]

    stats = new_stats()
    with _SingleWriter(DATASET_DIR / ".scraper.pid"):
        records = collect(sources, args.per_query, args.target, themes, stats)
    print(f"\n{len(records)} records in {METADATA_PATH}")
    print(format_stats(stats))


if __name__ == "__main__":
    main()
