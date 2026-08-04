"""Browser-driven real-outfit scraper -> `outfit_dataset/` (NOT apparel_dataset).

Point it at any URL. It drives a real Chromium via Playwright, scrolls the
page until lazy-loading stops producing new images, harvests every image
with its nearest linked item, and writes records in the standard
`outfit_dataset/` schema. Works on JS-rendered grid/feed sites that have no
usable JSON API -- which is most of them.

    # one-time: log in by hand, session persists in the profile dir
    python browser_outfit_scraper.py --login https://www.pinterest.com/

    # then scrape
    python browser_outfit_scraper.py --source pinterest --target 400 \
        --url "https://www.pinterest.com/search/pins/?q=streetwear%20outfit"

    python browser_outfit_scraper.py --urls-file targets.txt --source blogs

Records are DELIBERATELY UNLABELED -- images plus provenance only, no
garment/category/brand fields, not even placeholders. See
SCRAPING_PROCESS.md, "Real-outfit scraping (`outfit_dataset/`)".

Design notes:
  - **Per-item grouping, not per-page.** A board or search page holds many
    outfits, so one record per page would fuse unrelated outfits into one
    record. Each image is attributed to its nearest ancestor `<a href>`,
    which on virtually every grid site is the item/pin/post permalink.
    Images sharing an anchor become one record; images with no anchor fall
    back to the page URL.
  - **Largest declared srcset candidate, then a resolution ladder.** Grid
    sites serve thumbnails in `src` and real resolutions in `srcset`, in
    either `640w` or `2x` descriptor form -- scoring only the former ranks
    every density candidate at zero and silently collects the smallest
    thumbnail. Even parsed correctly, some sites declare *nothing* bigger:
    a logged-out Pinterest grid offers only 236px, which is below this
    project's 320px floor, so a DOM-only scraper collects zero images from
    it. Hence UPGRADE_LADDERS: a per-host list of higher-resolution URL
    rewrites tried best-first, falling back to the declared URL when a
    rewrite 404s (originals are sometimes a different extension).
  - **Persistent profile.** Most of these sites gate browsing behind a
    login. `--login` opens a headed browser so a human can authenticate
    once; the session is reused headless afterwards.
  - Dedup, size validation and download are shared with every other
    outfit scraper via outfit_scrape_common, so the same photo scraped
    from two different sources is caught once.

Note on site terms: this drives a real browser and therefore ignores
robots.txt, which SCRAPING_PROCESS.md rule 4 otherwise requires. That is a
deliberate, per-target decision by the project owner, not the default --
`--respect-robots` is available and sites vary. Keep rate limits polite
regardless; these are community sites and individuals' own photos.
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path

from dataset_utils import load_outfit_records, outfit_key, save_outfit_records_safe
from outfit_scrape_common import (
    HEADERS, download_images, format_stats, load_seen_hashes, new_stats, stable_id,
)

DATASET_ROOT = Path("outfit_dataset")
PROFILE_DIR = Path.home() / ".cache" / "fashion-tests" / "browser-profile"

MAX_IMAGES_PER_ITEM = 4
# Rendered size floor, in CSS px, applied in-page before anything is
# fetched. Cheap way to drop icons, avatars, logos and tracking pixels.
MIN_RENDER_SIDE = 200
SCROLL_PAUSE = 1.2
MAX_SCROLLS = 60
# Stop scrolling after this many consecutive rounds that add no new images.
SCROLL_PATIENCE = 3
# How far one scroll round advances, in CSS px. Deliberately close to a
# viewport height rather than a big jump: virtualized masonry grids
# (Pinterest) only mount the rows near the viewport, so leaping 12k px past
# unmounted rows harvests nothing from the region it skipped. Measured
# 2026-08-03 on one Pinterest search page: 2000px yields 180 images where
# 12000px yields 65, for 13s instead of 4s.
SCROLL_STEP = 2000
PAGE_TIMEOUT_MS = 60_000

# Junk that survives the size filter on most sites.
URL_BLOCKLIST = re.compile(
    r"(sprite|logo|avatar|favicon|placeholder|/emoji/|profile[-_]?pic|"
    r"\.svg($|\?)|/ads?/|doubleclick|analytics)", re.I)

# host substring -> (pattern, replacements tried best-first).
# Only for CDNs whose size lives in the path and whose full-size form is
# stable. Anything not listed is downloaded exactly as the page declared it.
UPGRADE_LADDERS = {
    "i.pinimg.com": (re.compile(r"/(\d+x\d*|originals)/"),
                     ["/originals/", "/1200x/", "/736x/", "/564x/"]),
    # wear.jp declares only `_276.jpg` in the grid -- 276x368, below the
    # 320px floor, so a DOM-faithful scrape of it collects ZERO images, the
    # same trap Pinterest sets. The width is a plain filename suffix and
    # bigger renditions are generated on demand: measured 2026-08-03,
    # `_1000.jpg` returns 1000x1334 and `_500.jpg` 500x667, while `_org.jpg`
    # is 403. Ladder down rather than up so one 404 doesn't lose the photo.
    "images.wear2.jp": (re.compile(r"_(\d+)\.jpg"),
                        ["_1000.jpg", "_750.jpg", "_500.jpg"]),
}

# host substring -> regex whose group 1 is the author handle in an ITEM url.
# Provenance rule 3 in SCRAPING_PROCESS.md requires an author on every
# record so a takedown can be traced without re-scraping; sites that put the
# handle in the permalink give it away for free.
AUTHOR_PATTERNS = {
    "wear.jp": re.compile(r"wear\.jp/([^/]+)/\d+"),
}


def author_for(item_url: str) -> str:
    for host, pattern in AUTHOR_PATTERNS.items():
        if host in item_url:
            match = pattern.search(item_url)
            if match:
                return match.group(1)
    return ""


def upgrade_candidates(url: str) -> list:
    """Best-first URL candidates for one image, declared URL always last."""
    for host, (pattern, replacements) in UPGRADE_LADDERS.items():
        if host not in url or not pattern.search(url):
            continue
        ladder = [pattern.sub(rep, url, count=1) for rep in replacements]
        # Preserve order, drop dupes, keep the page's own URL as the floor.
        seen, out = set(), []
        for candidate in ladder + [url]:
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
        return out
    return [url]

HARVEST_JS = """
(minSide) => {
  const out = [];
  const seen = new Set();

  // Handles BOTH srcset descriptor forms. Width ('640w') is the common
  // one, but many sites -- Pinterest among them -- use pixel density
  // ('1x', '2x') instead. Scoring only 'w' silently ranks every density
  // candidate at 0 and takes the FIRST, i.e. the smallest thumbnail.
  const largestFromSrcset = (raw) => {
    if (!raw) return null;
    let best = null, bestScore = -1;
    for (const part of raw.split(',')) {
      const bits = part.trim().split(/\\s+/);
      if (!bits[0]) continue;
      const desc = bits[1] || '';
      let score = 0;
      const mw = desc.match(/^(\\d+(?:\\.\\d+)?)w$/);
      const mx = desc.match(/^(\\d+(?:\\.\\d+)?)x$/);
      if (mw) score = parseFloat(mw[1]);
      else if (mx) score = parseFloat(mx[1]) * 1000;  // density ranks above bare urls
      if (score > bestScore) { bestScore = score; best = bits[0]; }
    }
    return best;
  };

  const push = (url, el, w, h) => {
    if (!url || url.startsWith('data:') || seen.has(url)) return;
    seen.add(url);
    const anchor = el.closest('a');
    out.push({
      url: new URL(url, document.baseURI).href,
      width: Math.round(w), height: Math.round(h),
      alt: (el.getAttribute && el.getAttribute('alt')) || '',
      item: anchor ? anchor.href : null,
    });
  };

  document.querySelectorAll('img').forEach((img) => {
    const rect = img.getBoundingClientRect();
    const w = img.naturalWidth || rect.width;
    const h = img.naturalHeight || rect.height;
    if (w < minSide || h < minSide) return;
    // Prefer the largest resolution the page itself declares.
    let url = largestFromSrcset(img.getAttribute('srcset'));
    if (!url) {
      const picture = img.closest('picture');
      if (picture) {
        for (const s of picture.querySelectorAll('source')) {
          const cand = largestFromSrcset(s.getAttribute('srcset'));
          if (cand) { url = cand; break; }
        }
      }
    }
    push(url || img.currentSrc || img.src, img, w, h);
  });

  // Some grids paint photos as CSS backgrounds rather than <img>.
  document.querySelectorAll('[style*="background-image"]').forEach((el) => {
    const rect = el.getBoundingClientRect();
    if (rect.width < minSide || rect.height < minSide) return;
    const m = (el.getAttribute('style') || '').match(/url\\(['"]?(.*?)['"]?\\)/);
    if (m) push(m[1], el, rect.width, rect.height);
  });

  return out;
}
"""


def robots_allows(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(f"{parts.scheme}://{parts.netloc}/robots.txt")
    try:
        parser.read()
    except Exception:
        return True  # unreachable robots.txt is not a prohibition
    return parser.can_fetch(HEADERS["User-Agent"], url)


def source_for(url: str) -> str:
    host = urllib.parse.urlsplit(url).netloc.lower()
    host = re.sub(r"^(www|m|mobile)\.", "", host)
    return host.split(".")[0] or "web"


def item_id_for(item_url: str) -> str:
    """Prefer the site's own item id when the URL exposes one."""
    for pattern in (r"/pin/(\d+)", r"/p/([A-Za-z0-9_-]{5,})",
                    r"/photos?/([A-Za-z0-9_-]{5,})", r"/status/(\d+)",
                    # wear.jp: /<handle>/<coordinate id>/
                    r"wear\.jp/[^/]+/(\d+)"):
        match = re.search(pattern, item_url)
        if match:
            return match.group(1)
    return stable_id(item_url)


def harvest_page(page, url, target_images, verbose=True):
    """Load `url`, scroll to exhaustion, return harvested image dicts."""
    page.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
    try:
        page.wait_for_load_state("networkidle", timeout=15_000)
    except Exception:
        pass  # infinite feeds never go idle; harmless

    found, stale = {}, 0
    for round_index in range(MAX_SCROLLS):
        # `before` must be captured before this round's harvest, not after:
        # nothing between a `wheel()` and the next `evaluate()` can change
        # the count, so comparing across the scroll alone made `stale`
        # increment every single round and capped every page at exactly
        # SCROLL_PATIENCE scrolls no matter how much it had left to give.
        before = len(found)
        for entry in page.evaluate(HARVEST_JS, MIN_RENDER_SIDE):
            if URL_BLOCKLIST.search(entry["url"]):
                continue
            found.setdefault(entry["url"], entry)

        if verbose:
            print(f"    scroll {round_index + 1}: {len(found)} images", end="\r")
        if len(found) >= target_images:
            break

        if round_index:  # round 0 has no prior round to be stale against
            stale = stale + 1 if len(found) == before else 0
            if stale >= SCROLL_PATIENCE:
                break

        page.mouse.wheel(0, SCROLL_STEP)
        time.sleep(SCROLL_PAUSE)

    if verbose:
        print(f"    harvested {len(found)} candidate images from {url[:70]}")
    return list(found.values())


def group_items(entries, page_url):
    """Group harvested images by their linked item, preserving order."""
    groups = {}
    for entry in entries:
        key = entry["item"] or page_url
        groups.setdefault(key, []).append(entry)
    return groups


def build_record(source, item_url, section, entries, paths, urls, hashes):
    titles = [e["alt"].strip() for e in entries if e.get("alt", "").strip()]
    return {
        "source": source,
        "source_id": item_id_for(item_url),
        "post_url": item_url,
        "author": author_for(item_url),
        "title": titles[0] if titles else "",
        "section": section,
        "created_utc": 0,
        "source_tags": [],
        "image_count": len(paths),
        "images": paths,
        "image_urls": urls,
        "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phash": hashes,
    }


def scrape_url(page, url, source, target_images, existing_ids, seen_hashes, stats):
    print(f"\n=== {url[:90]} (target {target_images} images) ===")
    entries = harvest_page(page, url, target_images * 2)
    groups = group_items(entries, url)
    print(f"    {len(groups)} distinct items")

    touched, images_here, items_here = {}, 0, 0
    for item_url, group in groups.items():
        if images_here >= target_images:
            break
        source_id = item_id_for(item_url)
        if f"{source}:{source_id}" in existing_ids:
            continue

        dest = DATASET_ROOT / source / source_id
        paths, urls, hashes = download_images(
            dest, [upgrade_candidates(e["url"]) for e in group], seen_hashes,
            stats, max_images=MAX_IMAGES_PER_ITEM,
        )
        if not paths:
            continue

        record = build_record(source, item_url, url, group, paths, urls, hashes)
        touched[outfit_key(record)] = record
        existing_ids.add(f"{source}:{source_id}")
        images_here += len(paths)
        items_here += 1
        print(f"  [{images_here}/{target_images}] {source_id} ({len(paths)} img) "
              f"{record['title'][:60]}")

        if len(touched) >= 10:
            save_outfit_records_safe(touched)
            print(f"  checkpointed {len(touched)} records")
            touched = {}

    if touched:
        save_outfit_records_safe(touched)
        print(f"  final checkpoint: {len(touched)} records")

    if images_here < target_images:
        print(f"  NOTE: yielded {images_here}/{target_images} images from "
              f"{len(groups)} items -- page exhausted or filters too tight.")
    return items_here, images_here


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", action="append", default=[],
                        help="page to scrape (repeatable)")
    parser.add_argument("--urls-file", help="file with one URL per line (# comments ok)")
    parser.add_argument("--source", help="dataset source label (default: from domain)")
    parser.add_argument("--target", type=int, default=200,
                        help="image target per URL (default 200)")
    parser.add_argument("--login", metavar="URL",
                        help="open a headed browser at URL to log in, then exit; "
                             "the session persists for later headless runs")
    parser.add_argument("--profile", default=str(PROFILE_DIR),
                        help="persistent browser profile dir")
    parser.add_argument("--headed", action="store_true", help="show the browser")
    parser.add_argument("--respect-robots", action="store_true",
                        help="skip URLs disallowed by robots.txt")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright not installed: pip install playwright && playwright install chromium")

    urls = list(args.url)
    if args.urls_file:
        for line in Path(args.urls_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if not urls and not args.login:
        sys.exit("nothing to do: pass --url, --urls-file, or --login")

    Path(args.profile).mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            args.profile,
            headless=not (args.headed or args.login),
            viewport={"width": 1440, "height": 900},
            user_agent=HEADERS["User-Agent"],
        )
        page = context.pages[0] if context.pages else context.new_page()

        if args.login:
            page.goto(args.login, timeout=PAGE_TIMEOUT_MS)
            print(f"Log in to {args.login} in the open browser window.")
            input("Press Enter here when finished -- the session will persist. ")
            context.close()
            return

        records = load_outfit_records()
        existing_ids = {outfit_key(r) for r in records}
        seen_hashes = load_seen_hashes(records)
        print(f"Existing outfit records: {len(records)} "
              f"({len(seen_hashes)} known image hashes)")

        stats = new_stats()
        summary = []
        for url in urls:
            if args.respect_robots and not robots_allows(url):
                print(f"\n=== SKIPPED (robots.txt disallows): {url[:80]}")
                continue
            source = args.source or source_for(url)
            try:
                items, images = scrape_url(
                    page, url, source, args.target, existing_ids, seen_hashes, stats)
            except Exception as error:
                print(f"  FAILED {type(error).__name__}: {error}")
                items, images = 0, 0
            summary.append((url, source, items, images))

        context.close()

    final = load_outfit_records()
    print("\n=== summary ===")
    for url, source, items, images in summary:
        print(f"  {source:<12} {items:4d} items  {images:4d} images  {url[:60]}")
    print(f"  skipped: {format_stats(stats)}")
    print(f"Total outfit_dataset records: {len(final)}, "
          f"images: {sum(r.get('image_count', 0) for r in final)}")


if __name__ == "__main__":
    main()
