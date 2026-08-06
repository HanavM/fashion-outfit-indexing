"""American Eagle (ae.com) men's clothing scraper — T-Shirts, Hoodies &
Sweatshirts, Jeans, Shorts. Target 50 colorway variants per category -> 200.

Site notes (first Akamai-fronted, JSON:API-embedded site in this pipeline;
closest prior analogue is New Balance, but much softer):

  - **Bot protection: medium (Akamai Bot Manager, no interactive
    challenge).** Plain `requests` works for both PLP and PDP *document*
    fetches, but ONLY with a full browser header set — User-Agent alone
    gets a 403 "Access Denied" Akamai edge page. The headers that matter
    are the `sec-fetch-*` navigation quartet plus `upgrade-insecure-
    requests`; drop them and even `robots.txt` 403s. Blocks are also
    intermittent (roughly 1 in 20 requests during this scrape), so every
    fetch retries with escalating backoff, same shape as the New Balance
    `goto_with_retry` pattern.
  - **The XHR API is NOT usable from plain `requests`.** The PLP's own
    pagination endpoint (`/ugp-api/browse/v1/category/{catId}?offset=N`,
    discovered by watching network traffic in patchright) returns 403 to
    every non-browser client because it requires Akamai's `_abck` sensor
    cookie, which is only minted by running the site's JS. Documented here
    so nobody re-derives it: it exists, it is the right endpoint, and it
    is not reachable without a real browser session.
  - **Everything needed is embedded in the server-rendered HTML instead.**
    Both PLP and PDP contain a bare `<script>` tag whose entire body is a
    JSON:API document (`{"data": {...}, "included": [...], "meta": {...}}`).
    Located by "first <script> whose stripped text starts with `{` and
    contains `"type":"plp"` / `"type":"pdp"`". This is far richer than the
    ld+json Product block that is also present (which carries only name /
    sku / color / material / price / one image).
      * PLP blob: `meta.totalProducts`, and `included` = 30 product
        objects (page 1 only — deeper pages are the browser-gated XHR
        above).
      * PDP blob: `data.attributes.copySections` = {details, material,
        size} each with `bullets` + `longDesc`, `data.attributes.
        breadcrumbs` = the site's own category path, and `included`
        product objects carrying `displayName`, `colorName`, `listPrice`,
        `salePrice`, `pdpImages`, `colorSwatches`, `modelSizeAndHeight`.
  - **Colorways (the thing HUF's notes warn about).** AE's model is the
    opposite of Shopify's: one PDP URL == exactly one colorway, and the
    product id IS the colorway id — `{style}_{color}`, e.g.
    `0195_2926_001` (style `0195_2926`, color `001` = Black). Sibling
    colorways of the same style are listed in the product object's
    `colorSwatches` array (id + productUrl + swatch image + color name),
    measured at 1–21 colorways per style on the live catalog. So there is
    no risk of silently collapsing colorways here, but there IS the mirror
    risk: a PLP page lists only *some* colorways of a style, so scraping
    the PLP alone under-counts the catalog. This scraper therefore walks
    `colorSwatches` breadth-first (round 1 = every PLP seed, round 2 = the
    2nd colorway of each seed's style, ...) so the 50 records per category
    stay spread across many styles instead of being eaten by one 21-color
    tee.
  - **Images: Adobe Scene7, and the default size is uselessly small.**
    `https://s7d2.scene7.com/is/image/aeo/{colorwayId}_{view}` with NO
    query string returns a ~6 KB thumbnail preset, not the full-res
    original. `?wid=N` is the only knob that matters (`fmt=jpg` is ignored
    — the CDN content-negotiates, returning webp to an `Accept: */*`
    request and jpeg when `Accept: image/jpeg` is sent, so this scraper
    sends the jpeg Accept header explicitly). `wid=940` yields ~940x1200,
    ~70 KB — the ~1200px long edge this pipeline wants. The unparameterised
    URL is what gets stored in `image_urls`, so full-res stays refetchable.
  - **Per-image view codes are free here** (lesson 5): the CDN filename
    suffix is `_of` (on-figure front), `_ob` (on-figure back), `_os`
    (on-figure side), `_f` (flat/laydown), `_d1`/`_d2`/`_d3` (detail
    crops), `_s` (colour swatch chip — excluded). These survive in
    `image_urls` without any schema change, in the same order as `images`.
    `modelSizeAndHeight` on the product object confirms which suffixes are
    on-model ("Des is 6'3\", wearing size large") and which are not.
  - Photos are MIXED: `_of/_ob/_os` are on-model studio shots, `_f` is a
    flat-lay, `_d*` are close crops. Verified by direct image inspection.
"""

import html
import json
import random
import re
import shutil
import time
from pathlib import Path

import requests

from dataset_utils import load_records, save_records_safe

BRAND = "americaneagle"
BASE = "https://www.ae.com"

# Full browser navigation header set. The sec-fetch-* quartet is load
# bearing: without it Akamai 403s even /robots.txt.
DOC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.9",
    "sec-ch-ua": '"Chromium";v="126", "Not)A;Brand";v="24", "Google Chrome";v="126"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
}
IMG_HEADERS = {
    "User-Agent": DOC_HEADERS["User-Agent"],
    "accept": "image/jpeg,image/*,*/*;q=0.8",
    "referer": BASE + "/",
}

# Category label -> PLP path. Labels are the site's own left-nav names.
CATEGORIES = {
    "T-Shirts": "/us/en/c/men/tops/t-shirts/cat90012",
    "Hoodies & Sweatshirts": "/us/en/c/men/tops/hoodies-sweatshirts/cat90020",
    "Jeans": "/us/en/c/men/bottoms/jeans/cat6430041",
    "Shorts": "/us/en/c/men/bottoms/shorts/cat5180435",
}
TARGET_PER_CATEGORY = 50
MAX_IMAGES = 6
IMAGE_WIDTH = 940          # -> ~940x1200, the pipeline's ~1200px long edge
CHECKPOINT_EVERY = 10
DISK_FLOOR_GIB = 3.0

DATASET_DIR = Path("apparel_dataset") / BRAND
TAG_STRIP_RE = re.compile(r"<[^>]+>")
SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.S)
BLOCK_MARKERS = ("Access Denied", "Reference&#32;&#35;", "errors.edgesuite.net")

SESSION = requests.Session()
SESSION.headers.update(DOC_HEADERS)


def free_gib():
    return shutil.disk_usage("/System/Volumes/Data").free / 2**30


def strip_html(text):
    text = TAG_STRIP_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def extract_blob(page_html, blob_type):
    """Pull the embedded JSON:API document of the given type out of the HTML.

    Returns None when the page is not the expected page — which is the
    explicit "did content actually arrive" sniff lesson 3 and lesson 12 ask
    for. HTTP 200 is NOT sufficient on this site: Akamai's block page and
    AE's own soft redirect to the homepage both return 200-ish HTML that
    contains no blob at all.
    """
    for script in SCRIPT_RE.findall(page_html):
        text = script.strip()
        if text.startswith("{") and f'"type":"{blob_type}"' in text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
    return None


_BLOB_CACHE = {}


def fetch_blob(url, blob_type, tries=5):
    """GET a document URL and return its embedded blob, retrying through
    Akamai's intermittent blocks with escalating backoff.

    Cached: seed PDPs get looked at twice (once to harvest colorSwatches,
    once to build the seed's own record), and there is no reason to hand
    Akamai a second identical request for that."""
    if url in _BLOB_CACHE:
        return _BLOB_CACHE[url]
    last = ""
    for attempt in range(tries):
        try:
            resp = SESSION.get(url, timeout=40)
            last = f"http {resp.status_code}"
            if resp.status_code == 200:
                if any(marker in resp.text[:2000] for marker in BLOCK_MARKERS):
                    last = "akamai block page"
                else:
                    blob = extract_blob(resp.text, blob_type)
                    if blob is not None:
                        _BLOB_CACHE[url] = blob
                        return blob
                    last = f"200 but no {blob_type} blob (soft redirect?)"
        except Exception as error:  # noqa: BLE001
            last = f"{type(error).__name__}: {error}"
        # Akamai's 403 here is overwhelmingly transient: measured on a
        # 14-request burst, the FIRST request 403'd and the next 13 all
        # returned 200. So retry almost immediately and only then escalate
        # — a 3s+7s+11s ladder (the New Balance shape) more than doubles
        # the wall-clock cost of this scrape for no extra success rate.
        time.sleep(0.6 + 2.5 * attempt + random.random())
    print(f"  ! giving up on {url} -- {last}")
    return None


def product_objects(blob):
    return {
        item["id"]: item["attributes"]
        for item in blob.get("included", [])
        if item.get("type") == "product"
    }


def bullets(copy_sections, key):
    section = (copy_sections or {}).get(key) or {}
    return [strip_html(b) for b in section.get("bullets", []) if strip_html(b)]


def build_record(colorway_id, blob, attrs, category_label):
    copy_sections = blob["data"]["attributes"].get("copySections") or {}
    details_bullets = bullets(copy_sections, "details")
    size_bullets = bullets(copy_sections, "size")
    material_bullets = bullets(copy_sections, "material")
    long_desc = strip_html((copy_sections.get("details") or {}).get("longDesc", ""))

    # pdpImages are protocol-relative and unparameterised (= full res).
    # Store them exactly as-is; download a width-capped variant.
    image_urls = []
    for raw in attrs.get("pdpImages") or []:
        url = "https:" + raw if raw.startswith("//") else raw
        if url.endswith("_s"):        # colour swatch chip, not a photo
            continue
        image_urls.append(url)
    image_urls = image_urls[:MAX_IMAGES]

    price = attrs.get("salePrice") or attrs.get("listPrice") or ""
    slug = (attrs.get("url") or "").rstrip("/").split("/")
    slug = slug[-2] if len(slug) >= 2 else colorway_id

    return {
        "brand": BRAND,
        "category": category_label,
        "name": attrs.get("displayName", ""),
        "color_name": attrs.get("colorName", ""),
        "price": f"{float(price):.2f}" if price != "" else "",
        "product_code": f"{BRAND}-{colorway_id}",
        "slug": slug,
        "product_url": BASE + "/us/en" + (attrs.get("url") or ""),
        "image_urls": image_urls,
        "details": {
            "description": long_desc,
            "features": details_bullets + size_bullets,
            "materials": material_bullets,
        },
    }


def download_images(record):
    product_dir = DATASET_DIR / record["slug"] / record["product_code"]
    product_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for index, url in enumerate(record["image_urls"]):
        dest = product_dir / f"image_{index}.jpg"
        if not dest.is_file():
            sized = f"{url}?wid={IMAGE_WIDTH}"
            try:
                resp = requests.get(sized, headers=IMG_HEADERS, timeout=40)
                resp.raise_for_status()
                if not resp.headers.get("content-type", "").startswith("image/"):
                    print(f"  image not an image: {sized}")
                    continue
                if len(resp.content) < 8000:
                    # Scene7 serves a tiny placeholder for missing assets.
                    print(f"  image suspiciously small ({len(resp.content)} B): {sized}")
                    continue
                dest.write_bytes(resp.content)
            except Exception as error:  # noqa: BLE001
                print(f"  image fetch failed: {sized} -- {error}")
                continue
        images.append(str(dest))
    record["images"] = images
    record["image_count"] = len(images)
    return record


def category_colorway_rounds(plp_path, label):
    """Yield colorway ids for a category, breadth-first across styles.

    Round 1 is every colorway the PLP itself listed; round N>1 is the Nth
    colorway of each of those styles, taken from each seed's own
    `colorSwatches`. This keeps a 21-colorway tee from eating the whole
    category while still filling the target from real sibling colorways.
    """
    blob = fetch_blob(BASE + plp_path, "plp")
    if blob is None:
        print(f"  !! could not load PLP for {label}")
        return
    total = blob.get("meta", {}).get("totalProducts")
    seeds = [item["id"] for item in blob.get("included", []) if item.get("type") == "product"]
    print(f"  PLP: {len(seeds)} seed colorways listed, site reports {total} products in category")

    yield from seeds

    # Sibling expansion. Sibling lists are resolved lazily as each seed's
    # PDP is fetched by the caller, so we re-fetch seeds here only if the
    # caller still needs more; in practice the caller caches PDP blobs.
    sibling_lists = []
    for seed in seeds:
        blob = fetch_blob(f"{BASE}/us/en/p/x/x/x/{seed}", "pdp")
        if blob is None:
            sibling_lists.append([])
            continue
        attrs = product_objects(blob).get(seed, {})
        sibling_lists.append([s["id"] for s in (attrs.get("colorSwatches") or [])])
        time.sleep(0.4)

    depth = 0
    while any(len(lst) > depth for lst in sibling_lists):
        for lst in sibling_lists:
            if len(lst) > depth:
                yield lst[depth]
        depth += 1


def main():
    print(f"Free disk at start: {free_gib():.1f} GiB")
    existing = load_records()
    all_codes = {r["product_code"] for r in existing}
    mine = {c for c in all_codes if c.startswith(BRAND + "-")}
    print(f"Existing records: {len(existing)} total, {len(mine)} for {BRAND}")

    touched = {}
    processed = 0
    summary = {}

    for label, plp_path in CATEGORIES.items():
        print(f"\n=== {label} ({plp_path}) ===")
        added = 0
        for colorway_id in category_colorway_rounds(plp_path, label):
            if added >= TARGET_PER_CATEGORY:
                break
            code = f"{BRAND}-{colorway_id}"
            if code in all_codes:
                continue

            if free_gib() < DISK_FLOOR_GIB:
                if touched:
                    save_records_safe(touched)
                print(f"\n!! STOPPING: only {free_gib():.1f} GiB free (floor {DISK_FLOOR_GIB}).")
                return

            blob = fetch_blob(f"{BASE}/us/en/p/x/x/x/{colorway_id}", "pdp")
            if blob is None:
                continue
            attrs = product_objects(blob).get(colorway_id)
            if not attrs:
                print(f"  ! {colorway_id}: pdp blob had no product object for this colorway")
                continue

            record = build_record(colorway_id, blob, attrs, label)
            if not record["image_urls"]:
                print(f"  ! {colorway_id}: no images listed, skipping")
                continue
            record = download_images(record)
            if not record["images"]:
                print(f"  ! {colorway_id}: no images downloaded, skipping")
                continue

            touched[code] = record
            all_codes.add(code)
            added += 1
            processed += 1
            print(f"  [{added}/{TARGET_PER_CATEGORY}] {record['name']} ({record['color_name']}) "
                  f"-- {colorway_id}, {record['image_count']} imgs")

            if len(touched) >= CHECKPOINT_EVERY:
                save_records_safe(touched)
                print(f"  checkpointed {len(touched)} records ({free_gib():.1f} GiB free)")
                touched = {}
            if processed % 50 == 0:
                print(f"  --- disk check: {free_gib():.1f} GiB free ---")
            time.sleep(0.4)

        summary[label] = added
        if added < TARGET_PER_CATEGORY:
            print(f"  NOTE: {label} reached only {added}/{TARGET_PER_CATEGORY}.")

    if touched:
        save_records_safe(touched)
        print(f"Final checkpoint: {len(touched)} records")

    final = load_records()
    count = sum(1 for r in final if r.get("brand") == BRAND)
    print(f"\nPer-category: {summary}")
    print(f"Total {BRAND} records in dataset: {count}")
    print(f"Free disk at end: {free_gib():.1f} GiB")


if __name__ == "__main__":
    main()
