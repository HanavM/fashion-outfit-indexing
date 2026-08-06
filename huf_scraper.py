"""HUF (hufworldwide.com) men's clothing scraper — T-Shirts, Hoodies and
Fleece, Tops, Bottoms. Target 50 colorway variants per section -> 200.

Site notes (fourth Shopify-storefront site in this pipeline, after
Champion, Stüssy and Dickies — but NOT the same record shape as those
three; see the colorway note below, it is the one real difference):
  - No bot protection at all. Plain `requests` with a normal desktop UA
    works for everything; same easiest tier as Champion/Stüssy/Dickies/
    Gap/Skechers. `robots.txt` is the stock Shopify one (`Allow: /` for
    `*`, only cart/checkout/admin disallowed) and even advertises a
    UCP/MCP agent endpoint, so catalog crawling is explicitly sanctioned.
  - Real collection handles scraped from the homepage's own nav links
    (`https://hufworldwide.com/`, `href="/collections/..."`), not
    guessed. Handles used: `mens-t-shirts` (183 products),
    `mens-hoodies-and-fleece` (61), `mens-tops` (53), `mens-bottoms`
    (32). Also present but not used: `mens-jackets` (33), `mens-shorts`
    (11), plus non-clothing `mens-hats`/`mens-socks`/`bags`/`jewelry`/
    `decks`. The storefront is NOT locale-scoped — bare
    `/collections/{handle}/products.json` works (unlike Dickies, which
    404s without `/en-us/`).
  - Standard Shopify storefront JSON API,
    `https://hufworldwide.com/collections/{handle}/products.json?
    limit=250&page=N`, paginated until a page returns zero products.
    Full product detail (description, all colorways, all images, price,
    SKUs) is inline in that one response — no PDP visit needed at all,
    same as Champion/Stüssy/Dickies.

  - **THE ONE REAL DIFFERENCE FROM EVERY PRIOR SHOPIFY BRAND HERE: a HUF
    product is NOT one colorway.** Champion/Stüssy/Dickies all had
    `options[0]` = Color with exactly one value per product, so "one API
    product = one dataset record" held. HUF's `options[0]` = Color
    carries 1-15 values on a single product (the Cromer Pant alone
    carries 15). Taking `variants[0]['option1']` as *the* color the way
    dickies_scraper.py does would silently collapse 327 real colorways
    down to 329 products' first colors and throw the rest away. So this
    scraper expands each product into one record per Color option value.
    Expanded reachable colorways: T-Shirts 247, Hoodies and Fleece 94,
    Bottoms 102, Tops 73 — all four comfortably above the 50 target.

  - **Per-colorway images are recovered by SKU-substring match on the CDN
    filename**, which is the only reliable mapping the API gives. A
    product's `images` array is a flat list covering every colorway
    interleaved, and only the *first* image of each colorway is linked
    back via `variant_ids` (the 2nd/3rd/... shots have `variant_ids: []`),
    so filtering on `variant_ids` alone yields exactly one image per
    colorway and loses the rest. But every filename embeds the variant
    SKU: variant `TS02678_BKWHT` -> `89-EMBROIDRED-S-S-TEE_BLACK-WHITE_
    TS02678_BKWHT_01.png`. Matching on the punctuation-stripped SKU
    recovers the full per-colorway gallery (3 shots for a tee, ~8 for a
    pant). Falls back to the variant's `featured_image`, then to the whole
    product gallery for single-colorway products; measured on the live
    catalog that fallback fires on only 3 of 558 colorways.

  - **Images are 2400x2400 PNGs, ~3 MB each raw.** Disk is the binding
    constraint on this machine, so they are fetched through Shopify's CDN
    resizer (`&width=800`, ~390 KB) rather than at full res. Note the CDN
    ignores `&format=jpg` on `/s/files/` URLs (it keeps serving
    `image/png`) and swapping the `.png` extension for `.jpg` in the path
    404s — width is the only knob that works. Files are still written as
    `image_N.jpg` per the pipeline's directory convention; the bytes are
    PNG, which PIL sniffs by content, not extension. Raw full-res URLs
    are preserved in `image_urls` per lesson 5.

  - `product_code` is `huf-{variant SKU}` (e.g. `huf-TS02678_BKWHT`).
    The numeric Shopify product `id` is NOT usable here because it is
    per-product, not per-colorway, and would collide across the expanded
    records. The `huf-` prefix also keeps these clear of the 764 bare
    numeric Shopify ids already in metadata.json from Champion/Stüssy/
    Dickies (checked: zero raw collisions today, prefixed anyway).

  - `body_html` is a mix of both prior shapes: an optional free-text
    intro paragraph followed by a `•`-bulleted `<br>` list. Split on the
    bullet character — text before the first bullet becomes
    `details.description`, the bullets become `details.features`, and any
    bullet containing a fibre percentage / fabric weight becomes
    `details.materials` as well (it is always the first bullet in
    practice, e.g. "100% cotton (6oz) short sleeve tee shirt").
"""

import html
import re
import shutil
import time
from pathlib import Path

import requests

from dataset_utils import load_records, save_records_safe

BASE = "https://hufworldwide.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

CATEGORIES = {
    "T-Shirts": "mens-t-shirts",
    "Hoodies and Fleece": "mens-hoodies-and-fleece",
    "Tops": "mens-tops",
    "Bottoms": "mens-bottoms",
}
TARGET_PER_CATEGORY = 50
MAX_IMAGES = 6
IMAGE_WIDTH = 800

DATASET_DIR = Path("apparel_dataset/huf")
TAG_STRIP_RE = re.compile(r"<[^>]+>")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
MATERIAL_RE = re.compile(r"\d+\s*%|\d+\s*/\s*\d+\s+\w+|\d+\s*(oz|gsm)", re.I)

# Stop before the disk is actually full -- several other scrapers write to
# this volume concurrently. See SCRAPING_PROCESS.md lesson 2.
MIN_FREE_GIB = 3.0


def free_gib():
    return shutil.disk_usage("/System/Volumes/Data").free / (1024 ** 3)


def strip_html(text):
    text = TAG_STRIP_RE.sub(" ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_body(body_html):
    """HUF's body_html = optional intro paragraph + a bullet list joined by
    `•` inside <br>-separated <p> content. Returns (description, features,
    materials)."""
    raw = html.unescape(body_html or "")
    raw = re.sub(r"<br\s*/?>|</p>|<p>", "\n", raw, flags=re.I)
    text = TAG_STRIP_RE.sub(" ", raw)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    description_parts, features = [], []
    for line in lines:
        if line.startswith("•"):
            bullet = line.lstrip("•").strip()
            if bullet:
                features.append(bullet)
        elif not features:
            description_parts.append(line)

    description = " ".join(description_parts).strip()
    if not description and features:
        # Bullets-only product (common on Tops) -- keep the bullets as the
        # description too so captioning has prose to work from.
        description = " ".join(features)
    materials = [f for f in features if MATERIAL_RE.search(f)]
    return description, features, materials


def fetch_collection_products(handle):
    page = 1
    while True:
        url = f"{BASE}/collections/{handle}/products.json"
        resp = requests.get(url, headers=HEADERS, params={"limit": 250, "page": page}, timeout=30)
        resp.raise_for_status()
        products = resp.json().get("products", [])
        if not products:
            return
        for product in products:
            yield product
        page += 1


def colorway_images(product, variant):
    """Per-colorway gallery, recovered by SKU-substring match on the CDN
    filename (see module docstring for why variant_ids doesn't work)."""
    images = product.get("images", [])
    key = NON_ALNUM_RE.sub("", (variant.get("sku") or "").lower())
    if key:
        matched = [im["src"] for im in images
                   if key in NON_ALNUM_RE.sub("", im["src"].rsplit("/", 1)[-1].lower())]
        if matched:
            return matched
    featured = variant.get("featured_image")
    if featured:
        return [featured["src"]]
    if len(product["options"][0]["values"]) == 1:
        return [im["src"] for im in images]
    return []


def expand_colorways(product):
    """One record-shaped dict per Color option value. HUF products carry
    1-15 colors each, unlike every prior Shopify brand in this pipeline."""
    first_variant_by_color = {}
    for variant in product.get("variants", []):
        first_variant_by_color.setdefault(variant.get("option1") or "", variant)
    return list(first_variant_by_color.items())


def build_record(product, variant, color_name, category_label):
    sku = variant.get("sku") or f"{product['id']}-{color_name}"
    description, features, materials = parse_body(product.get("body_html", ""))
    return {
        "brand": "huf",
        "category": category_label,
        "name": product.get("title", ""),
        "color_name": color_name,
        "price": variant.get("price", ""),
        "product_code": f"huf-{sku}",
        "slug": product["handle"],
        "product_url": f"{BASE}/products/{product['handle']}",
        "image_urls": colorway_images(product, variant),
        "details": {
            "description": description,
            "features": features,
            "materials": materials,
        },
    }


def download_images(record):
    product_dir = DATASET_DIR / record["slug"] / record["product_code"]
    product_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for i, url in enumerate(record["image_urls"][:MAX_IMAGES]):
        dest = product_dir / f"image_{i}.jpg"
        if not dest.is_file():
            sep = "&" if "?" in url else "?"
            try:
                resp = requests.get(f"{url}{sep}width={IMAGE_WIDTH}", headers=HEADERS, timeout=30)
                resp.raise_for_status()
                if len(resp.content) < 1000:
                    print(f"  suspiciously small image, skipping: {url}")
                    continue
                dest.write_bytes(resp.content)
            except Exception as error:
                print(f"  image fetch failed: {url} -- {error}")
                continue
        images.append(str(dest))
    record["images"] = images
    record["image_count"] = len(images)
    return record


def main():
    existing = load_records()
    all_codes = {str(r["product_code"]) for r in existing}
    huf_codes = {str(r["product_code"]) for r in existing if r.get("brand") == "huf"}
    print(f"Existing records: {len(existing)} total, {len(huf_codes)} huf")
    print(f"Free disk: {free_gib():.1f} GiB")

    touched = {}
    checkpoint_every = 10
    records_since_disk_check = 0
    stopped_for_disk = False

    for category_label, handle in CATEGORIES.items():
        if stopped_for_disk:
            break
        print(f"\n=== {category_label} ({handle}) ===")
        added = 0
        for product in fetch_collection_products(handle):
            if added >= TARGET_PER_CATEGORY or stopped_for_disk:
                break
            for color_name, variant in expand_colorways(product):
                if added >= TARGET_PER_CATEGORY:
                    break
                record = build_record(product, variant, color_name, category_label)
                code = record["product_code"]
                if code in all_codes:
                    continue
                if not record["image_urls"]:
                    print(f"  skipping {code} ({record['name']}) -- no colorway images resolved")
                    continue

                record = download_images(record)
                if not record["images"]:
                    print(f"  skipping {code} ({record['name']}) -- no images downloaded")
                    continue

                touched[code] = record
                all_codes.add(code)
                added += 1
                records_since_disk_check += 1
                print(f"  [{added}/{TARGET_PER_CATEGORY}] {record['name']} ({color_name}) "
                      f"-- {code}, {record['image_count']} imgs")

                if len(touched) >= checkpoint_every:
                    save_records_safe(touched)
                    print(f"  checkpointed {len(touched)} records")
                    touched = {}

                if records_since_disk_check >= 50:
                    records_since_disk_check = 0
                    free = free_gib()
                    print(f"  disk check: {free:.1f} GiB free")
                    if free < MIN_FREE_GIB:
                        print(f"  STOPPING: free disk {free:.1f} GiB < {MIN_FREE_GIB} GiB floor")
                        stopped_for_disk = True
                        break
                time.sleep(0.2)

        if added < TARGET_PER_CATEGORY and not stopped_for_disk:
            print(f"  NOTE: {category_label} only had {added} new colorway variants reachable.")

    if touched:
        save_records_safe(touched)
        print(f"Final checkpoint: {len(touched)} records")

    final = load_records()
    huf = [r for r in final if r.get("brand") == "huf"]
    print(f"\nTotal huf records in dataset: {len(huf)} (dataset total {len(final)})")
    by_cat = {}
    for r in huf:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    for cat, n in by_cat.items():
        print(f"  {cat}: {n}")
    print(f"Free disk: {free_gib():.1f} GiB")
    if stopped_for_disk:
        print("RUN STOPPED EARLY FOR DISK SAFETY")


if __name__ == "__main__":
    main()
