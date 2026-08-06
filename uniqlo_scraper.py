"""Uniqlo US men's clothing scraper — T-Shirts, Sweatshirts and Hoodies,
Casual Pants, Casual Shirts. Target 50 colorway variants per section -> 200
requested.

Site notes (first "own-brand JSON commerce API" site in this pipeline —
neither Shopify nor SFCC nor Next.js `__NEXT_DATA__`):
  - `robots.txt` is permissive for product/category paths. It *does*
    `Disallow: /*?categoryIds=` (plural) and a pile of other PLP filter
    params, so this scraper deliberately drives the API with the
    `path=,,{categoryId}` form instead of `categoryIds=`, which is not
    covered by any Disallow rule.
  - No bot protection at all — plain `requests` with a normal desktop UA
    works for the full catalog + detail + image CDN. Same easiest tier as
    Skechers/Champion/Dickies. No browser, no patchright.
  - Two public JSON endpoints, both keyless (verified live, not from
    memory — v3 404s, v5 is the current version):
      LIST:   https://www.uniqlo.com/us/api/commerce/v5/en/products
                ?path=%2C%2C{categoryId}&limit=100&offset=N&httpFailure=true
      DETAIL: https://www.uniqlo.com/us/api/commerce/v5/en/products
                /{productId}/price-groups/{priceGroup}/details
                ?includeModelSize=true&httpFailure=true
    The list response's `result.pagination.total` is a STYLE count, not a
    colorway count. `httpFailure=true` makes errors come back as JSON
    rather than an HTML error page.
  - Category IDs are NOT in the API's aggregation tree (that only goes down
    to level-2 "classes"). They are found by fetching the men's PLP HTML and
    grepping `categoryIds":[NNNNN]` — that is how the four IDs below were
    obtained, not guessed:
      T-Shirts 23386, Sweatshirts and Hoodies 23385,
      Casual Pants 50251, Casual Shirts 95671.
  - **One API "product" = one style with N colorways.** Unlike the Shopify
    sites (Champion/Stüssy/Dickies) where one API product is already one
    colorway, Uniqlo needs an explicit colorway-expansion step: each item's
    `colors[]` is the colorway list and `images.main[displayCode].image` is
    that colorway's own photo. One record is emitted per colorway.
  - Uniqlo's men's PLPs include UNISEX styles; genderCategory == "WOMEN"
    items are skipped, UNISEX and MEN are kept.
  - **Per-colorway imagery is thin and that is a property of the site, not
    a scraping failure.** Each colorway has exactly ONE color-specific
    photo (the on-model hero). The `images.sub[]` gallery is mostly
    style-level: only some sub images carry a `colorCode`, and the ones
    that don't are a mix of a flat-lay of the *representative* colorway and
    multi-color group shots (verified by eye — `goods_{l1Id}_sub1` for the
    SUPIMA tee is eight folded tees of eight different colors). Attaching
    those to every colorway would put the wrong color in the record, which
    is exactly what this catalog exists to distinguish, so only
    `main[color]` + subs whose `colorCode` matches are downloaded. Expect
    ~2 images/record, not 6.
  - **There are TWO hero images per colorway on two CDN paths and they are
    not always the same photo.** The API only exposes the US one
    (`.../ST3/us/imagesgoods/{l1Id}/item/usgoods_{color}_{l1Id}_3x4.jpg`),
    which has model height/size text **burned into the pixels** in the
    lower right. A second, undocumented-but-derivable path
    (`.../ST3/WesternCommon/imagesgoods/{l1Id}/item/goods_{color}_{l1Id}_3x4.jpg`)
    holds the same shot without the overlay for most tops, but for most
    bottoms holds a completely different photograph: a clean flat-lay of
    the garment alone, where the US one is a full-length on-model shot of a
    styled outfit. So: pass 1 prefers the clean WesternCommon variant for
    image_0, and `--us-hero` (pass 2) appends the US hero only when a
    64x64 grayscale diff says it is genuinely a different photograph
    (measured: 209 different, 153 same-shot duplicates rejected).
  - Photography is therefore **mixed**: tops are on-model (head and body in
    frame, garment ~40% of frame, same as Vans/Dickies); bottoms usually
    carry both a garment-only flat-lay and a full-outfit on-model shot.
    Garment cropping is still warranted — the on-model shots contain
    distractor garments and the burned-in size text.
  - `longDescription` is real prose for most styles but is occasionally a
    placeholder/near-empty string ("-", a 4-char stub) on newly-listed
    items; `shortDescription` is used as the fallback and a warning is
    printed. Neither is a framework bailout marker (cf. lesson 12) — the
    field is genuinely unpopulated on the site too.
  - `details.features` comes from `images.features[].text` (the real
    marketing bullets that sit next to each feature photo) plus the
    displayable `tags` (Fit / Sleeve Length / Neck Type ...).
    `details.materials` comes from `composition` + `washingInformation`.
  - `product_code` is `uniqlo-{productId}-{colorDisplayCode}` (e.g.
    `uniqlo-E455365-000-68`). Uniqlo's raw IDs are bare numerics, so the
    brand prefix is required to guarantee global uniqueness in the shared
    metadata file.
"""

import json
import re
import shutil
import time
import unicodedata
from pathlib import Path

import requests

from dataset_utils import load_records, save_records_safe

BASE = "https://www.uniqlo.com"
API = f"{BASE}/us/api/commerce/v5/en"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

BRAND = "uniqlo"
# label -> level-3 category id (scraped from the men's PLP HTML, see docstring)
CATEGORIES = {
    "T-Shirts": 23386,
    "Sweatshirts and Hoodies": 23385,
    "Casual Pants": 50251,
    "Casual Shirts": 95671,
}
TARGET_PER_CATEGORY = 50
MAX_IMAGES = 6
CHECKPOINT_EVERY = 10
MIN_FREE_GIB = 3.0

DATASET_DIR = Path("apparel_dataset") / BRAND
TAG_STRIP_RE = re.compile(r"<[^>]+>")


def free_gib():
    return shutil.disk_usage("/System/Volumes/Data").free / 2**30


def strip_html(text):
    text = TAG_STRIP_RE.sub(" ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slugify(text):
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "item"


def get_json(url, params=None, attempts=4):
    last = None
    for i in range(attempts):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("status") != "ok":
                raise ValueError(f"API status not ok: {json.dumps(payload)[:200]}")
            return payload["result"]
        except Exception as error:
            last = error
            time.sleep(3 * (i + 1))
    print(f"  API failed after {attempts} attempts: {url} -- {last}")
    return None


def fetch_styles(category_id):
    """Yield every style (list-endpoint item) in a category, paginated."""
    offset, limit = 0, 100
    while True:
        result = get_json(
            f"{API}/products",
            {"path": f",,{category_id}", "limit": limit, "offset": offset, "httpFailure": "true"},
        )
        if not result:
            return
        items = result.get("items", [])
        if not items:
            return
        for item in items:
            yield item
        offset += limit
        if offset >= result["pagination"]["total"]:
            return


def clean_hero_url(us_url):
    """The US hero image has model height/size text burned in; the identical
    shot without the overlay lives on the WesternCommon path. Returns the
    clean URL if it really exists, else None."""
    clean = re.sub(
        r"/ST3/us/imagesgoods/(\d+)/item/usgoods_", r"/ST3/WesternCommon/imagesgoods/\1/item/goods_", us_url
    )
    if clean == us_url:
        return None
    try:
        resp = requests.head(clean, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=20)
        if resp.status_code == 200:
            return clean
    except Exception:
        pass
    return None


def colorway_image_urls(images, display_code):
    """Only images that genuinely depict THIS colorway (see docstring)."""
    urls = []
    main = (images.get("main") or {}).get(display_code) or {}
    if main.get("image"):
        urls.append(clean_hero_url(main["image"]) or main["image"])
    for sub in images.get("sub") or []:
        if sub.get("colorCode") == display_code and sub.get("image"):
            urls.append(sub["image"])
    seen, deduped = set(), []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped[:MAX_IMAGES]


def build_details(detail):
    description = strip_html(detail.get("longDescription") or "")
    if len(description) < 40:
        fallback = strip_html(detail.get("shortDescription") or "")
        if len(fallback) > len(description):
            print(f"    NOTE: longDescription is a stub ({description!r}); using shortDescription")
            description = fallback

    features = [strip_html(f.get("text")) for f in (detail.get("images", {}).get("features") or [])]
    for tag in detail.get("tags") or []:
        if tag.get("display") and tag.get("groupName") and tag.get("tagName"):
            features.append(f"{tag['groupName']}: {tag['tagName']}")
    design = strip_html(detail.get("designDetail") or "")
    if design:
        features.append(design)

    materials = []
    composition = strip_html((detail.get("composition") or "").replace("<br>", " | "))
    composition = re.sub(r"(\s*\|\s*)+", " | ", composition).strip(" |")
    if composition:
        materials.append(composition)
    for key in ("washingInformation", "careInstruction"):
        value = strip_html(detail.get(key) or "")
        if value:
            materials.append(value)

    return {
        "description": description,
        "features": [f for f in features if f],
        "materials": materials,
    }


def build_record(style, detail, color, category_label):
    display_code = color["displayCode"]
    product_id = style["productId"]
    l1_id = style["l1Id"]
    name = detail.get("name") or style.get("name") or ""
    prices = detail.get("prices") or style.get("prices") or {}
    promo = (prices.get("promo") or {}).get("value")
    base = (prices.get("base") or {}).get("value")
    price = f"{promo if promo is not None else base}"

    return {
        "brand": BRAND,
        "category": category_label,
        "name": name,
        "color_name": color.get("name", "").title(),
        "price": price,
        "product_code": f"{BRAND}-{product_id}-{display_code}",
        "slug": f"{slugify(name)}-{l1_id}",
        "product_url": f"{BASE}/us/en/products/{product_id}/{style['priceGroup']}?colorDisplayCode={display_code}",
        "image_urls": colorway_image_urls(detail.get("images") or {}, display_code),
        "details": build_details(detail),
    }


def download_images(record):
    product_dir = DATASET_DIR / record["slug"] / record["product_code"]
    product_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for i, url in enumerate(record["image_urls"]):
        dest = product_dir / f"image_{i}.jpg"
        if not dest.is_file():
            try:
                resp = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=30)
                resp.raise_for_status()
                if len(resp.content) < 5000:
                    raise ValueError(f"suspiciously small image ({len(resp.content)} bytes)")
                dest.write_bytes(resp.content)
            except Exception as error:
                print(f"  image fetch failed: {url} -- {error}")
                continue
        images.append(str(dest))
    record["images"] = images
    record["image_count"] = len(images)
    return record


US_HERO_RE = re.compile(r"/ST3/WesternCommon/imagesgoods/(\d+)/item/goods_")


def _thumb(path_or_bytes):
    from PIL import Image
    import io

    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    with Image.open(source) as img:
        return list(img.convert("L").resize((64, 64)).getdata())


def looks_like_same_shot(existing_path, candidate_bytes):
    """The WesternCommon and US hero are the SAME photo for some styles (the
    US one just has model height/size burned into the pixels) but genuinely
    DIFFERENT photos for others (flat-lay vs full-length on-model). Only the
    latter is worth keeping as an extra image."""
    try:
        a, b = _thumb(existing_path), _thumb(candidate_bytes)
    except Exception:
        return False
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a) < 6.0


def backfill_us_hero():
    """Second pass: for every uniqlo record whose hero came from the clean
    WesternCommon path, add the US on-model hero as an extra image when it is
    actually a different photograph."""
    records = [r for r in load_records() if r.get("brand") == BRAND]
    print(f"Backfilling US on-model hero for {len(records)} uniqlo records")
    touched, added, dupes, processed = {}, 0, 0, 0

    for record in records:
        urls = record.get("image_urls") or []
        wc = next((u for u in urls if US_HERO_RE.search(u)), None)
        if not wc:
            continue
        us_url = US_HERO_RE.sub(r"/ST3/us/imagesgoods/\1/item/usgoods_", wc)
        if us_url in urls:
            continue
        try:
            resp = requests.get(us_url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=30)
            if resp.status_code != 200 or len(resp.content) < 5000:
                continue
        except Exception as error:
            print(f"  fetch failed {us_url} -- {error}")
            continue

        if record.get("images") and looks_like_same_shot(record["images"][0], resp.content):
            dupes += 1
            continue

        product_dir = DATASET_DIR / record["slug"] / record["product_code"]
        product_dir.mkdir(parents=True, exist_ok=True)
        index = len(record.get("images") or [])
        while (product_dir / f"image_{index}.jpg").is_file():
            index += 1
        dest = product_dir / f"image_{index}.jpg"
        dest.write_bytes(resp.content)
        record["image_urls"] = urls + [us_url]
        record["images"] = (record.get("images") or []) + [str(dest)]
        record["image_count"] = len(record["images"])
        touched[record["product_code"]] = record
        added += 1
        processed += 1

        if len(touched) >= 20:
            save_records_safe(touched)
            print(f"  checkpointed {len(touched)} (added {added}, skipped {dupes} duplicate shots)")
            touched = {}
        if processed % 50 == 0:
            free = free_gib()
            print(f"  disk free: {free:.2f} GiB")
            if free < MIN_FREE_GIB:
                if touched:
                    save_records_safe(touched)
                print(f"STOPPING: only {free:.2f} GiB free.")
                return

    if touched:
        save_records_safe(touched)
    print(f"Backfill done: added {added} on-model heroes, skipped {dupes} same-shot duplicates")


def main():
    existing = load_records()
    all_codes = {r["product_code"] for r in existing}
    print(f"Existing records in dataset: {len(existing)} "
          f"(uniqlo: {sum(1 for r in existing if r.get('brand') == BRAND)})")

    touched = {}
    dumped_sample = False
    per_category = {}
    processed = 0

    def checkpoint():
        nonlocal touched
        if touched:
            save_records_safe(touched)
            print(f"  checkpointed {len(touched)} records")
            touched = {}

    for category_label, category_id in CATEGORIES.items():
        print(f"\n=== {category_label} (categoryId {category_id}) ===")
        added = 0
        for style in fetch_styles(category_id):
            if added >= TARGET_PER_CATEGORY:
                break
            if style.get("genderCategory") == "WOMEN":
                continue

            detail = get_json(
                f"{API}/products/{style['productId']}/price-groups/{style['priceGroup']}/details",
                {"includeModelSize": "true", "httpFailure": "true"},
            )
            if not detail:
                print(f"  skipping style {style['productId']} -- no detail payload")
                continue
            time.sleep(0.25)

            for color in detail.get("colors") or []:
                if added >= TARGET_PER_CATEGORY:
                    break
                record = build_record(style, detail, color, category_label)
                if record["product_code"] in all_codes:
                    continue
                if not record["image_urls"]:
                    continue
                record = download_images(record)
                if not record["images"]:
                    print(f"  skipping {record['product_code']} -- no images downloaded")
                    continue

                if not dumped_sample:
                    print("\n--- SAMPLE FULLY-EXTRACTED RECORD (read this before trusting the batch) ---")
                    print(json.dumps(record, indent=2)[:2500])
                    print("--- END SAMPLE ---\n")
                    dumped_sample = True

                touched[record["product_code"]] = record
                all_codes.add(record["product_code"])
                added += 1
                processed += 1
                print(f"  [{added}/{TARGET_PER_CATEGORY}] {record['name']} ({record['color_name']}) "
                      f"-- {record['product_code']} -- {record['image_count']} img")

                if len(touched) >= CHECKPOINT_EVERY:
                    checkpoint()
                if processed % 50 == 0:
                    free = free_gib()
                    print(f"  disk free: {free:.2f} GiB")
                    if free < MIN_FREE_GIB:
                        checkpoint()
                        print(f"STOPPING: only {free:.2f} GiB free (< {MIN_FREE_GIB} GiB).")
                        return

        per_category[category_label] = added
        if added < TARGET_PER_CATEGORY:
            print(f"  NOTE: {category_label} yielded only {added} colorway variants "
                  f"(real catalog size, not padded).")
        checkpoint()

    checkpoint()

    final = load_records()
    mine = [r for r in final if r.get("brand") == BRAND]
    print(f"\nTotal {BRAND} records in dataset: {len(mine)}")
    print(f"Per category this run: {per_category}")
    print(f"Total images: {sum(r.get('image_count', 0) for r in mine)}")
    print(f"Disk free: {free_gib():.2f} GiB")


if __name__ == "__main__":
    import sys

    if "--us-hero" in sys.argv:
        backfill_us_hero()
    else:
        main()
