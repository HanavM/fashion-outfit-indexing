"""Brain Dead (wearebraindead.com) men's clothing scraper — T-shirt, Shirt,
Pant, Jacket. Target 50 colorway variants per category -> 200 requested;
the real catalog is much smaller than that (see "Shortfalls" below).

Site notes (sixth Shopify-storefront site in this pipeline, after Champion,
Stussy, Dickies, HUF and OBEY):

  - **Bot protection: none.** Plain `requests` + a normal desktop UA fetched
    the entire catalog and every image. No playwright/patchright needed at
    any point. Same easiest tier as Champion/Stussy/Dickies/HUF/OBEY.
    `robots.txt` is Shopify's newer agent-aware boilerplate: `User-agent: *`
    / `Allow: /`, explicitly stating "public product, collection, page,
    blog, policy, cart, and localized HTML is crawlable". The only
    prohibition is automated checkout/payment, which this scraper never
    touches. It also advertises `/agents.md` and a UCP/MCP endpoint at
    `/api/ucp/mcp` (not used — products.json is simpler and complete).

  - Storefront is **not** locale-scoped: bare
    `https://wearebraindead.com/collections/{handle}/products.json?limit=250
    &page=N` works (unlike Dickies, which 404s without `/en-us/`).

  - **GOTCHA A — `options[0]` is Size, not Color.** Every prior Shopify
    brand here had `options[0].name == "Color"`, which is why Champion/
    Stussy/Dickies all do `color = variants[0]["option1"]`. On Brain Dead
    the ONLY option is Size (verified: 0 of 510 catalog products have a
    non-Size first option). Copying that line here would have written "XS"
    into `color_name` on every single record. Brain Dead genuinely is one
    product = one colorway, but the colorway name lives in the **product
    title** after the final " - " (e.g. "Poplin Camp Collar Shirt -
    Chocolate"), mirrored in a `color:{name}` tag on ~84% of products and in
    a `<p>Color: ...</p>` line in `body_html` on only ~11%. The title tail
    is the only 100%-coverage source AND the richest ("Blue Multi" where the
    tag says just "blue"), so it is the primary; the tag is a fallback that
    in practice never fires. Verified: all 202 candidate products have a
    " - " in the title and no two share a title.

  - **GOTCHA B — `collections.json` product_counts are fiction.** The site's
    own collection index reports `shirt: 273`, `hoodie: 200`,
    `longsleeve: 339`, `tops: 1801`, `perk-collection: 3118` — but fetching
    those collections' `products.json` returns 45, 27, 4, 204 and 0
    products respectively. The whole store is 510 published products.
    Use `collections.json` to *enumerate handles*, never to size a category.

  - Because per-garment collections are small and inconsistently curated,
    categories here are built by crawling **all 97 collections**, deduping
    products by id, and grouping on the merchant's own `product_type` field
    (`t-shirt`, `shirt`, `pant`, `jacket`, ...) — which is the site's own
    taxonomy and matches its collection titles. Products appearing in the
    `women` / `womenswear` collections are excluded (34 products).

  - Full product detail (title, price, SKUs, every image URL, `body_html`)
    is inline in products.json — **no PDP visit needed**, same as
    Champion/Stussy/Dickies/HUF/OBEY.

  - `body_html` shape: one or more `<p>` prose paragraphs, occasionally with
    a `•`-bulleted `<br>` block, plus optional `Material:` / `Color:` label
    paragraphs. 0 of 202 products have an empty body; none use `<li>`.

  - **Images are already sane-sized here** — 1200x1500 JPEG, ~160 KB each,
    filenames ending `_optimized`. `&width=1200` is appended anyway (it is a
    no-op on already-1200px assets, and is the right default for any Shopify
    CDN per the OBEY lesson, where 3000x3750 PNGs cost 2.9 GB for 182
    records). Original unparameterised URLs are stored in `image_urls`.

  - Image CDN filenames are **self-describing view labels** —
    `..._Front_optimized.jpg`, `_Back_`, `_Side_`, `_Detail_`,
    `_Detail_Back_`, `_Detail_1_`. Per lesson 5 that signal dies when files
    are renamed to `image_N.jpg`; it is preserved here only because
    `image_urls` keeps the original URLs in the same order as `images`.

  - `product_code` = `braindead-{style SKU}`, where the style SKU is the
    variant SKU with its `-SIZE` suffix stripped
    (`BDW24T24003973BR15-XS` -> `BDW24T24003973BR15`). Every candidate
    product has a non-empty SKU and all stripped SKUs are unique. The
    numeric Shopify product id was avoided because three other brands
    already contribute bare-numeric ids to metadata.json.

  - Shortfalls (real inventory, not padded): the entire men's Brain Dead
    catalog holds 70 t-shirts, 37 shirts, 36 pants and 33 jackets. Only
    T-shirt can reach the 50 target.
"""

import html
import json
import re
import shutil
import time
from pathlib import Path

import requests

from dataset_utils import load_records, save_records_safe

BASE = "https://wearebraindead.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

BRAND = "braindead"
DATASET_DIR = Path("apparel_dataset") / BRAND

# dataset category label -> merchant `product_type` values that feed it.
# Labels are the site's own collection titles for those types.
CATEGORIES = {
    "T-shirt": {"t-shirt"},
    "Shirt": {"shirt"},
    "Pant": {"pant"},
    "Jacket": {"jacket"},
}
TARGET_PER_CATEGORY = 50
MAX_IMAGES = 6
IMAGE_WIDTH = 1200
EXCLUDE_COLLECTIONS = {"women", "womenswear", "kid"}

MIN_FREE_GIB = 3.0
TAG_STRIP_RE = re.compile(r"<[^>]+>")
SIZE_SUFFIX_RE = re.compile(
    r"-(?:XXXS|XXS|XS|S|M|L|XL|XXL|XXXL|2XL|3XL|4XL|OS|ONESIZE|\d{2,3})$", re.I
)
MATERIAL_RE = re.compile(
    r"(\d+\s*%|cotton|polyester|nylon|wool|denim|fleece|linen|leather|"
    r"cashmere|mohair|acrylic|rayon|viscose|spandex|elastane|oz\b|gsm)",
    re.I,
)


def free_gib():
    return shutil.disk_usage("/System/Volumes/Data").free / (1024**3)


def check_disk(context=""):
    free = free_gib()
    if free < MIN_FREE_GIB:
        raise SystemExit(
            f"STOPPING FOR DISK: only {free:.2f} GiB free (< {MIN_FREE_GIB} GiB) {context}"
        )
    return free


def strip_html(text):
    text = TAG_STRIP_RE.sub(" ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def get_json(url, params=None, attempts=4):
    for attempt in range(attempts):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as error:
            if attempt == attempts - 1:
                print(f"  giving up on {url}: {error}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def crawl_catalog():
    """Return (products_by_id, membership_by_id) across every collection.

    The per-garment collections are small and the collections.json counts are
    wrong (GOTCHA B), so the only reliable inventory is the union of all
    collections deduped by product id.
    """
    index = get_json(f"{BASE}/collections.json", {"limit": 250})
    handles = [c["handle"] for c in (index or {}).get("collections", [])]
    print(f"Collections advertised: {len(handles)}")

    products, membership = {}, {}
    for handle in handles:
        for page in range(1, 8):
            data = get_json(
                f"{BASE}/collections/{handle}/products.json",
                {"limit": 250, "page": page},
            )
            batch = (data or {}).get("products", [])
            if not batch:
                break
            for product in batch:
                products[product["id"]] = product
                membership.setdefault(product["id"], set()).add(handle)
            if len(batch) < 250:
                break
        time.sleep(0.1)
    print(f"Unique published products in catalog: {len(products)}")
    return products, membership


def color_from(product):
    """Colorway name. Title tail is the only 100%-coverage source here; the
    `color:` tag is a fallback that never actually fires on the selected set."""
    title = product.get("title", "")
    if " - " in title:
        return title.rsplit(" - ", 1)[1].strip()
    for tag in product.get("tags", []):
        if tag.lower().startswith("color:"):
            return tag.split(":", 1)[1].strip().title()
    return ""


def style_code(product):
    for variant in product.get("variants", []):
        sku = (variant.get("sku") or "").strip()
        if sku:
            return SIZE_SUFFIX_RE.sub("", sku)
    return ""


def parse_body(body_html):
    """<p> prose + optional bullet block + optional Material:/Color: labels."""
    blocks = re.split(r"</p\s*>|<br\s*/?>", body_html or "")
    description_parts, features, materials = [], [], []
    for raw in blocks:
        text = strip_html(raw).lstrip("•").strip()
        if not text:
            continue
        low = text.lower()
        if low.startswith("color:"):
            continue
        if low.startswith("material:") or low.startswith("fabric:"):
            value = text.split(":", 1)[1].strip()
            if value:
                materials.append(value)
            continue
        if "•" in raw or len(text) < 90:
            features.append(text)
            if MATERIAL_RE.search(text):
                materials.append(text)
        else:
            description_parts.append(text)

    description = " ".join(description_parts).strip()
    if not description:
        # No prose paragraph -- reuse the bullets so captioning has something.
        description = ". ".join(features)
    return {
        "description": description,
        "features": features,
        "materials": list(dict.fromkeys(materials)),
    }


def sized(url):
    """Ask the Shopify CDN for ~1200px on the long edge (OBEY lesson)."""
    return url + ("&" if "?" in url else "?") + f"width={IMAGE_WIDTH}"


def build_record(product, category_label):
    code = f"{BRAND}-{style_code(product)}"
    slug = product["handle"]
    variants = product.get("variants", [])
    image_urls = [img["src"] for img in product.get("images", [])][:MAX_IMAGES]
    return {
        "brand": BRAND,
        "category": category_label,
        "name": product.get("title", ""),
        "color_name": color_from(product),
        "price": variants[0]["price"] if variants else "",
        "product_code": code,
        "slug": slug,
        "product_url": f"{BASE}/products/{slug}",
        "image_count": 0,
        "images": [],
        "image_urls": image_urls,
        "details": parse_body(product.get("body_html", "")),
    }


def download_images(record):
    product_dir = DATASET_DIR / record["slug"] / record["product_code"]
    product_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, url in enumerate(record["image_urls"]):
        dest = product_dir / f"image_{i}.jpg"
        if not dest.is_file():
            try:
                resp = requests.get(sized(url), headers=HEADERS, timeout=45)
                resp.raise_for_status()
                if not resp.content.startswith((b"\xff\xd8", b"\x89PNG", b"RIFF")):
                    print(f"  not an image: {url}")
                    continue
                dest.write_bytes(resp.content)
            except Exception as error:
                print(f"  image fetch failed: {url} -- {error}")
                continue
        paths.append(str(dest))
    record["images"] = paths
    record["image_count"] = len(paths)
    return record


def main():
    print(f"Disk free at start: {check_disk('(startup)'):.2f} GiB")

    existing = load_records()
    all_codes = {r["product_code"] for r in existing}
    print(f"Existing records in metadata.json: {len(existing)}")

    products, membership = crawl_catalog()
    excluded = {
        pid
        for pid, handles in membership.items()
        if handles & EXCLUDE_COLLECTIONS
    }
    print(f"Excluded as women's/kids': {len(excluded)}")

    # sanity assertion for GOTCHA A -- never silently reuse the Dickies shape
    non_size = [
        p["title"]
        for p in products.values()
        if p.get("options") and p["options"][0]["name"] != "Size"
    ]
    print(f"Products whose options[0] is not Size: {len(non_size)}")

    touched, counts, dumped = {}, {}, False
    for category_label, types in CATEGORIES.items():
        pool = sorted(
            (
                p
                for pid, p in products.items()
                if pid not in excluded
                and (p.get("product_type") or "").lower() in types
                and p.get("images")
            ),
            key=lambda p: p["handle"],
        )
        print(f"\n=== {category_label}: {len(pool)} products in catalog ===")
        added = 0
        for product in pool:
            if added >= TARGET_PER_CATEGORY:
                break
            record = build_record(product, category_label)
            if not style_code(product):
                print(f"  skip (no SKU): {record['name']}")
                continue
            if record["product_code"] in all_codes:
                print(f"  skip (code exists): {record['product_code']}")
                continue
            record = download_images(record)
            if not record["images"]:
                print(f"  skip (no images downloaded): {record['name']}")
                continue

            if not dumped:
                print("\n--- FULL FIRST RECORD (read this before trusting the batch) ---")
                print(json.dumps(record, indent=2, ensure_ascii=False))
                print("--- end first record ---\n")
                dumped = True

            touched[record["product_code"]] = record
            all_codes.add(record["product_code"])
            added += 1
            print(
                f"  [{added}] {record['name']} | color={record['color_name']!r} "
                f"| {record['image_count']} imgs | {record['product_code']}"
            )

            if len(touched) >= 15:
                save_records_safe(touched)
                free = check_disk("(checkpoint)")
                print(f"  checkpointed {len(touched)} records | {free:.2f} GiB free")
                touched = {}
            time.sleep(0.15)

        counts[category_label] = added
        if added < TARGET_PER_CATEGORY:
            print(
                f"  SHORTFALL: {category_label} yielded {added}/{TARGET_PER_CATEGORY} "
                f"-- the whole men's catalog only has {len(pool)} of this type."
            )

    if touched:
        save_records_safe(touched)
        print(f"Final checkpoint: {len(touched)} records")

    final = load_records()
    mine = [r for r in final if r.get("brand") == BRAND]
    print(f"\n=== {BRAND}: {len(mine)} records, {sum(r['image_count'] for r in mine)} images ===")
    for label, n in counts.items():
        print(f"  {label:10s} {n}")
    codes = [r["product_code"] for r in final]
    print(f"Duplicate product_codes in metadata.json: {len(codes) - len(set(codes))}")
    print(f"Disk free at end: {free_gib():.2f} GiB")


if __name__ == "__main__":
    main()
