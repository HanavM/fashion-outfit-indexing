"""
Champion men's clothing scraper — Hoodies and Sweatshirts, T-Shirts and
Tops, Shorts, Pants and Joggers. Target 50 colorway variants per section ->
200 requested.

Site notes (first Shopify-storefront site in this pipeline):
  - No bot protection encountered at all — plain `requests` works for the
    entire catalog + product detail fetch, no browser needed anywhere
    (same easiest tier as Skechers/Gap).
  - Shopify's standard storefront JSON API does all the work:
    `https://www.champion.com/collections/{handle}/products.json?limit=250
    &page=N`, paginated until a page returns zero products. Collection
    handles found by scraping `<a href="/collections/...">` out of any
    catalog page's rendered nav (a 404 page still rendered the full site
    nav in this case, which is where the handle list below came from).
    Men's-scoped handles used here: `mens-hoodies-sweatshirts`,
    `mens-t-shirt-tops`, `mens-shorts`, `mens-pants` (picked over the
    unscoped `joggers`/`sweatpants` handles, which mix genders).
  - Each Shopify "product" in this catalog is already ONE colorway (color
    baked into the title/handle, `options[0]` = Color with exactly one
    value) — same "one PDP per colorway" shape as PacSun/Gap, not the
    multi-colorway-per-grouping shape Nike/New Balance use. No
    colorway-expansion step needed: one API product = one dataset record.
    Numeric `id` is the stable `product_code`; `handle` is the slug.
  - Full product detail (description, all variant images, all
    size/price/SKU data) is inline in the SAME `products.json` response —
    no separate PDP visit needed at all, unlike every other brand in this
    pipeline (Gap/PacSun/New Balance all need a second per-product fetch
    for description bullets or SFCC detail data). `body_html` is a single
    descriptive paragraph (no bullet list of materials/features on this
    site), stripped of HTML tags for `details.description`.
  - Images are self-describing by filename position, printed in the CDN
    URL itself (`..._Front1_...`, `..._Front2_...`, `..._Back1_...`,
    `..._Back2_...`, `..._Detail_...`, `..._Full_Length_...`) — same kind
    of real view-angle signal Skechers has, captured here via `image_urls`
    (the raw CDN URLs, filenames intact) alongside the renamed
    `image_N.jpg` files on disk, per SCRAPING_PROCESS.md lesson #5 (view
    signal is lost forever if you don't save it before renaming).
  - Price: `variants[0]['price']` (a plain decimal string, no parsing
    needed, unlike Adidas's label-prefixed price text).
"""

import html
import json
import re
import time
from pathlib import Path

import requests

from dataset_utils import load_records, save_records_safe

BASE = "https://www.champion.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

CATEGORIES = {
    "Hoodies and Sweatshirts": "mens-hoodies-sweatshirts",
    "T-Shirts and Tops": "mens-t-shirt-tops",
    "Shorts": "mens-shorts",
    "Pants and Joggers": "mens-pants",
}
TARGET_PER_CATEGORY = 50

DATASET_DIR = Path("apparel_dataset/champion")
TAG_STRIP_RE = re.compile(r"<[^>]+>")


def strip_html(text):
    text = TAG_STRIP_RE.sub(" ", text or "")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_collection_products(handle):
    """Generator over every product in a Shopify collection, paging until
    a page comes back empty."""
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


def build_record(product, category_label):
    product_code = str(product["id"])
    slug = product["handle"]
    variants = product.get("variants", [])
    color_name = variants[0]["option1"] if variants else ""
    price = variants[0]["price"] if variants else ""
    image_urls = [img["src"] for img in product.get("images", [])]

    return {
        "brand": "champion",
        "category": category_label,
        "name": product.get("title", ""),
        "color_name": color_name,
        "price": price,
        "product_code": product_code,
        "slug": slug,
        "product_url": f"{BASE}/products/{slug}",
        "image_urls": image_urls,
        "details": {"description": strip_html(product.get("body_html", ""))},
    }


def download_images(record):
    product_dir = DATASET_DIR / record["slug"] / record["product_code"]
    product_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for i, url in enumerate(record["image_urls"]):
        dest = product_dir / f"image_{i}.jpg"
        if not dest.is_file():
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
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
    existing_codes = {r["product_code"] for r in existing if r.get("brand") == "champion"}
    print(f"Existing champion records: {len(existing_codes)}")

    touched = {}
    checkpoint_every = 10

    for category_label, handle in CATEGORIES.items():
        print(f"\n=== {category_label} ({handle}) ===")
        added_this_category = 0
        for product in fetch_collection_products(handle):
            if added_this_category >= TARGET_PER_CATEGORY:
                break
            product_code = str(product["id"])
            if product_code in existing_codes:
                continue  # already scraped, doesn't count against target
            record = build_record(product, category_label)
            record = download_images(record)
            if not record["images"]:
                print(f"  skipping {product_code} ({record['name']}) -- no images downloaded")
                continue
            touched[product_code] = record
            existing_codes.add(product_code)
            added_this_category += 1
            print(f"  [{added_this_category}/{TARGET_PER_CATEGORY}] {record['name']} ({record['color_name']}) -- {product_code}")

            if len(touched) >= checkpoint_every:
                save_records_safe(touched)
                print(f"  checkpointed {len(touched)} records")
                touched = {}
            time.sleep(0.2)

        if added_this_category < TARGET_PER_CATEGORY:
            print(f"  NOTE: {category_label} only had {added_this_category} new colorway variants reachable "
                  f"(catalog-size-capped, same situation PacSun/Gap/Levi's sweaters hit).")

    if touched:
        save_records_safe(touched)
        print(f"Final checkpoint: {len(touched)} records")

    final = load_records()
    champion_count = sum(1 for r in final if r.get("brand") == "champion")
    print(f"\nTotal champion records in dataset: {champion_count}")


if __name__ == "__main__":
    main()
