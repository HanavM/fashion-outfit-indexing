"""Dickies men's clothing scraper — Pants, Shirts, Shorts, Coats and
Jackets. Target 50 colorway variants per section -> 200 requested.

Site notes (third Shopify-storefront site in this pipeline, after
Champion and Stüssy):
  - No bot protection encountered — plain `requests` works for the entire
    catalog + product detail fetch, same easiest tier as Champion/Stüssy/
    Gap/Skechers.
  - Real collection handles found from the homepage's own nav links
    (`https://www.dickies.com/`, `href="/en-us/collections/..."`), not
    guessed. Real handles used here: `mens-pants`, `mens-shirts`,
    `mens-shorts`, `mens-coats-jackets` (each has 100+ colorway variants
    per a `products.json?limit=250` count check before scraping --
    mens-pants/mens-shirts both hit the 250-per-page cap, meaning real
    inventory is even larger and pagination is needed to reach it).
  - Standard Shopify storefront JSON API, same pattern as Champion/
    Stüssy: `https://www.dickies.com/en-us/collections/{handle}/
    products.json?limit=250&page=N`, paginated until a page returns zero
    products. Note the `/en-us/` locale prefix in the URL path -- unlike
    Champion/Stüssy's bare `/collections/...`, Dickies's storefront is
    locale-scoped and the products.json endpoint 404s without it.
  - Each Shopify "product" here is already ONE colorway (`options[0]` =
    Color with exactly one value per product, color baked into every
    variant's `option1`) -- same "one API product = one dataset record"
    shape as Champion/Stüssy, no colorway-expansion step needed. Numeric
    `id` is the stable `product_code`; `handle` is the slug.
  - Full product detail (description, all images, size/price/SKU data)
    is inline in the SAME products.json response -- no separate PDP
    visit needed, same as Champion/Stüssy.
  - `body_html` is a real paragraph (not a bullet list like Stüssy's),
    stripped of HTML tags for `details.description`, same strip_html()
    approach.
  - Price: `variants[0]['price']`, plain decimal string.
  - Sizes/inseams are separate `options` entries (Size, Inseam for
    pants) nested under one colorway product -- consistent with the
    "one product = one color, many size variants" Shopify shape already
    established, no special handling needed since only the product-level
    fields (not per-size variant fields) are scraped.
"""

import html
import re
import time
from pathlib import Path

import requests

from dataset_utils import load_records, save_records_safe

BASE = "https://www.dickies.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

CATEGORIES = {
    "Pants": "mens-pants",
    "Shirts": "mens-shirts",
    "Shorts": "mens-shorts",
    "Coats and Jackets": "mens-coats-jackets",
}
TARGET_PER_CATEGORY = 50

DATASET_DIR = Path("apparel_dataset/dickies")
TAG_STRIP_RE = re.compile(r"<[^>]+>")


def strip_html(text):
    text = TAG_STRIP_RE.sub(" ", text or "")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_collection_products(handle):
    page = 1
    while True:
        url = f"{BASE}/en-us/collections/{handle}/products.json"
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
        "brand": "dickies",
        "category": category_label,
        "name": product.get("title", ""),
        "color_name": color_name,
        "price": price,
        "product_code": product_code,
        "slug": slug,
        "product_url": f"{BASE}/en-us/products/{slug}",
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
    existing_codes = {r["product_code"] for r in existing if r.get("brand") == "dickies"}
    print(f"Existing dickies records: {len(existing_codes)}")

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
                continue
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
                  f"(catalog-size-capped, same situation PacSun/Gap/Levi's sections hit).")

    if touched:
        save_records_safe(touched)
        print(f"Final checkpoint: {len(touched)} records")

    final = load_records()
    dickies_count = sum(1 for r in final if r.get("brand") == "dickies")
    print(f"\nTotal dickies records in dataset: {dickies_count}")


if __name__ == "__main__":
    main()
