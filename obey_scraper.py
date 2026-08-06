"""OBEY Clothing men's scraper — T-Shirts, Sweatshirts, Pants, Shorts.
Target 50 colorway variants per category -> 200 requested.

Site notes (fourth Shopify-storefront site in this pipeline, after
Champion, Stüssy and Dickies):
  - **`shop.obeyclothing.com` does not resolve** (curl exits 000 — no DNS/
    TLS). The live storefront is `https://obeyclothing.com`
    (`www.obeyclothing.com` 301s to the apex). Don't trust the `shop.`
    subdomain that older references mention.
  - **No bot protection at all** — plain `requests` with a normal desktop
    UA works for the whole catalog. Easiest tier, same as Champion/
    Stüssy/Dickies/Gap/Skechers. `robots.txt` is Shopify's newer
    agent-aware boilerplate: `User-agent: * / Allow: /`, with an explicit
    note that "public product, collection, page, blog, policy, cart and
    localized HTML is crawlable" (it only forbids *automated checkout*,
    which this scraper never touches).
  - **Real collection handles taken from the homepage nav** (`href=
    "/collections/..."`) and cross-checked against the site's own
    `https://obeyclothing.com/collections.json?limit=250` index (101
    collections) — not guessed. No locale prefix (unlike Dickies's
    `/en-us/`); bare `/collections/{handle}/products.json?limit=250&page=N`
    works.
  - **OBEY splits one merchandising category across several collections**
    — this is the main structural quirk here. `mens-t-shirts` alone is
    only 79 products, but `classic-t-shirts`, `heavyweight-t-shirt`,
    `pigment-t-shirts`, `pigmnet-ls-t-shirts` (sic — the site's own
    typo'd handle) and `sale-t-shirts` hold the rest; likewise
    sweatshirts are split across `men-sweatshirts` / `crewneck-fleece` /
    `pullover-hood` / `zip-hood` / `sale-sweatshirts`. So each dataset
    category here is the **de-duplicated union of several handles**,
    ordered so the main handle is consumed first. Union sizes measured
    before scraping: T-Shirts 181, Sweatshirts 64, Pants 41, Shorts 41.
  - **Pants and Shorts genuinely have fewer than 50 men's colorways
    live** (41 each; the `mens-all` catalog is 608 products total and its
    `BOTTOMS` tag covers only 91 of them, which matches 41+41+overlap).
    That shortfall is real inventory, not a pagination bug — it is
    recorded, not padded.
  - **Each Shopify "product" is already ONE colorway** (`options[0]` is
    `COLOR` with exactly one value, upper-cased e.g. `RAINFOREST`), same
    "one API product = one dataset record" shape as Champion/Stüssy/
    Dickies. No colorway-expansion step.
  - **`product_code` is prefixed `obey-`.** Raw Shopify ids here are bare
    numerics, and three other brands in this dataset already contribute
    bare-numeric Shopify ids, so the prefix removes any collision risk.
    `handle` is the slug.
  - **Full product detail is inline in the same products.json response** —
    no PDP visit needed. `body_html` is a `<ul>` bullet list (Stüssy-
    shaped, not Dickies's paragraph), typically ALL-CAPS, and the last
    bullet is a bare `SKU:165264442`. Bullets are split into
    `details.features`, the fabric/weight ones are also copied into
    `details.materials`, and `details.description` is the bullets joined
    into one sentence-ish string (there is no prose description anywhere
    on this site — verified on the PDP HTML, not assumed).
  - **Images are PNGs on the Shopify CDN**, named `{sku}_{colorcode}_N.png`.
    Saved as `image_N.jpg` by filename convention like every other brand
    here (bytes are left as-is, PIL/torch read them fine by content).
    Most products carry only 2-4 images, well under the 6-image cap.
  - Photography is **mixed flat-lay/product-only and on-model** —
    verified by opening actual downloaded files, not assumed.
"""

import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

from dataset_utils import load_records, save_records_safe

BASE = "https://obeyclothing.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}

BRAND = "obey"

# Each dataset category is the de-duplicated union of several real site
# collection handles (see module docstring); main handle listed first.
CATEGORIES = {
    "T-Shirts": [
        "mens-t-shirts",
        "classic-t-shirts",
        "heavyweight-t-shirt",
        "pigment-t-shirts",
        "pigmnet-ls-t-shirts",
        "sale-t-shirts",
    ],
    "Sweatshirts": [
        "men-sweatshirts",
        "crewneck-fleece",
        "pullover-hood",
        "zip-hood",
        "sale-sweatshirts",
    ],
    "Pants": ["men-bottoms-pants", "pants", "denim-pants"],
    "Shorts": ["shorts", "regular-fit-shorts", "relaxed-fit-shorts", "baggy-fit-shorts"],
}
TARGET_PER_CATEGORY = 50
MAX_IMAGES = 6
CHECKPOINT_EVERY = 10
DISK_CHECK_EVERY = 50
MIN_FREE_GIB = 3.0

DATASET_DIR = Path(f"apparel_dataset/{BRAND}")
TAG_STRIP_RE = re.compile(r"<[^>]+>")
MATERIAL_RE = re.compile(
    r"(%|COTTON|POLYESTER|NYLON|WOOL|ACRYLIC|SPANDEX|ELASTANE|RAYON|LINEN|"
    r"DENIM|FLEECE|TWILL|CANVAS|CORDUROY|OZ\b|GSM|GRAM)",
    re.I,
)
SKU_BULLET_RE = re.compile(r"^SKU\s*:", re.I)


def free_gib():
    out = subprocess.run(
        ["df", "-g", "/System/Volumes/Data"], capture_output=True, text=True
    ).stdout.splitlines()
    return float(out[-1].split()[3])


def strip_html(text):
    text = TAG_STRIP_RE.sub(" ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_bullets(body_html):
    """OBEY's body_html is a <ul> of ALL-CAPS bullets; last one is the SKU."""
    raw = re.findall(r"<li[^>]*>(.*?)</li>", body_html or "", re.S | re.I)
    bullets = [strip_html(item) for item in raw]
    bullets = [b for b in bullets if b and not SKU_BULLET_RE.match(b)]
    if not bullets:
        flat = strip_html(body_html)
        bullets = [flat] if flat else []
    return bullets


def fetch_collection_products(handle):
    page = 1
    while True:
        url = f"{BASE}/collections/{handle}/products.json"
        resp = requests.get(
            url, headers=HEADERS, params={"limit": 250, "page": page}, timeout=30
        )
        resp.raise_for_status()
        products = resp.json().get("products", [])
        if not products:
            return
        for product in products:
            yield product
        page += 1


def build_record(product, category_label):
    product_code = f"{BRAND}-{product['id']}"
    slug = product["handle"]
    variants = product.get("variants", [])
    color_name = variants[0].get("option1", "") if variants else ""
    price = variants[0].get("price", "") if variants else ""
    image_urls = [img["src"] for img in product.get("images", [])][:MAX_IMAGES]

    bullets = parse_bullets(product.get("body_html", ""))
    materials = [b for b in bullets if MATERIAL_RE.search(b)]

    return {
        "brand": BRAND,
        "category": category_label,
        "name": product.get("title", ""),
        "color_name": color_name,
        "price": price,
        "product_code": product_code,
        "slug": slug,
        "product_url": f"{BASE}/products/{slug}",
        "image_urls": image_urls,
        "details": {
            "description": ". ".join(bullets),
            "features": bullets,
            "materials": materials,
        },
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
    dump_only = "--dump-one" in sys.argv

    existing = load_records()
    all_codes = {r["product_code"] for r in existing}
    mine = {r["product_code"] for r in existing if r.get("brand") == BRAND}
    print(f"Existing records: {len(existing)} total, {len(mine)} for {BRAND}")

    touched = {}
    added_total = 0
    summary = {}

    for category_label, handles in CATEGORIES.items():
        print(f"\n=== {category_label} ({', '.join(handles)}) ===")
        added = 0
        seen_ids = set()
        for handle in handles:
            if added >= TARGET_PER_CATEGORY:
                break
            for product in fetch_collection_products(handle):
                if added >= TARGET_PER_CATEGORY:
                    break
                if product["id"] in seen_ids:
                    continue
                seen_ids.add(product["id"])
                product_code = f"{BRAND}-{product['id']}"
                if product_code in all_codes:
                    continue
                record = build_record(product, category_label)

                if dump_only:
                    record = download_images(record)
                    print(json.dumps(record, indent=2, ensure_ascii=False))
                    return

                record = download_images(record)
                if not record["images"]:
                    print(f"  skipping {product_code} ({record['name']}) -- no images")
                    continue
                touched[product_code] = record
                all_codes.add(product_code)
                added += 1
                added_total += 1
                print(
                    f"  [{added}/{TARGET_PER_CATEGORY}] {record['name']} "
                    f"({record['color_name']}) -- {product_code} "
                    f"{record['image_count']} imgs"
                )

                if len(touched) >= CHECKPOINT_EVERY:
                    save_records_safe(touched)
                    print(f"  checkpointed {len(touched)} records")
                    touched = {}

                if added_total % DISK_CHECK_EVERY == 0:
                    avail = free_gib()
                    print(f"  disk free: {avail} GiB")
                    if avail < MIN_FREE_GIB:
                        if touched:
                            save_records_safe(touched)
                        print(f"STOPPING: disk below {MIN_FREE_GIB} GiB")
                        return
                time.sleep(0.15)

        summary[category_label] = added
        if added < TARGET_PER_CATEGORY:
            print(
                f"  NOTE: {category_label} only had {added} colorway variants "
                f"across its real collections (catalog-size-capped, not padded)."
            )

    if touched:
        save_records_safe(touched)
        print(f"Final checkpoint: {len(touched)} records")

    final = load_records()
    count = sum(1 for r in final if r.get("brand") == BRAND)
    print(f"\nPer-category added: {summary}")
    print(f"Total {BRAND} records in dataset: {count}")


if __name__ == "__main__":
    main()
