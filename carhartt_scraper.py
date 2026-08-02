"""Carhartt WIP (carhartt-wip.com) men's clothing scraper -- Jackets and
Coats, Pants, Shirts, T-Shirts and Polos. Target 50 colorway variants per
section -> 200 requested (a section may fall short if its real catalog
is smaller, same "capped by catalog size" situation several prior brands
in this pipeline hit).

Site notes (first commercetools-backed site in this pipeline):
  - No real bot protection encountered -- plain `playwright` (headless,
    --disable-blink-features=AutomationControlled), same tier as Nike/Gap.
    A fixed-position `[data-rac]` overlay (cookie-consent/region-select
    modal) intercepts clicks on first load; removed via a one-line
    `page.evaluate()` before any interaction, no dismiss-button click
    needed.
  - Next.js App Router (React Server Components) -- NOT the classic
    `__NEXT_DATA__` script tag pattern (absent entirely here). Category/
    listing pages render product tiles as real `<a href="/en-de/p/...">`
    anchors in the DOM once loaded, though -- no need to parse the RSC
    stream directly.
  - URL scheme is commercetools-flavored: category listings live at
    `/en-de/c/{category-slug}` (discovered via real nav hrefs, not
    guessed -- `/en/men/clothing/jackets`-style guesses 404), paginated
    via `?page=N` (48 products/page). Each listing link is already a
    distinct per-colorway PDP (`/en-de/p/{slug}-{trailing-id}`), same
    "one PDP = one colorway, no separate expansion step" shape as
    PacSun/Gap -- unlike New Balance/Levi's where one PDP inlines every
    colorway via ld+json `hasVariant`.
  - Real, stable data source is a schema.org ld+json `Product` block
    (not `ProductGroup`) on every PDP: `name`, `image` (small preset
    URLs, upgraded below), `sku`, `size`, `material`, `color`, `brand`,
    `offers.price`/`priceCurrency`. No `description` field in the
    ld+json itself.
  - Description + feature bullets live in a native HTML `<details>`
    element (`Details` accordion) -- content is present in the DOM
    regardless of open/collapsed state (that's the whole point of a
    native `<details>` tag, unlike Adidas/New Balance's JS-gated
    accordions), so `element.innerText` reads it directly with no click
    needed. First `<details>` block only; `Material & Care`/`Size & Fit`/
    `Additional Information` are separate `<details>` blocks not scraped
    here (thin/redundant with `material` already in the ld+json).
  - Image CDN is Amplience (`cdn.media.amplience.net`), a different
    dynamic-imaging vendor than the Scene7 (Adobe)/Shopify CDNs seen on
    other brands. ld+json image URLs carry a small schema-markup preset
    query string (`?$google_struc_main_of$`) -- strip it and append
    `?w=1600&fmt=auto&qlt=default` for full-res, same "always strip
    existing query string first" convention as Gap/Levi's zoom sizing.
    **Enforces hotlink protection**: a plain `requests.get()` with only a
    User-Agent header gets a 403 (confirmed via direct testing) -- a
    `Referer` header pointing at the site's own domain is required and
    sufficient, no cookies/session state needed.
  - `product_code`: the ld+json `sku` field ("I037532_1") is NOT
    per-colorway unique (style-level only). The real stable per-colorway
    identifier is the shared filename prefix across all of a PDP's own
    images -- e.g. every image URL for one PDP shares "I037532_453_02"
    before the "-OF-NN"/"-ST-NN" view-type suffix. Extracted via regex
    from the first ld+json image URL; falls back to the PDP URL's
    trailing numeric ID if that regex ever fails to match (should not
    happen in practice, kept only as a defensive fallback).
"""

import re
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

from dataset_utils import load_records, save_records_safe

CATEGORIES = {
    "men-jackets-and-coats": "Jackets and Coats",
    "men-pants": "Pants",
    "men-shirts": "Shirts",
    "men-tshirts-and-polos": "T-Shirts and Polos",
}
TARGET_PER_CATEGORY = 50
CHECKPOINT_EVERY = 10

BASE = "https://www.carhartt-wip.com"
IMAGE_CODE_PATTERN = re.compile(r"/i/carhartt_wip/(.+?)-(?:OF|ST)-\d+")
TRAILING_ID_PATTERN = re.compile(r"-(\d+)$")


def remove_overlays(page):
    page.evaluate(
        "document.querySelectorAll('[data-rac]').forEach(e => { "
        "if (getComputedStyle(e).position === 'fixed') e.remove(); })"
    )


def collect_pdp_links(page, category_slug, already_have, target_count):
    """Paginate a category listing until target_count NEW (not already in
    already_have) product links are found or the listing is exhausted."""
    found = []
    seen = set()
    page_num = 1
    empty_pages_in_a_row = 0
    while len(found) < target_count and empty_pages_in_a_row < 2:
        url = f"{BASE}/en-de/c/{category_slug}" + (f"?page={page_num}" if page_num > 1 else "")
        page.goto(url, wait_until="load", timeout=30000)
        page.wait_for_timeout(1500)
        remove_overlays(page)
        hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.getAttribute('href'))")
        product_hrefs = sorted(set(h for h in hrefs if h and "/p/" in h))
        new_this_page = 0
        for href in product_hrefs:
            full_url = BASE + href if href.startswith("/") else href
            if full_url in seen:
                continue
            seen.add(full_url)
            new_this_page += 1
            found.append(full_url)
        print(f"  [{category_slug}] page {page_num}: {len(product_hrefs)} links, {new_this_page} new (total {len(found)})")
        empty_pages_in_a_row = empty_pages_in_a_row + 1 if new_this_page == 0 else 0
        page_num += 1
        if page_num > 15:  # sane upper bound, avoids infinite loop on an unexpected pagination bug
            break
    return found[:target_count]


def extract_product_code(image_urls, pdp_url):
    for url in image_urls:
        match = IMAGE_CODE_PATTERN.search(url)
        if match:
            return match.group(1)
    match = TRAILING_ID_PATTERN.search(pdp_url)
    if match:
        return f"carhartt-{match.group(1)}"
    return pdp_url.rstrip("/").rsplit("/", 1)[-1]


def upgrade_image_url(url):
    base = url.split("?")[0]
    return f"{base}?w=1600&fmt=auto&qlt=default"


def scrape_pdp(page, pdp_url):
    page.goto(pdp_url, wait_until="load", timeout=30000)
    page.wait_for_timeout(1500)

    ld_blocks = page.eval_on_selector_all('script[type="application/ld+json"]', "els => els.map(e => e.textContent)")
    product_data = None
    for block in ld_blocks:
        try:
            import json
            parsed = json.loads(block)
            if parsed.get("@type") == "Product":
                product_data = parsed
                break
        except (json.JSONDecodeError, TypeError):
            continue
    if product_data is None:
        return None

    details_texts = page.eval_on_selector_all("details", "els => els.map(e => e.innerText)")
    description = details_texts[0] if details_texts else ""
    # Strip the leading "Details" summary label and the trailing raw
    # image-code line the accordion text always ends with (e.g.
    # "I037132_3ZO_XX", matching product_code -- junk as a "feature").
    IMAGE_CODE_LINE = re.compile(r"^I\d{6}_")
    description_lines = [
        line for line in description.split("\n")
        if line.strip() and line.strip() != "Details" and not IMAGE_CODE_LINE.match(line.strip())
    ]
    features = description_lines[1:] if len(description_lines) > 1 else []
    description_text = description_lines[0] if description_lines else ""

    raw_images = product_data.get("image", []) or []
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    image_urls = [upgrade_image_url(u) for u in raw_images]

    product_code = extract_product_code(raw_images, pdp_url)
    offers = product_data.get("offers", {}) or {}

    return {
        "product_code": product_code,
        "name": product_data.get("name", ""),
        "color_name": product_data.get("color", ""),
        "price": offers.get("price"),
        "currency": offers.get("priceCurrency"),
        "image_urls": image_urls,
        "details": {
            "description": description_text,
            "features": features,
            "materials": [product_data.get("material", "")] if product_data.get("material") else [],
        },
        "product_url": pdp_url,
    }


def download_image(url, dest_path, session):
    if dest_path.exists():
        return True
    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(response.content)
        return True
    except Exception as error:
        print(f"    [warn] image download failed {url}: {error}")
        return False


def slugify(name):
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "product"


def main():
    existing = load_records()
    existing_codes = {r["product_code"] for r in existing}

    session = requests.Session()
    # Amplience CDN enforces hotlink protection -- a bare User-Agent gets a
    # 403; a Referer from the site's own domain is required (confirmed via
    # direct testing, not assumed).
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Referer": f"{BASE}/",
    })

    touched = {}
    total_new = 0
    category_counts = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        page = browser.new_page(viewport={"width": 1400, "height": 1000})

        for category_slug, category_label in CATEGORIES.items():
            print(f"\n=== Category: {category_label} ({category_slug}) ===")
            pdp_links = collect_pdp_links(page, category_slug, existing_codes, TARGET_PER_CATEGORY + 15)
            category_count = 0

            for pdp_url in pdp_links:
                if category_count >= TARGET_PER_CATEGORY:
                    break
                try:
                    record = scrape_pdp(page, pdp_url)
                except Exception as error:
                    print(f"  [warn] PDP failed {pdp_url}: {error}")
                    continue
                if record is None:
                    print(f"  [warn] no ld+json Product block: {pdp_url}")
                    continue
                if record["product_code"] in existing_codes:
                    continue  # already scraped in a prior run

                slug = slugify(record["name"])
                image_dir = Path("apparel_dataset") / "carhartt" / slug / record["product_code"]
                local_paths = []
                for index, image_url in enumerate(record["image_urls"]):
                    dest = image_dir / f"image_{index}.jpg"
                    if download_image(image_url, dest, session):
                        local_paths.append(str(dest))

                if not local_paths:
                    print(f"  [warn] no images downloaded, skipping: {pdp_url}")
                    continue

                full_record = {
                    "brand": "carhartt",
                    "category": category_label,
                    "name": record["name"],
                    "color_name": record["color_name"],
                    "price": f"{record['price']} {record['currency']}" if record["price"] else "",
                    "product_code": record["product_code"],
                    "slug": slug,
                    "product_url": record["product_url"],
                    "image_count": len(local_paths),
                    "images": local_paths,
                    "image_urls": record["image_urls"],
                    "details": record["details"],
                }
                touched[record["product_code"]] = full_record
                existing_codes.add(record["product_code"])
                category_count += 1
                total_new += 1
                print(f"  [{category_count}/{TARGET_PER_CATEGORY}] {record['product_code']}: {record['name']} ({record['color_name']}) -- {len(local_paths)} images")

                if total_new % CHECKPOINT_EVERY == 0:
                    save_records_safe(touched)
                    print(f"  [checkpoint] saved ({total_new} new records so far)")

            category_counts[category_label] = category_count

        browser.close()

    save_records_safe(touched)
    print(f"\nDone. {total_new} new records scraped.")
    for label, count in category_counts.items():
        print(f"  {label}: {count}/{TARGET_PER_CATEGORY}")


if __name__ == "__main__":
    main()
