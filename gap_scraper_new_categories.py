"""
Gap men's clothing scraper, part 2 — Jackets, Hats, Socks.

Extends gap_scraper.py's pattern to three visually-distinct, unambiguous
categories not yet in the dataset (see build_hierarchy.py's docstring for
why these were chosen over things like "jeans": each of these has a
silhouette that doesn't overlap with anything else already scraped, so it
won't confuse a SigLIP-style classifier the way a "jeans vs. pants" split
would).

Site notes specific to this run (delta from gap_scraper.py's notes):
  - Unlike T-Shirts/Shorts/Pants/Sweaters, no `cid` was findable for these
    three categories through the site nav (Gap's mega-menu is client-side
    rendered, and WebFetch only sees the server-rendered shell — matches
    the same client-rendering trap gap_scraper.py already documented for
    PDP accordions, just hitting the nav this time instead).
  - Workaround: the same CATEGORY_API accepts a `keyword` param instead of
    `cid`+`department` (the two are mutually exclusive — passing both
    returns `"Invalid filter parameter(s)"`). A keyword search returns
    products across every division (men/women/kids/baby) with no
    department filter; each product carries a `webProductType` field
    (e.g. "mens jackets", "womens jackets", "boys jackets") that IS
    reliable for post-filtering to men's-only. Confirmed empirically
    against live API responses, not assumed.
  - Jackets is the thin category here: keyword="jacket" alone only surfaces
    19 men's colorways. Several synonym keywords are queried and merged
    (dedup by ccId) to reach a reasonable pool: jacket/coat/outerwear/
    puffer/windbreaker. Hats and Socks each clear 50 on their own keyword.
  - `pageSize=200` was NOT enough to cover "hat" in one page (6 pages of
    results across all divisions) — this scraper paginates pageNumber
    until pageNumberTotal is exhausted, same defensive pattern as
    gap_scraper.py's cid-based iterator.
"""

import hashlib, time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

from dataset_utils import load_records, save_records_safe
from gap_scraper import (
    slugify, download, best_image_urls, fetch_details,
    CATEGORY_API, HEADERS,
)

OUTPUT_DIR = Path("apparel_dataset/gap")
TARGET_PER_CATEGORY = 50
PAGE_SIZE = 200

# category label -> (webProductType exact match, [keywords to query and merge])
CATEGORY_KEYWORDS = {
    "Jackets": ("mens jackets", ["jacket", "coat", "outerwear", "puffer", "windbreaker"]),
    "Hats": ("mens hats", ["hat", "beanie", "cap"]),
    "Socks": ("mens socks", ["sock"]),
}


def iter_colorways_by_keyword(keyword: str, expected_wpt: str):
    """Same flattening as gap_scraper.iter_colorways, but keyed off a
    `keyword` search instead of `cid`, filtered to the exact men's
    webProductType (keyword search has no department filter of its own)."""
    page_number = 0
    while True:
        params = {
            "pageSize": PAGE_SIZE,
            "pageNumber": page_number,
            "ignoreInventory": "false",
            "keyword": keyword,
            "vendor": "constructorio",
            "client_id": 0,
            "session_id": 0,
            "includeMarketingFlagsDetails": "true",
            "enableDynamicFacets": "true",
            "enableDynamicPhoto": "true",
            "brand": "gap",
            "locale": "en_US",
            "market": "us",
        }
        r = requests.get(CATEGORY_API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()

        products = data.get("products") or []
        if not products:
            return

        for product in products:
            if product.get("webProductType") != expected_wpt:
                continue

            style_name = product.get("styleName", "")
            for sc in product.get("styleColors") or []:
                images = sc.get("images") or []
                if not images:
                    continue
                yield {
                    "product_code": sc["ccId"],
                    "name": sc.get("styleName") or style_name,
                    "color_name": sc.get("ccName", ""),
                    "price": f"${sc['effectivePrice']}" if sc.get("effectivePrice") else "",
                    "image_urls": best_image_urls(images),
                }

        pagination = data.get("pagination") or {}
        total_pages = int(pagination.get("pageNumberTotal", 1) or 1)
        page_number += 1
        if page_number >= total_pages:
            return
        time.sleep(0.3)


def iter_category_colorways(category: str):
    expected_wpt, keywords = CATEGORY_KEYWORDS[category]
    seen_in_run = set()
    for keyword in keywords:
        for colorway in iter_colorways_by_keyword(keyword, expected_wpt):
            if colorway["product_code"] in seen_in_run:
                continue
            seen_in_run.add(colorway["product_code"])
            yield colorway


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    database = load_records()
    seen_codes = {p["product_code"] for p in database}
    print(f"Starting with {len(database)} existing records.\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        for category in CATEGORY_KEYWORDS:
            print(f"\n=== {category} (keyword search) ===")
            already = sum(1 for p in database if p.get("brand") == "gap" and p.get("category") == category)
            target_new = TARGET_PER_CATEGORY - already
            if target_new <= 0:
                print(f"{category} already has {already} records, skipping.")
                continue

            added = 0
            for colorway in iter_category_colorways(category):
                if added >= target_new:
                    break

                code = colorway["product_code"]
                if code in seen_codes:
                    continue

                print(f"[{added + 1}/{target_new}] {category}: {colorway['name']} - {colorway['color_name']} ({code})")

                try:
                    details, product_url = fetch_details(page, code)
                except Exception as e:
                    print(f"  [warn] detail fetch failed: {e}")
                    continue

                slug = slugify(colorway["name"] or code)
                item_dir = OUTPUT_DIR / slug / code
                item_dir.mkdir(parents=True, exist_ok=True)

                saved = []
                seen_hashes: set[str] = set()
                for i, img_url in enumerate(colorway["image_urls"]):
                    dest = item_dir / f"image_{i}.jpg"
                    if download(img_url, dest):
                        h = hashlib.md5(dest.read_bytes()).hexdigest()
                        if h in seen_hashes:
                            dest.unlink()
                        else:
                            seen_hashes.add(h)
                            saved.append(str(dest))

                if not saved:
                    print("  [warn] no images downloaded, skipping")
                    continue

                new_record = {
                    "brand": "gap",
                    "category": category,
                    "name": colorway["name"],
                    "color_name": colorway["color_name"],
                    "price": colorway["price"],
                    "product_code": code,
                    "slug": slug,
                    "product_url": product_url,
                    "image_count": len(saved),
                    "images": saved,
                    "image_urls": colorway["image_urls"],
                    "details": details,
                }
                database.append(new_record)
                seen_codes.add(code)
                added += 1

                database = save_records_safe({code: new_record})
                print(f"  [checkpoint] {len(database)} total records "
                      f"({added}/{target_new} added for {category})")

                time.sleep(0.3)

            print(f"\n{category}: {added} new records added "
                  f"({already + added}/{TARGET_PER_CATEGORY} total).")

        browser.close()

    total_imgs = sum(
        p["image_count"] for p in database
        if p.get("brand") == "gap" and p.get("category") in CATEGORY_KEYWORDS
    )
    print(f"\nDone. {len(database)} total records in dataset.")
    print(f"New-category images downloaded: {total_imgs}")


if __name__ == "__main__":
    main()
