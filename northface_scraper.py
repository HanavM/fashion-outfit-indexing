"""
The North Face (thenorthface.com) men's clothing scraper -- Jackets and
Vests, Fleece, Hoodies and Sweatshirts, Pants. Deliberately weighted
toward outerwear (two of the four categories are jacket/fleece families):
the existing catalog is thin on jackets and coats, and TNF's whole
catalog is outerwear-first, which is the reason this brand was added.

Site notes (VF Corp brand -- same parent and same Nuxt.js storefront as
Vans; `vans_scraper.py` was a near-direct template):
  - Bot protection: Akamai Bot Manager. Plain `requests` gets an edge
    "Access Denied" (errors.edgesuite.net) on *every* path including
    robots.txt. `patchright` (headed, channel="chrome") passes cleanly
    with no interactive challenge screen at all -- same softer tier as
    Vans / New Balance, NOT Levi's behavioral-challenge interstitial.
  - robots.txt (readable only through patchright) does NOT disallow
    catalog browsing. It disallows `/*/c/*filters=*`, `/*/c/*sort=*`,
    `/*/search`, cart/checkout. Plain `?page=N` pagination is allowed,
    which is all this scraper uses.
  - Real category paths come from the commerce sitemap
    (sitemap.xml -> sitemaps/commerce/commerce-en-us.xml), not guessed --
    same lesson as Vans, where guessed paths produced real in-app 404s.
  - Data source is schema.org ld+json in a `@graph`, two shapes:
      * Listing pages (`/en-us/c/...`): `CollectionPage` ->
        `mainEntity.itemListElement`, 48 items/page, real `?page=N`
        pagination (verified: page 1 and 2 share zero product codes).
        Each item carries name/url/price/images -- but only for ONE
        colorway per style (the default), so the listing alone cannot
        satisfy this dataset's one-record-per-colorway requirement.
      * Product pages (`/en-us/p/...`): a `ProductGroup` whose
        `hasVariant` is SIZE variants of one fixed colorway (same shape
        as Vans, useless for colorways), PLUS a separate `Product` node
        that IS the currently-selected colorway: `sku`/`mpn`/`productID`
        all equal the colorway code (e.g. `NF0A88XU2EK` = style
        `NF0A88XU` + color `2EK`), with its own `color`, `image` array
        and `offers.price`.
  - COLORWAY SIBLINGS (the non-obvious part, and the difference from
    vans_scraper.py): the sibling colorways of a style live only in the
    flat `__NUXT_DATA__` payload array, not in any ld+json node and not
    in a normal `<a href>` swatch list. They appear as a fixed 4-tuple
    run of adjacent strings: full colorway code, human color label,
    3-char color value, and the `?color=` URL -- matched by SWATCH_RE
    below. The *currently selected* color is often absent from that run
    (its URL string gets deduped out of the payload), so the selected
    colorway must always be unioned in from the `Product` node's own
    sku/color. Visiting `?color={value}` re-renders the whole PDP for
    that colorway, ld+json included.
  - Images: Cloudinary-style CDN (`assets.thenorthface.com/images/...`).
    ld+json serves `t_Thumbnail` (600x698 png). Rewriting the transform
    segment to `t_img/c_fill,g_center,f_auto,h_2500,w_2000` yields a real
    2000x2500 JPEG (~280KB, verified by downloading and checking actual
    pixel dimensions). Do NOT use bare `t_img`: it returns the untouched
    3.5MB source PNG. View codes ARE embedded in the image path
    (`-HERO`, `-HERO2`, `-HERO3`, `-BACK`, `-ALT1..`, `-MODEL34`) --
    and survive the rename to image_N.jpg because the stored `image_urls`
    stay positionally aligned with `images` (lesson 5) -- no extra schema
    field was added for them.
  - Product-detail bullets are server-rendered, no click needed:
    `data-test-id="product-details-bulletin"` <ul><li> list, same
    selector as Vans.

Usage:
    python3 northface_scraper.py
"""

import json
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import requests
from patchright.sync_api import sync_playwright

from dataset_utils import load_records, save_records_safe

BASE = "https://www.thenorthface.com"
CATEGORY_URLS = {
    "Jackets and Vests": (f"{BASE}/en-us/c/mens/mens-jackets-and-vests-211702", 50),
    "Fleece": (f"{BASE}/en-us/c/mens/mens-fleece-299285", 50),
    "Hoodies and Sweatshirts": (f"{BASE}/en-us/c/mens/mens-tops/mens-hoodies-and-sweatshirts-224211", 50),
    "Pants": (f"{BASE}/en-us/c/mens/mens-bottoms/mens-pants-224219", 50),
}
DATASET_ROOT = Path("apparel_dataset")
BRAND = "northface"

CHECKPOINT_EVERY = 10
MAX_IMAGES = 6
MAX_LISTING_PAGES = 12
MIN_FREE_GIB = 3.0

FULL_RES_TRANSFORM = "t_img/c_fill,g_center,f_auto,h_2500,w_2000"

# Adjacent-string run in the flat __NUXT_DATA__ payload:
#   "<colorway code>","<color label>","<3-char value>","<?color= url>"
SWATCH_RE = re.compile(
    r'"(NF[A-Z0-9]{8,12})","([^"]{1,60})","([A-Z0-9]{3})","(/en-us/p/[^"]*\?color=\3)"'
)


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def free_gib():
    return shutil.disk_usage("/System/Volumes/Data").free / (1024 ** 3)


def check_disk(context=""):
    free = free_gib()
    print(f"  [disk] {free:.2f} GiB free {context}")
    if free < MIN_FREE_GIB:
        raise SystemExit(
            f"ABORT: only {free:.2f} GiB free (floor {MIN_FREE_GIB} GiB). "
            "Checkpoint already saved; stopped for disk."
        )


@contextmanager
def new_browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, channel="chrome")
        try:
            yield browser
        finally:
            browser.close()


def is_blocked(page):
    """Akamai's edge block renders as an 'Access Denied' title. Checked by
    title rather than HTTP status because the edge returns a real 200-ish
    looking document body (lesson 3: never trust status alone)."""
    try:
        title = page.title() or ""
    except Exception:
        return True
    lowered = title.lower()
    return "access denied" in lowered or "something went wrong" in lowered


def goto_with_retry(page, url, attempts=4, settle=2.5):
    for attempt in range(attempts):
        try:
            page.goto(url, timeout=45000)
        except Exception as error:
            print(f"    [warn] goto failed ({error}), retrying...")
            time.sleep(5 * (attempt + 1))
            continue
        for _ in range(8):
            if not is_blocked(page):
                time.sleep(settle)
                return True
            time.sleep(2)
        wait = 5 * (attempt + 1)
        print(f"    [blocked] {url}, backing off {wait}s (attempt {attempt + 1}/{attempts})")
        time.sleep(wait)
    return False


def extract_ldjson(html):
    nodes = []
    for raw in re.findall(
        r'<script type="application/ld\+json"[^>]*>(.*?)</script>', html, re.DOTALL
    ):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "@graph" in data:
            nodes.extend(data["@graph"])
        elif isinstance(data, dict):
            nodes.append(data)
    return nodes


def full_res(url):
    """Rewrite the CDN transform segment to the verified 2000x2500 JPEG
    preset. Bare `t_img` would return the multi-MB source PNG instead."""
    return re.sub(r"/images/t_[^/]+(?:/[^/v][^/]*)?/(v\d+/)", f"/images/{FULL_RES_TRANSFORM}/\\1", url, count=1)


def parse_bulletin(html):
    m = re.search(r'data-test-id="product-details-bulletin"[^>]*>(.*?)</ul>', html, re.DOTALL)
    if not m:
        return []
    bullets = [
        re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", li)).strip()
        for li in re.findall(r"<li[^>]*>(.*?)</li>", m.group(1), re.DOTALL)
    ]
    return [b for b in bullets if b]


def parse_pdp(html):
    """Return (product_node, swatches, bullets) or (None, [], []).

    Content sniff (lesson 3/12): a page that loaded but has no ld+json
    `Product` node with an `sku` is NOT a usable PDP -- treated as a
    failure rather than silently producing an empty record.
    """
    product = None
    for node in extract_ldjson(html):
        if node.get("@type") == "Product" and node.get("sku"):
            product = node
            break
    if product is None:
        return None, [], []
    swatches = [
        {"code": code, "color": label, "value": value, "url": BASE + url}
        for code, label, value, url in SWATCH_RE.findall(html)
    ]
    return product, swatches, parse_bulletin(html)


def record_from_product(category, product, bullets, pdp_url):
    image_urls = []
    for image in product.get("image", []):
        url = image.get("url", "") if isinstance(image, dict) else str(image)
        if not url:
            continue
        image_urls.append(full_res(url))
    image_urls = image_urls[:MAX_IMAGES]

    price = product.get("offers", {}).get("price")
    name = product.get("name", "")
    return {
        "brand": BRAND,
        "category": category,
        "name": name,
        "color_name": product.get("color", ""),
        "price": f"${price}" if price is not None else "",
        "product_code": product["sku"],
        "slug": slugify(name),
        "product_url": pdp_url,
        "image_urls": image_urls,
        "details": {
            "description": product.get("description", "") or "",
            "features": bullets,
            "materials": [],
        },
    }


def scrape_listing_page(page, category_url, page_number):
    url = f"{category_url}?page={page_number}" if page_number > 1 else category_url
    if not goto_with_retry(page, url):
        print(f"  [error] could not load {url} after retries")
        return []
    for node in extract_ldjson(page.content()):
        if node.get("@type") == "CollectionPage":
            return node.get("mainEntity", {}).get("itemListElement", [])
    return []


def scrape_style(page, category, style_url, target_remaining, existing_codes):
    """Visit a style's PDP, emit one record per colorway (default colorway
    from the first load, siblings via ?color=)."""
    records = []
    if not goto_with_retry(page, style_url):
        print(f"    [skip] PDP unreachable: {style_url}")
        return records
    product, swatches, bullets = parse_pdp(page.content())
    if product is None:
        print(f"    [skip] no ld+json Product node (not a real PDP): {style_url}")
        return records

    if product["sku"] not in existing_codes:
        records.append(record_from_product(category, product, bullets, style_url))
        existing_codes.add(product["sku"])
        print(f"    + {product['sku']} {product.get('color', '')}")

    for swatch in swatches:
        if len(records) >= target_remaining:
            break
        if swatch["code"] in existing_codes:
            continue
        if not goto_with_retry(page, swatch["url"]):
            print(f"    [skip] colorway unreachable: {swatch['url']}")
            continue
        variant, _, variant_bullets = parse_pdp(page.content())
        if variant is None or variant["sku"] in existing_codes:
            continue
        records.append(
            record_from_product(category, variant, variant_bullets or bullets, swatch["url"])
        )
        existing_codes.add(variant["sku"])
        print(f"    + {variant['sku']} {variant.get('color', '')}")
    return records


def download_images(record):
    product_dir = DATASET_ROOT / BRAND / record["slug"] / record["product_code"]
    product_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for index, url in enumerate(record["image_urls"]):
        dest = product_dir / f"image_{index}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            saved_paths.append(str(dest))
            continue
        try:
            response = requests.get(
                url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            )
            response.raise_for_status()
            if not response.content.startswith(b"\xff\xd8") and not response.content.startswith(b"\x89PNG"):
                print(f"    [warn] non-image body for {url}")
                continue
            dest.write_bytes(response.content)
            saved_paths.append(str(dest))
        except Exception as error:
            print(f"    [warn] image download failed ({url}): {error}")
    record["images"] = saved_paths
    record["image_count"] = len(saved_paths)
    return record


def main():
    existing = load_records()
    all_codes = {r["product_code"] for r in existing}
    print(f"{len(existing)} records already in metadata.json ({len(all_codes)} unique codes).")
    mine = [r for r in existing if r.get("brand") == BRAND]
    print(f"{len(mine)} existing '{BRAND}' records will be skipped.")
    check_disk("at start")

    touched = {}
    seen_styles = set()
    saved_since_check = 0
    dumped_sample = False

    with new_browser() as browser:
        listing_page = browser.new_page()
        pdp_page = browser.new_page()

        for category, (url, target) in CATEGORY_URLS.items():
            have = sum(1 for r in existing if r.get("brand") == BRAND and r.get("category") == category)
            remaining = max(0, target - have)
            if remaining == 0:
                print(f"[{category}] already at target ({have}), skipping.")
                continue
            print(f"\n=== {category}: need {remaining} more (have {have}) ===")

            collected = 0
            page_number = 1
            while collected < remaining and page_number <= MAX_LISTING_PAGES:
                items = scrape_listing_page(listing_page, url, page_number)
                if not items:
                    print(f"  [{category}] page {page_number} empty -- end of catalog.")
                    break
                for item in items:
                    if collected >= remaining:
                        break
                    product = item.get("item", {})
                    style_url = product.get("url", "")
                    if not style_url:
                        continue
                    # Categories genuinely overlap (fleece jackets are listed
                    # under both "Jackets and Vests" and "Fleece"); walking the
                    # same style twice would only re-find codes already taken.
                    if style_url in seen_styles:
                        continue
                    seen_styles.add(style_url)
                    print(f"  [{category}] {product.get('name', '')}")
                    new_records = scrape_style(
                        pdp_page, category, style_url, remaining - collected, all_codes
                    )
                    for record in new_records:
                        download_images(record)
                        if not dumped_sample:
                            print("\n----- SAMPLE FULLY-EXTRACTED RECORD -----")
                            print(json.dumps(record, indent=2, ensure_ascii=False))
                            print("----- END SAMPLE -----\n")
                            dumped_sample = True
                        touched[record["product_code"]] = record
                        collected += 1
                        saved_since_check += 1
                        if len(touched) % CHECKPOINT_EVERY == 0:
                            save_records_safe(touched)
                            print(f"  [checkpoint] {len(touched)} records saved")
                        if saved_since_check >= 50:
                            save_records_safe(touched)
                            check_disk(f"after {len(touched)} records")
                            saved_since_check = 0
                page_number += 1

            save_records_safe(touched)
            got = sum(1 for r in touched.values() if r["category"] == category)
            print(f"[{category}] {got}/{remaining} new records (target {target})")

    save_records_safe(touched)
    print(f"\nDone. {len(touched)} new {BRAND} records saved.")
    for category in CATEGORY_URLS:
        print(f"  {category}: {sum(1 for r in touched.values() if r['category'] == category)}")
    check_disk("at end")


if __name__ == "__main__":
    main()
