"""
Vans (vans.com) men's shoes + clothing scraper -- Shoes, Hoodies and
Jackets, Shirts. Biased toward footwear (target 100 shoes vs 50 each of
the two clothing categories, ~200 total) since every apparel-only brand
added this session (Champion, Levi's, Carhartt, Stussy) left footwear
coverage stuck at the original 4 shoe brands (Nike, Adidas, New Balance,
Skechers) -- Vans's real value here is shoe diversity, not more clothing.

Site notes (VF Corp brand, Nuxt.js storefront):
  - Real bot protection: Akamai Bot Manager, same edge "Access Denied"
    (errors.edgesuite.net) signature as Levi's/New Balance on plain
    `requests`. Unlike Levi's interactive behavioral-challenge
    interstitial, plain `patchright` (headed, channel="chrome") gets
    through cleanly with no visible challenge screen at all -- softer
    tier than Levi's, closer to New Balance's binary-but-patchright-
    passes block.
  - Real, stable data source is schema.org ld+json, in TWO different
    shapes depending on page type -- check which one you're on:
      * Category listing pages (`/en-us/c/...`): a `CollectionPage` node
        with `mainEntity.itemListElement`, 48 products per page
        (`?page=N` pagination, confirmed real -- different products per
        page, not a silent no-op). Each item already has name, url,
        category, price, and a full image array (HERO + ALT1-4) --
        enough to build a record WITHOUT visiting the PDP at all, except
        for description/materials text.
      * Product pages (`/en-us/p/...`): a `ProductGroup` node.
        `hasVariant` here is SIZE variants of ONE fixed colorway (color
        is constant across every entry), matching PacSun's "one PDP per
        colorway" pattern, not New Balance's "one PDP has every
        colorway" pattern -- each `/en-us/c/.../page=N` listing item IS
        already a distinct colorway/product, one PDP visit per item.
        `productGroupID` (e.g. "VN000D9RWVD") is the stable
        `product_code`, same code embedded in the listing page's image
        filenames and PDP URL slug.
  - Real product-detail bullets are SERVER-RENDERED, no click needed:
    `data-test-id="product-details-bulletin"` on the PDP, a plain
    `<ul><li>` list -- readable directly from `page.content()`.
  - Images are on a Cloudinary-style dynamic-imaging CDN
    (`assets.vans.com/images/t_img/...`). The listing page's embedded
    image URLs are low-res thumbnails; PDP visits reveal the SAME image
    IDs (`{SKU}-HERO`, `{SKU}-ALT1..4`) served through a different
    transform prefix (`c_fill,g_center,f_auto,h_2500,w_2000`) for a real
    2000x2500 full image -- construct full-res URLs directly from the
    HERO/ALT id fragments already present in the listing page, no need
    to re-scrape image URLs from the PDP.
  - Category paths discovered via the real commerce sitemap
    (`vans.com/sitemap.xml` -> `sitemaps/commerce/commerce-en-us.xml`),
    not guessed -- initial guesses (`/en-us/mens-shoes`) 404'd inside
    the Nuxt app itself (a real in-app 404, not a bot block -- confirmed
    by inspecting `__NUXT_DATA__`'s `statusCode`).

Usage:
    python3 vans_scraper.py
"""

import re
import time
from contextlib import contextmanager
from pathlib import Path

import requests
from patchright.sync_api import sync_playwright

from dataset_utils import load_records, save_records_safe

BASE = "https://www.vans.com"
CATEGORY_URLS = {
    "Shoes": (f"{BASE}/en-us/c/mens/shoes-1100", 100),
    "Hoodies and Jackets": (f"{BASE}/en-us/c/mens/clothing/hoodies-and-jackets-1204", 50),
    "Shirts": (f"{BASE}/en-us/c/mens/clothing/shirts-5810", 50),
}
DATASET_ROOT = Path("apparel_dataset")
BRAND = "vans"

PRODUCTS_PER_PAGE = 48
CHECKPOINT_EVERY = 10

SKU_RE = re.compile(r"-([A-Z0-9]{6,})$")


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@contextmanager
def new_browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, channel="chrome")
        try:
            yield browser
        finally:
            browser.close()


def is_blocked(page):
    try:
        title = page.title()
    except Exception:
        return True
    return "access denied" in (title or "").lower()


def goto_with_retry(page, url, attempts=4):
    for attempt in range(attempts):
        try:
            page.goto(url, timeout=30000)
        except Exception as error:
            print(f"  [warn] goto failed ({error}), retrying...")
            time.sleep(5 * (attempt + 1))
            continue
        for _ in range(6):
            if not is_blocked(page):
                return True
            time.sleep(2)
        if not is_blocked(page):
            return True
        wait = 5 * (attempt + 1)
        print(f"  [blocked] {url}, backing off {wait}s (attempt {attempt + 1}/{attempts})")
        time.sleep(wait)
    return False


def extract_ldjson(html):
    import json
    matches = re.findall(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', html, re.DOTALL)
    for raw in matches:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "@graph" in data:
            return data["@graph"]
    return []


def sku_from_url(url):
    slug = url.rstrip("/").split("/")[-1].split("?")[0]
    m = SKU_RE.search(slug)
    return m.group(1) if m else slug


def full_res_image_urls(thumbnail_urls):
    """Listing-page image URLs use a low-res transform prefix; rewrite to
    the full 2000x2500 transform confirmed present on real PDP pages,
    keeping the same {SKU}-{VIEW} image id and version segment."""
    urls = []
    for u in thumbnail_urls:
        m = re.search(r"/images/t_[^/]+/(v\d+/[A-Za-z0-9_-]+/[^/]+)$", u)
        if not m:
            urls.append(u)
            continue
        urls.append(f"https://assets.vans.com/images/t_img/c_fill,g_center,f_auto,h_2500,w_2000/{m.group(1)}")
    return urls


def scrape_listing_page(page, category_url, page_number):
    url = f"{category_url}?page={page_number}" if page_number > 1 else category_url
    if not goto_with_retry(page, url):
        print(f"  [error] could not load {url} after retries")
        return []
    graph = extract_ldjson(page.content())
    for node in graph:
        if node.get("@type") == "CollectionPage":
            items = node.get("mainEntity", {}).get("itemListElement", [])
            return items
    return []


def scrape_pdp_details(page, pdp_url):
    if not goto_with_retry(page, pdp_url):
        return {"description": "", "features": [], "color_name": ""}
    html = page.content()

    bullets = []
    bulletin_match = re.search(
        r'data-test-id="product-details-bulletin"[^>]*>(.*?)</ul>', html, re.DOTALL
    )
    if bulletin_match:
        bullets = [
            re.sub(r"<[^>]+>", "", li).strip()
            for li in re.findall(r"<li[^>]*>(.*?)</li>", bulletin_match.group(1), re.DOTALL)
        ]
        bullets = [b for b in bullets if b]

    color_name = ""
    description = ""
    for node in extract_ldjson(html):
        if node.get("@type") == "ProductGroup":
            description = node.get("description", "")
            variants = node.get("hasVariant", [])
            if variants:
                color_name = variants[0].get("color", "")
            break

    return {"description": description, "features": bullets, "color_name": color_name}


def scrape_category(category, category_url, target_count, existing_codes):
    records = []
    page_number = 1
    with new_browser() as browser:
        listing_page = browser.new_page()
        pdp_page = browser.new_page()

        while len(records) < target_count:
            items = scrape_listing_page(listing_page, category_url, page_number)
            if not items:
                print(f"  [{category}] page {page_number} returned no items, stopping (reached end of catalog)")
                break

            for item in items:
                if len(records) >= target_count:
                    break
                product = item.get("item", {})
                pdp_url = product.get("url", "")
                code = sku_from_url(pdp_url)
                if not code or code in existing_codes:
                    continue

                name = product.get("name", "")
                price = product.get("offers", {}).get("price")
                image_urls = full_res_image_urls([img.get("url", "") for img in product.get("image", [])])

                print(f"  [{category}] {code}: {name}")
                details = scrape_pdp_details(pdp_page, pdp_url)

                records.append({
                    "brand": BRAND,
                    "category": category,
                    "name": name,
                    "color_name": details["color_name"],
                    "price": f"${price}" if price is not None else "",
                    "product_code": code,
                    "slug": slugify(name),
                    "product_url": pdp_url,
                    "image_urls": image_urls,
                    "details": {
                        "description": details["description"],
                        "features": details["features"],
                        "materials": [],
                    },
                })
                existing_codes.add(code)

            page_number += 1
            if page_number > 20:
                print(f"  [{category}] hit page-count safety limit, stopping")
                break

    return records


def download_images(record):
    product_dir = DATASET_ROOT / BRAND / record["slug"] / record["product_code"]
    product_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for index, url in enumerate(record["image_urls"]):
        dest = product_dir / f"image_{index}.jpg"
        if dest.exists():
            saved_paths.append(str(dest))
            continue
        try:
            response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            dest.write_bytes(response.content)
            saved_paths.append(str(dest))
        except Exception as error:
            print(f"    [warn] image download failed ({url}): {error}")
    record["images"] = saved_paths
    record["image_count"] = len(saved_paths)
    return record


def main():
    existing = load_records()
    existing_codes = {r["product_code"] for r in existing if r.get("brand") == BRAND}
    print(f"{len(existing_codes)} Vans records already present, skipping those.")

    touched = {}
    total_new = 0
    for category, (url, target_count) in CATEGORY_URLS.items():
        already_in_category = sum(
            1 for r in existing if r.get("brand") == BRAND and r.get("category") == category
        )
        remaining_target = max(0, target_count - already_in_category)
        if remaining_target == 0:
            print(f"[{category}] already at target ({already_in_category}), skipping.")
            continue
        print(f"\n=== {category}: scraping up to {remaining_target} more (have {already_in_category}) ===")
        new_records = scrape_category(category, url, remaining_target, existing_codes)
        print(f"[{category}] scraped {len(new_records)} new records, downloading images...")

        for i, record in enumerate(new_records, start=1):
            record = download_images(record)
            touched[record["product_code"]] = record
            total_new += 1
            if len(touched) % CHECKPOINT_EVERY == 0:
                save_records_safe(touched)
                print(f"  [checkpoint] saved {len(touched)} records so far")

    save_records_safe(touched)
    print(f"\nDone. {total_new} new Vans records scraped and saved.")


if __name__ == "__main__":
    main()
