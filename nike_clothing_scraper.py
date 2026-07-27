"""
Nike Clothing Scraper — men's Tops & T-Shirts, Shorts, Hoodies & Pullovers,
Pants & Tights. Top 50 PRODUCTS per category (one record per product, not
one per colorway like nike_scraper.py's shoe scrape) -> 200 new records.

Category pages/paths confirmed live against the product_wall search API:
    Tops and T-Shirts      /w/mens-tops-t-shirts-9om13znik1
    Shorts                 /w/mens-shorts-38fphznik1
    Hoodies and Pullovers  /w/mens-hoodies-and-pullovers-6riveznik1
    Pants and Tights       /w/mens-pants-and-tights-2kq19znik1

Workflow (same pattern as nike_scraper.py):
  1. product_wall search API (paginated via `anchor`) with path=<category path>
     -> productGroupings, each already carrying one representative PDP URL +
     that grouping's default/selected product info -> take only that one
     product per grouping (not every colorway) until 50 unique groupKeys
     collected per category.
  2. Visit that one PDP URL per grouping (Playwright, __NEXT_DATA__) to pull
     the full title/color/price/gallery for the *selected* variant only.
  3. Download images to apparel_dataset/nike/{slug}/{product_code}/image_N.jpg
  4. Append records straight into apparel_dataset/metadata.json (checkpoint
     after every grouping), tagged brand="nike", category=<section name>.
"""

import hashlib, json, re, time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

from dataset_utils import load_records, save_records_safe

SEARCH_API = (
    "https://api.nike.com/discover/product_wall/v1/marketplace/US/language/en/"
    "consumerChannelId/d9a5bc42-4b9c-4976-858a-f159cf99c647"
)
SEARCH_HEADERS = {
    "nike-api-caller-id": "nike:dotcom:browse:wall.client:2.0",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "accept": "*/*",
    "referer": "https://www.nike.com/",
}

CATEGORIES = {
    "Tops and T-Shirts": "/w/mens-tops-t-shirts-9om13znik1",
    "Shorts": "/w/mens-shorts-38fphznik1",
    "Hoodies and Pullovers": "/w/mens-hoodies-and-pullovers-6riveznik1",
    "Pants and Tights": "/w/mens-pants-and-tights-2kq19znik1",
}

OUTPUT_DIR = Path("apparel_dataset/nike")
TARGET_PRODUCTS_PER_CATEGORY = 50
PAGE_COUNT = 24


# ── helpers (identical to nike_scraper.py) ──────────────────────────────────

def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s.strip("-")[:60]


def hi_res(url: str) -> str:
    return re.sub(r"/a/images/t_[^/]+/", "/a/images/f_auto,q_auto:best/", url)


def clean_color(raw: str) -> str:
    return re.sub(r"\s*/\s*", " / ", raw or "").strip()


def format_price(prices: dict) -> str:
    val = (prices or {}).get("currentPrice")
    if val is None:
        return ""
    return f"${val:.0f}" if float(val).is_integer() else f"${val:.2f}"


def clean_product_url(url: str, group_key: str) -> str:
    if group_key:
        url = url.replace(f"-{group_key}/", "/")
    return url


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"      [warn] {e}")
        return False


# ── step 1: search API → incrementally page groupings on demand ────────────

def iter_groupings(category_path: str, category_name: str):
    """Yields {group_key, pdp_url} dicts one at a time, paging the search API
    (queryType=PRODUCTS, matching nike_scraper.py's validated pattern) as
    needed, until the API reports no more pages. Deliberately a generator
    (not a fixed pre-collected batch) so the caller can keep pulling more
    groupings if some turn out to be cross-category duplicates already
    scraped under a different section (e.g. a hoodie listed under both
    "Tops and T-Shirts" and "Hoodies and Pullovers")."""
    seen_group_keys = set()
    anchor = 0

    while True:
        params = {
            "path": category_path,
            "searchTerms": category_name,
            "queryType": "PRODUCTS",
            "anchor": anchor,
            "count": PAGE_COUNT,
        }
        r = requests.get(SEARCH_API, params=params, headers=SEARCH_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        page_groupings = data.get("productGroupings") or []
        if not page_groupings:
            return

        for g in page_groupings:
            products = g.get("products") or []
            if not products:
                continue
            group_key = products[0]["groupKey"]
            if group_key in seen_group_keys:
                continue
            seen_group_keys.add(group_key)
            yield {"group_key": group_key, "pdp_url": products[0]["pdpUrl"]["url"]}

        if not data.get("pages", {}).get("next"):
            return
        anchor += PAGE_COUNT
        time.sleep(0.2)


# ── step 2: visit PDP → extract only the selected/default variant ──────────

def extract_next_data(page) -> dict | None:
    try:
        raw = page.query_selector("script#__NEXT_DATA__").inner_text()
        return json.loads(raw)
    except Exception:
        return None


def parse_selected_variant(data: dict) -> dict | None:
    """Return {product_code, name, color_name, price, product_url, image_urls}
    for ONLY the selected/default product of this grouping (not every colorway)."""
    try:
        page_props = data["props"]["pageProps"]
        selected = page_props["selectedProduct"]
        code = selected["styleColor"]
        p = selected
        info = p.get("productInfo") or {}
        title = info.get("title") or ""
        group_key = p.get("groupKey", "")
        product_url = clean_product_url((p.get("pdpUrl") or {}).get("url", ""), group_key)

        image_urls = []
        for card in p.get("contentImages") or []:
            if card.get("cardType") != "image":
                continue
            props = card.get("properties") or {}
            portrait = (props.get("portrait") or {}).get("url")
            squarish = (props.get("squarish") or {}).get("url")
            url = portrait or squarish
            if url:
                image_urls.append(hi_res(url))

        return {
            "name": title,
            "color_name": clean_color(p.get("colorDescription", "")),
            "price": format_price(p.get("prices")),
            "product_code": code,
            "slug": slugify(title),
            "product_url": product_url,
            "image_urls": image_urls,
        }
    except Exception as e:
        print(f"      [warn] couldn't parse selectedProduct: {e}")
        return None


# ── main ─────────────────────────────────────────────────────────────────────

def load_db() -> list[dict]:
    return load_records()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    database = load_db()
    seen_codes = {p["product_code"] for p in database}
    print(f"Starting with {len(database)} existing records.\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = context.new_page()

        for category, path in CATEGORIES.items():
            print(f"\n=== {category} ({path}) ===")
            already_in_category = sum(
                1 for p in database if p.get("brand") == "nike" and p.get("category") == category
            )
            added_this_category = 0
            target_new = TARGET_PRODUCTS_PER_CATEGORY - already_in_category
            if target_new <= 0:
                print(f"{category} already has {already_in_category} records, skipping.")
                continue

            for g in iter_groupings(path, category):
                if added_this_category >= target_new:
                    break

                url = g["pdp_url"]
                print(f"[{added_this_category + 1}/{target_new}] {category}: {url.split('/t/')[-1][:60]}")

                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=45000)
                except Exception as e:
                    print(f"  [warn] goto failed: {e}")
                    continue
                time.sleep(2.0)

                next_data = extract_next_data(page)
                if not next_data:
                    print("  [warn] no __NEXT_DATA__ found, skipping")
                    continue

                variant = parse_selected_variant(next_data)
                if not variant:
                    continue
                if variant["product_code"] in seen_codes:
                    print(f"  [skip] {variant['product_code']} already in dataset (cross-category duplicate)")
                    continue

                code = variant["product_code"]
                slug = variant["slug"]
                item_dir = OUTPUT_DIR / slug / code
                item_dir.mkdir(parents=True, exist_ok=True)

                saved = []
                seen_hashes: set[str] = set()
                for i, img_url in enumerate(variant["image_urls"]):
                    dest = item_dir / f"image_{i}.jpg"
                    if download(img_url, dest):
                        h = hashlib.md5(dest.read_bytes()).hexdigest()
                        if h in seen_hashes:
                            dest.unlink()
                        else:
                            seen_hashes.add(h)
                            saved.append(str(dest))

                new_record = {
                    "brand": "nike",
                    "category": category,
                    "name": variant["name"],
                    "color_name": variant["color_name"],
                    "price": variant["price"],
                    "product_code": code,
                    "slug": slug,
                    "product_url": variant["product_url"],
                    "image_count": len(saved),
                    "images": saved,
                    "image_urls": variant["image_urls"],
                }
                database.append(new_record)
                seen_codes.add(code)
                added_this_category += 1

                database = save_records_safe({code: new_record})
                print(f"  [checkpoint] {len(database)} total records saved "
                      f"({added_this_category}/{target_new} added for {category})")

                time.sleep(0.3)

            print(f"\n{category}: {added_this_category} new records added "
                  f"({already_in_category + added_this_category}/{TARGET_PRODUCTS_PER_CATEGORY} total).")

        browser.close()

    total_imgs = sum(p["image_count"] for p in database if p.get("brand") == "nike" and p.get("category") in CATEGORIES)
    print(f"\nDone. {len(database)} total records in dataset.")
    print(f"New Nike clothing images downloaded: {total_imgs}")


if __name__ == "__main__":
    main()
