"""
Nike Shoes Scraper — Full Detail Version

Workflow:
  1. Nike product_wall search API (paginated via `anchor`) → collect product
     "groupings" (base models). Each grouping already lists every color-variant
     product code plus a representative PDP URL — no separate variant-discovery
     step needed like Adidas required.
  2. Visit ONE PDP URL per grouping (Playwright, to get the embedded
     __NEXT_DATA__ JSON) → this single page load contains full metadata
     (title, color, price, gallery images) for EVERY color variant of that
     model, not just the one we navigated to.
  3. Download images to nike_catalog/{slug}/{product_code}/image_N.jpg
  4. Write nike_products.json incrementally (checkpoint after every grouping)
     so a partial run doesn't lose data.

Each unique color variant becomes its own database entry.
"""

import hashlib, json, re, time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

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
SEARCH_PARAMS = {"path": "/w?q=shoes&vst=shoes", "searchTerms": "Shoes", "queryType": "PRODUCTS"}

OUTPUT_DIR = Path("nike_catalog")
DB_FILE = Path("nike_products.json")
TARGET_VARIANTS = 180   # stop once we've queued at least this many variants (finish current page)
PAGE_COUNT = 24


# ── helpers ──────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s.strip("-")[:60]


def hi_res(url: str) -> str:
    """Swap Nike's low-res 't_default' transform for a full-resolution one."""
    return re.sub(r"/a/images/t_[^/]+/", "/a/images/f_auto,q_auto:best/", url)


def clean_color(raw: str) -> str:
    return re.sub(r"\s*/\s*", " / ", raw or "").strip()


def format_price(prices: dict) -> str:
    val = (prices or {}).get("currentPrice")
    if val is None:
        return ""
    return f"${val:.0f}" if float(val).is_integer() else f"${val:.2f}"


def clean_product_url(url: str, group_key: str) -> str:
    """Strip the trailing '-{groupKey}' hash segment from the PDP path."""
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


# ── step 1: search API → groupings ──────────────────────────────────────────

def collect_groupings(target_variants: int) -> list[dict]:
    """Page through the search API, returning groupings until we've queued
    at least `target_variants` color variants (finishing the current page)."""
    groupings = []
    seen_group_keys = set()
    total_variants = 0
    anchor = 0

    while True:
        params = {**SEARCH_PARAMS, "anchor": anchor, "count": PAGE_COUNT}
        r = requests.get(SEARCH_API, params=params, headers=SEARCH_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        page_groupings = data.get("productGroupings") or []
        if not page_groupings:
            break

        for g in page_groupings:
            products = g.get("products") or []
            if not products:
                continue
            group_key = products[0]["groupKey"]
            if group_key in seen_group_keys:
                continue
            seen_group_keys.add(group_key)
            groupings.append({
                "group_key": group_key,
                "pdp_url": products[0]["pdpUrl"]["url"],
                "variant_count": len(products),
            })
            total_variants += len(products)

        print(f"  anchor {anchor}: +{len(page_groupings)} groupings, "
              f"{total_variants} variants queued so far")

        if total_variants >= target_variants:
            break
        if not data.get("pages", {}).get("next"):
            break
        anchor += PAGE_COUNT
        time.sleep(0.2)

    return groupings


# ── step 2: visit one PDP per grouping → extract all variants ──────────────

def extract_next_data(page) -> dict | None:
    try:
        raw = page.query_selector("script#__NEXT_DATA__").inner_text()
        return json.loads(raw)
    except Exception:
        return None


def parse_variants_from_next_data(data: dict) -> list[dict]:
    """Return list of {product_code, name, color_name, price, product_url, image_urls}."""
    try:
        page_props = data["props"]["pageProps"]
        selected = page_props["selectedProduct"]
        group_idx = selected.get("groupIndex", 0)
        groups = page_props.get("productGroups") or []
        products = groups[group_idx]["products"] if groups else {selected["styleColor"]: selected}
    except Exception as e:
        print(f"      [warn] couldn't parse productGroups: {e}")
        return []

    variants = []
    for code, p in products.items():
        try:
            info = p.get("productInfo") or {}
            title = info.get("title") or ""
            group_key = p.get("groupKey", "")
            product_url = clean_product_url(
                (p.get("pdpUrl") or {}).get("url", ""), group_key
            )
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

            variants.append({
                "name": title,
                "color_name": clean_color(p.get("colorDescription", "")),
                "price": format_price(p.get("prices")),
                "product_code": code,
                "slug": slugify(title),
                "product_url": product_url,
                "image_urls": image_urls,
            })
        except Exception as e:
            print(f"      [warn] skipping variant {code}: {e}")

    return variants


# ── main ─────────────────────────────────────────────────────────────────────

def load_existing_db() -> tuple[list[dict], set[str]]:
    if DB_FILE.exists():
        db = json.loads(DB_FILE.read_text())
        return db, {p["product_code"] for p in db}
    return [], set()


def save_db(database: list[dict]):
    DB_FILE.write_text(json.dumps(database, indent=2, ensure_ascii=False))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Collecting groupings from search API...")
    groupings = collect_groupings(TARGET_VARIANTS)
    total_expected = sum(g["variant_count"] for g in groupings)
    print(f"\n{len(groupings)} groupings queued, ~{total_expected} color variants expected.\n")

    database, seen_codes = load_existing_db()
    print(f"Resuming with {len(database)} variants already saved.\n" if database else "")

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

        for idx, g in enumerate(groupings, 1):
            url = g["pdp_url"]
            print(f"[{idx}/{len(groupings)}] {url.split('/t/')[-1][:60]}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
            except Exception as e:
                print(f"  [warn] goto failed: {e}")
                continue
            time.sleep(2.5)

            next_data = extract_next_data(page)
            if not next_data:
                print("  [warn] no __NEXT_DATA__ found, skipping grouping")
                continue

            variants = parse_variants_from_next_data(next_data)
            new_variants = [v for v in variants if v["product_code"] not in seen_codes]
            print(f"  → {len(variants)} variants on page, {len(new_variants)} new")

            for v in new_variants:
                code = v["product_code"]
                slug = v["slug"]
                shoe_dir = OUTPUT_DIR / slug / code
                shoe_dir.mkdir(parents=True, exist_ok=True)

                saved = []
                seen_hashes: set[str] = set()
                for i, img_url in enumerate(v["image_urls"]):
                    dest = shoe_dir / f"image_{i}.jpg"
                    if download(img_url, dest):
                        h = hashlib.md5(dest.read_bytes()).hexdigest()
                        if h in seen_hashes:
                            dest.unlink()
                        else:
                            seen_hashes.add(h)
                            saved.append(str(dest))

                database.append({
                    "name": v["name"],
                    "color_name": v["color_name"],
                    "price": v["price"],
                    "product_code": code,
                    "slug": slug,
                    "product_url": v["product_url"],
                    "image_count": len(saved),
                    "images": saved,
                    "image_urls": v["image_urls"],
                })
                seen_codes.add(code)

            # checkpoint after every grouping
            save_db(database)
            print(f"  [checkpoint] {len(database)} total variants saved")

            time.sleep(0.3)

        browser.close()

    save_db(database)
    total_imgs = sum(p["image_count"] for p in database)
    print(f"\nDone. {len(database)} color variants, {total_imgs} images total.")
    print(f"Database → {DB_FILE}")
    print(f"Images   → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
