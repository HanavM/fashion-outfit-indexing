"""
Gap men's clothing scraper — T-Shirts, Shorts, Pants, Sweaters.
Target 50 colorway variants per section -> 200 requested (Sweaters only has
26 colorways in the whole category, same "capped by catalog size" situation
PacSun's mens-sweaters hit, so the realistic total is 176).

Site notes (first Gap Inc. site in this pipeline):
  - No bot protection encountered at all — plain `requests` works for both
    the category listing and the PDP, no browser/Playwright/patchright
    needed anywhere in this scraper (same tier as Skechers).
  - Category listing comes from a clean JSON API, not an embedded blob or
    HTML scrape: `https://api.gap.com/commerce/search/products/v2/cc
    ?pageSize=200&pageNumber=0&cid={cid}&department=75&vendor=constructorio
    &client_id=0&session_id=0&brand=gap&locale=en_US&market=us` — cid is the
    men's-category id (T-Shirts=5225, Shorts=5156, Pants=80799,
    Sweaters=5180; department=75 for all four). pageSize=200 covers every
    category tested here in one page (largest was 171 colorways), but the
    fetch still pages by pageNumber until pageNumberTotal is exhausted for
    safety on a category that grows past that later.
  - One API product ("style") bundles every colorway as a `styleColors`
    entry directly inline — no separate per-colorway page load needed to
    discover them (unlike PacSun, where each colorway was already a
    separate PDP/URL requiring its own grid-fragment page). `ccId` (a
    9-digit code) is the stable per-colorway product_code; `styleId` is the
    shared product-family id.
  - Image URLs are relative paths (e.g. `/webcontent/0061/457/472/
    cn61457472.jpg`) served off `https://www1.assets-gap.com` — prefixing
    `www.gap.com` also works (same Akamai image origin) but assets-gap.com
    is the dedicated image CDN.
  - Each position (camera angle) in a colorway's `images` list carries ~10
    resolution/crop variants of the SAME shot (thumbnail, quicklook, "OVI"
    hero, etc.) — the one to keep is whichever type is exactly "Z" (position
    1) or ends in "_Z" (`AV1_Z`, `AV2_Z`, ...), Gap's zoom/full-res variant;
    picking "highest score"-style richness by eye, "Z" was consistently the
    largest (1500x2000 in spot checks) vs. ~text-thumbnail-sized others.
  - PDP url is simply `https://www.gap.com/browse/product.do?pid={ccId}`
    (no styleId or category needed in the URL at all).
  - "Product details" and "Fabric & care" bullets sit in a React Suspense
    boundary that a plain `requests.get()` always receives as a
    `BAILOUT_TO_CLIENT_SIDE_RENDERING` placeholder with an EMPTY `<ul>` —
    the heading text ("Product details") is present in the raw HTML either
    way, which is a trap: checking for that substring alone looks like
    success even though the actual bullet content never rendered. A plain
    headless `playwright` page (no anti-automation flags, no patchright
    needed — Gap has no bot protection on either the catalog API or the
    PDP) with a couple seconds' wait renders the real content client-side.
    No separate `enrich_gap_details.py` stage needed, but this scraper uses
    Playwright for the per-colorway PDP visit instead of `requests`.
"""

import hashlib, html, json, re, time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

from dataset_utils import load_records, save_records_safe

CATEGORY_API = "https://api.gap.com/commerce/search/products/v2/cc"
IMAGE_BASE = "https://www1.assets-gap.com"
PDP_BASE = "https://www.gap.com/browse/product.do"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

CATEGORIES = {
    "5225": "T-Shirts",
    "5156": "Shorts",
    "80799": "Pants",
    "5180": "Sweaters",
}
DEPARTMENT = "75"

OUTPUT_DIR = Path("apparel_dataset/gap")
TARGET_PER_CATEGORY = 50
PAGE_SIZE = 200


def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s.strip("-")[:60]


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        r = requests.get(url, timeout=20, headers=HEADERS)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"      [warn] {e}")
        return False


def image_url(path: str) -> str:
    return IMAGE_BASE + (path if path.startswith("/") else "/" + path)


def best_image_urls(images: list[dict]) -> list[str]:
    """One full-res URL per camera-angle position: the `Z` (position 1) or
    `AVn_Z` (later positions) type is Gap's zoom/full-res variant; every
    other type at that position is a smaller crop/thumbnail of the same
    shot."""
    by_position: dict[int, list[dict]] = {}
    for img in images:
        by_position.setdefault(img["position"], []).append(img)

    urls = []
    for pos in sorted(by_position):
        variants = by_position[pos]
        zoom = next((v for v in variants if v["type"] == "Z" or v["type"].endswith("_Z")), None)
        chosen = zoom or variants[0]
        urls.append(image_url(chosen["path"]))
    return urls


def iter_colorways(cid: str, category: str):
    """Yields flattened per-colorway dicts, paging pageNumber until the API
    reports every page fetched (pageSize=200 covers every category tested
    here in one page, but this still pages defensively)."""
    page_number = 0
    while True:
        params = {
            "pageSize": PAGE_SIZE,
            "pageNumber": page_number,
            "ignoreInventory": "false",
            "cid": cid,
            "vendor": "constructorio",
            "client_id": 0,
            "session_id": 0,
            "includeMarketingFlagsDetails": "true",
            "enableDynamicFacets": "true",
            "enableDynamicPhoto": "true",
            "brand": "gap",
            "locale": "en_US",
            "market": "us",
            "department": DEPARTMENT,
        }
        r = requests.get(CATEGORY_API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()

        products = data.get("products") or []
        if not products:
            return

        for product in products:
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


def extract_accordion(page_html: str, title: str) -> list[str]:
    m = re.search(rf'title="{re.escape(title)}".*?<ul[^>]*>(.*?)</ul>', page_html, re.S)
    if not m:
        return []
    out = []
    for li in re.findall(r"<li[^>]*>(.*?)</li>", m.group(1), re.S):
        text = re.sub(r"<[^>]+>", "", li).strip()
        text = re.sub(r"\s+", " ", html.unescape(text))
        if text:
            out.append(text)
    return out


def fetch_details(page, ccid: str) -> tuple[dict, str]:
    url = f"{PDP_BASE}?pid={ccid}"
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    time.sleep(2.0)
    html = page.content()
    return {
        "features": extract_accordion(html, "Product details"),
        "materials": extract_accordion(html, "Fabric &amp; care"),
    }, url


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    database = load_records()
    seen_codes = {p["product_code"] for p in database}
    print(f"Starting with {len(database)} existing records.\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        for cid, category in CATEGORIES.items():
            print(f"\n=== {category} (cid={cid}) ===")
            already = sum(1 for p in database if p.get("brand") == "gap" and p.get("category") == category)
            target_new = TARGET_PER_CATEGORY - already
            if target_new <= 0:
                print(f"{category} already has {already} records, skipping.")
                continue

            added = 0
            for colorway in iter_colorways(cid, category):
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
        if p.get("brand") == "gap" and p.get("category") in CATEGORIES.values()
    )
    print(f"\nDone. {len(database)} total records in dataset.")
    print(f"Gap images downloaded: {total_imgs}")


if __name__ == "__main__":
    main()
