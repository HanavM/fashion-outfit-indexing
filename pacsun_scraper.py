"""
PacSun men's clothing scraper — Pants, Graphic Tees, Sweaters, Shorts.
Target 50 variants (colorways) per section -> 200 new records.

Site notes (see SCRAPING_PROCESS.md for the full write-up once this run is
folded in):
  - Plain `playwright` headless (and even non-headless plain playwright) gets
    an "Access to this page has been denied" bot-block page. `patchright`
    (headed, channel="chrome") gets through cleanly — same mitigation as
    New Balance.
  - SFCC (Salesforce Commerce Cloud) site, same family as New Balance:
      * Category pages page via `Search-UpdateGrid?cgid={cgid}&start=N&sz=12`
        (an HTML fragment, not JSON — fetched via page.goto in the same
        browser context so it inherits the bot-check pass).
      * Each PDP embeds a `ProductGroup` ld+json block. Unlike New Balance/
        Nike, PacSun's "hasVariant" list is SIZES of ONE fixed colorway, not
        multiple colors — each colorway is already its own separate PDP/URL
        (the category grid lists colorways as distinct tiles), so one PDP
        visit = one dataset record, no groupKey/colorway-expansion step
        needed.
      * `productGroupID` in the ld+json (13-digit code) is the stable
        per-colorway product_code.
      * Human-readable color name is NOT in the ld+json (only a numeric
        swatch code like "349") — it's in an
        `aria-label="Select Color MEDIUM INDIGO"` attribute on the rendered
        page instead.
      * Description + "Fit & Sizing" bullets + "Care & Composition" bullets
        (materials) are all server-rendered in accordions — no click needed,
        unlike Adidas/New Balance's JS-gated accordions.
"""

import hashlib, json, re, time
from pathlib import Path

import requests
from patchright.sync_api import sync_playwright

from dataset_utils import load_records, save_records_safe

BASE = "https://www.pacsun.com"
GRID_URL = BASE + "/on/demandware.store/Sites-pacsun-Site/default/Search-UpdateGrid"

CATEGORIES = {
    "mens-pants": "Pants",
    "mens-graphic-tees": "Graphic Tees",
    "mens-sweaters": "Sweaters",
    "mens-shorts": "Shorts",
}

OUTPUT_DIR = Path("apparel_dataset/pacsun")
TARGET_PER_CATEGORY = 50
PAGE_SIZE = 12


def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s.strip("-")[:60]


def hi_res(url: str) -> str:
    return re.sub(r"[?&]sw=\d+", "?sw=1200", url) if "sw=" in url else url + "?sw=1200"


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


def extract_ld_product_group(html: str) -> dict | None:
    for block in re.findall(
        r'<script type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S
    ):
        try:
            data = json.loads(block)
        except Exception:
            continue
        if data.get("@type") == "ProductGroup":
            return data
    return None


def extract_accordion_bullets(html: str, collapse_id: str) -> list[str]:
    m = re.search(
        rf'id="{collapse_id}"[^>]*>(.*?)</div>\s*</div>', html, re.S
    )
    if not m:
        return []
    return [
        re.sub(r"\s+", " ", li).strip()
        for li in re.findall(r"<li>(.*?)</li>", m.group(1), re.S)
        if li.strip()
    ]


def parse_pdp(html: str) -> dict | None:
    group = extract_ld_product_group(html)
    if not group or not group.get("hasVariant"):
        return None

    variant = group["hasVariant"][0]
    price = (variant.get("offers") or {}).get("price")
    price_str = f"${price:.2f}" if isinstance(price, (int, float)) else ""

    color_match = re.search(r'Select Color ([A-Za-z0-9 /\-\.]+)"', html)
    color_name = color_match.group(1).strip().title() if color_match else ""

    image_urls = [hi_res(u) for u in (variant.get("image") or [])]

    features = extract_accordion_bullets(html, "collapse-fit-sizing")
    materials = extract_accordion_bullets(html, "collapse-care-composition")

    return {
        "name": group.get("name", ""),
        "color_name": color_name,
        "price": price_str,
        "product_code": group.get("productGroupID", ""),
        "product_url": group.get("url", ""),
        "image_urls": image_urls,
        "details": {
            "description": (group.get("description") or "").strip(),
            "features": features,
            "materials": materials,
        },
    }


def iter_category_pdp_urls(page, cgid: str):
    """Yields unique /pacsun/...html PDP paths, paging the SFCC grid fragment
    until a page returns no new URLs (matches New Balance/Nike's page-until-
    exhausted pattern rather than assuming a fixed count)."""
    seen = set()
    start = 0
    while True:
        url = f"{GRID_URL}?cgid={cgid}&start={start}&sz={PAGE_SIZE}"
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"  [warn] grid fetch failed at start={start}: {e}")
            return
        time.sleep(1.0)
        html = page.content()
        hrefs = re.findall(r'href="(/pacsun/[a-z0-9\-]+\.html)"', html)
        new_this_page = [h for h in hrefs if h not in seen]
        if not new_this_page:
            return
        for h in new_this_page:
            seen.add(h)
            yield h
        start += PAGE_SIZE


def load_db() -> list[dict]:
    return load_records()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    database = load_db()
    seen_codes = {p["product_code"] for p in database}
    print(f"Starting with {len(database)} existing records.\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, channel="chrome")
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        for cgid, category in CATEGORIES.items():
            print(f"\n=== {category} ({cgid}) ===")
            already = sum(
                1 for p in database if p.get("brand") == "pacsun" and p.get("category") == category
            )
            target_new = TARGET_PER_CATEGORY - already
            if target_new <= 0:
                print(f"{category} already has {already} records, skipping.")
                continue

            added = 0
            for href in iter_category_pdp_urls(page, cgid):
                if added >= target_new:
                    break

                pdp_url = BASE + href
                print(f"[{added + 1}/{target_new}] {category}: {href}")

                try:
                    page.goto(pdp_url, wait_until="domcontentloaded", timeout=30000)
                except Exception as e:
                    print(f"  [warn] goto failed: {e}")
                    continue
                time.sleep(1.5)

                html = page.content()
                if "Access to this page has been denied" in html:
                    print("  [warn] bot-blocked, backing off 15s")
                    time.sleep(15)
                    continue

                variant = parse_pdp(html)
                if not variant or not variant["product_code"]:
                    print("  [warn] couldn't parse ProductGroup, skipping")
                    continue

                code = variant["product_code"]
                if code in seen_codes:
                    print(f"  [skip] {code} already in dataset")
                    continue

                slug = slugify(variant["name"] or code)
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
                    "brand": "pacsun",
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
                    "details": variant["details"],
                }
                database.append(new_record)
                seen_codes.add(code)
                added += 1

                database = save_records_safe({code: new_record})
                print(f"  [checkpoint] {len(database)} total records "
                      f"({added}/{target_new} added for {category})")

                time.sleep(0.5)

            print(f"\n{category}: {added} new records added "
                  f"({already + added}/{TARGET_PER_CATEGORY} total).")

        browser.close()

    total_imgs = sum(
        p["image_count"] for p in database
        if p.get("brand") == "pacsun" and p.get("category") in CATEGORIES.values()
    )
    print(f"\nDone. {len(database)} total records in dataset.")
    print(f"PacSun images downloaded: {total_imgs}")


if __name__ == "__main__":
    main()
