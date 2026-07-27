"""
Skechers Shoes Scraper — Full Detail Version

Skechers (Salesforce Commerce Cloud, same platform as New Balance) has no
bot protection at all — plain `requests` works for both the search grid and
product pages, no browser automation needed.

Workflow:
  1. Search-UpdateGrid AJAX endpoint (paginated via `startIndex`) → the grid
     HTML already contains a direct href for EVERY color variant (both the
     tile's default color and each color-swatch), so no separate
     variant-discovery step is needed.
  2. Visit each color variant's PDP → parse the embedded ld+json `Product`
     block, which has the full image gallery, name, color, price, and sku
     for that exact colorway already.
  3. Download images to skechers_catalog/{slug}/{sku}/image_N.jpg
  4. Write skechers_products.json incrementally (checkpoint every N variants)
     so a partial run doesn't lose data.

Each unique color variant becomes its own database entry.
"""

import hashlib, json, re, time
from pathlib import Path

import requests

SEARCH_URL = "https://www.skechers.com/search/?q=shoes&sz=48"
GRID_ENDPOINT = "https://www.skechers.com/on/demandware.store/Sites-USSkechers-Site/en_US/Search-UpdateGrid"
PAGE_SIZE = 48

OUTPUT_DIR = Path("skechers_catalog")
DB_FILE = Path("skechers_products.json")
TARGET_VARIANTS = 180
CHECKPOINT_EVERY = 10

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

VARIANT_HREF_RE = re.compile(r'href="(/[a-z0-9-]+/[A-Za-z0-9]+_[A-Za-z0-9]+\.html)"')


# ── helpers ──────────────────────────────────────────────────────────────────

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


# ── step 1: search grid → variant PDP URLs ──────────────────────────────────

def collect_variant_urls(target_variants: int) -> list[str]:
    urls, seen = [], set()
    start = 0

    while len(urls) < target_variants:
        resp = requests.get(
            GRID_ENDPOINT, params={"q": "shoes", "page": start // PAGE_SIZE, "startIndex": start},
            headers=HEADERS, timeout=20,
        )
        if resp.status_code != 200:
            print(f"  [warn] grid fetch failed at start={start}: {resp.status_code}")
            break
        found = VARIANT_HREF_RE.findall(resp.text)
        if not found:
            break
        new_this_page = 0
        for path in found:
            if path not in seen:
                seen.add(path)
                urls.append("https://www.skechers.com" + path)
                new_this_page += 1
        print(f"  grid start={start}: +{new_this_page} new variants ({len(urls)} total)")
        if new_this_page == 0:
            break
        start += PAGE_SIZE
        time.sleep(0.2)

    return urls


# ── step 2: variant PDP → ld+json Product block ─────────────────────────────

def parse_product_ldjson(html: str) -> dict | None:
    for m in re.finditer(r'<script type="application/ld\+json">(.*?)</script>', html, re.S):
        try:
            data = json.loads(m.group(1))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("@type") == "Product":
            return data
    return None


def fetch_variant_detail(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
    except Exception as e:
        print(f"    [warn] request failed: {e}")
        return None
    if r.status_code != 200:
        print(f"    [warn] status {r.status_code}")
        return None

    product = parse_product_ldjson(r.text)
    if not product:
        print("    [warn] no ld+json Product block found")
        return None

    return {
        "name": product.get("name", ""),
        "color_name": product.get("color", ""),
        "price": f"${product.get('offers', {}).get('price', '')}" if product.get("offers", {}).get("price") else "",
        "sku": product.get("sku", ""),
        "product_url": product.get("offers", {}).get("url", url),
        "image_urls": [
            re.sub(r";width=[^/]*", "", u)  # strip the resize transform for full-res original
            for u in (product.get("image") or [])
        ],
    }


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
    database, seen_codes = load_existing_db()
    if database:
        print(f"Resuming with {len(database)} variants already saved.\n")

    print("Collecting color-variant URLs from search grid...")
    variant_urls = collect_variant_urls(TARGET_VARIANTS)
    print(f"\n{len(variant_urls)} variant URLs queued.\n")

    for idx, url in enumerate(variant_urls, 1):
        if len(database) >= TARGET_VARIANTS:
            print(f"Reached target of {TARGET_VARIANTS} variants, stopping.")
            break

        print(f"[{idx}/{len(variant_urls)}] {url.split('.com')[-1][:60]}")
        detail = fetch_variant_detail(url)
        if not detail or not detail["sku"] or not detail["image_urls"]:
            print("  [warn] incomplete detail, skipping")
            continue
        if detail["sku"] in seen_codes:
            print("  already have this sku, skipping")
            continue

        slug = slugify(detail["name"])
        shoe_dir = OUTPUT_DIR / slug / detail["sku"]
        shoe_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        seen_hashes: set[str] = set()
        for i, img_url in enumerate(detail["image_urls"]):
            dest = shoe_dir / f"image_{i}.jpg"
            if download(img_url, dest):
                h = hashlib.md5(dest.read_bytes()).hexdigest()
                if h in seen_hashes:
                    dest.unlink()
                else:
                    seen_hashes.add(h)
                    saved.append(str(dest))

        database.append({
            "name": detail["name"],
            "color_name": detail["color_name"],
            "price": detail["price"],
            "product_code": detail["sku"],
            "slug": slug,
            "product_url": detail["product_url"],
            "image_count": len(saved),
            "images": saved,
            "image_urls": detail["image_urls"],
        })
        seen_codes.add(detail["sku"])

        if len(database) % CHECKPOINT_EVERY == 0:
            save_db(database)
            print(f"  [checkpoint] {len(database)} total variants saved")

        time.sleep(0.15)

    save_db(database)
    total_imgs = sum(p["image_count"] for p in database)
    print(f"\nDone. {len(database)} color variants, {total_imgs} images total.")
    print(f"Database → {DB_FILE}")
    print(f"Images   → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
