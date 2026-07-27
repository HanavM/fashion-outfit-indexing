"""
New Balance Shoes Scraper — Full Detail Version

New Balance (Salesforce Commerce Cloud) has Akamai bot protection that blocks
category/search/API pages under plain Playwright — even with stealth flags,
the CDP connection itself gets fingerprinted. `patchright` (a CDP-stealth
Playwright fork) gets through cleanly, so it's used here instead.

Workflow:
  1. Search-UpdateGrid AJAX endpoint (paginated via `start`/`sz`) → collect
     master-product PDP URLs from the search grid tiles.
  2. Visit each master PDP once → parse the embedded ld+json `ProductGroup`
     block, which lists every color/size SKU combo. Dedupe by color to get
     one representative style-code per color variant.
  3. For each color, visit the `Product-Variation` JSON endpoint (same
     mechanism the site's own color-swatch buttons use) → full-res image
     gallery, name, and price for that exact colorway.
  4. Download images to newbalance_catalog/{slug}/{style_code}/image_N.jpg
  5. Write newbalance_products.json incrementally (checkpoint after every
     grouping) so a partial run doesn't lose data.

Each unique color variant becomes its own database entry.
"""

import hashlib, json, re, time
from pathlib import Path

import requests
from patchright.sync_api import sync_playwright

SEARCH_URL = "https://www.newbalance.com/shoes/?searchKey=shoes&sm=Search%20Bar%20and%20Type%20Text"
GRID_ENDPOINT = "https://www.newbalance.com/on/demandware.store/Sites-NBUS-Site/en_US/Search-UpdateGrid"
CGID = "4004704"
PAGE_SIZE = 18

OUTPUT_DIR = Path("newbalance_catalog")
DB_FILE = Path("newbalance_products.json")
TARGET_VARIANTS = 180


# ── helpers ──────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s.strip("-")[:60]


def clean_color(raw: str) -> str:
    return re.sub(r"\s+with\s+", " / ", raw or "", flags=re.I).strip()


def format_price(val) -> str:
    if val is None:
        return ""
    return f"${val:.0f}" if float(val).is_integer() else f"${val:.2f}"


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


# ── step 1: search grid → master PDP URLs ───────────────────────────────────

def collect_master_urls(page, target_variants: int) -> list[str]:
    """Page through the search grid, returning master PDP URLs. Since we don't
    know variant-count-per-master ahead of time, over-collect a bit relative
    to target_variants (many masters have 3-8 colors)."""
    urls, seen = [], set()
    start = 0
    approx_needed_masters = max(20, target_variants // 4)  # rough heuristic

    while len(urls) < approx_needed_masters:
        resp = page.goto(
            f"{GRID_ENDPOINT}?cgid={CGID}&start={start}&sz={PAGE_SIZE}",
            wait_until="domcontentloaded", timeout=45000,
        )
        if not resp or resp.status != 200:
            print(f"  [warn] grid fetch failed at start={start}: "
                  f"{resp.status if resp else 'no response'}")
            break
        html = page.content()
        found = re.findall(r'href="(/pd/[^"?]+/([A-Za-z0-9_-]+)\.html)[^"]*"', html)
        if not found:
            break
        new_this_page = 0
        for path, master_id in found:
            full_url = "https://www.newbalance.com" + path
            if master_id not in seen:
                seen.add(master_id)
                urls.append(full_url)
                new_this_page += 1
        print(f"  grid start={start}: +{new_this_page} new masters "
              f"({len(urls)} total)")
        if new_this_page == 0:
            break
        start += PAGE_SIZE
        time.sleep(0.3)

    return urls


# ── step 2: master PDP → unique colors via ld+json ──────────────────────────

def parse_product_group(html: str) -> dict | None:
    for m in re.finditer(
        r'<script type="application/ld\+json">(.*?)</script>', html, re.S
    ):
        try:
            data = json.loads(m.group(1))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("@type") == "ProductGroup":
            return data
    return None


def unique_colors_from_group(group: dict) -> list[dict]:
    """Dedupe hasVariant SKUs by color, return [{style_id, color_name}]."""
    seen_colors = {}
    for variant in group.get("hasVariant", []):
        sku = variant.get("sku", "")
        m = re.match(r"^([A-Za-z0-9]+)-", sku)
        style_id = m.group(1) if m else sku
        color = variant.get("color", "")
        if color and color not in seen_colors:
            seen_colors[color] = style_id
    return [{"style_id": v, "color_name": k} for k, v in seen_colors.items()]


# ── step 3: Product-Variation JSON → full gallery for one color ────────────

def build_variation_url(master_id: str, style_id: str) -> str:
    return (
        "https://www.newbalance.com/on/demandware.store/Sites-NBUS-Site/en_US/"
        f"Product-Variation?dwvar_{master_id}_style={style_id}&"
        f"dwvar_{master_id}_width=D&pid={master_id}&quantity=1"
    )


def fetch_variant_detail(page, master_id: str, style_id: str) -> dict | None:
    url = build_variation_url(master_id, style_id)
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        print(f"      [warn] goto failed: {e}")
        return None
    if not resp or resp.status != 200:
        print(f"      [warn] variation fetch status {resp.status if resp else None}")
        return None
    try:
        text = page.evaluate("document.body.innerText")
        data = json.loads(text)
    except Exception as e:
        print(f"      [warn] JSON parse failed: {e}")
        return None

    p = data.get("product", {})
    images = [
        img.get("zoomsrc") or img.get("src")
        for img in (p.get("images", {}).get("productDetail") or [])
    ]
    images = [i for i in images if i]

    return {
        "name": f"{p.get('brand', 'New Balance')} {p.get('productName', '')}".strip(),
        "price": format_price((p.get("price", {}).get("sales") or {}).get("value")),
        "image_urls": images,
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

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, channel="chrome")
        context = browser.new_context(viewport={"width": 1440, "height": 900}, locale="en-US")
        page = context.new_page()

        print("Warming up session on homepage...")
        page.goto("https://www.newbalance.com/", wait_until="domcontentloaded", timeout=60000)
        time.sleep(3)

        print("Collecting master product URLs from search grid...")
        master_urls = collect_master_urls(page, TARGET_VARIANTS)
        print(f"\n{len(master_urls)} master products queued.\n")

        total_variants = len(database)
        for idx, master_url in enumerate(master_urls, 1):
            if total_variants >= TARGET_VARIANTS:
                print(f"Reached target of {TARGET_VARIANTS} variants, stopping.")
                break

            print(f"[{idx}/{len(master_urls)}] {master_url.split('/pd/')[-1][:60]}")
            try:
                resp = page.goto(master_url, wait_until="domcontentloaded", timeout=45000)
            except Exception as e:
                print(f"  [warn] goto failed: {e}")
                continue
            if not resp or resp.status != 200:
                print(f"  [warn] status {resp.status if resp else None}, skipping")
                continue
            time.sleep(1.5)

            html = page.content()
            group = parse_product_group(html)
            if not group:
                print("  [warn] no ld+json ProductGroup found, skipping")
                continue

            master_id = group.get("productGroupID", "")
            model_name = group.get("name", "")
            slug = slugify(f"new-balance-{model_name}") if model_name else slugify(master_id)
            colors = unique_colors_from_group(group)
            new_colors = [c for c in colors if c["style_id"] not in seen_codes]
            print(f"  → {len(colors)} colors, {len(new_colors)} new")

            for c in new_colors:
                style_id = c["style_id"]
                detail = fetch_variant_detail(page, master_id, style_id)
                if not detail or not detail["image_urls"]:
                    print(f"    [warn] no detail/images for {style_id}, skipping")
                    continue

                shoe_dir = OUTPUT_DIR / slug / style_id
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

                product_url = f"https://www.newbalance.com/pd/{master_url.split('/pd/')[-1].split('/')[0]}/{master_id}.html?dwvar_{master_id}_style={style_id}"

                database.append({
                    "name": detail["name"] or model_name,
                    "color_name": clean_color(c["color_name"]),
                    "price": detail["price"],
                    "product_code": style_id,
                    "slug": slug,
                    "product_url": product_url,
                    "image_count": len(saved),
                    "images": saved,
                    "image_urls": detail["image_urls"],
                })
                seen_codes.add(style_id)
                total_variants += 1
                time.sleep(0.3)

            save_db(database)
            print(f"  [checkpoint] {len(database)} total variants saved")

        browser.close()

    save_db(database)
    total_imgs = sum(p["image_count"] for p in database)
    print(f"\nDone. {len(database)} color variants, {total_imgs} images total.")
    print(f"Database → {DB_FILE}")
    print(f"Images   → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
