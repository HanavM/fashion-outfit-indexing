"""
Adidas Originals Shoes Scraper — Full Detail Version

Workflow:
  1. Listing page  → collect all product URLs
  2. Each product  → collect ALL color variant URLs + shoe name
  3. Each color URL → scrape color name + all gallery images
  4. Download images to adidas_catalog/{shoe_slug}/{product_code}/
  5. Write adidas_products.json

Each unique color variant becomes its own database entry.
"""

import hashlib, json, re, time, requests
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE_URL   = "https://www.adidas.com/us/originals-shoes"
OUTPUT_DIR = Path("adidas_catalog")
DB_FILE    = Path("adidas_products.json")


# ── helpers ────────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:60]


def hi_res(url: str) -> str:
    """Upgrade Adidas CDN image to max resolution."""
    # Replace any single or paired dimension transform with h_2000
    url = re.sub(r"w_\d+,h_\d+", "w_2000,h_2000", url)
    url = re.sub(r"\bw_\d+\b", "w_2000", url)
    url = re.sub(r"\?.*$", "", url)
    return url


def product_code(url: str) -> str:
    m = re.search(r"/([A-Z0-9]{4,8})\.html", url)
    return m.group(1) if m else ""


def shoe_slug_from_url(url: str) -> str:
    """Extract the model slug, e.g. 'taekwondo-mei-ballet-shoes'."""
    m = re.search(r"/us/([^/]+)/[A-Z0-9]+\.html", url)
    return m.group(1) if m else ""


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


# ── listing page ───────────────────────────────────────────────────────────────

def collect_listing_urls(page) -> list[str]:
    """Scroll the listing page and return all product detail URLs."""
    # Wait for the page to fully settle before touching it
    time.sleep(3)
    # First scroll triggers the product grid
    page.evaluate("window.scrollBy(0, 600)")
    time.sleep(2)

    prev = 0
    for i in range(60):
        try:
            page.evaluate("window.scrollBy(0, 900)")
            time.sleep(1.2)
            count = len(page.query_selector_all("article[data-testid='plp-product-card']"))
        except Exception:
            # Page navigated mid-scroll — wait and re-check
            time.sleep(2)
            try:
                count = len(page.query_selector_all("article[data-testid='plp-product-card']"))
            except Exception:
                break
        print(f"  scroll {i+1}: {count} products", end="\r")
        if count == prev and count > 0 and i > 3:
            break
        prev = count
    print()

    cards = page.query_selector_all("article[data-testid='plp-product-card']")
    urls, seen = [], set()
    for card in cards:
        link = card.query_selector("a[href*='.html']")
        if not link:
            continue
        href = link.get_attribute("href") or ""
        if href.startswith("/"):
            href = "https://www.adidas.com" + href
        if href and href not in seen:
            seen.add(href)
            urls.append(href)
    return urls


# ── product detail page ────────────────────────────────────────────────────────

def goto_safe(page, url: str):
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
    except Exception:
        pass
    time.sleep(3.5)


def get_shoe_name(page) -> str:
    try:
        for sel in ["h1[data-testid='product-title']", "h1[class*='name']", "h1"]:
            el = page.query_selector(sel)
            if el:
                t = el.inner_text().strip()
                if t:
                    return t
    except Exception:
        pass
    return ""


def get_price(page) -> str:
    try:
        for sel in [
            "[data-testid='main-price'] span:last-child",
            "[data-testid='main-price']",
            "[data-testid='sale-price']",
            "[class*='price'] span:last-child",
        ]:
            el = page.query_selector(sel)
            if el:
                t = el.inner_text().strip()
                if t and ("$" in t or t[0].isdigit()):
                    return t
    except Exception:
        pass
    return ""


def get_color_name(page) -> str:
    """
    Adidas shows the selected color like 'Silver Metallic / Grey One / Cloud White'.
    Try several known data-testid / class patterns.
    """
    for sel in [
        "[data-testid='pdp-color-description']",
        "[data-testid='color-chip-label']",
        "[data-testid='selected-color-label']",
        "[data-testid='colorway-label']",
        "[class*='color-description']",
        "[class*='colorDescription']",
        "[class*='color_description']",
        "[class*='colorway']",
    ]:
        el = page.query_selector(sel)
        if el:
            t = el.inner_text().strip()
            if t:
                return t

    # Fallback: search page text for "Color: ..." or a pattern with slashes (e.g. "Black / White")
    # scan all small text nodes that look like color descriptions
    try:
        texts = page.evaluate("""() => {
            const els = document.querySelectorAll('p, span, div');
            const out = [];
            for (const el of els) {
                const t = el.childNodes.length === 1 && el.firstChild.nodeType === 3
                    ? el.textContent.trim() : '';
                if (t && t.includes(' / ') && t.length < 120) out.push(t);
            }
            return [...new Set(out)].slice(0, 5);
        }""")
        if texts:
            return texts[0]
    except Exception:
        pass
    return ""


def get_gallery_images(page, code: str) -> list[str]:
    """
    Collect all unique gallery images for this product code.

    Deduplication key = filename (e.g. SAMBA_OG_KI2181_HM3.jpg).
    The same image appears under many CDN transform URLs; filename is the
    true unique identifier. Only view images (HM1–HM9, HB1–HB9) are kept.
    """
    # Try to click "show more" to expose extra gallery images
    for sel in [
        "button:has-text('Show more')",
        "button:has-text('show more')",
        "[data-testid='show-more']",
        "[class*='show-more']",
        "[class*='showMore']",
    ]:
        try:
            btn = page.query_selector(sel)
            if btn:
                btn.click()
                time.sleep(1.5)
                break
        except Exception:
            pass

    code_upper = code.upper()
    # Map filename → canonical hi-res URL (dedup by filename)
    by_filename: dict[str, str] = {}

    def add(raw_url: str):
        if code_upper not in raw_url.upper():
            return
        m = re.search(r'/([^/?#]+\.jpg)', raw_url)
        if not m:
            return
        filename = m.group(1)
        fl = filename.lower()
        if "hover" in fl or "_plp_" in fl:
            return
        by_filename[filename] = hi_res(raw_url)

    try:
        for img in page.query_selector_all("img[src*='assets.adidas.com']"):
            try:
                add(img.get_attribute("src") or "")
            except Exception:
                pass
    except Exception:
        pass

    try:
        for m in re.finditer(
            r'https://assets\.adidas\.com/images/[^"\'> ]+\.jpg',
            page.content(), re.IGNORECASE,
        ):
            add(m.group(0))
    except Exception:
        pass

    return list(by_filename.values())


def get_color_variants(page, current_url: str) -> list[str]:
    slug = shoe_slug_from_url(current_url)
    if not slug:
        return []

    variant_urls: set[str] = set()
    try:
        links = page.query_selector_all(f"a[href*='/{slug}/']")
        for link in links:
            try:
                href = link.get_attribute("href") or ""
            except Exception:
                continue
            if href.startswith("/"):
                href = "https://www.adidas.com" + href
            if re.search(r"/us/" + re.escape(slug) + r"/[A-Z0-9]+\.html", href):
                variant_urls.add(href)
    except Exception:
        pass

    try:
        html = page.content()
        for m in re.finditer(r'href="(/us/' + re.escape(slug) + r'/[A-Z0-9]+\.html)"', html):
            variant_urls.add("https://www.adidas.com" + m.group(1))
    except Exception:
        pass

    return list(variant_urls)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    database: list[dict] = []

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

        # ── Step 1: listing page → seed URLs ──────────────────────────────────
        print(f"Loading listing page…")
        goto_safe(page, BASE_URL)
        print("Scrolling to collect products…")
        seed_urls = collect_listing_urls(page)
        print(f"Collected {len(seed_urls)} seed URLs from listing.\n")

        # ── Step 2: visit each seed to discover all color variant URLs ─────────
        print("Discovering color variants for each shoe…")
        all_variant_urls: set[str] = set(seed_urls)

        for i, url in enumerate(seed_urls, 1):
            print(f"  [{i}/{len(seed_urls)}] {url.split('/us/')[1][:60]}", end="")
            goto_safe(page, url)
            variants = get_color_variants(page, url)
            new = [v for v in variants if v not in all_variant_urls]
            all_variant_urls.update(variants)
            print(f"  → +{len(new)} new variants")

        print(f"\nTotal unique color variants to scrape: {len(all_variant_urls)}\n")

        # ── Step 3: scrape every color variant ────────────────────────────────
        all_urls = sorted(all_variant_urls)
        for idx, url in enumerate(all_urls, 1):
            code = product_code(url)
            slug = shoe_slug_from_url(url)
            print(f"[{idx}/{len(all_urls)}] {slug[:40]}  {code}")

            goto_safe(page, url)

            name       = get_shoe_name(page)
            color_name = get_color_name(page)
            price      = get_price(page)
            img_urls   = get_gallery_images(page, code)

            print(f"  name:   {name}")
            print(f"  color:  {color_name or '(not found)'}")
            print(f"  price:  {price or '(not found)'}")
            print(f"  images: {len(img_urls)}")

            # Save images to adidas_catalog/{shoe_slug}/{product_code}/
            shoe_dir = OUTPUT_DIR / slugify(slug) / code
            shoe_dir.mkdir(parents=True, exist_ok=True)

            saved = []
            seen_hashes: set[str] = set()
            for i, img_url in enumerate(img_urls):
                dest = shoe_dir / f"image_{i}.jpg"
                if download(img_url, dest):
                    h = hashlib.md5(dest.read_bytes()).hexdigest()
                    if h in seen_hashes:
                        dest.unlink()
                    else:
                        seen_hashes.add(h)
                        saved.append(str(dest))

            database.append({
                "name":         name or slug.replace("-", " ").title(),
                "color_name":   color_name,
                "price":        price,
                "product_code": code,
                "slug":         slug,
                "product_url":  url,
                "image_count":  len(saved),
                "images":       saved,
                "image_urls":   img_urls,
            })

            time.sleep(0.3)

        browser.close()

    # ── Step 4: write database ─────────────────────────────────────────────────
    DB_FILE.write_text(json.dumps(database, indent=2, ensure_ascii=False))
    total_imgs = sum(p["image_count"] for p in database)
    print(f"\nDone. {len(database)} color variants, {total_imgs} images total.")
    print(f"Database → {DB_FILE}")
    print(f"Images   → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
