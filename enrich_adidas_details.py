"""
Enrich adidas_products.json with each variant's "Details" accordion, AND
backfill name/color_name/price — these were never scraped originally because
adidas_products.json was lost mid-run before this project started, so
adidas_catalog/ only ever had slug/product_code from the folder structure.

adidas_products.json doesn't exist yet in that case — this script creates it
from scratch by walking adidas_catalog/ and visiting each PDP once.

Adds:
    name, color_name, price          (backfilled if missing/empty)
    details = {"description": str, "bullets": [str, ...]}

Usage:
    python enrich_adidas_details.py            # full run
    python enrich_adidas_details.py --limit 15 # test batch
"""

import argparse, json, re, time
from pathlib import Path

from playwright.sync_api import sync_playwright

CATALOG_DIR = Path("adidas_catalog")
DB_FILE = Path("adidas_products.json")
CHECKPOINT_EVERY = 10


def load_or_build_db() -> list[dict]:
    if DB_FILE.exists():
        return json.loads(DB_FILE.read_text())

    database = []
    for slug_dir in sorted(CATALOG_DIR.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        for code_dir in sorted(slug_dir.iterdir()):
            if not code_dir.is_dir():
                continue
            code = code_dir.name
            images = sorted(code_dir.glob("image_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
            database.append({
                "name": "",
                "color_name": "",
                "price": "",
                "product_code": code,
                "slug": slug,
                "product_url": f"https://www.adidas.com/us/{slug}/{code}.html",
                "image_count": len(images),
                "images": [str(i) for i in images],
                "image_urls": [],
            })
    return database


def remove_overlays(page):
    try:
        page.evaluate(
            "document.querySelectorAll('[data-mf-id=\"cookie-consent-mf\"], "
            "dialog[open], [data-mf-id^=\"ap/\"]').forEach(e=>e.remove())"
        )
    except Exception:
        pass


def goto_safe(page, url: str):
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
    except Exception:
        pass
    time.sleep(3)
    remove_overlays(page)
    try:
        page.wait_for_selector("h1", timeout=8000)
    except Exception:
        pass


def get_text(page, sel: str) -> str:
    el = page.query_selector(sel)
    if not el:
        return ""
    try:
        return el.inner_text().strip()
    except Exception:
        return ""


def extract_details_accordion(page) -> dict:
    """Click the 'Details' accordion and parse description + bullets + color/code line."""
    try:
        for i in range(6):
            page.evaluate("window.scrollBy(0, 800)")
            time.sleep(0.4)
        remove_overlays(page)
        btn = page.query_selector('button:has-text("Details")')
        if btn:
            btn.click(timeout=8000)
            time.sleep(1)
    except Exception as e:
        print(f"    [warn] couldn't open Details accordion: {e}")
        remove_overlays(page)
        try:
            btn = page.query_selector('button:has-text("Details")')
            if btn:
                btn.click(timeout=5000)
                time.sleep(1)
        except Exception as e2:
            print(f"    [warn] retry also failed: {e2}")

    result = {"description": "", "bullets": [], "product_color_line": ""}
    try:
        for acc in page.query_selector_all('[data-testid*="accordion"]'):
            text = acc.inner_text()
            if text.startswith("Description"):
                # strip the "Description" header line
                result["description"] = "\n".join(text.split("\n")[1:]).strip()
            elif text.startswith("Details"):
                lines = [l.strip() for l in text.split("\n")[1:] if l.strip()]
                bullets = [l for l in lines if not l.startswith("Product color:") and not l.startswith("Product code:")]
                color_line = next((l for l in lines if l.startswith("Product color:")), "")
                result["bullets"] = bullets
                result["product_color_line"] = color_line
    except Exception as e:
        print(f"    [warn] accordion parse failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process first N variants (test)")
    args = parser.parse_args()

    database = load_or_build_db()
    if not DB_FILE.exists():
        DB_FILE.write_text(json.dumps(database, indent=2, ensure_ascii=False))
        print(f"Reconstructed {len(database)} variant records from adidas_catalog/ folder.\n")

    todo = [p for p in database if "details" not in p]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} variants to enrich (of {len(database)} total).\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, channel="chrome",
                                      args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()

        for idx, p in enumerate(todo, 1):
            print(f"[{idx}/{len(todo)}] {p['product_code']} — {p['slug'][:40]}")
            goto_safe(page, p["product_url"])

            title = get_text(page, "h1")
            price_raw = get_text(page, "[data-testid='main-price']") or get_text(page, "[data-testid='sale-price']")
            price_match = re.search(r"\$[\d,]+(?:\.\d+)?", price_raw)
            price = price_match.group(0) if price_match else ""

            details = extract_details_accordion(page)

            if not p.get("name"):
                p["name"] = title
            if not p.get("price"):
                p["price"] = price
            if not p.get("color_name") and details["product_color_line"]:
                p["color_name"] = details["product_color_line"].replace("Product color:", "").strip()

            p["details"] = {"description": details["description"], "bullets": details["bullets"]}

            if idx % CHECKPOINT_EVERY == 0:
                DB_FILE.write_text(json.dumps(database, indent=2, ensure_ascii=False))
                print(f"  [checkpoint] saved")

            time.sleep(0.3)

        browser.close()

    DB_FILE.write_text(json.dumps(database, indent=2, ensure_ascii=False))
    enriched = sum(1 for p in database if "details" in p)
    named = sum(1 for p in database if p.get("name"))
    print(f"\nDone. {enriched}/{len(database)} variants have details. {named}/{len(database)} have names.")


if __name__ == "__main__":
    main()
