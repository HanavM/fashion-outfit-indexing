"""
Enrich nike_products.json with each variant's "Product Details" section.

Nike's PDP embeds this directly in __NEXT_DATA__ (productInfo.productDescription,
.featuresAndBenefits, .productDetails) — no click needed, the accordion UI just
reveals text that's already server-rendered into the page.

Adds a "details" field to each record:
    {"description": str, "features": [str, ...], "product_details": [str, ...]}

Usage:
    python enrich_nike_details.py            # full run
    python enrich_nike_details.py --limit 15 # test batch
"""

import argparse, json, re, time
from pathlib import Path

from playwright.sync_api import sync_playwright

DB_FILE = Path("nike_products.json")
CHECKPOINT_EVERY = 10


def extract_details(page) -> dict | None:
    try:
        raw = page.query_selector("script#__NEXT_DATA__").inner_text()
        data = json.loads(raw)
        info = data["props"]["pageProps"]["selectedProduct"]["productInfo"]
    except Exception as e:
        print(f"    [warn] parse failed: {e}")
        return None

    features = []
    for section in info.get("featuresAndBenefits") or []:
        features.extend(section.get("body") or [])

    product_details = []
    for section in info.get("productDetails") or []:
        product_details.extend(section.get("body") or [])

    return {
        "description": info.get("productDescription", ""),
        "features": features,
        "product_details": product_details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process first N variants (test)")
    args = parser.parse_args()

    database = json.loads(DB_FILE.read_text())
    todo = [p for p in database if "details" not in p]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} variants to enrich (of {len(database)} total).\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, channel="chrome",
                                      args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()

        for idx, p in enumerate(todo, 1):
            print(f"[{idx}/{len(todo)}] {p['product_code']} — {p['name'][:40]}")
            try:
                page.goto(p["product_url"], wait_until="domcontentloaded", timeout=45000)
            except Exception as e:
                print(f"  [warn] goto failed: {e}")
                continue
            time.sleep(1.5)

            details = extract_details(page)
            if details:
                p["details"] = details
            else:
                p["details"] = {"description": "", "features": [], "product_details": []}

            if idx % CHECKPOINT_EVERY == 0:
                DB_FILE.write_text(json.dumps(database, indent=2, ensure_ascii=False))
                print(f"  [checkpoint] saved")

            time.sleep(0.2)

        browser.close()

    DB_FILE.write_text(json.dumps(database, indent=2, ensure_ascii=False))
    enriched = sum(1 for p in database if "details" in p)
    print(f"\nDone. {enriched}/{len(database)} variants have details.")


if __name__ == "__main__":
    main()
