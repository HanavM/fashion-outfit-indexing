"""
Enrich apparel_dataset/metadata.json records with each variant's "Product
Details" section, generalized from enrich_nike_details.py to operate on the
shared apparel_dataset (instead of a per-brand nike_products.json), scoped by
default to Nike clothing records missing "details" (the 200 new
Tops/Shorts/Hoodies/Pants records) rather than the already-enriched shoes.

Nike's PDP embeds this directly in __NEXT_DATA__ (productInfo.productDescription,
.featuresAndBenefits, .productDetails) — no click needed.

Usage:
    python enrich_apparel_details.py             # all Nike records missing details
    python enrich_apparel_details.py --limit 15  # test batch
"""

import argparse, json, time

from playwright.sync_api import sync_playwright

from dataset_utils import load_records, save_records_safe

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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--brand", default="nike")
    args = parser.parse_args()

    database = load_records()
    todo = [p for p in database if "details" not in p and p.get("brand") == args.brand]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} records to enrich (of {len(database)} total).\n")

    touched = {}
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
            touched[p["product_code"]] = p

            if idx % CHECKPOINT_EVERY == 0:
                save_records_safe(touched)
                print(f"  [checkpoint] saved")

            time.sleep(0.2)

        browser.close()

    database = save_records_safe(touched)
    enriched = sum(1 for p in database if "details" in p and p.get("brand") == args.brand)
    print(f"\nDone. {enriched} {args.brand} records have details.")


if __name__ == "__main__":
    main()
