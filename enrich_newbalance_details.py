"""
Enrich newbalance_products.json with each variant's "Product Details" accordion
(Features + Material bullets) plus the model "Description" text.

New Balance needs patchright (Akamai bot protection blocks plain Playwright).

Adds a "details" field to each record:
    {"description": str, "features": [str, ...], "materials": [str, ...]}

Usage:
    python enrich_newbalance_details.py            # full run
    python enrich_newbalance_details.py --limit 15 # test batch
"""

import argparse, json, re, time
from pathlib import Path

from patchright.sync_api import sync_playwright

DB_FILE = Path("newbalance_products.json")
CHECKPOINT_EVERY = 10


def remove_modal(page):
    try:
        page.evaluate(
            "document.querySelectorAll('.storepage.discount_modal, .background').forEach(e=>e.remove())"
        )
    except Exception:
        pass


def is_blocked(page) -> bool:
    """Detect Akamai's 'Oops! Something went wrong' block page."""
    try:
        title = page.title()
        if "Oops" in title:
            return True
        body_start = page.evaluate("document.body.innerText.slice(0, 200)")
        return "Oops! Something went wrong" in body_start
    except Exception:
        return False


def goto_with_retry(page, url: str, max_attempts: int = 4) -> bool:
    """Navigate to url, retrying with backoff if Akamai serves a block page."""
    for attempt in range(1, max_attempts + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
        except Exception as e:
            print(f"    [warn] goto failed (attempt {attempt}): {e}")
            time.sleep(3 * attempt)
            continue
        time.sleep(2)
        if not is_blocked(page):
            return True
        wait = 5 * attempt
        print(f"    [warn] blocked (attempt {attempt}/{max_attempts}), backing off {wait}s")
        time.sleep(wait)
    return False


def extract_details(page) -> dict:
    result = {"description": "", "features": [], "materials": []}

    try:
        for i in range(6):
            page.evaluate("window.scrollBy(0, 800)")
            time.sleep(0.4)
        remove_modal(page)
    except Exception:
        pass

    # Read Description before clicking Product Details — these accordions
    # are mutually exclusive (opening one collapses the other), and
    # inner_text() only returns currently-visible text.
    try:
        desc_acc = page.query_selector('accordion-component[data-title="Description"]')
        if desc_acc:
            text = desc_acc.inner_text()
            text = re.sub(r"^Description\s*", "", text).strip()
            text = re.sub(r"^Looking for other options\?[^\n]*\n+", "", text).strip()
            result["description"] = text
    except Exception as e:
        print(f"    [warn] description parse failed: {e}")

    try:
        btn = page.query_selector('button:has-text("Product Details")')
        if btn:
            btn.click(timeout=8000)
            time.sleep(1)
    except Exception as e:
        print(f"    [warn] couldn't open Product Details accordion: {e}")

    try:
        details_acc = page.query_selector('accordion-component[data-title="Product Details"]')
        if details_acc:
            headers = details_acc.query_selector_all("h3.product-details-header")
            for h in headers:
                label = h.inner_text().strip().lower()
                ul = h.evaluate_handle("el => el.nextElementSibling")
                items_text = ul.as_element().inner_text() if ul.as_element() else ""
                items = [l.strip() for l in items_text.split("\n") if l.strip()]
                if "material" in label:
                    result["materials"].extend(items)
                elif "feature" in label:
                    result["features"].extend(items)
    except Exception as e:
        print(f"    [warn] details accordion parse failed: {e}")

    return result


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
        browser = pw.chromium.launch(headless=False, channel="chrome")
        context = browser.new_context(viewport={"width": 1440, "height": 900}, locale="en-US")
        page = context.new_page()

        print("Warming up session on homepage...")
        page.goto("https://www.newbalance.com/", wait_until="domcontentloaded", timeout=60000)
        time.sleep(3)
        remove_modal(page)

        skipped_blocked = 0
        for idx, p in enumerate(todo, 1):
            print(f"[{idx}/{len(todo)}] {p['product_code']} — {p['name'][:40]}")
            if not goto_with_retry(page, p["product_url"]):
                print(f"  [warn] still blocked after retries, leaving unenriched for next run")
                skipped_blocked += 1
                continue
            remove_modal(page)

            p["details"] = extract_details(page)

            if idx % CHECKPOINT_EVERY == 0:
                DB_FILE.write_text(json.dumps(database, indent=2, ensure_ascii=False))
                print(f"  [checkpoint] saved")

            time.sleep(1.0)

        browser.close()

    DB_FILE.write_text(json.dumps(database, indent=2, ensure_ascii=False))
    enriched = sum(1 for p in database if "details" in p)
    print(f"\nDone. {enriched}/{len(database)} variants have details. "
          f"{skipped_blocked} skipped due to persistent blocking (re-run to retry).")


if __name__ == "__main__":
    main()
