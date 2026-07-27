"""
Enrich skechers_products.json with each variant's "Key Features" and
"Design Details" bullet lists.

Skechers has no bot protection at all — these sections are static HTML,
already present in the page source, so plain `requests` is enough (no
browser automation needed).

Adds a "details" field to each record:
    {"key_features": [str, ...], "design_details": [str, ...]}

Usage:
    python enrich_skechers_details.py            # full run
    python enrich_skechers_details.py --limit 15 # test batch
"""

import argparse, json, re, time
from pathlib import Path

import requests

DB_FILE = Path("skechers_products.json")
CHECKPOINT_EVERY = 20
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

SECTION_RE = re.compile(
    r'(Key Features|Design Details)\s*</h2>.*?<ul class="c-product-features-details__bullets[^"]*">(.*?)</ul>',
    re.S,
)
BULLET_RE = re.compile(r'<li class="c-product-features-details__bullet">\s*(.*?)\s*</li>', re.S)


def extract_details(html: str) -> dict:
    result = {"key_features": [], "design_details": []}
    for label, body in SECTION_RE.findall(html):
        bullets = [re.sub(r"\s+", " ", b).strip() for b in BULLET_RE.findall(body)]
        bullets = [re.sub(r"&reg;|&trade;|&amp;", "", b).strip() for b in bullets]
        if label == "Key Features":
            result["key_features"] = bullets
        else:
            result["design_details"] = bullets
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

    for idx, p in enumerate(todo, 1):
        print(f"[{idx}/{len(todo)}] {p['product_code']} — {p['name'][:40]}")
        try:
            r = requests.get(p["product_url"], headers=HEADERS, timeout=20)
        except Exception as e:
            print(f"  [warn] request failed: {e}")
            continue
        if r.status_code != 200:
            print(f"  [warn] status {r.status_code}")
            continue

        p["details"] = extract_details(r.text)

        if idx % CHECKPOINT_EVERY == 0:
            DB_FILE.write_text(json.dumps(database, indent=2, ensure_ascii=False))
            print(f"  [checkpoint] saved")

        time.sleep(0.15)

    DB_FILE.write_text(json.dumps(database, indent=2, ensure_ascii=False))
    enriched = sum(1 for p in database if "details" in p)
    print(f"\nDone. {enriched}/{len(database)} variants have details.")


if __name__ == "__main__":
    main()
