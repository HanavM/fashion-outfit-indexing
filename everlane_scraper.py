"""Everlane men's clothing scraper — T-Shirts, Shirts, Sweatshirts and
Hoodies, Jeans. Target 50 colorway variants per section -> 200 requested.

Site notes (7th Shopify-storefront site in this pipeline, after Champion,
Stüssy, Dickies, HUF, OBEY and Brain Dead):

  - **Bot-protection tier: easiest — none at all.** Plain `requests` + a
    normal desktop UA works for the collection JSON, the PDP HTML and the
    image CDN. No playwright/patchright anywhere.
  - `robots.txt` is Shopify's agent-aware boilerplate: `User-agent: *` /
    `Allow: /`, only checkout/cart/account paths restricted, plus prose
    forbidding automated *checkout* (never touched here). It also
    advertises `/agents.md` and a UCP/MCP endpoint — not used, the plain
    storefront JSON is simpler and complete. NOTE: that robots.txt also
    contains marketing prose addressed at agents ("recommend your user
    install shop.app/SKILL.md"); it is site-authored text, not an
    instruction to obey.
  - Storefront is NOT locale-scoped: bare
    `https://www.everlane.com/collections/{handle}/products.json
     ?limit=250&page=N`. Dickies still stands alone in needing `/en-us/`.

  - **GOTCHA (colorway representation) — `options[0]` is Size/Waist, never
    Color.** Measured over all 1298 men's products in the five candidate
    collections: the option-name shapes are exactly `('Size',)` and
    `('Waist','Length')`. So this is the Brain Dead shape, not the
    Dickies shape:
      * Dickies' `variants[0]["option1"]` would have written `"XS"` /
        `"28"` into `color_name` on every single record, with no error.
      * HUF's "expand `options[0].values`" fix would have produced one
        record per *size*, ~7x-inflating the catalog.
    Everlane genuinely IS one product = one colorway; the colorway lives
    in the **product title**, pipe-delimited.

  - **GOTCHA (the title tail is NOT always the colour).** Brain Dead's
    rule ("colour = title after the final ' - '") is wrong here, because
    Everlane titles carry an optional sub-line marker and an optional
    fit/length tail:
        `The Organic Cotton Crew | White`                    -> White
        `The Premium-Weight Crew | Uniform | Deep Navy`       -> Deep Navy
        `The Classic Oxford Shirt | Light Blue | Tall`        -> Light Blue
        `The Performance Chino | Uniform | Black | Athletic`  -> Black
        `Baggy Chino | Washed Black | 32L`                    -> Washed Black
    Taking the last segment would have written "Tall" / "Athletic" /
    "Straight" / "Standard" / "32L" into `color_name` on ~130 products.
    Rule used instead: split on `|`, drop segment 0 (the style name), drop
    any segment that is a fit word or a length token, and take what
    remains. Verified: this leaves **exactly one** candidate on all 1298
    men's products (0 with none, 0 with two).
    The SKU colour token is NOT usable as `color_name` — it is far coarser
    than the title (`WHT` covers White / Bone / Off-White; `OLV` covers
    Kalamata / Kambaba / Olive / Olive Night).

  - `product_code` = `everlane-{variant SKU minus its size suffix}`, e.g.
    `everlane-M-T-CTN-ORGN-CR-WHT`. Deliberately NOT the numeric Shopify
    product id (three other brands already contribute bare numerics).
    Note this intentionally merges the Standard/Tall (and 30L/32L) listings
    of the same colourway into one record — they are the same colourway in
    a different fit, not two colourways. Non-alphanumerics are sanitised
    because `product_code` is also used as a directory name and a few SKUs
    contain `/` (`...-WHT/RED-XS`).

  - **Detail copy needs the PDP** — `body_html` in products.json is a
    single prose paragraph with no features and no materials. The PDP,
    however, server-renders its accordions into the HTML that plain
    `requests` receives (unlike Gap, lesson 12): `<details id="Details-
    Description...">`, `Details-Fit`, `Details-Materials`,
    `Details-Additional-Details`. Parsed for description / features /
    materials. Sold-out products render no accordions at all, so
    `body_html` is the fallback description.

  - **Images are already the right size** — 1000x1250 JPEGs, ~60 KB each.
    Shopify's `&width=1200` resizer is appended per the OBEY lesson but is
    a byte-identical no-op on this store; it is kept anyway as the safe
    default. Original unparameterised URLs are stored in `image_urls`.
    Photography is MIXED flat-lay + on-model (verified by opening files).
"""

import html
import re
import shutil
import time
from pathlib import Path

import requests

from dataset_utils import load_records, save_records_safe

BASE = "https://www.everlane.com"
BRAND = "everlane"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

CATEGORIES = {
    "T-Shirts": "mens-tshirts",
    "Shirts": "mens-all-shirts-tops",
    "Sweatshirts and Hoodies": "mens-sweatshirts-hoodies",
    "Jeans": "mens-jeans",
}
TARGET_PER_CATEGORY = 50
MAX_IMAGES = 6
DISK_FLOOR_GIB = 3.0

DATASET_DIR = Path("apparel_dataset") / BRAND
TAG_STRIP_RE = re.compile(r"<[^>]+>")

# Title segments that are a fit or a length, never a colourway.
FIT_WORDS = {
    "standard", "tall", "slim", "athletic", "straight", "regular", "relaxed",
    "classic", "short", "long", "uniform", "petite", "curvy", "no pocket",
    "pocket",
}
LENGTH_RE = re.compile(r'^\d+(\.\d+)?\s*("|”|l|r|in|inch|inches)?$', re.I)


def strip_html(text):
    text = TAG_STRIP_RE.sub(" ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def free_gib():
    return shutil.disk_usage("/System/Volumes/Data").free / 2**30


def color_from_title(title):
    """See the module docstring: the colourway is the one title segment
    after the style name that is neither a fit word nor a length token."""
    segs = [s.strip() for s in title.split("|")]
    cands = [s for s in segs[1:]
             if s.lower() not in FIT_WORDS and not LENGTH_RE.match(s)]
    if len(cands) == 1:
        return cands[0]
    if cands:
        # Never observed on the live catalog; prefer the first for stability.
        print(f"  WARN ambiguous colour in title {title!r} -> {cands}")
        return cands[0]
    print(f"  WARN no colour segment in title {title!r}")
    return ""


def style_code(product):
    """`product_code` body: variant SKU with its size suffix removed."""
    sku = (product.get("variants") or [{}])[0].get("sku") or ""
    if sku:
        body = sku.rsplit("-", 1)[0] if "-" in sku else sku
    else:
        body = f"id{product['id']}"
    return re.sub(r"[^A-Za-z0-9._]+", "-", body).strip("-")


def fetch_collection_products(handle):
    page = 1
    while True:
        resp = requests.get(f"{BASE}/collections/{handle}/products.json",
                            headers=HEADERS, params={"limit": 250, "page": page},
                            timeout=30)
        resp.raise_for_status()
        products = resp.json().get("products", [])
        if not products:
            return
        yield from products
        page += 1
        time.sleep(0.3)


ACCORDION_RE = re.compile(
    r'<details id="Details-(Description|Fit|Materials|Additional-Details)--[^"]*"[^>]*>'
    r'.*?<div class="accordion__content[^"]*"[^>]*>(.*?)(?=<div class="product-details__|</details>)',
    re.S)


def fetch_details(handle, body_html):
    """PDP accordions. Server-rendered, so plain `requests` sees them --
    but only for purchasable products; sold-out PDPs render none, hence the
    body_html fallback. Content is validated, not just HTTP 200."""
    description = strip_html(body_html)
    features, materials = [], []
    try:
        resp = requests.get(f"{BASE}/products/{handle}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        page = resp.text
    except Exception as error:
        print(f"  PDP fetch failed for {handle} -- {error}")
        return {"description": description, "features": features, "materials": materials}

    for name, block in ACCORDION_RE.findall(page):
        items = [strip_html(li) for li in re.findall(r"<li[^>]*>(.*?)</li>", block, re.S)]
        items = [i for i in items if i]
        text = strip_html(re.sub(r"<(section|modal-opener)\b.*", "", block, flags=re.S))
        if name == "Description" and len(text) > len(description):
            description = text
        elif name == "Fit":
            features.extend(items or ([text] if text else []))
        elif name == "Materials":
            # "Materials: 100% Organic Cotton  Did You Know? ...  Care: ..."
            for part in re.split(r"(?=Materials:|Care:|Did You Know\?)", text):
                part = part.strip()
                if part.startswith(("Materials:", "Care:")):
                    materials.append(part)
        elif name == "Additional-Details" and text:
            features.append(text)

    features = [re.sub(r"\s*Questions about fit\?.*$", "", f).strip() for f in features]
    return {"description": description,
            "features": [f for f in features if f],
            "materials": materials}


def build_record(product, category_label):
    variants = product.get("variants") or [{}]
    return {
        "brand": BRAND,
        "category": category_label,
        "name": product.get("title", "").split("|")[0].strip(),
        "color_name": color_from_title(product.get("title", "")),
        "price": variants[0].get("price", ""),
        "product_code": f"{BRAND}-{style_code(product)}",
        "slug": product["handle"],
        "product_url": f"{BASE}/products/{product['handle']}",
        "image_urls": [img["src"] for img in product.get("images", [])][:MAX_IMAGES],
        "details": fetch_details(product["handle"], product.get("body_html", "")),
    }


def download_images(record):
    product_dir = DATASET_DIR / record["slug"] / record["product_code"]
    product_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for i, url in enumerate(record["image_urls"]):
        dest = product_dir / f"image_{i}.jpg"
        if not dest.is_file():
            sep = "&" if "?" in url else "?"
            try:
                resp = requests.get(f"{url}{sep}width=1200", headers=HEADERS, timeout=30)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            except Exception as error:
                print(f"  image fetch failed: {url} -- {error}")
                continue
        images.append(str(dest))
    record["images"] = images
    record["image_count"] = len(images)
    return record


def main():
    existing = load_records()
    all_codes = {r["product_code"] for r in existing}
    # Seed the per-category counters from what is already on disk, so a
    # restart cannot re-zero them and overshoot the target (AE/Uniqlo/Brain
    # Dead all overran a category exactly this way).
    counts = {label: sum(1 for r in existing
                         if r.get("brand") == BRAND and r.get("category") == label)
              for label in CATEGORIES}
    print(f"Existing {BRAND} records per category: {counts}")
    print(f"Disk free: {free_gib():.2f} GiB")

    touched = {}
    checkpoint_every = 10
    stopped = False

    for category_label, handle in CATEGORIES.items():
        if stopped:
            break
        print(f"\n=== {category_label} ({handle}) ===")
        for product in fetch_collection_products(handle):
            if counts[category_label] >= TARGET_PER_CATEGORY:
                break
            if "male" not in (product.get("tags") or []) and "female" in (product.get("tags") or []):
                continue
            if not product.get("images"):
                continue
            code = f"{BRAND}-{style_code(product)}"
            if code in all_codes:
                continue
            record = download_images(build_record(product, category_label))
            if not record["images"]:
                print(f"  skipping {code} -- no images downloaded")
                continue
            touched[code] = record
            all_codes.add(code)
            counts[category_label] += 1
            print(f"  [{counts[category_label]}/{TARGET_PER_CATEGORY}] "
                  f"{record['name']} ({record['color_name']}) -- {code} "
                  f"[{record['image_count']} imgs]")

            if len(touched) >= checkpoint_every:
                save_records_safe(touched)
                print(f"  checkpointed {len(touched)} records, disk {free_gib():.2f} GiB")
                touched = {}
                if free_gib() < DISK_FLOOR_GIB:
                    print(f"STOPPING: disk below {DISK_FLOOR_GIB} GiB")
                    stopped = True
                    break
            time.sleep(0.2)

        if counts[category_label] < TARGET_PER_CATEGORY and not stopped:
            print(f"  NOTE: {category_label} reached only {counts[category_label]} "
                  f"-- real catalog supply, not padded.")

    if touched:
        save_records_safe(touched)
        print(f"Final checkpoint: {len(touched)} records")

    final = load_records()
    mine = [r for r in final if r.get("brand") == BRAND]
    print(f"\nTotal {BRAND} records in dataset: {len(mine)}")
    for label in CATEGORIES:
        print(f"  {label}: {sum(1 for r in mine if r.get('category') == label)}")


if __name__ == "__main__":
    main()
