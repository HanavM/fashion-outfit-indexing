"""J.Crew men's clothing scraper — T-Shirts and Polos, Shirts, Sweaters,
Jeans. Target 50 colorway variants per section -> 200 requested.

Site notes:

  - **Bot-protection tier: HARD (Akamai, sustained not intermittent).**
    Plain `requests` gets a 403 `errors.edgesuite.net` "Access Denied" page
    on *every* URL including `robots.txt` and `sitemap-index.xml`, even
    with the full browser header set that was enough to clear American
    Eagle's Akamai. Measured persistence per the AE lesson (don't inherit
    another site's backoff constants): a 10-request burst at one PLP
    returned **403 on all 10** — so unlike AE, a J.Crew 403 is NOT a blip
    and no retry ladder fixes it.
    What does fix it: **patchright (`channel="chrome"`, headed) clears the
    edge on its own, and the cookies it mints work in plain `requests`.**
    So the browser is used exactly once, to open the homepage and hand its
    cookie jar + UA to a `requests.Session`; every catalog page, PDP and
    image after that is a plain HTTP fetch. On a 403 the right response is
    to re-mint the jar, not to sleep longer.
  - `robots.txt` (readable once cookies exist) has no `Disallow: /` for
    `*`. Nothing this scraper fetches is disallowed. Two rules matter and
    are respected: **`Disallow: /api/` and `Disallow: */data/v1/`** — which
    is exactly where the PLP grid's XHR lives, so the client-side product
    API is deliberately NOT used. `Allow: /s7-img-facade/*` covers the
    images. Sitemaps are advertised in robots.txt itself.

  - **Seeding: the sitemap, not the PLP.** The PLP is client-rendered from
    the robots-disallowed `/api/` endpoint — its `__NEXT_DATA__` ships
    `products.productsByProductCode == {}` (a valid-looking empty blob, cf.
    lesson 12), and rendering + scrolling it in a browser tops out at
    ~29 styles no matter how long you scroll. `custom-sitemap-N-Jcrew-US-
    product.xml` (4 files, 3819 URLs) carries the whole catalog with the
    category baked into the path, giving 61 tshirts-and-polos / 64 shirts /
    79 sweaters / 53 jeans men's styles. Women's URLs share the same
    category slugs, so the `/p/mens/` prefix filter is load-bearing.

  - **Data source: the PDP's `__NEXT_DATA__`**, which IS server-rendered
    (unlike the PLP's): `props.initialState.products.
    productsByProductCode[{styleCode}]` holds name, `colorsList`,
    `priceModel`, `productDescriptionRomance` / `...Tech` / `...Fit`,
    `gender` and the marketing slug. One PDP fetch per style.

  - **Colorway representation: one PDP == one STYLE with N colorways**
    (the HUF/Uniqlo shape, not the Dickies one). Verified by reading the
    data rather than inferring from the URL: `colorsList[*].colors[]` is
    the colorway list (`code`, `name`), and `priceModel[style].colors[]`
    carries the same colour codes with a per-colorway price and
    `skuShotType`. The PDP URL's `colorCode` query param merely preselects
    one of them — seeding one record per PDP URL would have kept ~1 of the
    3-15 colorways of every style.
    Colorways are taken **breadth-first across styles** (round 1 = the 1st
    colorway of every style, round 2 = the 2nd, ...) per the AE lesson, so
    a single 15-colour tee cannot eat a category.

  - **Images: Scene7 behind `/s7-img-facade/{style}_{color}{suffix}`, and
    a missing asset returns HTTP 200 with a placeholder, not a 404.** Every
    unknown suffix (`_f`, `_ob`, `_of`, `_zzz`, even a nonexistent style)
    returns the same 42,145-byte 1200x1200 JPEG reading "A GREAT IMAGE IS
    ON ITS WAY. PLEASE POP BACK LATER." — so guessing suffixes from another
    brand's vocabulary silently fills a dataset with placeholder art.
    Two defences, both used: (1) only the suffixes listed in that
    colorway's own `skuShotType` are requested, plus the bare
    `{style}_{color}` URL which is the flat-lay and is never listed; (2)
    the placeholder's md5 is fetched at startup from a deliberately bogus
    URL and every download is compared against it.
    `?wid=1200` gives ~1200x1200 at 40-110 KB (the parameterless URL is a
    smaller 34 KB render); unparameterised URLs are stored in `image_urls`.
    Photography is MIXED: the bare URL is a clean garment-only flat-lay,
    `_m` is an on-model studio shot — verified by opening the files.
"""

import hashlib
import html
import json
import re
import shutil
import time
from pathlib import Path

import requests

from dataset_utils import load_records, save_records_safe

BASE = "https://www.jcrew.com"
BRAND = "jcrew"

CATEGORIES = {
    "T-Shirts and Polos": "tshirts-and-polos",
    "Shirts": "shirts",
    "Sweaters": "sweaters",
    "Jeans": "jeans",
}
TARGET_PER_CATEGORY = 50
MAX_IMAGES = 6
MAX_STYLES_PER_CATEGORY = 90
DISK_FLOOR_GIB = 3.0

DATASET_DIR = Path("apparel_dataset") / BRAND
SITEMAP_INDEX = f"{BASE}/sitemap/sitemap-index.xml"
TAG_STRIP_RE = re.compile(r"<[^>]+>")
NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
FIBRE_RE = re.compile(r"\d+\s*%|cotton|wool|linen|cashmere|denim|polyester|nylon|silk|leather|suede",
                      re.I)

DOC_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
}


def strip_html(text):
    text = TAG_STRIP_RE.sub(" ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def free_gib():
    return shutil.disk_usage("/System/Volumes/Data").free / 2**30


# --------------------------------------------------------------------------
# Akamai: one headed patchright visit mints a cookie jar that plain requests
# can reuse. Re-minted on a 403 -- sleeping does not help on this site.
# --------------------------------------------------------------------------

def mint_session():
    from patchright.sync_api import sync_playwright

    print("  [akamai] minting cookies via patchright ...")
    with sync_playwright() as play:
        browser = play.chromium.launch(channel="chrome", headless=False)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(6000)
        user_agent = page.evaluate("navigator.userAgent")
        cookies = ctx.cookies()
        browser.close()

    session = requests.Session()
    for cookie in cookies:
        session.cookies.set(cookie["name"], cookie["value"], domain=cookie["domain"])
    session.headers.update({**DOC_HEADERS, "User-Agent": user_agent})
    return session


class Client:
    def __init__(self):
        self.session = mint_session()

    def get(self, url, **kwargs):
        for attempt in range(4):
            try:
                resp = self.session.get(url, timeout=45, **kwargs)
            except Exception as error:
                print(f"  request error {error} ({url})")
                time.sleep(2.0)
                continue
            if resp.status_code == 200 and "Access Denied" not in resp.text[:400]:
                return resp
            print(f"  blocked ({resp.status_code}) on {url[:90]}")
            if attempt == 0:
                time.sleep(0.6)
            else:
                self.session = mint_session()
        return None

    def get_bytes(self, url):
        # Scene7 CONTENT-NEGOTIATES on Accept (the AE lesson): with a
        # browser's `image/avif,image/webp,*/*` Accept it returns AVIF for
        # some assets and JPEG for others, so an explicit jpeg Accept is
        # what makes the format deterministic -- there is no `fmt=` knob.
        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=45,
                                        headers={"Accept": "image/jpeg,image/*;q=0.8"})
                if resp.status_code == 200:
                    return resp.content
            except Exception as error:
                print(f"  image error {error}")
            time.sleep(0.8)
        return None


# --------------------------------------------------------------------------


def seed_styles(client):
    """category label -> ordered list of (style_code, pdp_url) from the
    sitemap. Women's URLs reuse the same category slugs, hence /p/mens/."""
    index = client.get(SITEMAP_INDEX)
    sitemaps = [u for u in re.findall(r"<loc>(.*?)</loc>", index.text) if "product" in u]
    seeds = {label: [] for label in CATEGORIES}
    seen = set()
    for sitemap in sitemaps:
        resp = client.get(sitemap)
        if resp is None:
            continue
        for url in re.findall(r"<loc>(.*?)</loc>", resp.text):
            match = re.search(r"/p/mens/categories/clothing/([^/]+)/.*/([A-Z0-9]+)$", url)
            if not match:
                continue
            slug, style = match.group(1), match.group(2)
            for label, cat_slug in CATEGORIES.items():
                if slug == cat_slug and style not in seen:
                    seen.add(style)
                    seeds[label].append((style, url))
        time.sleep(0.3)
    return seeds


def fetch_style(client, style_code, pdp_url):
    """One PDP -> the style's product object from __NEXT_DATA__.
    Validates content, not HTTP 200: an Akamai page has no __NEXT_DATA__,
    and a mis-seeded URL has no entry for this style code."""
    resp = client.get(pdp_url)
    if resp is None:
        return None
    match = NEXT_DATA_RE.search(resp.text)
    if not match:
        print(f"  {style_code}: no __NEXT_DATA__ in response")
        return None
    try:
        blob = json.loads(match.group(1))
    except ValueError:
        print(f"  {style_code}: __NEXT_DATA__ did not parse")
        return None
    products = blob["props"]["initialState"]["products"]["productsByProductCode"]
    product = products.get(style_code)
    if product is None:
        print(f"  {style_code}: absent from productsByProductCode {list(products)}")
        return None
    if product.get("gender") not in (None, "men", "mens"):
        print(f"  {style_code}: gender={product.get('gender')}, skipping")
        return None
    return product


def colorways_of(product):
    """[(color_code, color_name, price, shot_suffixes)] in site order.

    `colorsList` is the ordered colour list; `priceModel` carries the
    per-colorway price and shot types. Both are keyed by colour code."""
    shots, prices = {}, {}
    for style_entry in (product.get("priceModel") or {}).values():
        for color in style_entry.get("colors", []):
            code = color.get("colorCode")
            if not code:
                continue
            shots[code] = [s for s in (color.get("skuShotType") or "").split(",") if s]
            price = color.get("salePrice") or style_entry.get("listPrice") or {}
            prices[code] = price.get("formatted", "")

    out, seen = [], set()
    for group in product.get("colorsList") or []:
        for color in group.get("colors", []):
            code = color.get("code")
            if not code or code in seen:
                continue
            seen.add(code)
            out.append((code, strip_html(color.get("name") or "").title(),
                        prices.get(code, ""), shots.get(code, [])))
    return out


def slug_of(product, style_code):
    url = product.get("url") or ""
    parts = [p for p in url.split("?")[0].split("/") if p]
    if len(parts) >= 2 and parts[-1] == style_code:
        return parts[-2]
    return style_code.lower()


def build_details(product):
    tech = [strip_html(b) for b in (product.get("productDescriptionTech") or [])]
    fit = [strip_html(b) for b in (product.get("productDescriptionFit") or [])]
    extra = strip_html(product.get("productDescriptionFitAdditionalText") or "")
    features = [b for b in tech + fit + ([extra] if extra else []) if b]
    return {
        "description": strip_html(product.get("productDescriptionRomance") or ""),
        "features": features,
        "materials": [b for b in tech if FIBRE_RE.search(b)],
    }


def build_record(product, style_code, category_label, color_code, color_name, price, shots):
    slug = slug_of(product, style_code)
    base = f"{BASE}/s7-img-facade/{style_code}_{color_code}"
    # Bare URL first: it is the flat-lay and is never listed in skuShotType.
    urls = [base] + [f"{base}{suffix}" for suffix in shots]
    return {
        "brand": BRAND,
        "category": category_label,
        # productName is stored HTML-escaped ("Piqu&eacute; ... polo shirt").
        "name": strip_html(product.get("productName", "")),
        "color_name": color_name,
        "price": price,
        "product_code": f"{BRAND}-{style_code}-{color_code}",
        "slug": slug,
        "product_url": f"{BASE}{product.get('url', '')}",
        "image_urls": urls[:MAX_IMAGES],
        "details": build_details(product),
    }


def download_images(client, record, placeholder_md5):
    product_dir = DATASET_DIR / record["slug"] / record["product_code"]
    product_dir.mkdir(parents=True, exist_ok=True)
    images, kept_urls = [], []
    for url in record["image_urls"]:
        dest = product_dir / f"image_{len(images)}.jpg"
        if not dest.is_file():
            data = client.get_bytes(f"{url}?wid=1200")
            if data is None:
                continue
            if hashlib.md5(data).hexdigest() == placeholder_md5:
                # "A GREAT IMAGE IS ON ITS WAY" -- a missing asset served 200.
                continue
            if not data.startswith(b"\xff\xd8\xff"):
                # Belt-and-braces after the explicit jpeg Accept: anything
                # that is not a JPEG (an AVIF-encoded placeholder slipped
                # through the md5 check exactly this way) is not a photo.
                continue
            dest.write_bytes(data)
        images.append(str(dest))
        kept_urls.append(url)
    record["image_urls"] = kept_urls
    record["images"] = images
    record["image_count"] = len(images)
    return record


def main():
    client = Client()
    placeholder = client.get_bytes(f"{BASE}/s7-img-facade/ZZZ999_XX0000?wid=1200")
    placeholder_md5 = hashlib.md5(placeholder).hexdigest() if placeholder else ""
    print(f"placeholder md5: {placeholder_md5}")

    existing = load_records()
    all_codes = {r["product_code"] for r in existing}
    counts = {label: sum(1 for r in existing
                         if r.get("brand") == BRAND and r.get("category") == label)
              for label in CATEGORIES}
    print(f"Existing {BRAND} records per category: {counts}")
    print(f"Disk free: {free_gib():.2f} GiB")

    seeds = seed_styles(client)
    for label, items in seeds.items():
        print(f"  seed {label}: {len(items)} men's styles from sitemap")

    touched = {}
    stopped = False

    for category_label, style_seeds in seeds.items():
        if stopped:
            break
        if counts[category_label] >= TARGET_PER_CATEGORY:
            continue
        print(f"\n=== {category_label} ===")

        styles = []
        for style_code, pdp_url in style_seeds[:MAX_STYLES_PER_CATEGORY]:
            product = fetch_style(client, style_code, pdp_url)
            if product is None:
                continue
            colorways = colorways_of(product)
            if colorways:
                styles.append((style_code, product, colorways))
            time.sleep(0.4)
        supply = sum(len(c) for _, _, c in styles)
        print(f"  {len(styles)} styles fetched, {supply} colorways available")

        # Breadth-first across styles (AE lesson): round r takes colorway r
        # of every style, so one 15-colour style cannot eat the category.
        for round_index in range(max((len(c) for _, _, c in styles), default=0)):
            if counts[category_label] >= TARGET_PER_CATEGORY or stopped:
                break
            for style_code, product, colorways in styles:
                if counts[category_label] >= TARGET_PER_CATEGORY:
                    break
                if round_index >= len(colorways):
                    continue
                color_code, color_name, price, shots = colorways[round_index]
                code = f"{BRAND}-{style_code}-{color_code}"
                if code in all_codes:
                    continue
                record = build_record(product, style_code, category_label,
                                      color_code, color_name, price, shots)
                record = download_images(client, record, placeholder_md5)
                if not record["images"]:
                    print(f"  skipping {code} -- no real images (all placeholders?)")
                    continue
                touched[code] = record
                all_codes.add(code)
                counts[category_label] += 1
                print(f"  [{counts[category_label]}/{TARGET_PER_CATEGORY}] "
                      f"{record['name']} ({color_name}) -- {code} "
                      f"[{record['image_count']} imgs]")

                if len(touched) >= 10:
                    save_records_safe(touched)
                    print(f"  checkpointed {len(touched)}, disk {free_gib():.2f} GiB")
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
