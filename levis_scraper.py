"""
Levi's (levi.com) men's clothing/accessories scraper — Jeans, Jean Jackets,
Shirts, Accessories. Target 50 colorway variants per section -> 200
requested (a section may fall short if its real catalog is smaller, same
"capped by catalog size" situation PacSun's mens-sweaters and Gap's
mens-sweaters hit).

Site notes (first Levi Strauss & Co. site in this pipeline):
  - Real bot protection, Akamai Bot Manager tier — HARDER than New Balance's
    Akamai: plain `requests` gets an immediate Akamai edge "Access Denied"
    (errors.edgesuite.net) on every page, and even `patchright` (headed,
    channel="chrome") initially loads an interactive Akamai *behavioral
    challenge* interstitial (`sec-if-cpt-container`, "Powered and protected
    by Akamai") instead of a flat block. Unlike New Balance where the block
    is binary (through or blocked), here patchright's traffic looks human
    enough that the challenge auto-resolves and the page reloads itself
    within ~3-9s if you just wait (poll `page.title()` in a loop, don't
    treat the first response as final) -- no manual interaction needed, but
    a short fixed sleep alone is fragile; poll until the title changes away
    from empty/"Access Denied".
  - Vue.js SSR app (not Next.js/Nuxt globals despite `data-v-*` hydration
    markers), own `window.__LSCO_INITIAL_STATE__` blob -- but it gets
    **deleted from `window` after hydration completes** (confirmed:
    `Object.getOwnPropertyNames(window)` no longer lists it a few seconds
    after load), so don't rely on reading it via `page.evaluate` after any
    wait -- it's a red herring here, not a stable data source.
  - The real, stable data source is a schema.org **ld+json `ProductGroup`**
    block on every PDP (`<script type="application/ld+json">`), same family
    as New Balance/PacSun's `ProductGroup` pattern: `hasVariant` is a list of
    per-colorway `Product` objects, each with its own `sku` (stable
    `product_code`), `color`, `image` (list of full CDN URLs, no query
    string), `offers.price`, and a shared `description` inherited from the
    parent product family. **One PDP visit yields every colorway inline**,
    same efficiency as Gap's `styleColors` bundling -- no separate grid-
    fragment or per-colorway page needed once the ld+json is parsed.
  - Category *listing* pages are capped at exactly 38 product-detail-page
    links per URL regardless of category (confirmed identical count across
    jeans/jean-jackets/shirts/accessories main category pages) -- no
    infinite-scroll or "load more" trigger increases this. To get past 38
    unique products per section, harvest PDP links across MULTIPLE listing
    URLs for the same section (fit/subcategory pages for jeans, colorgroup
    facet pages for jean jackets/shirts, sub-department pages for
    accessories) and dedupe by PDP href -- see LISTING_URLS below.
  - Image CDN is Scene7 (`lscoglobal.scene7.com`), same dynamic-imaging
    service family as other Adobe-Scene7-backed retailers. ld+json image
    URLs are usually bare but SOMETIMES already carry a Scene7 preset
    query string (e.g. `...GLO_CM_DA?$qv_desktop_full$`) -- appending
    `?fmt=jpeg&wid=1500&hei=2000` blindly produces a malformed double-`?`
    URL that Scene7 403s on. Always strip any existing query string first
    (`url.split("?")[0]`) before appending the explicit-size params, same
    convention as Gap's zoom-variant sizing otherwise. No CDN-embedded
    view/angle codes were found (unlike Skechers/Adidas) -- images are
    just sequentially ordered in the list.
  - `hasVariant` on the ld+json `ProductGroup` sometimes contains a bare
    list of `{"url": ...}` sibling-colorway links mixed in alongside the
    real per-colorway `Product` dicts (not nested under its own key) --
    must filter to `isinstance(v, dict)` before reading `v["sku"]` or it
    throws `'list' object has no attribute 'get'` on every single PDP.
  - "Composition & Care" bullet text is lazy-rendered below the fold --
    absent from `page.content()` until the page is scrolled, even though
    it's not behind an accordion click (no JS-gated click needed, unlike
    Adidas/New Balance -- just scroll-into-view).
    No separate `enrich_levis_details.py` stage needed once scrolled.
  - PDP visits must go through the SAME Akamai challenge-wait logic as the
    category listing pages -- it is not a one-time cookie/session unlock;
    every fresh `page.goto()` to a new PDP can re-trigger the interstitial.
"""

import re
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import requests
from patchright.sync_api import sync_playwright

from dataset_utils import load_records, save_records_safe


class WatchdogTimeout(Exception):
    pass


@contextmanager
def watchdog(seconds: int):
    """Hard wall-clock timeout independent of Playwright's own `timeout=`
    param -- observed in practice that a PDP visit can wedge a renderer
    process (confirmed via `ps`: one Chrome renderer pegged at 144% CPU,
    13 minutes of accumulated CPU time, no exception ever surfacing)
    well past goto()'s own 60s timeout, most likely Akamai's challenge
    script repeatedly calling `location.reload(true)` out from under
    Playwright's navigation tracking. A SIGALRM-based watchdog forces
    forward progress regardless of what's actually stuck inside the
    browser automation layer."""
    def _handler(signum, frame):
        raise WatchdogTimeout(f"exceeded {seconds}s wall-clock limit")
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

OUTPUT_DIR = Path("apparel_dataset/levis")
TARGET_PER_CATEGORY = 50

# One or more listing URLs per section, harvested across fit/color/sub-
# department pages since any single listing URL caps at 38 PDP links.
LISTING_URLS = {
    "Jeans": [
        "https://www.levi.com/US/en_US/clothing/men/jeans/c/levi_clothing_men_jeans",
        "https://www.levi.com/US/en_US/clothing/men/jeans/bootcut/c/levi_clothing_men_jeans_bootcut",
        "https://www.levi.com/US/en_US/clothing/men/jeans/loose/c/levi_clothing_men_jeans_loose",
        "https://www.levi.com/US/en_US/clothing/men/jeans/relaxed/c/levi_clothing_men_jeans_relaxed",
        "https://www.levi.com/US/en_US/clothing/men/jeans/slim/c/levi_clothing_men_jeans_slim",
        "https://www.levi.com/US/en_US/clothing/men/jeans/straight/c/levi_clothing_men_jeans_straight",
        "https://www.levi.com/US/en_US/clothing/men/jeans/taper/c/levi_clothing_men_jeans_taper",
    ],
    "Jean Jackets": [
        "https://www.levi.com/US/en_US/clothing/men/c/levi_clothing_men/facets/productitemtype/trucker%20jean%20jacket",
        "https://www.levi.com/US/en_US/clothing/men/outerwear/c/levi_clothing_men_outerwear/facets/colorgroup/blue/productitemtype/trucker%20jean%20jacket",
        "https://www.levi.com/US/en_US/clothing/men/outerwear/c/levi_clothing_men_outerwear/facets/colorgroup/black/productitemtype/trucker%20jean%20jacket",
        "https://www.levi.com/US/en_US/clothing/men/outerwear/c/levi_clothing_men_outerwear/facets/colorgroup/light%20wash/productitemtype/trucker%20jean%20jacket",
        "https://www.levi.com/US/en_US/clothing/men/outerwear/c/levi_clothing_men_outerwear/facets/colorgroup/medium%20wash/productitemtype/trucker%20jean%20jacket",
        "https://www.levi.com/US/en_US/clothing/men/outerwear/c/levi_clothing_men_outerwear/facets/colorgroup/dark%20indigo/productitemtype/trucker%20jean%20jacket",
        "https://www.levi.com/US/en_US/clothing/men/outerwear/c/levi_clothing_men_outerwear/facets/feature-materialtype/denim/productitemtype/trucker%20jean%20jacket",
    ],
    "Shirts": [
        "https://www.levi.com/US/en_US/clothing/men/shirts/c/levi_clothing_men_shirts",
        "https://www.levi.com/US/en_US/clothing/men/shirts/c/levi_clothing_men_shirts/facets/colorgroup/blue",
        "https://www.levi.com/US/en_US/clothing/men/shirts/c/levi_clothing_men_shirts/facets/colorgroup/black",
        "https://www.levi.com/US/en_US/clothing/men/shirts/c/levi_clothing_men_shirts/facets/colorgroup/white",
        "https://www.levi.com/US/en_US/clothing/men/shirts/c/levi_clothing_men_shirts/facets/colorgroup/red",
        "https://www.levi.com/US/en_US/clothing/men/shirts/c/levi_clothing_men_shirts/facets/colorgroup/green",
        "https://www.levi.com/US/en_US/clothing/men/shirts/c/levi_clothing_men_shirts/facets/colorgroup/multi-color",
    ],
    "Accessories": [
        "https://www.levi.com/US/en_US/accessories/men/c/levi_accessories_men",
        "https://www.levi.com/US/en_US/accessories/men/c/levi_accessories_men?page=1",
        "https://www.levi.com/US/en_US/accessories/men/c/levi_accessories_men?page=2",
        "https://www.levi.com/US/en_US/accessories/men/bags-backpacks/c/levi_accessories_men_bags_backpacks",
        "https://www.levi.com/US/en_US/accessories/men/belts/c/levi_accessories_men_belts_suspenders",
        "https://www.levi.com/US/en_US/accessories/men/hats/c/levi_accessories_men_hats",
        "https://www.levi.com/US/en_US/accessories/men/c/levi_accessories_men/facets/productitemtype/wallets",
        "https://www.levi.com/US/en_US/accessories/men/c/levi_accessories_men/facets/productitemtype/socks",
        "https://www.levi.com/US/en_US/accessories/men/c/levi_accessories_men/facets/productitemtype/bandanas",
        "https://www.levi.com/US/en_US/accessories/men/c/levi_accessories_men/facets/productitemtype/bags",
        "https://www.levi.com/US/en_US/accessories/men/c/levi_accessories_men/facets/productitemtype/underwear",
    ],
}

PDP_LINK_RE = re.compile(r'href="(/US/en_US/[^"]*?/p/[A-Za-z0-9]+)"')
LDJSON_RE = re.compile(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', re.S)


def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s.strip("-")[:60]


def wait_past_challenge(page, timeout_rounds=8):
    """Akamai serves an interactive behavioral-challenge interstitial (not
    a flat block) that self-resolves and reloads within a few seconds --
    poll title rather than trusting the first response."""
    for _ in range(timeout_rounds):
        page.wait_for_timeout(1500)
        title = page.title()
        if title and "Access Denied" not in title:
            return True
    return False


BACKOFF_SECONDS = [20, 45, 90]  # escalating waits, same shape as New
# Balance's is_blocked()/goto_with_retry() Akamai mitigation in this
# pipeline -- Akamai can escalate from a self-resolving challenge to a
# sustained hard block mid-session (confirmed live: every single Jean
# Jackets listing URL failed "challenge never cleared" back-to-back after
# ~85 successful Jeans PDP visits), and the documented fix for that tier
# is patience with real backoff, not immediate give-up.


def goto_and_clear_challenge(page, url: str) -> bool:
    """Navigate + wait past the Akamai challenge, retrying with escalating
    backoff if the challenge doesn't clear (sustained block) rather than
    giving up on the first miss."""
    for attempt, backoff in enumerate([0] + BACKOFF_SECONDS):
        if backoff:
            print(f"  [backoff] waiting {backoff}s before retry {attempt}/{len(BACKOFF_SECONDS)} for {url}")
            page.wait_for_timeout(backoff * 1000)
        page.goto(url, timeout=60000)
        if wait_past_challenge(page):
            return True
    return False


def harvest_pdp_links(page, url: str) -> list[str]:
    if not goto_and_clear_challenge(page, url):
        print(f"  [warn] challenge never cleared for {url} (after backoff retries)")
        return []
    for _ in range(6):
        page.mouse.wheel(0, 3000)
        page.wait_for_timeout(300)
    html = page.content()
    return sorted(set(PDP_LINK_RE.findall(html)))


def extract_composition(html: str) -> list[str]:
    m = re.search(r'Composition\s*&amp;\s*Care</h3>.*?<ul[^>]*>(.*?)</ul>', html, re.S)
    if not m:
        return []
    items = re.findall(r"<li[^>]*>(.*?)</li>", m.group(1), re.S)
    out = []
    for li in items:
        text = re.sub(r"<[^>]+>", "", li).strip()
        if text:
            out.append(text)
    return out


def fetch_pdp_variants(page, pdp_url: str, category: str) -> list[dict]:
    if not goto_and_clear_challenge(page, pdp_url):
        print(f"  [warn] challenge never cleared for {pdp_url} (after backoff retries)")
        return []
    # "Composition & Care" is lazy-rendered below the fold -- doesn't
    # appear in the DOM until scrolled into view.
    for _ in range(6):
        page.mouse.wheel(0, 2000)
        page.wait_for_timeout(250)
    html = page.content()

    variants = []
    for block in LDJSON_RE.findall(html):
        if '"@type":"ProductGroup"' not in block and '"ProductGroup"' not in block:
            continue
        import json
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        groups = data if isinstance(data, list) else [data]
        for pg in groups:
            if pg.get("@type") != "ProductGroup":
                continue
            materials = extract_composition(html)
            for v in pg.get("hasVariant", []):
                if not isinstance(v, dict):
                    # hasVariant sometimes also carries a bare list of
                    # {"url": ...} sibling-colorway links alongside the
                    # real Product dicts -- not a Product itself, skip.
                    continue
                sku = str(v.get("sku", "")).strip()
                images = v.get("image") or []
                offers = v.get("offers") or {}
                if not sku or not images:
                    continue
                variants.append({
                    "product_code": sku,
                    "name": v.get("name") or pg.get("name", ""),
                    "color_name": v.get("color", ""),
                    "price": f"${offers.get('price')}" if offers.get("price") else "",
                    "image_urls": [img.split("?")[0] + "?fmt=jpeg&wid=1500&hei=2000" for img in images],
                    "details": {
                        "description": v.get("description") or pg.get("description", ""),
                        "features": [],
                        "materials": materials,
                    },
                    "product_url": pdp_url,
                    "category": category,
                })
    return variants


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        r = requests.get(url, timeout=25, headers=HEADERS)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"      [warn] {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    database = load_records()
    seen_codes = {p["product_code"] for p in database}
    print(f"Starting with {len(database)} existing records.\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, channel="chrome")
        context = browser.new_context(viewport={"width": 1400, "height": 1200})
        page = context.new_page()

        def recover_page(old_page):
            """After a watchdog-forced timeout the page/renderer can be
            left in a broken navigation state -- close it (best-effort,
            it may already be wedged) and open a fresh one on the same
            context rather than trying to keep reusing a stuck page."""
            try:
                old_page.close()
            except Exception:
                pass
            return context.new_page()

        for category, listing_urls in LISTING_URLS.items():
            print(f"\n=== {category} ===")
            already = sum(1 for p in database if p.get("brand") == "levis" and p.get("category") == category)
            target_new = TARGET_PER_CATEGORY - already
            if target_new <= 0:
                print(f"{category} already has {already} records, skipping.")
                continue

            pdp_links: list[str] = []
            seen_links = set()
            for listing_url in listing_urls:
                try:
                    with watchdog(300):
                        links = harvest_pdp_links(page, listing_url)
                except Exception as e:
                    print(f"  [warn] listing harvest wedged/failed for {listing_url}: {e}")
                    page = recover_page(page)
                    continue
                new_links = [l for l in links if l not in seen_links]
                seen_links.update(new_links)
                pdp_links.extend(new_links)
                print(f"  harvested {len(new_links)} new PDP links from {listing_url} (total {len(pdp_links)})")
                time.sleep(0.3)

            added = 0
            visited_pdp_urls = {p["product_url"] for p in database if p.get("brand") == "levis"}
            for pdp_path in pdp_links:
                if added >= target_new:
                    break
                pdp_url = "https://www.levi.com" + pdp_path
                if pdp_url in visited_pdp_urls:
                    # Every colorway from this PDP is already in the
                    # dataset from a prior run -- skip the network visit
                    # entirely instead of re-fetching (and potentially
                    # re-hitting the Akamai challenge) just to discard it
                    # at the per-variant seen_codes check below.
                    continue
                try:
                    with watchdog(300):
                        variants = fetch_pdp_variants(page, pdp_url, category)
                except Exception as e:
                    print(f"  [warn] PDP fetch wedged/failed for {pdp_url}: {e}")
                    page = recover_page(page)
                    continue

                for variant in variants:
                    if added >= target_new:
                        break
                    code = variant["product_code"]
                    if code in seen_codes:
                        continue

                    print(f"[{added + 1}/{target_new}] {category}: {variant['name']} - {variant['color_name']} ({code})")

                    slug = slugify(variant["name"] or code)
                    item_dir = OUTPUT_DIR / slug / code
                    item_dir.mkdir(parents=True, exist_ok=True)

                    saved = []
                    for i, img_url in enumerate(variant["image_urls"]):
                        dest = item_dir / f"image_{i}.jpg"
                        if download(img_url, dest):
                            saved.append(str(dest))

                    if not saved:
                        print("  [warn] no images downloaded, skipping")
                        continue

                    new_record = {
                        "brand": "levis",
                        "category": category,
                        "name": variant["name"],
                        "color_name": variant["color_name"],
                        "price": variant["price"],
                        "product_code": code,
                        "slug": slug,
                        "product_url": variant["product_url"],
                        "image_count": len(saved),
                        "images": saved,
                        "image_urls": variant["image_urls"],
                        "details": variant["details"],
                    }
                    database.append(new_record)
                    seen_codes.add(code)
                    added += 1

                    database = save_records_safe({code: new_record})
                    print(f"  [checkpoint] {len(database)} total records "
                          f"({added}/{target_new} added for {category})")

                time.sleep(0.3)

            print(f"\n{category}: {added} new records added "
                  f"({already + added}/{TARGET_PER_CATEGORY} total, "
                  f"{len(pdp_links)} PDPs discovered).")

        browser.close()

    total_imgs = sum(
        p["image_count"] for p in database
        if p.get("brand") == "levis" and p.get("category") in LISTING_URLS
    )
    print(f"\nDone. {len(database)} total records in dataset.")
    print(f"Levi's images downloaded: {total_imgs}")


if __name__ == "__main__":
    main()
