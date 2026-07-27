"""Debug: dump all adidas CDN image URLs from a single product page."""
import re, time
from playwright.sync_api import sync_playwright

URL = "https://www.adidas.com/us/gazelle-indoor-shoes/IH9653.html"

with sync_playwright() as pw:
    browser = pw.chromium.launch(
        headless=False, channel="chrome",
        args=["--disable-blink-features=AutomationControllers"],
    )
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    page = ctx.new_page()

    try:
        page.goto(URL, wait_until="domcontentloaded", timeout=45000)
    except Exception:
        pass
    time.sleep(3)

    # Scroll a bit to trigger lazy loads
    page.evaluate("window.scrollBy(0, 800)")
    time.sleep(2)

    print("=== <img> src attributes ===")
    for img in page.query_selector_all("img"):
        src = img.get_attribute("src") or ""
        if "assets.adidas.com" in src:
            print(f"  SRC: {src}")

    print("\n=== Full HTML scan for adidas CDN URLs ===")
    html = page.content()
    found = set()
    for m in re.finditer(r'https://assets\.adidas\.com/images/[^"\'> \n]+\.jpg', html, re.IGNORECASE):
        u = m.group(0)
        fname = re.search(r'/([^/?#]+\.jpg)', u)
        key = fname.group(1) if fname else u
        if key not in found:
            found.add(key)
            print(f"  {u[:120]}")

    print(f"\nTotal unique filenames from HTML: {len(found)}")

    browser.close()
