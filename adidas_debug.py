"""Check what img elements exist inside product cards."""
import time
from playwright.sync_api import sync_playwright

URL = "https://www.adidas.com/us/originals-shoes"

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=False, channel="chrome",
                                  args=["--disable-blink-features=AutomationControlled"])
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    page = context.new_page()

    try:
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    except Exception:
        pass

    # Trigger product grid
    page.evaluate("window.scrollBy(0, 600)")
    time.sleep(2)
    page.evaluate("window.scrollBy(0, 600)")
    time.sleep(2)

    cards = page.query_selector_all("article[data-testid='plp-product-card']")
    print(f"Cards found: {len(cards)}")

    if cards:
        card = cards[0]
        print("\n--- First card img count ---")
        imgs = card.query_selector_all("img")
        print(f"  img elements: {len(imgs)}")
        for img in imgs:
            src    = img.get_attribute("src") or "(none)"
            srcset = img.get_attribute("srcset") or "(none)"
            dsrc   = img.get_attribute("data-src") or "(none)"
            print(f"  src:     {src[:100]}")
            print(f"  srcset:  {srcset[:100]}")
            print(f"  data-src:{dsrc[:100]}")
            print()

        print("\n--- All img tags in card outerHTML snippet ---")
        html = card.inner_html()
        # find all img tags
        import re
        for m in re.finditer(r'<img[^>]+>', html):
            print(" ", m.group()[:200])

    browser.close()
    print("Done.")
