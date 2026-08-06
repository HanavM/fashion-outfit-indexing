# OBEY Clothing — scraping notes (to be merged into SCRAPING_PROCESS.md)

Scraper: `obey_scraper.py`. Brand token: `obey`. Categories: T-Shirts,
Sweatshirts, Pants, Shorts. Target 50 colorway variants each (200 requested).

## Bot-protection tier: easiest (plain `requests`, no protection)

Same tier as Champion / Stüssy / Dickies / Gap / Skechers. Fourth
Shopify-storefront site in this pipeline. No headless browser needed at any
point — plain `requests` + a normal desktop UA fetched the entire catalog and
every image.

## Domain gotcha

`shop.obeyclothing.com` **does not resolve** (curl exits 000 — no DNS/TLS
handshake). The live storefront is the apex `https://obeyclothing.com`;
`www.obeyclothing.com` 301s to it. Don't trust the `shop.` subdomain.

`robots.txt` is Shopify's newer agent-aware boilerplate: `User-agent: * /
Allow: /`, plus prose confirming "public product, collection, page, blog,
policy, cart, and localized HTML is crawlable". The only prohibition is on
*automated checkout/payment*, which this scraper never touches. It also
advertises a UCP/MCP endpoint (`/api/ucp/mcp`) and `/agents.md` — not needed
here, `products.json` is simpler, but worth knowing that Shopify stores are
starting to publish agent endpoints.

## Data source

Standard Shopify storefront JSON:
`https://obeyclothing.com/collections/{handle}/products.json?limit=250&page=N`,
paginated until a page returns zero products. **No locale prefix** (unlike
Dickies's `/en-us/`). Full product detail — title, colorway, price, all image
URLs, `body_html` — is inline in the same response, so no PDP visit is needed.

Collection handles were taken from the homepage's own nav (`href=
"/collections/..."`) and cross-checked against the site's own
`https://obeyclothing.com/collections.json?limit=250` index (101 collections).
Nothing was guessed.

## Structural quirk: one category is split across many collections

This is the main thing that differs from Dickies/Champion/Stüssy. OBEY does
**not** have one big collection per garment type. `mens-t-shirts` is only 79
products; the rest of the tees live in `classic-t-shirts`,
`heavyweight-t-shirt`, `pigment-t-shirts`, `pigmnet-ls-t-shirts` (sic — the
site's own typo'd handle, `pigmnet`) and `sale-t-shirts`. Sweatshirts are
split across `men-sweatshirts` / `crewneck-fleece` / `pullover-hood` /
`zip-hood` / `sale-sweatshirts`. So each dataset category here is a
**de-duplicated union of several handles** (main handle consumed first), not a
single handle.

De-duplicated union sizes measured before scraping:

| dataset category | handles | unique products |
|---|---|---|
| T-Shirts | mens-t-shirts, classic-t-shirts, heavyweight-t-shirt, pigment-t-shirts, pigmnet-ls-t-shirts, sale-t-shirts | 181 |
| Sweatshirts | men-sweatshirts, crewneck-fleece, pullover-hood, zip-hood, sale-sweatshirts | 64 |
| Pants | men-bottoms-pants, pants, denim-pants | 41 |
| Shorts | shorts, regular-fit-shorts, relaxed-fit-shorts, baggy-fit-shorts | 41 |

(Men's-only. Jackets 21 and Shirts 18 were rejected as too small to be worth a
category slot.)

## Record shape

Each Shopify "product" is already ONE colorway: `options[0]` is `COLOR` with
exactly one value (ALL-CAPS, e.g. `RAINFOREST`), sizes are `options[1]`. Same
"one API product = one dataset record" shape as Champion/Stüssy/Dickies — no
colorway-expansion step.

`product_code` is **prefixed `obey-`** (`obey-8919254925490`). Raw ids are bare
numerics and three other brands already contribute bare-numeric Shopify ids to
`metadata.json`, so the prefix removes all collision risk. Verified against the
full existing code set before the first save. `handle` is the slug.

## `body_html` is a bullet list, not prose

Stüssy-shaped, not Dickies-shaped: a `<ul>` of short ALL-CAPS bullets, with the
last bullet a bare `SKU:165264442`. There is **no prose description anywhere on
this site** (checked the PDP HTML too, not assumed). Handling:

- `details.features` — every bullet, SKU bullet dropped
- `details.materials` — the subset matching fabric/weight patterns
  (`%`, COTTON, POLYESTER, FLEECE, TWILL, OZ, GSM, …)
- `details.description` — bullets joined with `. ` (best available stand-in)

Downstream captioning should expect SHOUTY, telegraphic source copy here.

## Result

| category | records | note |
|---|---|---|
| T-Shirts | 50 | target met |
| Sweatshirts | 50 | target met |
| Pants | 41 | shortfall 9 — whole men's catalog only has 41 |
| Shorts | 41 | shortfall 9 — whole men's catalog only has 41 |
| **total** | **182 / 200** | 524 images, 2.9 GB on disk |

Images per record range 1-6 (cap 6 rarely binds; median ~3). No duplicate
`product_code` anywhere in `metadata.json` (3116 records total after this run).
Every record has non-empty `details.description` / `features` / `materials`.

## Photography: flat-lay / product-only, NOT on-model

Verified by opening actual downloaded files, not assumed. Front and back
laid-flat shots of the garment on a transparent/white background, no model, no
lifestyle framing — same situation as Carhartt and Stüssy, opposite of
Dickies/Vans. **Garment cropping (`segment_apparel.py`) is therefore not
needed** for this brand: there is no model body/face competing with the garment
in frame.

Bottoms (denim especially) carry **macro fabric/hardware close-ups** in the
gallery — e.g. a full-frame shot of one back pocket and its OBEY woven tag.
Those are useless as retrieval targets for a whole-garment query, so
`classify_views.py` should be run for this brand and its material-closeup
class relied on when building galleries.

## Gotcha worth flagging: OBEY images are ~20x larger than every other brand

Images are **3000x3750 PNGs** served from the Shopify CDN as
`{sku}_{colorcode}_N.png` (saved as `image_N.jpg` by this pipeline's filename
convention — the bytes are left untouched PNG, which PIL/torch read fine by
content sniffing, but the extension is a lie).

Real cost measured: **~5.3 MB per image**, versus ~270 KB/image for Dickies.
One OBEY record therefore costs as much disk as ~20 Dickies records' worth of
a single image. On a machine where disk is the binding constraint this matters
a lot.

**Fix for any future re-run:** Shopify CDN URLs accept a resize parameter —
append `&width=1200` (or `&width=1600`) to the `src`, e.g.
`...165264442_RFR_1.png?v=1783207110&width=1200`. That returns a properly
downscaled image at a fraction of the bytes and is still larger than most other
brands' native product photos in this dataset. This scraper did NOT do that (it
saved originals); doing it from the start is recommended for the next
large-PNG Shopify storefront.

Most products carry only 2-4 images, comfortably under the 6-image cap, so the
cap rarely binds — the per-image size is the whole problem.

## Shortfalls (real inventory, not padded)

Pants and Shorts genuinely have only 41 men's colorways live each, across all
of their respective collections. The full men's catalog (`mens-all`, paginated)
is 608 products and its `BOTTOMS` tag covers only 91 of them, which matches
41 + 41 with overlap — so this is real inventory size, not a pagination bug.
