# HUF (hufworldwide.com) — scraping notes

To be merged into `SCRAPING_PROCESS.md` centrally (that file was not edited;
five scraper agents were running concurrently).

Scraper: `huf_scraper.py`. Brand token: `huf`.

## Feasibility probe result

Primary target passed on the first probe — no fallback to Brain Dead or
Element was needed.

- `robots.txt` is the stock Shopify one: `User-agent: *` / `Allow: /`, with
  only `/admin`, `/cart/`, `/checkout*`, `/orders` disallowed. Crawling the
  catalog is explicitly permitted. The file even advertises a UCP/MCP agent
  endpoint (`https://hufworldwide.com/api/ucp/mcp`) and an `agents.md` — the
  first site in this pipeline to publish agent-facing endpoints at all.
  (Not used here: the plain storefront JSON API is simpler and complete.
  Worth knowing it exists if a future task needs live pricing/stock.)
- **No bot protection whatsoever.** Plain `requests` + a normal desktop UA
  for both catalog and images. No playwright/patchright needed. Same
  easiest tier as Champion / Stüssy / Dickies / Gap / Skechers.
- Collection handles were scraped from the homepage's own nav
  (`href="/collections/..."`), not guessed. The storefront is **not**
  locale-scoped — bare `/collections/{handle}/products.json` works, unlike
  Dickies which 404s without `/en-us/`.

## Data source — Shopify storefront JSON API (4th such site in the pipeline)

`https://hufworldwide.com/collections/{handle}/products.json?limit=250&page=N`,
paginated until a page returns zero products. Every field needed
(description, all colorways, all images, price, SKUs) is inline in that one
response — **no PDP visit needed at all**, same as Champion/Stüssy/Dickies.

Real collection sizes measured at scrape time (products, before colorway
expansion):

| handle | products | colorways w/ images |
|---|---|---|
| `mens-t-shirts` | 183 | 247 |
| `mens-hoodies-and-fleece` | 61 | 94 |
| `mens-tops` | 53 | 73 |
| `mens-bottoms` | 32 | 102 |
| `mens-jackets` (unused) | 33 | 42 |
| `mens-shorts` (unused) | 11 | — |

Category labels used are the site's own nav names: `T-Shirts`,
`Hoodies and Fleece`, `Tops`, `Bottoms`. `Tops` is HUF's name for
shirts/sweaters/knits (resort shirts, cardigans, thermals); `Bottoms` is
pants/denim/some shorts.

## GOTCHA 1 (the important one) — a HUF product is NOT one colorway

Every prior Shopify brand in this pipeline (Champion, Stüssy, Dickies) had
`options[0]` = Color with **exactly one value per product**, which is why
those scrapers all do `color = variants[0]['option1']` and treat one API
product as one dataset record.

**That assumption is false on HUF.** `options[0]` carries 1–15 color values
on a single product (the Cromer Pant alone carries 15). Copying the Dickies
pattern verbatim would have silently collapsed 558 real colorways down to
329 products' *first* colors and thrown the rest away — with no error, no
empty field, and a plausible-looking result. It only surfaced because the
per-collection product counts (32 bottoms, 33 jackets) were too small to
hit the 50 target, which prompted a look at `options[0]['values']`.

Fix: expand each product into one record per Color option value, keyed on
the first variant carrying that `option1`.

**Generalisable lesson**: "Shopify storefront ⇒ one product = one colorway"
is a property of the individual merchant's catalog setup, not of Shopify.
Always check `len(options[0]['values'])` across the whole collection before
reusing a prior Shopify scraper's record-shape assumption. A too-small
category count is a useful smell for this.

## GOTCHA 2 — per-colorway images must be recovered by SKU filename match

A product's `images` array is a flat list interleaving every colorway. The
obvious mapping — `image['variant_ids']` — **only links the first image of
each colorway**; the 2nd/3rd/... shots carry `variant_ids: []`. Filtering on
`variant_ids` therefore yields exactly one image per colorway and silently
discards the rest of the gallery.

What works: every CDN filename embeds the variant SKU.

```
variant sku TS02678_BKWHT
  -> 89-EMBROIDRED-S-S-TEE_BLACK-WHITE_TS02678_BKWHT_01.png
  -> ..._BKWHT_02.png, ..._BKWHT_03.png
```

Matching on the punctuation-stripped SKU against the punctuation-stripped
filename recovers the full per-colorway gallery (3 shots for a tee, ~8 for a
pant). Fallbacks: the variant's `featured_image`, then the whole product
gallery when the product has a single colorway. Measured on the live
catalog, the fallback fires on only 3 of 558 colorways.

Filenames also carry a trailing position index (`_01`, `_02`, …) and the
colour name in words — a weak but real view/order signal. Raw full-res URLs
are preserved in `image_urls` per lesson 5.

## GOTCHA 3 — 3 MB PNGs, and the CDN only honours `width`

HUF's images are **2400x2400 PNGs, ~3.1 MB each raw** — far heavier than any
prior brand (existing Dickies images average ~270 KB). With disk as the
binding constraint they are fetched through Shopify's CDN resizer at
`&width=800` (~390 KB each), which cut this brand's footprint by ~8x.

Two things that do **not** work on `cdn.shopify.com/s/files/` URLs:
- `&format=jpg` is silently ignored — the response is still
  `Content-Type: image/png`.
- Swapping the `.png` path extension for `.jpg` returns **404**.

`width` is the only knob. Files are still written as `image_N.jpg` per the
pipeline's directory convention; the bytes are PNG, which PIL sniffs by
content rather than extension, so downstream stages are unaffected.

## `product_code`

`huf-{variant SKU}` (e.g. `huf-TS02678_BKWHT`). The numeric Shopify product
`id` is unusable here because it is per-product, not per-colorway, and would
collide across the expanded records. Verified zero raw collisions against
the 764 bare-numeric Shopify ids already in `metadata.json` from
Champion/Stüssy/Dickies; prefixed anyway per convention.

## `body_html` shape

A hybrid of both prior shapes: an optional free-text intro paragraph
followed by a `•`-bulleted `<br>` list (Stüssy was bullets-only, Champion and
Dickies were prose-only). Parsed by splitting on the bullet character:

- text before the first bullet → `details.description`
- bullets → `details.features`
- bullets containing a fibre percentage / fabric weight → also
  `details.materials` (in practice always the first bullet, e.g.
  `"100% cotton (6oz) short sleeve tee shirt"`)
- when there is no intro paragraph, the joined bullets are reused as the
  description so captioning still has prose to work from

Only 1 of 329 products across the four collections had an empty `body_html`.

## Photos: MIXED flat-lay and on-model — cropping IS needed

Verified by direct image inspection across categories, not assumed.

Within a single colorway's gallery, the **early positions are clean flat-lay
product shots on white** (garment only, no model) and the **later positions
are full-body or three-quarter on-model studio shots**, also on white. So
HUF is not in the Carhartt/Stüssy "skip cropping entirely" bucket, and not
in the Champion/Dickies "uniformly on-model" bucket either — it is genuinely
mixed, the same situation as Vans.

Implication for the later pass: `segment_apparel.py --brand huf` should be
run for real. Note it will be a partial no-op on the flat-lay frames (the
garment already fills the frame), which is harmless. `CATEGORY_LABELS` will
need entries for `"Hoodies and Fleece"`, `"Tops"` and `"Bottoms"` — the
closest existing phrasings are `"Hoodies and Sweatshirts"`,
`"T-Shirts and Tops"` and `"Pants"` respectively.
