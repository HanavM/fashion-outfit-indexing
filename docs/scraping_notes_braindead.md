# Brain Dead (wearebraindead.com) — scraping notes

To be merged into `SCRAPING_PROCESS.md` centrally (that file was not edited —
several scraper agents were running concurrently).

Scraper: `braindead_scraper.py`. Brand token: `braindead`.
Categories: T-shirt, Shirt, Pant, Jacket.

## Feasibility probe result

Primary target passed on the first probe — no fallback to Noah or Golf Wang
was needed.

- `robots.txt` is Shopify's newer agent-aware boilerplate: `User-agent: *` /
  `Allow: /`, with prose stating "public product, collection, page, blog,
  policy, cart, and localized HTML is crawlable". The only prohibition is
  automated checkout/payment, which this scraper never touches. It also
  advertises `/agents.md` and a UCP/MCP endpoint (`/api/ucp/mcp`) — the
  third site in this pipeline to do so, after HUF and OBEY. Not used;
  `products.json` is simpler and complete.
- **Bot-protection tier: easiest — none at all.** Plain `requests` + a normal
  desktop UA for both catalog and images. No playwright/patchright needed.
  Same tier as Champion / Stüssy / Dickies / HUF / OBEY / Gap / Skechers.
- Storefront is **not** locale-scoped: bare
  `https://wearebraindead.com/collections/{handle}/products.json?limit=250&page=N`
  works (Dickies still stands alone in needing `/en-us/`).
- Data source: Shopify storefront JSON API — the 6th such site here. Title,
  price, SKUs, every image URL and `body_html` are all inline. **No PDP
  visit needed at any point.**

## GOTCHA A (the important one) — `options[0]` is **Size**, not Color

Every prior Shopify brand in this pipeline had `options[0].name == "Color"`,
which is why Champion / Stüssy / Dickies all do
`color = variants[0]["option1"]`, and why HUF's fix was to *expand*
`options[0]["values"]` into multiple colorways.

**Brain Dead is a third shape.** Its products carry exactly one option and it
is `Size`. Verified across the entire 510-product catalog: **zero** products
have a non-`Size` first option. So:

- Copying the Dickies line would have written **`"XS"` into `color_name` on
  every single record** — no error, no empty field, plausible-looking output.
- Copying the HUF fix would have expanded each product into one record per
  *size*, ~6x-inflating the catalog with duplicate colorways.

Brain Dead genuinely **is** one product = one colorway, but the colorway name
lives in the **product title after the final `" - "`**
(`"Poplin Camp Collar Shirt - Chocolate"` → `Chocolate`). It is mirrored in a
`color:{name}` tag on ~84% of products and in a `<p>Color: …</p>` line in
`body_html` on only ~11%. The title tail is the only 100%-coverage source AND
the richest — where the tag normalises to `blue`, the title says
`Blue Multi`. Verified: all 202 candidate products have a `" - "` in the
title, and no two candidates share a title.

**Generalised lesson (extends HUF's):** the colorway axis on a Shopify
storefront can live in `options[0]` (Dickies), in `options[0]` with many
values (HUF), or **not be a Shopify option at all** (Brain Dead). Before
reusing any prior Shopify scraper's record shape, print
`Counter(tuple(o["name"] for o in p["options"]))` over the whole collection.
It is one line and it distinguishes all three cases immediately.

## GOTCHA B — `collections.json` `products_count` is fiction on this store

The site's own collection index reports counts that are wildly wrong:

| handle | `products_count` says | `products.json` actually returns |
|---|---|---|
| `shirt` | 273 | 45 |
| `hoodie` | 200 | 27 |
| `longsleeve` | 339 | 4 |
| `tops` | 1801 | 204 |
| `perk-collection` | 3118 | 0 |
| `vest` | 32 | 0 |

The whole store is **510 published products**. Presumably these are
smart/automated collections whose counts include unpublished or
draft-channel items. **Use `collections.json` to enumerate handles, never to
size a category.** Trusting it here would have produced a plan for a
1000-record scrape that the catalog cannot support.

## Category construction — union of all 97 collections, grouped by `product_type`

Because the per-garment collections are small and inconsistently curated
(and their counts are fiction), categories were not taken from a single
handle. Instead: crawl **all 97 collections**, dedupe by product `id`, drop
anything appearing in `women` / `womenswear` (34 products), then group on the
merchant's own `product_type` field — which is the site's own taxonomy and
matches its collection titles.

Full men's-catalog inventory by `product_type` (clothing only):

| product_type | products (= colorways) |
|---|---|
| t-shirt | 70 |
| shirt | 37 |
| pant | 36 |
| jacket | 33 |
| hoodie | 26 |
| sweater | 17 |
| shorts | 16 |
| longsleeve / pullover / sweatshirt / sweatpant | 4 / 2 / 2 / 2 |

The four largest were taken: T-shirt, Shirt, Pant, Jacket — a tee, a woven
shirt, a bottoms category and an outerwear category.

## `product_code`

`braindead-{style SKU}`, where the style SKU is the variant SKU with its
`-SIZE` suffix stripped (`BDW24T24003973BR15-XS` → `BDW24T24003973BR15`).
Every candidate product has a non-empty SKU and all stripped SKUs are unique.
The numeric Shopify product `id` was avoided because three other brands
already contribute bare-numeric ids to `metadata.json`. Verified 0 duplicate
`product_code`s across the whole file after the run.

## `body_html` shape

Prose `<p>` paragraphs (Dickies-shaped), occasionally with a `•`-bulleted
`<br>` block appended (HUF-shaped), plus optional `Material:` / `Color:`
label paragraphs. No `<li>` anywhere. 0 of 202 products have an empty body.
Parsed as: long paragraphs → `description`; short paragraphs and `•` bullets
→ `features`; `Material:` value and any fibre/weight-matching bullet →
`materials`; `Color:` line dropped (already captured as `color_name`).

Result: **0 records with an empty description**, but 117/176 have empty
`features` and 140/176 empty `materials` — most Brain Dead copy is two prose
paragraphs with no spec list at all. Downstream captioning should not expect
a materials field here.

## Images — already sane-sized, unlike OBEY/HUF

Brain Dead serves **1200x1500 JPEGs, ~160-250 KB each**, with filenames
ending `_optimized`. `&width=1200` was appended per the OBEY lesson; on this
store it is a **no-op** (byte-identical response) because the assets are
already 1200px. Appending it anyway is still the right default — it costs
nothing and it is the only thing standing between a future large-PNG Shopify
store and another 2.9 GB run.

Original unparameterised URLs are stored in `image_urls`, resized bytes on
disk, per the brief.

**CDN filenames are self-describing view labels** — `_Front_optimized.jpg`,
`_Back_`, `_Side_`, `_Detail_`, `_Detail_Back_`, `_Detail_1_`. Per lesson 5
that signal dies when files are renamed to `image_N.jpg`; it survives here
only because `image_urls` preserves the original URLs in the same order as
`images`. **Caveat: the labels are not universal** — lifestyle shots use raw
camera filenames instead (`1V9A9396-optimized.jpg`, `Type00-15.5_19-…`), so
a view classifier cannot rely on filenames alone.

## Photography: MIXED flat-lay and on-model — cropping IS needed

Verified by opening actual downloaded files across all four categories, not
assumed.

- **Early gallery positions (0-2): clean flat-lay product shots on pure
  white** — garment only, no model, front and back. Same as Carhartt / OBEY /
  Stüssy.
- **Later positions (3-5): genuine on-model lifestyle photography** — full
  body, outdoors, real environments (concrete walls, brick patios), model's
  face and shoes in frame, ambient colour cast. Also **macro fabric/hardware
  closeups** that fill the frame with pattern and a single pocket.

So Brain Dead sits in the mixed bucket with HUF and Vans, not the
flat-lay-only bucket with OBEY/Carhartt/Stüssy.

Implications for the later coordinated pass:
- `segment_apparel.py --brand braindead` **should be run for real** — the
  on-model frames have a competing body/face. It will be a partial no-op on
  the flat-lay frames, which is harmless.
- `classify_views.py` should also be run and its material-closeup class
  relied on — the macro fabric shots are useless as whole-garment retrieval
  targets.
- `CATEGORY_LABELS` will need entries for `"T-shirt"`, `"Shirt"`, `"Pant"`,
  `"Jacket"`. Closest existing phrasings: `"T-Shirts and Tops"`,
  `"T-Shirts and Tops"` / `"Shirts"`, `"Pants"`, `"Jackets and Vests"`.

## Result

| category | records | catalog size | note |
|---|---|---|---|
| T-shirt | 70 | 70 | over target — see below |
| Shirt | 37 | 37 | shortfall 13 — whole men's catalog is 37 |
| Pant | 36 | 36 | shortfall 14 — whole men's catalog is 36 |
| Jacket | 33 | 33 | shortfall 17 — whole men's catalog is 33 |
| **total** | **176 / 200** | | 864 images, 220 MB |

Average image size 248 KB; 211 MB on disk for the brand's image tree.
Images per record 2-6, with the 6-cap binding on 99/176 records.

Three of four categories are **genuinely capped by real inventory** — the
entire published Brain Dead store is 510 products of which only ~200 are
men's clothing. Nothing was padded.

**T-shirt overrun (honest note):** the run was interrupted at the 3 GiB disk
floor after 30 T-shirt records, then resumed. The per-category cap counts
only records *added this run*, and the resume skipped the 30 already-saved
ones as "code exists" — so it added a further 40 before hitting the cap,
leaving 70 T-shirt records instead of 50 (i.e. the entire T-shirt catalog).
These are all real, unique, non-duplicated colorways, so they were kept
rather than deleted. Anyone re-using this scraper for a resumable run should
seed the per-category counter from existing records of that brand+category,
not from zero.

## Disk

Started at 4.22 GiB free. The first run **stopped itself at 2.97 GiB** at a
checkpoint boundary (the hard floor in the brief) with 30 records safely
saved — the pressure was from concurrent scrapers, not this one, whose own
footprint at that point was 21 MB. Disk recovered to ~5 GiB a few minutes
later and the run was resumed; the scraper is idempotent (skips existing
`product_code`s and existing files on disk). Ended at 3.59 GiB free.
