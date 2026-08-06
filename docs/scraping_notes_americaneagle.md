# American Eagle (ae.com) — scraping notes

To be merged into `SCRAPING_PROCESS.md` centrally (that file was not edited;
several scraper agents were running concurrently).

Scraper: `americaneagle_scraper.py`. Brand token: `americaneagle`.
Categories: T-Shirts, Hoodies & Sweatshirts, Jeans, Shorts (site's own
left-nav names). Target 50 colorway variants each (200 requested).

## Feasibility probe result — primary target passed, but only just

Probed in the required order. Result per candidate:

| candidate | robots.txt | verdict |
|---|---|---|
| **ae.com** (primary) | 403 to a bare-UA curl, **200 with a full browser header set**; `Allow` by default, only `/browse/`, `*/search/`, `*/s/`, `/cms*`, `*/favorites`, session/tracking paths disallowed | **used** |
| urbanoutfitters.com | 403 — **DataDome** interstitial (`geo.captcha-delivery.com`, "Please enable JS and disable any ad blocker") on robots.txt itself | not needed |
| everlane.com | 200, stock agent-aware Shopify boilerplate, fully crawlable | not needed |

Nothing this scraper touches is disallowed. `/browse/` in robots.txt is a
legacy top-level site path; the JSON endpoint discussed below lives at
`/ugp-api/browse/...`, which does not match that prefix (and is not used
anyway — see below).

Worth recording for whoever picks Urban Outfitters later: it is the first
**DataDome** site the pipeline has met. That is a harder tier than Levi's —
the block fires on `robots.txt`, before any product page, so there is no
"soft" path in with plain `requests`.

## Bot-protection tier: MEDIUM (Akamai Bot Manager, no interactive challenge)

Roughly New Balance's tier, but much softer in practice — no patchright
needed for the actual scrape.

Two separate facts, both non-obvious:

1. **Plain `requests` works for document (HTML) fetches — but only with a
   full browser header set.** A normal `User-Agent` alone gets an Akamai
   "Access Denied" edge page (403, ~400 bytes, `errors.edgesuite.net`
   reference) on *every* URL including `robots.txt`. The headers that flip
   it are the navigation `sec-fetch-*` quartet
   (`dest: document` / `mode: navigate` / `site: none` / `user: ?1`) plus
   `upgrade-insecure-requests: 1` and `sec-ch-ua*`. This is a cheap fix and
   it removes the need for a browser entirely.
2. **Blocks are intermittent, not sticky.** Measured on a 14-request burst
   against one PDP: request 1 → 403, requests 2–14 → 200. So unlike New
   Balance (where a block meant a sustained streak and the practical fix was
   waiting tens of minutes), AE's 403 is best treated as a transient blip
   and retried after well under a second. The first version of this scraper
   used New Balance's 3s/7s/11s ladder and ran at ~1.5 records/min; dropping
   to 0.6s/3.1s/5.6s roughly quadrupled throughput with no change in
   eventual success rate. **Copying New Balance's backoff constants was the
   single biggest wall-clock mistake in this scrape.**

A headless browser (patchright, `channel="chrome"`, `headless=False`) was
used only during the probe, to watch network traffic. It works fine, it is
just unnecessary.

## Data source — an embedded JSON:API document in the server-rendered HTML

**The XHR API exists and is NOT usable.** PLP pagination calls
`GET https://www.ae.com/ugp-api/browse/v1/category/{catId}?offset=N`
(found by capturing XHR while scrolling a PLP in patchright). Called from
plain `requests` with any header combination it returns 403 — it requires
Akamai's `_abck` sensor cookie, which is only minted by executing the site's
JS. Recorded here so nobody re-derives it: right endpoint, wrong door.

What to use instead: **both PLP and PDP embed a complete JSON:API document
in a bare `<script>` tag** whose entire body is JSON
(`{"data": {...}, "included": [...], "meta": {...}}`). Locate it as "first
`<script>` whose stripped text starts with `{` and contains
`"type":"plp"` / `"type":"pdp"`". There is no `__NEXT_DATA__` and no
`window.__X__` assignment — a `__NEXT_DATA__`-shaped grep finds nothing and
would send you down the CSS-selector path unnecessarily (lesson 4 applies,
the blob just isn't where the usual frameworks put it).

An ld+json `Product` block is also present on PDPs, but it is strictly
poorer (name / sku / color / one material string / price / one image). Use
the JSON:API blob.

Contents:

- **PLP blob** — `meta.totalProducts` (real category size) and `included`
  = **exactly 30 product objects, page 1 only**. Deeper pages are the
  browser-gated XHR above; `?page=2`, `?start=30`, `?offset=30&rows=60` are
  all silently ignored by the server-rendered response (still `offset: 0,
  rows: 30`) — a trap, because they all return HTTP 200 with a valid blob
  and look like they worked.
- **PDP blob** — `data.attributes.copySections` = `{details, material,
  size}`, each with `bullets[]` plus `details.longDesc` (the marketing
  one-liner); `data.attributes.breadcrumbs` = the site's own category path;
  `included` product objects with `displayName`, `colorName`, `colorId`,
  `styleId`, `listPrice`, `salePrice`, `pdpImages[]`, `colorSwatches[]`,
  `modelSizeAndHeight`, `categoryL4[]`.

Nice property: **the PDP URL slug is ignored** — only the trailing colorway
id matters, so `https://www.ae.com/us/en/p/x/x/x/0195_2926_001` returns the
full correct PDP. This means a colorway id harvested from a swatch array can
be fetched without first resolving its marketing slug (contrast New Balance,
where a non-derivable `master_id` in the URL path cost real recovery time).

Real category sizes reported by `meta.totalProducts` at scrape time:
T-Shirts 260, Hoodies & Sweatshirts 72, Jeans 132, Shorts 144
(Tops overall 621).

## Colorways — the opposite failure mode from HUF's

AE is the **inverse** of the Shopify pattern HUF's notes warn about. There,
one API product silently hid up to 15 colorways. Here:

- **one PDP URL == exactly one colorway**, and the product id *is* the
  colorway id: `{style}_{color}`, e.g. `0195_2926_001` = style `0195_2926`,
  color `001` (Black). `colorId` and `styleId` are also separate fields on
  the product object, so this is confirmed by the data, not inferred from
  the string shape.
- sibling colorways live in `colorSwatches[]` on the product object
  (`id`, `productUrl`, swatch `imageUrl`, colour `name`), and the array
  **includes the product itself**.

Verified by fetching six PLP seeds and counting: styles carried 1, 4, 7, 10,
13 and 20 siblings. So there is no colorway-collapse risk — but there is the
mirror risk, and it is just as silent: **a PLP page lists only some
colorways of a style**, so seeding purely from the PLP under-samples the
catalog and makes categories look smaller than they are.

Handling: walk `colorSwatches` **breadth-first**. Round 1 = every PLP seed
colorway; round 2 = the 2nd colorway of each seed's style; round 3 = the
3rd; etc., until the category target is met. This matters because the naive
depth-first alternative (exhaust each style's colorways before moving on)
lets a single 21-colorway tee eat half a category — bad for a catalog whose
whole purpose is telling colorway siblings apart *across* many styles.

## Images — Scene7, and the default size is a thumbnail, not the original

`https://s7d2.scene7.com/is/image/aeo/{colorwayId}_{view}` — **no query
string at all returns a ~6 KB preset thumbnail**, not the full-res asset.
This is the opposite of OBEY's problem (4000x5000 PNGs by default) and it
fails silently in the other direction: you would ship a dataset of 6 KB
images and nothing would error.

Knobs, measured:

- `?wid=N` — the only one that matters. `wid=940` → 940x1206, ~60 KB avg.
  `wid=1600` → 1600x2052, ~68 KB.
- `fmt=jpg` — **ignored**. Scene7 content-negotiates instead: `Accept: */*`
  gets `image/webp`, `Accept: image/jpeg,...` gets `image/jpeg`. So the
  format knob is the request header, not the query string.

Used here: `?wid=940` (≈1200px long edge per the pipeline target) with an
explicit jpeg `Accept`. The **unparameterised** URL is what gets stored in
`image_urls`, so full-res stays refetchable.

Scene7 is *not* behind Akamai — images fetch fine with a bare UA.

## Per-image view codes come free (lesson 5)

The CDN filename suffix is a genuine, human-readable view label:

| suffix | meaning |
|---|---|
| `_of` | on-figure front (on-model, studio, white/off-white seamless) |
| `_ob` | on-figure back (on-model, studio) |
| `_os` | on-figure side (on-model, studio) |
| `_f` | flat laydown front, garment only on off-white |
| `_b` | flat laydown **back** (common on jeans/bottoms) |
| `_l1` | **lifestyle** — on-model on a real location background (brick wall, street). The only frame type in this brand that is not shot on seamless. Appears mostly on Jeans. |
| `_d1` `_d2` `_d3` | **alternate** shots — mixed, NOT reliably detail crops |
| `_s` | colour swatch chip — **excluded**, it is a UI chip not a photo |

These survive in `image_urls` in the same order as `images`, so no schema
change was needed to keep the signal. `_of`, `_ob`, `_os`, `_f` and `_s` are
trustworthy. This is the second brand after Skechers where camera angle is
readable from metadata alone, and the first apparel brand where it is.

**Caveat, found by opening the files rather than trusting the metadata:**
`_d*` is *not* a detail crop and the site's own model annotation does not
tell you so. The product object's `modelSizeAndHeight` is a `::`-delimited
per-suffix map, and on the AE Boxy Fit Sublime tee it listed `_of/_ob/_os/
_d2` as model shots and `_f`/`_d1` as empty — yet `_d1` on that exact
product is a **full-length lifestyle shot with the model in frame**, and
`_f` really is the flat-lay. So `modelSizeAndHeight` corroborates `_f`
(no model) but is unreliable for `_d*`. Treat `_d*` as "alternate frame,
contents unknown"; if a downstream stage needs a hard on-model/flat split,
use `_f` (flat) vs `_of|_ob|_os` (on-model) and send `_d*` through the
classifier.

## Photos: MIXED flat-lay and on-model — cropping IS needed

Verified by opening actual downloaded files, not assumed. `_of` is a
three-quarter on-model studio shot with the model's face filling much of
the frame; `_f` is a clean garment-only laydown on off-white. Same bucket as
Vans and HUF: `segment_apparel.py --brand americaneagle` should be run for
real, and will be a harmless partial no-op on the `_f` frames.

`CATEGORY_LABELS` will need entries for `"Hoodies & Sweatshirts"`,
`"T-Shirts"`, `"Jeans"`, `"Shorts"` — note the **ampersand**, since this
scrape uses AE's own nav spelling rather than the "and" form other brands
in this dataset use.

## `product_code`

`americaneagle-{colorwayId}`, e.g. `americaneagle-0195_2926_001`. The
colorway id is already globally unique within AE, and the underscore-heavy
shape collides with nothing in the existing file; the brand prefix is added
per convention anyway. Checked against every existing `product_code` before
the first save.

## Results

| category | records | note |
|---|---|---|
| T-Shirts | **70** | see below — not padding, an artifact of a mid-run restart |
| Hoodies & Sweatshirts | **47** | **shortfall: 3.** The category's real colorway supply ran out — `meta.totalProducts` was 72, and after breadth-first expansion of all 30 PLP seeds plus every one of their `colorSwatches` siblings, only 47 distinct unseen colorways existed. Not padded. |
| Jeans | **50** | full |
| Shorts | **50** | full |
| **total** | **217** | 150 distinct styles |

Why T-Shirts has 70 and not 50: the first run was stopped after 20 T-Shirt
records to retune the retry backoff (see tier notes above). Those 20 were
already committed to `metadata.json`, so on restart they counted as
"existing" and the fresh run's own 50-record counter started from zero.
All 70 are real, distinct, non-duplicate colorways — nothing was padded or
double-counted — but the category is 20 over target.

Images: **1146 files, 100% of expected (0 missing on disk, 0 records with
zero images), avg 74 KB, 87 MB total** for the brand. Per-record image
counts: 6 imgs x142, 5 x46, 4 x5, 3 x1, 2 x18, 1 x5 — the low counts are
real (multipacks and a few colorways with a short gallery), not failed
downloads. Exactly one image fetch failed across the whole run (a Scene7
read timeout on `1192_2849_410_d1`).

Every record has non-empty `description`, `features` and `materials`.

## Gotchas worth promoting to SCRAPING_PROCESS.md

1. **Don't reuse another Akamai site's backoff constants.** New Balance's
   blocks were sustained and needed a long ladder; AE's are single-request
   blips. Measure the block's *persistence* (fire a burst at one URL and log
   the status sequence) before choosing constants — it is a 30-second
   experiment that was worth roughly 4x on wall clock here.
2. **A CDN's parameterless URL is not necessarily the original.** OBEY's
   lesson was "ask for a smaller image"; AE's is the same lesson inverted —
   Scene7's default preset is a 6 KB thumbnail, and only an explicit `wid`
   gets a usable image. Always check the *pixel dimensions* of one
   downloaded file before trusting a whole batch, in both directions.
3. **Pagination params that are silently ignored return HTTP 200 and a
   valid-looking blob.** AE's PLP echoes `offset: 0, rows: 30` no matter
   what you pass. The tell is in the response's own `meta`, not the status
   code — a sibling of lesson 12.
4. **A colorway-per-URL site can under-report colorways just as silently as
   a colorway-per-product site can over-collapse them.** HUF's smell was a
   too-small category count; AE's is the same smell with the opposite cause.
   Either way the check is the same: find where the site enumerates sibling
   colorways (`options[0].values` on Shopify, `colorSwatches` here) and
   compare it against what the listing page gave you.
5. **A scraper whose per-category counter resets on restart will overshoot
   that category.** The counter is per-run, but the dedupe set is
   per-dataset, so records committed before a restart are skipped as
   "existing" without counting toward the target. Harmless (no duplicates),
   but it produced 70 T-Shirts against a target of 50. If exact per-category
   counts matter, seed the counter from the number of already-present
   records for that `(brand, category)` pair rather than from zero.
