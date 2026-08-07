# J.Crew (jcrew.com) — scraping notes (2026-08-06)

To be merged into `SCRAPING_PROCESS.md` centrally (that file was not edited;
several scraper agents were running concurrently).

Scraper: `jcrew_scraper.py`. Brand token: `jcrew`.
Categories: T-Shirts and Polos, Shirts, Sweaters, Jeans (Jeans is the bottoms
category). Target 50 colorways each — **all four reached 50, no shortfall.**

## Feasibility probe result — passed, but it is the second-hardest tier yet

| candidate | robots.txt with a full browser header set | verdict |
|---|---|---|
| **jcrew.com** (priority 1) | **403** Akamai "Access Denied" (`errors.edgesuite.net`) — intermittently 200 | **used** (patchright clears it) |
| everlane.com (priority 2) | 200, agent-aware Shopify boilerplate | also used, see its own notes |

`robots.txt` is readable once cookies exist and has **no `Disallow: /` for
`*`**. Two of its rules are directly relevant and are respected:

- **`Disallow: /api/` and `Disallow: */data/v1/`** — which is exactly where
  the PLP grid's client-side product API lives. That endpoint was therefore
  deliberately **not** used, even though it is the obvious JSON source. The
  seeding path below exists because of this rule, not because the API was
  hard to find.
- `Allow: /s7-img-facade/*` covers the images. Sitemaps are advertised in
  robots.txt itself.

## Bot-protection tier: HARD — Akamai, sustained, not intermittent

Measured rather than assumed, per the AE lesson about not inheriting another
site's backoff constants:

- Plain `requests` with the **full** AE-style header set (`sec-fetch-*`
  navigation quartet, `upgrade-insecure-requests`, `sec-ch-ua*`, real UA)
  gets a 403 ~400-byte edge page on `robots.txt`, the homepage, every PLP,
  every PDP and `sitemap-index.xml`.
- **Persistence test: a 10-request burst at one PLP returned 403 on all
  10.** So this is the opposite of AE, whose 403s were single-request blips
  that a 0.6 s retry cleared. No retry ladder fixes J.Crew. (Two 200s were
  observed on `robots.txt` early in the probe, which is exactly the kind of
  flicker that would mislead a short probe — the burst test is what settled
  it.)
- **What works: patchright (`channel="chrome"`, headed) clears the edge, and
  the cookies it mints work in plain `requests`.** So the browser is used
  exactly once, to open the homepage and hand its cookie jar + UA to a
  `requests.Session`; every sitemap, PDP and image after that is plain HTTP.
  On a 403 the correct response is to **re-mint the jar, not sleep longer** —
  that is the scraper's retry policy.
- Measured across the whole 200-record run: **zero blocks** after the initial
  mint. One cookie jar covered ~230 sitemap+PDP fetches and 865 image
  downloads.

Generalised: this is a third distinct Akamai behaviour in this pipeline —
New Balance (sustained, wait it out), AE (blips, retry fast), J.Crew
(sustained, but a browser-minted cookie jar transplants cleanly into
`requests`). Try the transplant before concluding a site needs a browser
for every page.

## Seeding: the sitemap, NOT the PLP

The PLP is a trap in three separate ways, all of which return HTTP 200:

1. Its `__NEXT_DATA__` **is present and parses**, but
   `props.initialState.products.productsByProductCode` is `{}` — the grid is
   client-rendered from the robots-disallowed `/api/` endpoint. A
   `__NEXT_DATA__`-shaped grep "succeeds" and yields nothing (a sibling of
   lesson 12).
2. Rendering the PLP in a browser and scrolling tops out at **~29 styles**
   for Jeans no matter how long you scroll (25 scroll steps, 22 s of waiting,
   count flat from step 10).
3. Harvesting `a[href*='/p/']` from that render gives one `colorCode` per
   style — the AE under-sampling failure.

`custom-sitemap-{0..3}-Jcrew-US-product.xml` (advertised in robots.txt, 4
files, **3819 URLs**) carries the whole catalog with the category baked into
the URL path. Men's style counts at scrape time:

| category slug | men's styles |
|---|---|
| `sweaters` | 79 |
| `shirts` | 64 |
| `tshirts-and-polos` | 61 |
| `jeans` | 53 |
| `coats-and-jackets` | 45 |
| `shorts` / `pants-and-chinos` | 39 / 33 |

**Women's URLs reuse the same category slugs** (`/p/womens/categories/
clothing/jeans/...`), so the `/p/mens/` prefix filter is load-bearing; the
product object's own `gender` field is checked as a second gate.

## Data source — the PDP's `__NEXT_DATA__` (which *is* server-rendered)

`props.initialState.products.productsByProductCode[{styleCode}]` →
`productName`, `colorsList`, `priceModel`, `productDescriptionRomance`
(prose), `productDescriptionTech` (bullet list), `productDescriptionFit`,
`gender`, and the marketing slug in `url`. One PDP fetch per style; nothing
else is needed. An ld+json block also exists and is strictly poorer.

Nice property (same as AE): the PDP URL's slug is cosmetic, only the
trailing style code matters.

## Colorways — one PDP == one STYLE with N colorways

Verified from the data, not inferred from the URL shape:

- `colorsList[*].colors[]` is the ordered colorway list (`code`, `name`).
- `priceModel[style].colors[]` repeats the same colour codes with a
  per-colorway price and a `skuShotType` string.
- The `colorCode` query param on a PDP link merely **preselects** one of
  them.

So this is the HUF/Uniqlo shape. Seeding one record per sitemap URL would
have kept 1 of the 3-15 colorways of every style. Measured supply after
expansion: T-Shirts and Polos 53 styles → 159 colorways, Shirts 49 → 159,
Sweaters 72 → 180, **Jeans 53 → 70** (denim is nearly one-colorway-per-style
here, unlike the tees).

Colorways are taken **breadth-first across styles** per the AE lesson (round
1 = colorway 0 of every style, round 2 = colorway 1, …). Consequence worth
knowing: because supply was ample, the 200 records span **199 distinct
styles** — i.e. this brand contributes maximum style diversity and almost no
colorway *siblings*. If a future pass specifically wants sibling pairs for
colorway-discrimination training, run the expansion depth-first on a subset
instead.

## GOTCHA (the important one) — a missing Scene7 asset returns HTTP 200 with a placeholder

Images live at `https://www.jcrew.com/s7-img-facade/{style}_{color}{suffix}`.
**Every unknown suffix returns 200 with the same 42,145-byte 1200x1200 JPEG
reading "A GREAT IMAGE IS ON ITS WAY. PLEASE POP BACK LATER."** — verified on
`_f`, `_ob`, `_of`, `_zzz`, `_qqq99` and even a wholly nonexistent style
code. There is no 404 anywhere.

This is a silent, plausible-looking failure: borrowing AE's `_f`/`_of`/`_ob`
view vocabulary (the natural thing to do, since both sites are Scene7) fills
a dataset with placeholder art and nothing errors. Two defences, both used:

1. **Only the suffixes in that colorway's own `skuShotType`** are requested,
   plus the bare `{style}_{color}` URL. The bare URL is not listed in
   `skuShotType` and is the **clean garment-only flat-lay** — it is the
   single most valuable frame per record and would be missed by trusting
   `skuShotType` alone.
2. The placeholder's md5 is fetched at startup **from a deliberately bogus
   URL** and every download is compared against it. Result: 0 placeholder
   files on disk out of 865.

### Sub-gotcha — Scene7 content-negotiates, and that broke the md5 check first time

With a browser-style `Accept: image/avif,image/webp,*/*`, Scene7 returns
**AVIF for some assets and JPEG for others**. The placeholder came back AVIF
(9.7 kB, `ftyp…` magic) whose md5 differs from the JPEG placeholder's, so the
md5 check sailed straight past it and a placeholder landed on disk looking
like a valid `.jpg`. (It was briefly mis-diagnosed as MP4 video — `ftyp` is
the ISO-BMFF box both formats share.)

Fix, and the general rule: **there is no `fmt=` query knob on Scene7 — the
format knob is the `Accept` request header** (AE's notes say the same). The
image fetch now sends `Accept: image/jpeg,image/*;q=0.8`, which makes the
response deterministic, and a JPEG magic-byte check (`ff d8 ff`) backstops
the md5 check. Confirmed after the fix: 0 non-JPEG files in 865.

`?wid=1200` gives ~1200x1200 at 40-110 kB. The parameterless URL is a
*smaller* 34 kB render, so — as with AE's Scene7 — the bare URL is not the
original. Unparameterised URLs are stored in `image_urls`, in the same order
as `images`.

### Shot-type vocabulary actually observed (a weak lesson-5 signal)

| suffix | count in this run | what it is |
|---|---|---|
| (bare) | 200 | clean garment-only flat-lay on off-white — **100% coverage** |
| `_d1` / `_d2` / `_d3` | 163 / 154 / 74 | alternate frames, mixed on-model and detail |
| `_m` | 156 | on-model studio shot, model's face in frame |
| `_d12`, `_d4`, `_d5`, `_d7`, `_d8` | 42 / 17 / 8 / 8 / 9 | further alternates |
| `_ed`, `_edw`, `_edm` | 17 / 2 / 1 | **editorial/lifestyle** — model on a real street location |
| `_v` | 14 | alternate |

Only `(bare)` and `_m` are trustworthy labels; `_d*` is "alternate frame,
contents unknown", the same caveat AE's notes record for its own `_d*`.

## Other smaller gotchas

- **`productName` is stored HTML-escaped** (`Piqu&eacute; johnny-collar polo
  shirt`, `Wallace &amp; Barnes ...`). It is passed through `strip_html()` /
  `html.unescape()`; the colour names need the same treatment.
- Colour names arrive SHOUTED (`HTHR FLANNEL`) and are `.title()`-cased.
  Some are truncated by the site itself (`Beige Multi Hisoka Stri`,
  `Bright Medium Indigo Wa`) — that truncation is in the source data.
- `productDescriptionTech` doubles as the materials source: bullets matching
  a fibre-percentage / fabric-word regex are copied into `details.materials`.
  All 200 records ended with non-empty description, features **and**
  materials.

## `product_code`

`jcrew-{styleCode}-{colorCode}`, e.g. `jcrew-CV298-WT0002`. The pair is
already globally unique within J.Crew and collides with nothing in
`metadata.json`; the brand prefix is added per convention. Verified 0
duplicate `product_code`s across all 4079 records in the file after the run.

## Photography: MIXED flat-lay and on-model — cropping IS needed

Verified by opening actual downloaded files across categories, not assumed.

- The bare-suffix frame (present on **every** record) is a clean, well-lit,
  garment-only flat-lay on off-white — the Carhartt/Stüssy situation.
- `_m` and most `_d*` frames are on-model studio shots, waist-up or
  full-length, model's face in frame.
- `_ed*` frames are genuine outdoor lifestyle photography (street, traffic,
  ambient colour cast) — the only non-seamless frame type.

Same bucket as Vans / HUF / Brain Dead / AE / Everlane:
`segment_apparel.py --brand jcrew` should be run for real, and will be a
harmless partial no-op on the flat-lay frames.

`CATEGORY_LABELS` will need entries for `"T-Shirts and Polos"`, `"Shirts"`,
`"Sweaters"` and `"Jeans"` — `Shirts`, `Sweaters` and `Jeans` already exist
verbatim from Dickies / PacSun / Levi's.

## Results

| category | records | note |
|---|---|---|
| T-Shirts and Polos | 50 | full (159 colorways available) |
| Shirts | 50 | full (159 available) |
| Sweaters | 50 | full (180 available) |
| Jeans | 50 | full (70 available — the tightest) |
| **total** | **200 / 200** | no shortfall, nothing padded |

Images: **865 files, 0 missing on disk, 0 records with zero images, 0
placeholders, 0 non-JPEG, avg 233 KB, 197 MB total.** Per-record counts:
4 x60, 5 x54, 6 x48, 2 x19, 1 x14, 3 x5. The low counts are real — those
colorways carry a short `skuShotType` list, and a few had alternates that
resolved to placeholders and were correctly dropped. 199 distinct styles,
154 distinct colour names.

Disk: 10.19 GiB free at start, 5.5 GiB at end — most of that drop is the
other scrapers running concurrently; this brand's own footprint is 197 MB.
The 3 GiB floor was never reached.

## Gotchas worth promoting to SCRAPING_PROCESS.md

1. **A browser-minted cookie jar can often be transplanted into `requests`.**
   On a hard-Akamai site this converts "browser for every page" into "browser
   once". Try it before writing a fully browser-driven scraper — and on a
   403, re-mint rather than backing off, because a sustained block is not a
   timing problem.
2. **Measure whether a block is a blip or sustained before choosing retry
   constants.** AE's 403s cleared on request 2 of 10; J.Crew's did not clear
   in 10. Same vendor, opposite correct response. The burst test is 30
   seconds.
3. **A CDN that answers 200 for assets that do not exist will silently fill
   a dataset.** Fetch a deliberately bogus URL at startup, hash the response,
   and compare every download against it. Cheap, and it also catches the case
   where the site swaps in a new placeholder later.
4. **Content negotiation can defeat a byte-hash check.** The same missing
   asset came back as AVIF under a browser `Accept` and as JPEG under an
   explicit one, so the two have different hashes. Pin the format via the
   request header (Scene7 ignores `fmt=`) and keep a magic-byte check as a
   backstop.
5. **A robots.txt `Disallow` can rule out the *best* data source.** J.Crew's
   `/api/` and `/data/v1/` rules cover the PLP's own product API, so the
   correct route was the sitemap plus server-rendered PDPs. Read the
   Disallow list before designing the crawl, not after.
6. **Client-rendered listing pages can hide behind a valid `__NEXT_DATA__`.**
   J.Crew's PLP ships the blob with an empty product map — the presence of
   the framework's data island proves nothing about whether the data is in
   it.
