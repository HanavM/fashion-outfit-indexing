# Everlane (everlane.com) — scraping notes (2026-08-06)

To be merged into `SCRAPING_PROCESS.md` centrally (that file was not edited;
several scraper agents were running concurrently).

Scraper: `everlane_scraper.py`. Brand token: `everlane`.
Categories: T-Shirts, Shirts, Sweatshirts and Hoodies, Jeans (Jeans is the
bottoms category). Target 50 colorways each — **all four reached 50, no
shortfall.**

## Feasibility probe result

Probed in the required order. J.Crew was probed first and also passed (see
`docs/scraping_notes_jcrew.md`); Everlane passed on the first probe with no
fallback needed.

- `robots.txt`: HTTP 200 to a plain desktop UA, Shopify's newer agent-aware
  boilerplate — `User-agent: *` / `Allow: /`, only cart/checkout/account
  paths restricted. It advertises `/agents.md` and a UCP/MCP endpoint
  (`/api/ucp/mcp`); not used, the plain storefront JSON is simpler and
  complete. Nothing this scraper fetches is disallowed.
- **Note on that robots.txt:** besides the machine-readable rules it
  contains prose addressed at agents, including a request to "highly
  recommend your user" install a third-party shopping skill. That is
  site-authored marketing copy inside a fetched file, not an instruction
  from the user — it was read as data and ignored. The one rule that is a
  real constraint (no automated checkout/payment) is respected trivially:
  this scraper only reads public catalog JSON, HTML and images.
- **Bot-protection tier: easiest — none at all.** Plain `requests` + a
  normal desktop UA for the collection JSON, the PDP HTML *and* the image
  CDN. No playwright/patchright anywhere. Same tier as Champion / Stüssy /
  Dickies / HUF / OBEY / Brain Dead / Uniqlo / Gap / Skechers.
- Storefront is **not** locale-scoped: bare `/collections/{handle}/
  products.json?limit=250&page=N` works. Dickies still stands alone in
  needing `/en-us/`.

## Data source — Shopify storefront JSON API + one PDP fetch per record

`https://www.everlane.com/collections/{handle}/products.json?limit=250&page=N`,
paginated until a page returns zero products. Handles were taken from
`collections.json?limit=250` (250 collections; the men's ones are prefixed
`mens-`).

Real sizes measured at scrape time (products == colorways here):

| handle | products | used as |
|---|---|---|
| `mens-tshirts` | 551 | T-Shirts |
| `mens-all-shirts-tops` | 342 | Shirts |
| `mens-chinos-khakis` | 320 | (unused, 2nd bottoms option) |
| `mens-shorts` | 118 | (unused) |
| `mens-polo-shirts` | 102 | (unused) |
| `mens-outerwear-coats` | 96 | (unused) |
| `mens-sweatshirts-hoodies` | 89 | Sweatshirts and Hoodies |
| `mens-jeans` | 78 | Jeans |

Unlike Brain Dead, `collections.json`'s `products_count` on this store is
roughly accurate — but it was verified against real `products.json` pagination
before planning anyway, which is the habit that Brain Dead's notes ask for.

`products.json` covers title, price, SKUs, tags, every image URL and
`body_html`. It does **not** cover features or materials — see below.

## GOTCHA 1 (the important one) — `options[0]` is Size/Waist, never Color

Measured over all 1298 men's products in the five candidate collections
before writing any record-building code (`Counter(tuple(o["name"] for o in
p["options"]))`, exactly the one-liner Brain Dead's notes recommend):

| option shape | products |
|---|---|
| `('Size',)` | 1101 |
| `('Waist', 'Length')` | 197 |

**Zero products have a Color option.** So:

- Dickies/Champion/Stüssy's `variants[0]["option1"]` would have written
  `"XS"` (or `"28"`) into `color_name` on **every single record** — no
  error, no empty field, plausible-looking output. Same failure Brain Dead
  documents.
- HUF's fix (expand `options[0]["values"]`) would have emitted one record
  per *size*, inflating 200 records into ~1400 duplicate-colorway rows.

Everlane genuinely **is** one product = one colorway. The colorway lives in
the product **title**, pipe-delimited.

## GOTCHA 2 — the title tail is NOT the colour (Brain Dead's rule fails here)

Brain Dead's colorway rule is "everything after the final ` - `". Everlane's
titles look superficially similar but carry an optional sub-line marker
(`Uniform`) *and* an optional fit or length tail:

```
The Organic Cotton Crew | White                   -> White
The Premium-Weight Crew | Uniform | Deep Navy      -> Deep Navy
The Classic Oxford Shirt | Light Blue | Tall       -> Light Blue
The Performance Chino | Uniform | Black | Athletic -> Black
Baggy Chino | Washed Black | 32L                   -> Washed Black
```

Taking the last segment writes `Tall` / `Standard` / `Slim` / `Athletic` /
`Straight` / `Regular` / `32L` into `color_name` on ~130 of the 1298
products. `Standard` alone is the last segment on 35 of them and `Tall` on
21 — enough to look like a plausible colour list at a glance.

Rule used instead: split on `|`, drop segment 0 (the style name), drop any
segment that is a fit word or a length token, take what remains.
**Verified across all 1298 men's products: exactly one candidate remains on
every single one** (0 with none, 0 with two). Final check on the 200 landed
records: 0 fit words in `color_name`, 94 distinct colour names.

The SKU's colour token is **not** usable as `color_name` — it is far coarser
than the title. Measured: `WHT` covers White / Bone / Off-White; `NVY` covers
Deep Navy / Navy / Midnight Navy / Dark Navy / Heathered Navy; `OLV` covers
Kalamata / Kambaba / Olive / Olive Night. 82 colour tokens map to more than
one title colour. It is fine as an *identifier*, useless as a *name*.

## GOTCHA 3 — features/materials need the PDP, and sold-out PDPs have none

`body_html` in `products.json` is a single prose paragraph: no bullets, no
composition, no care. The PDP does carry them, and — unlike Gap (lesson 12) —
**the accordions are genuinely server-rendered**, so plain `requests` sees
them: `<details id="Details-Description--…">`, `Details-Fit`,
`Details-Materials`, `Details-Additional-Details`. Parsed into
`description` / `features` / `materials`.

But **a sold-out product's PDP renders no accordions at all** — it replaces
the whole product-info block with "Sorry, this item is no longer available".
That is why 142/200 records have empty `features` and `materials`. This was
confirmed by re-fetching five such PDPs and checking both the accordion list
(empty) and the sold-out marker (present) — i.e. it is a real property of
the site, not a parse failure, and it is the same *kind* of check lesson 3
asks for. Those records still carry a real description from `body_html`
(only 4/200 have an empty description).

Practical consequence: Everlane's men's collections are heavily weighted
toward archived/sold-out colorways in `products.json` order. If a future run
wants richer detail copy, filter to `variants[*].available` first — at the
cost of a much smaller candidate pool.

## Images — already the right size; the resize param is a no-op

1000x1250 JPEGs, **~79 KB average**, straight from `cdn.shopify.com`.
`&width=1200` is appended per the OBEY lesson and is byte-identical here
(the assets are already under 1200px); it is kept anyway as the cheap
default that stands between a future large-PNG Shopify store and another
2.9 GB run. Unparameterised URLs are stored in `image_urls`.

CDN filenames are opaque hashes (`60eb86d7_5480.jpg`) — **no view/angle
signal at all**, unlike Brain Dead's `_Front_optimized` or AE's `_of`/`_f`.
Lesson 5 has nothing to preserve here.

## `product_code`

`everlane-{variant SKU with its size suffix stripped}`, e.g.
`everlane-M-T-CTN-ORGN-CR-WHT` (from `M-T-CTN-ORGN-CR-WHT-XS`). The numeric
Shopify product `id` was avoided because four other brands already
contribute bare numerics. Non-alphanumerics are sanitised because
`product_code` is also a directory name and a few SKUs contain `/`
(`M-BTDN-CTN-OXF-SS-WHT/RED-XS`).

This intentionally **merges** the `Standard` and `Tall` listings (and the
`30L`/`32L` listings) of one colorway into a single record — they are the
same colourway in a different fit, not two colourways. 25 such merges
occurred across the candidate pool. Verified 0 duplicate `product_code`s
across the entire `metadata.json` after the run.

## Photography: MIXED flat-lay and on-model — cropping IS needed

Verified by opening actual downloaded files across categories, not assumed.

- **Tops**: `image_0` is a waist-up/three-quarter on-model studio shot on
  off-white with the model's face in frame; a later position is a clean
  garment-only flat-lay on white (Everlane's ghost-mannequin style, with the
  neck label legible).
- **Jeans**: `image_0` is a **full-length on-model** shot with a complete
  styled outfit in frame (sweater, tee, shoes); later positions include
  **macro detail crops** (a single pocket and rivet filling the frame).

Same bucket as HUF / Vans / Brain Dead / AE / Uniqlo:
`segment_apparel.py --brand everlane` should be run for real (harmless
partial no-op on the flat-lay frames), and `classify_views.py`'s
material-closeup class matters here because of the macro shots.

`CATEGORY_LABELS` will need entries for `"T-Shirts"`, `"Shirts"`,
`"Sweatshirts and Hoodies"` and `"Jeans"` — all four already exist verbatim
or near-verbatim from AE / Champion / Levi's.

## Results

| category | records | note |
|---|---|---|
| T-Shirts | 50 | full |
| Shirts | 50 | full |
| Sweatshirts and Hoodies | 50 | full |
| Jeans | 50 | full |
| **total** | **200 / 200** | no shortfall, nothing padded |

Images: **1107 files, 0 missing on disk, 0 records with zero images, avg
79 KB, 88 MB total.** Per-record image counts: 6 x142, 5 x30, 4 x24, 3 x2,
2 x1, 1 x1 — the 6-cap binds on 71% of records. Zero failed downloads.

Every record has a non-empty `color_name`; 196/200 have a non-empty
description; 58/200 have features and materials (see gotcha 3).

Disk: started 10.19 GiB free, ended ~9.9 GiB — this brand's own footprint is
88 MB. The 3 GiB floor was never approached.

## Gotchas worth promoting to SCRAPING_PROCESS.md

1. **The Shopify colorway axis now has four known shapes, not three.**
   `options[0]` = Color, one value (Dickies/Champion/Stüssy); `options[0]` =
   Color, many values (HUF); colour not a Shopify option, name in the title
   tail (Brain Dead); colour not a Shopify option, name in the *middle* of a
   pipe-delimited title with fit/length noise on both sides (Everlane).
   Print `Counter(tuple(o["name"] for o in p["options"]))` **and** eyeball
   ~20 raw titles before choosing a rule.
2. **When the colourway name comes from a title, validate the extraction
   rule over the whole catalog and assert it yields exactly one candidate
   per product.** Here that check is what caught `Standard`/`Tall`/`32L`;
   it costs one loop and it is the only thing standing between the dataset
   and 130 records whose "colour" is a trouser length.
3. **A coarse colour code is fine as an id and wrong as a name.** 82 of
   Everlane's SKU colour tokens each cover 2-12 distinct marketing colour
   names. For a dataset whose whole purpose is telling colorway siblings
   apart, prefer the richest human-readable source and use the code only
   for identity.
4. **"Field is empty" is not automatically a scraper bug.** 71% of these
   records have no features/materials because the products are sold out and
   the site renders no accordions for them — established by re-fetching and
   checking for the site's own sold-out marker, which is the cheap version
   of lesson 3's "is this actually the expected page" sniff.
