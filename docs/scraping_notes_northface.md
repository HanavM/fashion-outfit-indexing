# The North Face — scraping notes (`northface_scraper.py`)

To be merged into `SCRAPING_PROCESS.md` centrally (five agents were running
concurrently, so that file was deliberately not edited).

Brand token: `northface`. Categories targeted: **Jackets and Vests, Fleece,
Hoodies and Sweatshirts, Pants** (50 each = 200). Deliberately weighted
toward outerwear — two of the four categories are jacket/fleece families,
which is the reason this brand was added at all (the catalog was thin on
jackets and coats).

## RESULT — 143/200, stopped early for disk (not a site limitation)

| Category | Got | Target | Note |
|---|---|---|---|
| Jackets and Vests | 50 | 50 | complete |
| Fleece | 50 | 50 | complete |
| Hoodies and Sweatshirts | 43 | 50 | **7 short** — cut off mid-category by the disk floor |
| Pants | 0 | 50 | **never started** — cut off by the disk floor |

**143 records, 639 images on disk, 178 MB consumed** (avg 4.4 images per
record; the 6-image cap bound on only 32 records — TNF typically serves
3-5 images per colorway).

**The shortfall is NOT a catalog-size cap.** Unlike PacSun/Gap Sweaters or
Levi's Accessories, TNF's Hoodies and Pants catalogs are both far larger
than 50; the run was killed by the machine's disk floor. Free space on
`/System/Volumes/Data` fell from 8.2 GiB at start to 2.4 GiB while five
other scrapers ran concurrently — TNF's own 178 MB was a rounding error in
that. Per the standing rule (stop below 3 GiB), the scraper was stopped
with its checkpoint saved.

**Re-running is safe and cheap when disk allows.** `northface_scraper.py`
is idempotent: it counts existing records per category, skips completed
ones, and skips any `product_code` already present. A re-run picks up at
Hoodies 43/50 and then does Pants. Two smaller loose ends a re-run also
fixes for free: (a) 3 colorway image directories exist on disk whose
records fell inside the last unsaved checkpoint interval (8 orphan JPEGs,
harmless); (b) 4 records have fewer `images` than `image_urls` (1-2 vs 3-6)
from transient image-download failures — `download_images()` skips files
that already exist, so a re-run only fetches the missing ones.

Data integrity of what landed was verified explicitly, not assumed:
metadata.json parses; all 3679 `product_code`s in the file are unique;
every one of the 631 referenced image paths exists on disk;
`image_count == len(images)` on every record; and zero records have an
empty `color_name`, `price`, `details.description`, or `details.features`.

### Operational gotcha worth recording (cost ~20 min)

The first run died at 123 records for a reason unrelated to the site: the
scraper was launched as a *foreground child of a harness-managed background
shell*, and when that wrapper shell was reaped the Python process died with
it. Two fixes, both applied on restart:
- launch long scrapes with `nohup ... & disown` so they survive the parent;
- **run Python with `-u`**. Redirected stdout is block-buffered, so
  `logs/{brand}_scrape.log` stayed **completely empty for 50 minutes** while
  the scrape ran normally. Any `tail -f`-based progress monitor sees nothing
  and looks exactly like a hung job. Progress had to be inferred from
  `find apparel_dataset/{brand} -mindepth 2 -maxdepth 2 -type d | wc -l`
  instead. Nothing was lost (checkpointing worked — 123 records were on
  disk), but it made a healthy run look dead.

## Feasibility probe

- `robots.txt` fetched with plain `requests` → Akamai edge **"Access
  Denied"** (`errors.edgesuite.net`), i.e. the file itself is unreadable
  without a real browser. Fetched successfully through `patchright`.
  It is **not** `Disallow: /` for `*`. Real disallows: `/*?CountryPref*`,
  `/*/c/*filters=*`, `/*/c/*sort=*`, `/*/c/*storeFilter=*`,
  `/*/explore/*?*`, `/*/search`, `/*/size-chart`, `/*/warranty-retail`,
  `*/cart`, `*/checkout*`, `*/order/success`, `*/product/review`.
  Amazon's `AmazonProductDiscoverybot` and the ThousandEyes crawler get
  `Disallow: /`; the generic `*` agent does not. Catalog browsing and
  `?page=N` pagination — the only two things this scraper does — are
  allowed. No fallback to Columbia or Cotopaxi was needed.
- Applied lesson 4 before writing any selector code: found schema.org
  ld+json (`@graph`) on both listing and product pages, plus a
  `__NUXT_DATA__` payload. No CSS-selector scraping was needed for
  anything except the product-detail bullet list.

## Bot-protection tier

**Akamai Bot Manager, "Vans tier"** — the same softer tier as Vans and New
Balance, *not* Levi's. Plain `requests` gets an edge Access Denied on every
path (including `robots.txt`). `patchright` headed with `channel="chrome"`
passes cleanly on the very first navigation with **no interactive
behavioral-challenge interstitial at all** (unlike Levi's
`sec-if-cpt-container` screen). No intermittent mid-run blocks were
observed. `goto_with_retry()` + an `is_blocked()` title sniff are still
implemented, defensively, on the New Balance precedent.

VF Corp parent-company inference held: `vans_scraper.py` was a near-direct
template. Same Nuxt.js storefront, same `/en-us/c/...` and `/en-us/p/...`
URL shapes, same `CollectionPage` → `mainEntity.itemListElement` listing
node at 48 items/page, same real `?page=N` pagination, same
`data-test-id="product-details-bulletin"` selector for detail bullets.

## Data source

- **Category paths from the commerce sitemap**, not guessed:
  `sitemap.xml` → `sitemaps/commerce/commerce-en-us.xml` (1387 locs, 185
  `/en-us/c/` category URLs). Same discipline as Vans, where guessed paths
  produced genuine in-app 404s.
- **Listing pages** (`/en-us/c/...`): ld+json `CollectionPage` →
  `mainEntity.itemListElement`, 48 items/page. Pagination verified real
  (pages 1 and 2 shared zero product codes across 96 items).
- **Product pages** (`/en-us/p/...`): the ld+json `@graph` has *two*
  product nodes and only one of them is useful:
  - `ProductGroup.hasVariant` is **size** variants of one fixed colorway
    (color constant across every entry) — same dead end as Vans.
  - a sibling `Product` node **is** the currently-selected colorway:
    `sku` == `mpn` == `productID` == the colorway code (`NF0A88XU2EK` =
    style `NF0A88XU` + color `2EK`), with its own `color`, `image` array
    and `offers.price`. That node is the record source.

## The one genuinely new gotcha (worth adding to SCRAPING_PROCESS.md)

**The listing page shows only ONE colorway per style — the default — so a
Vans-style "one listing item = one record" loop silently produces a
per-style dataset, not a per-colorway one.** Across 96 listing items there
were 96 unique URLs and 96 unique colorway codes: one per style, no
siblings. This is a *quiet* failure mode: the codes are all distinct and
every record looks valid, so nothing in the output signals that ~4x the
colorway siblings were never seen.

**Where the siblings actually live**: not in any ld+json node, and not in
an `<a href>` swatch list either. They exist only inside the flat
`__NUXT_DATA__` payload array, as a run of four adjacent JSON strings:

```
"NF0A88XUJK3","TNF Black","JK3","/en-us/p/.../mens-hydrenalite-down-jacket-NF0A88XU?color=JK3"
```

i.e. `full colorway code, human color label, 3-char color value, ?color= URL`,
matched by:

```python
SWATCH_RE = re.compile(
    r'"(NF[A-Z0-9]{8,12})","([^"]{1,60})","([A-Z0-9]{3})","(/en-us/p/[^"]*\?color=\3)"'
)
```

The `\3` backreference is what makes this safe — it anchors the URL to the
color value from the same run, so unrelated adjacent strings in the flat
payload can't produce a false match.

**Second-order trap**: the *currently selected* colorway is usually
**absent** from that run (Nuxt dedupes its URL string out of the payload
because it's already the canonical page URL). On the probe style, 5
colorways existed but `SWATCH_RE` found only 4. The selected colorway must
always be unioned in from the `Product` node's own `sku`/`color`, or you
lose exactly one colorway per style — again silently.

Navigating to `?color={value}` re-renders the whole PDP for that colorway,
ld+json included, so each sibling costs one page load and yields a complete
record.

## Images

- CDN: `assets.thenorthface.com/images/{transform}/v{version}/{CODE}-{VIEW}/{Name}-{VIEW}.png`
  (Cloudinary-style dynamic imaging, same family as Vans's
  `assets.vans.com/images/t_img/...`).
- ld+json serves the `t_Thumbnail` transform: **600x698 PNG**, too small.
- **Do NOT use bare `t_img`** — it returns the untouched source PNG:
  2150x2500, **3.5 MB per image**. On a disk-constrained machine that is a
  ~4 GB mistake at 200 records.
- Correct transform (verified by downloading and checking real pixel
  dimensions, not inferred from the URL):
  `t_img/c_fill,g_center,f_auto,h_2500,w_2000` → **2000x2500 JPEG,
  ~250 KB**. `f_auto` returns JPEG to a `requests` client (no
  `Accept: image/webp`), so writing it as `.jpg` is correct.
- View codes are embedded in the image path (`-HERO`, `-HERO2`, `-HERO3`,
  `-BACK`, `-ALT1..`, `-MODEL34`). No extra schema field was added for
  them (the task specified an exact schema), but per lesson 5 the signal is
  **not lost**: `image_urls` is stored positionally aligned with `images`,
  so `image_N.jpg` → view code is recoverable from `image_urls[N]`.
- Capped at 6 images per record (disk constraint). In practice TNF serves
  3-5 images per colorway, so the cap rarely bound.

## Photo style — on-model, cropping WILL be needed

**Verified by opening real downloaded images from all three scraped
categories (a puffer jacket, a fleece full-zip, a hoodie), not assumed.**
The photos are
studio shots on a live model against a clean near-white seamless: the
model's full torso, head and face are in frame, and the model wears other
garments (pants, beanie) that are not the product. The `-MODEL34` and
`-BACK` views are the same setup at other angles. There are **no** flat-lay
/ laydown product shots in this set.

So this brand is like Vans/Dickies/Nike, **not** like Carhartt/Stüssy:
`segment_apparel.py` garment cropping should be run for real on all four
categories. New `CATEGORY_LABELS` entries will be needed for
`"Jackets and Vests"` (jacket + vest phrasings; the vest phrasing matters —
TNF's vest count is non-trivial and a "a jacket"-only label list would
mislabel them) and `"Fleece"` (fleece jacket / fleece pullover / fleece
sweatshirt phrasings — TNF's "Fleece" bucket contains both full-zips and
pullovers). `"Hoodies and Sweatshirts"` and `"Pants"` already exist.

## Category overlap (minor, handled)

TNF's own taxonomy genuinely double-lists items: fleece jackets appear
under both `mens-jackets-and-vests` and `mens-fleece`. Handled two ways:
a global `product_code` set (so no record is ever emitted twice) and a
`seen_styles` URL set (so an already-walked style is not re-fetched from a
later category). Category assignment is therefore first-come — a fleece
jacket reached first via Jackets and Vests is filed there.

## `product_code` collision check

TNF codes are `NF0A` + 4-6 alphanumerics + 3-char color (e.g.
`NF0A88XU2EK`) — already namespaced by the `NF` prefix. Checked against the
full existing file before the first save: **zero** existing codes started
with `NF`, and all existing codes were unique. No `{brand}-` prefix was
needed. Note the Vans-shares-a-platform concern did not materialise: Vans
codes are `VN...`, TNF codes are `NF...` — the shared VF commerce platform
uses a brand-prefixed ID space.

## Safety / conventions followed

- All metadata writes go through `dataset_utils.load_records()` /
  `save_records_safe()`. No `read_text()` → mutate → `write_text()`
  anywhere (lessons 11 and 14).
- Checkpoint every 10 records, plus a forced save + `df` check every 50
  (lesson 2). Hard abort with a saved checkpoint if free space < 3 GiB.
- Content sniff, not HTTP 200 (lessons 3 and 12): a page with no ld+json
  `Product` node carrying an `sku` is treated as a failed fetch and
  skipped, never turned into an empty record. Image bodies are magic-byte
  checked (`\xff\xd8` / `\x89PNG`) before being written, so an Akamai HTML
  block page can't land on disk as a `.jpg`.
- One fully-extracted record was dumped and read before trusting the batch.
- No captioning / segmentation / hierarchy / eval script was run.
