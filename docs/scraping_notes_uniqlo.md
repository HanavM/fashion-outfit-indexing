# Uniqlo US — scraping notes (2026-08-05)

To be merged into `SCRAPING_PROCESS.md` centrally. Scraper:
`uniqlo_scraper.py`. Brand token: `uniqlo`.

## Bot-protection tier: easiest (no protection at all)

Plain `requests` + a normal desktop UA works for **everything**: the PLP
HTML, both JSON APIs, and the image CDN. No browser, no playwright, no
patchright, no cookies, no `Referer` needed. Same tier as
Skechers/Champion/Dickies. `robots.txt` has no `Disallow: /` and does not
block product, category or API paths.

One robots.txt subtlety worth keeping: Uniqlo blocks a list of PLP filter
query params, including `Disallow: /*?categoryIds=` (**plural**). The
commerce API accepts two equivalent forms of category selection —
`categoryId={id}` / `categoryIds={id}` and `path=,,{id}`. The scraper uses
the `path=,,{id}` form on purpose, so no request it makes matches a
Disallow rule.

## Data source: Uniqlo's own public commerce API (v5, keyless)

There is **no** `__NEXT_DATA__`, no ld+json `ProductGroup`, and no Shopify
`products.json` on this site — but there is a first-party JSON commerce API
that returns strictly more than the rendered page shows. Verified live
(v3 404s; v5 is current):

```
LIST    https://www.uniqlo.com/us/api/commerce/v5/en/products
          ?path=%2C%2C{categoryId}&limit=100&offset={n}&httpFailure=true
DETAIL  https://www.uniqlo.com/us/api/commerce/v5/en/products
          /{productId}/price-groups/{priceGroup}/details
          ?includeModelSize=true&httpFailure=true
```

- `httpFailure=true` makes errors come back as JSON (`{"status":"nok"}`)
  instead of an HTML error page — check `status == "ok"`, not just HTTP 200.
- LIST `result.pagination.total` is a **style** count, not a colorway count.
- LIST already carries `colors[]`, `images.main/{colorCode}`, `images.sub[]`,
  prices, sizes and gender. DETAIL adds `longDescription`,
  `shortDescription`, `composition`, `washingInformation`,
  `images.features[]` (photo + marketing bullet), `tags[]` (Fit / Sleeve
  Length / Neck Type / Material) and breadcrumbs.

### Finding category IDs — the API tree does not have them

`result.aggregations.tree` only goes down to level-2 "classes"
(Outerwear, Bottoms, …). Level-3 category IDs must be pulled out of the
men's PLP HTML: fetch e.g. `https://www.uniqlo.com/us/en/men/tops/t-shirts`
and grep `categoryIds":[NNNNN]`. IDs used in this run:

| Site category | id | styles |
|---|---|---|
| T-Shirts (`men/tops/t-shirts`) | 23386 | 38 |
| Sweatshirts and Hoodies (`men/tops/sweatshirts-and-hoodies`) | 23385 | 15 |
| Casual Pants (`men/bottoms/casual-pants`) | 50251 | 52 |
| Casual Shirts (`men/shirts-and-polos/casual-shirts`) | 95671 | 66 |

(`men/bottoms/jeans` has no `categoryIds` marker in its HTML — it is a
landing page, not a real PLP. If a grep for the ID comes back empty, the
`path=,,` param silently degrades to "the whole catalog, 1391 styles";
always assert the ID parsed before using it.)

## One API product = one style with N colorways (colorway expansion needed)

Unlike the Shopify sites in this pipeline (Champion/Stüssy/Dickies), where
one API product is already one colorway, a Uniqlo product is a style.
`item.colors[]` is the colorway list and `images.main[color.displayCode]`
is that colorway's own photo, so an explicit expansion loop emits one
record per colorway. `product_code` = `uniqlo-{productId}-{displayCode}`
(e.g. `uniqlo-E455365-000-68`) — the brand prefix is required because the
raw IDs are bare numerics.

Men's PLPs contain UNISEX styles too; the scraper keeps `MEN` and `UNISEX`
and skips `genderCategory == "WOMEN"`.

## Gotcha 1 — two hero images per colorway live on two CDN paths, and they are NOT always the same photo

The API only ever gives you the **US** hero:
`https://image.uniqlo.com/UQ/ST3/us/imagesgoods/{l1Id}/item/usgoods_{color}_{l1Id}_3x4.jpg`
— which has `Height:6'1"/185cm  Size:M` rendered **into the pixels** at the
lower right, over the garment. A second hero for the same colorway is
reachable at a **derivable but undocumented** path:
`https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/{l1Id}/item/goods_{color}_{l1Id}_3x4.jpg`

I initially assumed (from one T-shirt) that this was the same photo minus
the overlay. It is only sometimes:

- For most **tops**, the two are the same shot and the WesternCommon one is
  strictly better (no burned-in text).
- For most **bottoms** (and a good share of shirts), they are two entirely
  different photographs: WesternCommon is a clean **flat-lay / ghost
  mannequin** of the garment alone, US is a **full-length on-model** shot
  with a whole styled outfit (jacket, shirt, belt, boots) in frame.

So the WesternCommon path is worth ~1 extra genuinely-colorway-correct
image per record for roughly half the catalog. The scraper does this in
two passes: pass 1 prefers WesternCommon for `image_0` (falling back to the
US URL), pass 2 (`uniqlo_scraper.py --us-hero`) re-fetches the US hero and
appends it **only when it is actually a different photograph**, judged by a
64x64 grayscale mean-abs-diff against the already-downloaded file
(threshold 6/255). Measured on this run: 209 added as genuinely different,
153 correctly rejected as the same shot.

Two takeaways generalisable beyond Uniqlo: **(a)** when a CDN has a
locale-prefixed and a common/global variant of the same asset path, check
both — one may carry a different photo, not just different overlays; and
**(b)** don't conclude "same image" from one example, diff the pixels.

## Gotcha 2 — only ~1 genuinely-colorway-specific photo exists per colorway

This is a property of Uniqlo's photography, not a scraping failure, and it
is the single most important thing to know before scraping this brand.

- `images.main[displayCode]` — exactly one on-model hero per colorway.
- `images.sub[]` — mostly **style-level**. Only some entries carry a
  `colorCode`. The ones that don't are a mix of (a) a flat-lay of the
  *representative* colorway only and (b) multi-color group shots — e.g.
  `goods_455365_sub1_3x4.jpg` for the SUPIMA tee is eight folded tees in
  eight different colors. Verified by downloading and looking at them.

Attaching colorless subs to every colorway would put the wrong color into
records whose entire purpose is colorway discrimination, so only
`main[color]` plus subs with a matching `colorCode` are downloaded. That
alone yields ~1.4 images/record; the second hero from gotcha 1 brings it to
**2.0** (738 images / 374 records; distribution 1:138, 2:155, 3:48, 4:21,
5:10, 6:2). Nothing else on the site is colorway-specific — any future
"make Uniqlo records richer" idea has to come from somewhere other than
these fields.

## Gotcha 3 — `longDescription` is occasionally a real stub

Most styles have 1.4-1.8 kB of prose. A few newly-listed styles return a
4- or 49-character stub (e.g. `"-"`). This is **not** a framework bailout
placeholder (cf. lesson 12) — the field is genuinely unpopulated on the
live PDP too. Handled by falling back to `shortDescription` and printing a
warning when the long description is under 40 chars.

## Photography style: MIXED — on-model AND flat-lay, per record

Confirmed by opening actual downloaded files, not assumed:

- **Tops** (T-shirts, sweats, most shirts): the colorway hero is an
  on-model waist-up shot, model's face in frame, studio-white background,
  garment ~40% of the frame. Same situation as Vans/Dickies.
- **Bottoms**: the record usually holds BOTH a clean flat-lay/ghost-mannequin
  of the garment alone (`image_0`, WesternCommon path) and a full-length
  on-model shot of a complete styled outfit (appended by the `--us-hero`
  pass).

So `segment_apparel.py` cropping is warranted for this brand, but it is
worth knowing that a subset of Uniqlo images are already
garment-isolated — the on-model ones (a) contain distractor garments
(jacket, belt, boots on the pants shots) and (b) carry burned-in
height/size text, and those are the ones that need cropping.

## Cost / footprint

Two API calls per style (list page + detail) plus one HEAD and one GET per
image. Hero JPEGs are ~300 kB each at ~1500x2000. Rate-limited at 0.25 s
between detail calls; zero 4xx/5xx across the whole run.

## What actually landed

374 records / 738 images / 189 MB, all four categories, zero failed
downloads, zero missing files, zero `product_code` collisions:

| Category | records |
|---|---|
| T-Shirts | 100 |
| Casual Pants | 100 |
| Casual Shirts | 100 |
| Sweatshirts and Hoodies | 74 |

74 for Sweatshirts and Hoodies is the **entire** non-women colorway
inventory of that category (15 styles, 73-74 colorways) — it is not padded
and cannot be raised without adding `men/tops/fleece` (5 styles).

## Resume caveat in this scraper (worth copying carefully)

The per-category counter counts **newly added** records, not records
already on disk for that category. Two overlapping invocations of this
script therefore each added their own 50 per category (each one's
`existing_codes` snapshot predated the other's writes), landing 100 per
category instead of 50. Nothing was lost, duplicated or padded — dedupe is
by `product_code` and every extra record is a real distinct colorway — but
the target was overshot. If exact per-category caps matter on a resume,
seed the counter from the existing records for that `brand`+`category` at
startup rather than from zero.
