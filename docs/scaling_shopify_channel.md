# Scaling the catalog via the generic Shopify storefront channel

**Status:** research note, 2026-09-15. Nothing in this was built or run at scale.
**Method:** every number tagged **[M]** was measured by me against live stores on
2026-09-15 with polite single requests (~1 req/sec, one store at a time). Numbers
tagged **[S]** are sourced from a cited third party. Numbers tagged **[E]** are my
estimates and are labelled as such. Sample sizes are given everywhere.

**One gap, stated up front:** §5.5 (EU/UK/Japan TDM) is the only part of this
document not grounded in a source read today — the session's web-search budget ran
out. §5.4 (case law) was sourced by a second agent that did have search access, and
every entry there carries a *[PRIMARY]* / *[SECONDARY]* / *[UNVERIFIED]* label.
Everything else, including every ToS quotation in §5.1-§5.3 and §5.6, comes from a
document or an HTTP response I fetched on 2026-09-15.

**Two premises in the brief were wrong and are corrected here:**

- *"~4.4 GiB of local disk free"* — `df` reports **1.6 GiB available** on
  `/System/Volumes/Data` (228 GiB volume, 100% capacity). **[M]**
- *"images average about 1 MB"* — the existing `apparel_dataset` images average
  **172 KB** (median 133 KB, n=15,974, 2.62 GiB total). `apparel_dataset` is
  4.1 GiB on disk overall and `outfit_dataset` is 7.2 GiB. **[M]**

The storage constraint is therefore **much tighter than stated** (1.6 GiB, not 4.4)
but the per-image cost is **~6x cheaper than stated** (172 KB, not 1 MB). Section 4
works the real arithmetic.

---

## 1. The endpoint — verified behaviour

### 1.1 What it returns

`GET https://<store>/products.json?limit=250&page=N` returns
`{"products": [...]}`. Every product object has exactly these 13 top-level keys,
identical across all 25 stores that served it **[M]**:

```
id  title  handle  body_html  published_at  created_at  updated_at
vendor  product_type  tags  variants  images  options
```

This is the same shape `champion_scraper.py` already consumes. `images[]` carries
`src` (full CDN URL), `width`, `height`, `position`, `alt`, `variant_ids`.
`variants[]` carries `price`, `sku`, `option1..3`, `available`. `body_html` is the
full description. **No second PDP fetch is needed for anything** — confirmed again
here, and the reason `SCRAPING_PROCESS.md` calls this the easiest source in the
pipeline.

There is also a per-product form, `GET /products/<handle>.json`, which returns
`{"product": {...}}` with three extra keys (`image`, `published_scope`,
`template_suffix`). **[M]**

### 1.2 Images and resolution

Native resolutions observed: overwhelmingly **1600x2000 or 2048x2048** **[M]**.
The Shopify CDN accepts `?width=N` on any image URL and resizes server-side.
Measured byte sizes, n=79 real product images sampled across 16 stores **[M]**:

| Request | Mean | Median | **Images per GiB** (mean-based) |
|---|---|---|---|
| native (no param) | 773.9 KB | 361.8 KB | **1,355** |
| `?width=1024` | 197.0 KB | 123.9 KB | **5,322** |
| `?width=768` | 116.5 KB | 70.5 KB | **8,997** |
| `?width=512` | 52.7 KB | 33.5 KB | **19,904** |

Mean is far above median because a minority of stores serve very large originals;
use the mean for capacity planning. `?width=` above the native width is a no-op
(4096 returned the same bytes as 2048) **[M]**.

WebP is available but **is not the storage lever it looks like**: requesting
`Accept: image/webp` returned `content-type: image/webp` at 180,914 bytes versus
186,128 bytes for JPEG at the same `width=1024` — a **2.8% saving** **[M]**. The
`&format=webp` query param was ignored entirely; only the `Accept` header works.
Resolution, not codec, is the lever.

### 1.3 Pagination and the hard caps

- `limit` **silently clamps to 250**. `limit=251`, `500` and `1000` all returned
  exactly 250 products with HTTP 200 and no warning **[M]**.
- `?page=N` works and is not deprecated on this endpoint; pagination terminates
  with an empty `products` array **[M]**.
- **The hard cap is `page x limit <= 25,000`.** Exceeding it returns HTTP 400 with
  the literal body **[M]**:

  ```json
  {"errors":"Page * Limit exceeds the 25000 limit."}
  ```

  Verified on both sides of the boundary: at `limit=250`, page 100 returns 250
  products and page 101 returns the 400; at `limit=50`, page 500 returns 50 and
  page 501 returns the 400. A smaller `limit` does **not** buy more depth — it is
  the product, not the page number, that is capped.

- **The cap is escapable two ways.** `/collections/<handle>/products.json` gets a
  fresh 25,000 window per collection (this is what the existing Champion / Stussy /
  Dickies scrapers already do). And `sitemap.xml` -> `sitemap_products_*.xml`
  enumerates every product URL with no cap at all: **kith.com's sitemap lists
  36,410 product URLs against a 25,000 cap on `/products.json`, and
  `/products/<handle>.json` resolved correctly for URLs drawn from it** **[M]**.
  The sitemap route costs 1 request per product instead of 1 per 250, so it is a
  fallback for the handful of stores that exceed 25,000, not the default.

- `/products.json` and `/collections/all/products.json` return the **same catalog
  in different sort orders** (different first handle for the same `limit=5`) **[M]**.

### 1.4 Rate limits

**No rate limiting observed.** 20 sequential requests with zero delay completed in
3.6 s, all HTTP 200, no 429 **[M]**. A parallel probe by a second researcher fired
40 requests at one store in 7 s with the same result **[M, second sample]**.

This is a genuine measurement and it is also a weak one — it says nothing about
what happens at 1,000 req/s from one IP, and Shopify's CDN (Cloudflare) can change
posture at any time. It is not a licence to hammer.

### 1.5 Is it on by default, and how does a disabled store fail?

**Probed 46 fashion domains; 38 confirmed Shopify** (by `cdn.shopify.com` /
`Shopify.theme` markers in the homepage HTML) **[M]**.

| Outcome | Count | Share of the 38 |
|---|---|---|
| `/products.json` serves JSON on the custom domain | **25** | **66%** |
| Refused (404 / 403 / HTML with 200) | 13 | 34% |

**But the refusals are almost entirely headless storefronts, not a disabled
endpoint.** Of 11 refusers inspected, 10 carry unambiguous headless markers **[M]**:

- **Shopify Hydrogen**: skims.com, mejuri.com, fashionnova.com, cutsclothing.com, suitsupply.com
- **Next.js** (`__NEXT_DATA__` / `_next/static`): gymshark.com, bombas.com, vuoriclothing.com, mizzenandmain.com
- **Genuine block** (classic Liquid theme, still 404s): wearlively.com — 1 of 11

A headless storefront is a custom front-end on the custom domain; the Shopify
backend still exists and **still serves `/products.json` on its `*.myshopify.com`
origin**. Two ways to find that origin, both measured:

1. **`/.well-known/ucp`** (see §1.6) returns the myshopify handle. Worked on
   skims -> `skimsbody`, mejuri -> `77123f`, fashionnova -> `fnova`,
   cutsclothing -> `cuts-clothing` **[M]**.
2. Grepping `([a-z0-9-]+)\.myshopify\.com` out of the homepage HTML — recovered
   vuori -> `vuori-clothing`, buckmason -> `buck-mason-usa` **[M]**. (`Shopify.shop`
   is the canonical variable.)

**All 6 recovered origins served `/products.json` with 250 real products** and the
correct brand in `vendor` **[M]**. Two more (gymshark, bombas) were recovered by
guessing the handle directly. Vuori's origin returned 250 products but **zero
images** — the images are attached on a different sales channel.

**Revised reachability: 33 of 38 Shopify fashion stores (87%) are reachable, 32
(84%) with images** **[M]**.

The failure modes, precisely:
- **HTTP 404 + JSON body** `{"errors":"Not Found"}` (22 bytes) — store does not exist.
- **HTTP 404 + HTML** — headless front-end; the path isn't routed. Recoverable.
- **HTTP 403 + HTML** — WAF/edge block (gymshark on the custom domain).
- **HTTP 200 + HTML** — a soft failure worth guarding against: kotn.com returned
  `text/html` with a 200, and bodenusa.com **redirected the whole request** to
  `us.boden.com/?limit=5`, dropping the path. Any harvester must check
  `content-type`, not just the status code **[M]**.
- **HTTP 401** — password-protected store (pre-launch).
- **HTTP 402** — Shopify "payment required", frozen store.

### 1.6 The thing that changed: Shopify now ships an agent policy

This is the most consequential finding and it is new. Shopify is rolling out a new
default storefront `robots.txt` template. Two templates are in circulation: the older
`# we use Shopify as our ecommerce platform` header (taylorstitch, stussy, champion)
and the new agent-aware one (noahny, kith). The new template opens with **[M]**:

```
# Shopify storefront. Public product, collection, page, blog, policy, cart, and localized HTML is crawlable.
# Agent instructions: https://noahny.com/agents.md
# UCP discovery: https://noahny.com/.well-known/ucp
# UCP/MCP endpoint: https://noahny.com/api/ucp/mcp
...
# Checkouts are for humans. Do NOT complete checkout, payment, or order placement automatically
# Terms of Service: https://www.shopify.com/legal/terms
# Contact: bots@shopify.com
```

And `/agents.md`, served by default, contains a section headed **"Read-Only
Browsing (No Authentication Required)"** that names the endpoints explicitly **[M]**:

```
### Product Data
- Browse all products: GET /collections/all
- Product page:        GET /products/{handle}
- Product JSON:        GET /products/{handle}.json
- Collection JSON:     GET /collections/{handle}/products.json
- Search:              GET /search?q={query}&type=product
```

`/agents.md` returned 200 with real content on **4 of 4** classic-theme stores
tested (taylorstitch, stussy, champion, kith) and 404 on the headless ones
(skims, gymshark) **[M]**.

**What this changes:** the platform operator now publicly documents the product
JSON endpoints as read-only, no-auth, and crawlable, and places its automation
restriction specifically on **checkout**, not on catalog reading. That materially
improves the *platform-ToS* position for reading (§5.1). **It does not touch the
copyright position on the images at all** (§5.3), which is where the actual risk is.

**None of the four `robots.txt` files I captured in full — taylorstitch (177 lines,
156 Disallow), stussy (149 / 133), champion (178 / 148), kith (116 / 50) — contains
the string `.json` anywhere at all** **[M]**. Champion's only `/products` rule targets
specific remote-rendered URLs (`Disallow: /products/*-[a-f0-9]{8}-remote`), not the
JSON endpoint. And `/agents.md` served real content on all four, including the three
still on the *old* robots template — so the agent policy is shipping ahead of the
robots.txt rollout.

**The sanctioned channel is not anonymous.** I probed the UCP/MCP endpoint
(`POST https://noahnyc.myshopify.com/api/ucp/mcp`). `tools/list` returned 200 and a
full tool catalog including `search_catalog` and `get_product`. But calling
`search_catalog` returned **[M]**:

```json
{"error":{"code":-32001,"message":"UCP discovery failed",
 "data":{"code":"invalid_profile_url","content":"Unable to fetch agent profile: Missing profile uri"}}}
```

It requires a resolvable **UCP agent profile URI** — a declared identity — and
`agents.md` states the MCP endpoint is rate-limited per IP. It is built for
buy-on-behalf-of-a-user shopping, not for bulk corpus construction. Useful to know
it exists; not a route to millions of images.

### 1.7 Measured per-store yields

First page only (`limit=250`), so **250 means "hit the cap, has more"** **[M]**:

| Store | Products (p1) | Images (p1) |
|---|---|---|
| allbirds.com | 250 (cap) | 1,103 |
| stussy.com | 250 (cap) | 1,011 |
| champion.com | 250 (cap) | 1,332 |
| kith.com | 250 (cap) | 1,157 |
| rothys.com | 250 (cap) | 1,545 |
| aloyoga.com | 250 (cap) | 1,157 |
| thepangaia.com | 234 | 766 |
| untuckit.com | 250 (cap) | 1,122 |
| johnelliott.com | 200 | 1,445 |
| everlane.com | 250 (cap) | 1,266 |
| goodamerican.com | 250 (cap) | 1,550 |
| taylorstitch.com | 250 (cap) | 2,112 |
| naadam.co | 250 (cap) | 2,689 |
| marinelayer.com | 250 (cap) | 828 |
| outerknown.com | 250 (cap) | 2,032 |
| fahertybrand.com | 250 (cap) | 1,349 |
| girlfriend.com | 165 | 1,083 |
| tentree.com | 250 (cap) | 1,043 |
| frankandoak.com | 106 | 716 |
| knowledgecottonapparel.com | 250 (cap) | 833 |
| percivalclo.com | 250 (cap) | 1,391 |
| finisterre.com | 250 (cap) | 1,542 |
| corridornyc.com | 250 (cap) | 923 |
| noahny.com | 250 (cap) | 1,888 |
| aimeleondore.com | 250 (cap) | 1,536 |

**22 of 25 hit the 250 cap on page 1**, so most of these catalogs are substantially
larger than the table shows.

**One store enumerated in full**, to ground the rest: **taylorstitch.com =
3,813 products / 29,881 images, in 17 requests, in about 40 seconds** **[M]**.
That single mid-size brand is **1.48x the entire current 20,205-image corpus**.

**Kith exceeds the cap**: 25,000 products via `/products.json`, 36,410 via the
sitemap **[M]**.

**Images per product: 6.51**, measured over 240 products across 8 stores; per-store
range 3.77 (stussy) to 9.10 (taylorstitch) **[M]**.

**Metadata size: 7,541 bytes/product raw JSON; 1,118 bytes/product** trimmed to
`{id, handle, title, vendor, product_type, tags, price, image_urls[]}`, n=4,000
products across 16 stores **[M]**.

### 1.8 Data-quality caveat, since it will bite the taxonomy work

`product_type` is **not a usable taxonomy across stores**. Observed values **[M]**:

- allbirds: `Apparel`, `Shoes`, `Socks`, and `return,package_protect` (a non-product)
- rothys: the full product *name* as the type (`The Braid Clog`, `The Bucket Bag`)
- aloyoga: colon-delimited paths (`Accessories:Bags:Fanny`, `Accessories:Hats:Caps`)
- stussy: inconsistent case (`ACCESSORIES` and `Accessories` as separate values)
- champion: includes `Pillows`, `Eyewear` — not apparel
- everlane, rothys: empty string on some products

A second researcher measured this independently on a random store sample: **only
19 of 49 stores populated `product_type` on more than half their products (~39%
recall)**, and naive substring matching on titles produced real false positives —
`"dress"` matching inside *ad**dress***, `"ring"` inside *du**ring*** **[M, second
sample]**. Word-boundary regex scored per-product, then thresholded on the
*fraction* of a store's products that match, was the approach that worked.

Also note: gift cards, shipping-protection SKUs, and bundles appear as ordinary
products and must be filtered.

### 1.9 A second, more reachable channel on the same platform

Shopify's **Storefront GraphQL API accepts tokenless requests** — no
`X-Shopify-Storefront-Access-Token` header at all. I verified this myself:

`POST https://<store>/api/2026-07/graphql.json` with only `Content-Type:
application/json` and a plain `{products(first:3){...}}` query returned **HTTP 200
with real products and image URLs** on fashionnova.com, skims.com, kith.com and
taylorstitch.com **[M]**. It 404'd on gymshark.com, mizzenandmain.com and
suitsupply.com (headless front-ends that don't proxy `/api/` to Shopify).

**This is strictly more reachable than `/products.json` on some stores** —
fashionnova and skims both 404 on `/products.json` but answer GraphQL on the same
custom domain **[M]**. Shopify documents tokenless access as intentional, with a
1,000 query-complexity budget. A second researcher independently measured the same
on kith, carbon38, rothys, allbirds and fashionnova **[M, second sample]**, and
measured the scope limits: `metafields` and `totalInventory` return `ACCESS_DENIED`
without a real token, but title, description, tags, vendor, productType, variants,
prices and image transforms all work.

Practical consequence: a harvester should try `/products.json` first (1 request per
250 products, simplest schema, already implemented in this repo) and fall back to
tokenless GraphQL, then to the `*.myshopify.com` origin.

### 1.10 The sanctioned cross-merchant channel that already exists

Following the UCP trail in §1.6 leads somewhere that changes the framing of this
whole document. Shopify documents two catalog APIs for agents:

- **Storefront Catalog MCP** — `POST https://{storedomain}/api/ucp/mcp`, one
  merchant. "Use Storefront Catalog MCP whenever buyer intent and query are scoped
  to a single merchant." **[S, shopify.dev]**
- **Global Catalog MCP** — **`POST https://catalog.shopify.com/api/ucp/mcp`**,
  **across all Shopify merchants**: "Use Global Catalog MCP when your agent needs to
  discover products across multiple Shopify merchants." **[S, shopify.dev]**

I probed the global endpoint. `tools/list` returned **HTTP 200 anonymously**, with
three tools — `search_catalog` ("Search for products across multiple Shopify stores
in the global catalog"), `get_product`, and `lookup_catalog` **[M]**. Calling
`search_catalog` without credentials returned **HTTP 422** **[M]**:

```json
{"error":{"code":-32001,"message":"UCP discovery failed",
 "data":{"code":"invalid_profile_url","content":"Unable to fetch agent profile: Missing profile uri"}}}
```

Every request must carry a `meta.ucp-agent.profile` URL pointing at a registered
UCP agent profile **[S]**. Responses include product media (image URLs) **[S]**.

**Its limits make it a search API, not a harvest API** **[S]**: `limit` has a
maximum of **50**, and "You can paginate up to 1,000 results. Beyond that depth,
`has_next_page` is `false`." So a single query yields at most 1,000 products. You
would need thousands of distinct queries to build a corpus, and the shape strongly
implies it is monitored per agent profile.

**Why it matters anyway.** This is Shopify offering, under its own name, the thing
the mass-harvest plan is trying to approximate: cross-merchant product discovery
with images, from one endpoint, with no per-store scraping. The price is that **you
declare who you are** instead of being anonymous. Shopify's documentation for these
endpoints states no restriction on indexing, caching, storing or AI training **[S]**
— which is not the same as permission, and the API ToS in §5.1 has not gone away.

**Three auth tiers exist** *[S, /docs/agents/profiles/auth-and-rate-limiting]*:
`TOKEN` (a Dev Dashboard credential, JWT Bearer), `SIGNED` (HTTP signatures per the
UCP spec, ECDSA P-256, verified against your hosted `/.well-known/ucp` profile — the
same mechanism I found on merchant storefronts), and `ANONYMOUS` (no credentials,
catalog/cart/checkout-build tools only). My 422 above shows even the anonymous tier
wants a resolvable profile URI.

**What the terms say: nothing. That is a verified absence, not a green light.** The
documentation is silent on all three questions that matter — whether merchants
opt in or can exclude themselves from the Global Catalog, what terms govern use of
the returned data and images, and whether **any right to display product imagery is
conferred** *[S, verified absence]*.

**Do not over-read this, and here is the critical warning.** UCP plausibly solves
the **display** problem. It very likely makes the **training** problem *worse*.
Shopify's API License and Terms of Use (last updated Feb. 27, 2026) define "Shopify
API" broadly enough to cover the UCP endpoints — "all software, including routines,
data structures, object classes, protocols, programs, templates, libraries and
interfaces, application programming interfaces (APIs), SDKs… made available by
Shopify" *[S]*. So §2.3.14 ("build any commerce or product index") and §2.3.24
("train, fine tune, or improve any machine learning") apply on their face — **and,
unlike anonymous scraping, registering an agent profile creates an unambiguous
contract you have affirmatively accepted.** You would trade a weak contract argument
for a strong one against yourself.

**This is still the least-researched lead here and probably the most important
one**, because it is the only route found that could plausibly end with a *document
to produce* when Apple asks (§5.6). But the question to answer first is not "how do
I register" — it is **"does Global Catalog access carry any licence to display
merchant imagery, and how does it interact with §2.3.24?"** That should be put to
Shopify in writing before any code is written.

---

## 2. Enumeration — how to find the stores

Ranked by what is actually usable. The reachability and classification figures here
come from a second researcher's independent probing, flagged **[M, second sample]**.

### 2.1 What works

**1. A free public list — 1,902,012 Shopify domains, $0.**
The GitHub repo `growthenginenowoslawski/shopify-master-list` carries a 53 MB gzip
(367 MB uncompressed) of 1.9M domains, exported 2026-07-20, 100% marked Shopify and
active. A random 60-domain sample probed two months later: **50 (83%) still serve
`/products.json`**, 4 x HTTP 402 (frozen), 4 dead, 1 x 404, 1 x 403 **[M, second
sample]**. Columns include `merchant_name`, `estimated_yearly_sales`,
`products_sold`, `average_product_price`, `instagram_followers`, `tags`.

**Caveat that matters: those column names are a verbatim StoreLeads schema.** This
is a redistributed copy of a paid commercial export. Using it is not the same legal
posture as crawling the web yourself, and it should be weighed alongside §5 — it is
arguably the *worst*-positioned item in this whole document, because unlike public
product pages it is somebody's proprietary database sitting in the open.

There is **no category column**, so apparel classification is on you (§2.3).

**2. StoreLeads Pro — $250 for one month, pre-classified.**
StoreLeads publishes **3,077,514 live Shopify stores**, of which **27.0% —
831,725 — sell Apparel** **[S]**. The **$250/mo Pro tier includes CSV export and
API access** for two platforms **[S]**. This is the licensed version of item 1,
it is pre-categorised, and one month is enough to export and cancel. **Highest
precision per dollar in this document.**

**3. Common Crawl via Athena — ~$0.25, 1-2 days of engineering.**
Common Crawl is in the AWS Open Data programme: **not requester-pays, free egress**
**[S]**. The current crawl (CC-MAIN-2026-34) covers 2.14B pages / 40.2M hosts; the
columnar index is 0.20 TiB and a full scan costs ~$1.00-1.50 at Athena's $5/TB
**[S, Common Crawl's own estimate]**. You cannot grep the index for
`cdn.shopify.com` — **the cc-index parquet schema has 34 columns and none contain
page content or response headers** **[M, second sample]**. The workable trick is
that `url_path` *is* indexed, so requiring `/collections/` and `/products/`
co-occurrence discriminates Shopify from WooCommerce (`/product/`, singular).
Published prior art using DNS resolution against Shopify's IP got **465,000+
domains from 14 archives** **[S]**; expected yield is **~380K Shopify hosts per
single crawl** **[E]**.

Fewer stores than the free list, for more work — but fully independent, cheap to
re-run monthly, and legally the cleanest of the three. **This is the freshness
channel, not the bootstrap.**

### 2.2 What does not work — tested, not assumed

- **Certificate Transparency: dead.** `*.myshopify.com` is served by a **single
  Let's Encrypt wildcard certificate**, so no per-store CT entries are generated.
  crt.sh yields 1,933 unique hostnames and **nothing after 2017** **[M, second
  sample]**. Custom-domain certs (e.g. `CN=allbirds.com`, one SAN) carry no Shopify
  marker at all.
- **DNS brute force: dead.** The zone is wildcarded.
  `zzqxnonexistentstore12345.myshopify.com` and a real store both CNAME to
  `shops.myshopify.com` -> `23.227.38.74` with `status: NOERROR`. Real and fake are
  **byte-identical at the DNS layer** **[M, second sample]**.
- **HTTP brute force of `<name>.myshopify.com`: works but pointless.** The oracle is
  clean (a 22-byte `{"errors":"Not Found"}` vs. real JSON) and unrate-limited, but
  a 6-character lowercase-alphanumeric keyspace is 36^6 ~= 2.2 billion candidates.
  Strictly worse than downloading a 1.9M-row list **[M, second sample]**.
- **Wappalyzer and PublicWWW: structurally incapable.** Wappalyzer is credit-metered
  (Business tier = 20,000 lookups/mo -> ~8 years to cover 2M stores). PublicWWW's
  Enterprise tier ($499/mo) allows 1M-row searches but caps **CSV export at 10,000
  rows** **[S]**.
- **BuiltWith**: real export on all tiers but Basic ($295) gives 2 lookups/month;
  Pro is $495 **[S]**. 2x StoreLeads' price with a worse ecommerce category model.
- **SimilarTech**: `www.similartech.com` failed DNS resolution entirely **[M, second
  sample]**. Possibly defunct.
- **Shopify's own surfaces**: no global store-discovery API exists — Shopify's docs
  say so explicitly **[S]**. `shop.app` does publish a storefront sitemap
  (29 shards x 50,000 = **~1.45M stores**) **[M, second sample]**, but 67.6% of
  handles are opaque 10-character IDs, only 23% of vanity handles match the actual
  myshopify handle, resolution requires an ~870 KB page fetch each, and shop.app
  **429'd on the second consecutive request**. Hold in reserve.

### 2.3 Filtering to fashion

Two findings from independent measurement, both of which will save real time **[M,
second sample]**:

- **`product_type` alone has ~39% recall** — only 19 of 49 stores populated it on
  more than half their products. This matches what I measured directly in §1.8. You
  must also use `title` and `tags`.
- **Naive substring matching is unusable.** Real false positives observed: a facial-
  serum store matched `cap`/`hat`/`sandal`/`tee`; a garden-plant store matched
  `dress` (inside *ad**dress***); a dirt-bike-parts store matched `ring` (inside
  *du**ring***).

What worked: **word-boundary regex** (`\b(shirts?|dress(es)?|hoodie|...)\b`) scored
**per product**, then thresholding on the **share of a store's products that
match**. At a 0.5 threshold this flagged 31% of stores as fashion — against
StoreLeads' independently derived **27.0% Apparel** **[S]**. Two unrelated methods
within 4 points is the strongest validation available here.

One judgement call to make explicitly: branded-merch stores (restaurants,
nonprofits, bands selling tees and hoodies) score high and *are* technically
apparel. For a retrieval corpus they are probably noise. A `vendor`-diversity or
minimum-product-count filter separates them.

---

## 3. Other platforms with the same property

Measurements in this section are a second researcher's, flagged **[M, second
sample]**; sample sizes are 2-7 stores per platform from a single vantage point, so
treat reachability as directional.

| # | Platform | Exact endpoint | Public by default? | Est. reachable fashion stores |
|---|---|---|---|---|
| 1 | **Shopify** | `POST /api/2026-07/graphql.json` (tokenless); `GET /products.json?limit=250&page=N` | **Yes** | ~1M |
| 2 | **WooCommerce** | `GET /wp-json/wc/store/v1/products?per_page=100&page=N` | **Yes** | ~350-400K |
| 3 | **Squarespace** | `GET /<store-page>?format=json` | **Yes** | ~100-200K |
| 4 | **Magento / Adobe Commerce** | `POST /graphql` | **Yes** | ~15K |
| 5 | **Wix** | *no documented public path* | Unknown — **untested** | unknown |
| 6 | **Shopware 6** | `POST /store-api/product` + `sw-access-key` | No — needs a key | ~10-20K |
| 7 | **BigCommerce** | `POST /graphql` + a token | No — needs a token | ~5K |
| 8 | **SFCC / Demandware** | `GET /s/<site>/dw/shop/v23_2/product_search?client_id=` | **No** — per-site allow-list | ~0 |
| 9 | **PrestaShop** | `GET /api/products` | **No** — off by default | ~0 |
| 10 | **commercetools** | `/product-projections/search` | **No** — OAuth2 always | 0 |
| 11 | **Ecwid / Square / Webflow / OpenCart** | token-gated or nonexistent | **No** | ~0 |
| 12 | **Amazon / Etsy** | approval-gated seller APIs | **No** | 0 |

**WooCommerce is the strongest true analogue.** `/wp-json/wc/store/v1/products`
returned **200 JSON, no auth** on both stores tested, and it is **on by default
because WooCommerce's own cart and checkout blocks call it** — disabling it breaks
the storefront. `per_page` caps at **100** (101 -> HTTP 400 `rest_out_of_bounds`),
but the response headers hand you `x-wp-total`, `x-wp-totalpages` and RFC 5988
`Link: rel="next"`, and deep paging is uncapped. Each product carries `images[]`
with `src`, `thumbnail` and a full `srcset`. **Resolution is not URL-manipulable**
— you get the fixed sizes WordPress generated at upload, unlike Shopify's `?width=`.
Do not confuse it with `/wp-json/wc/v3/products`, which is the admin API and
returned **401** `woocommerce_rest_cannot_view`.

**Squarespace** has the best image story of any platform here — `assetUrl` plus
`originalSize` (3094x4641 in the sample), and the CDN takes `?format=100w|1000w|
2500w|original`. But **catalog pagination is unsolved**: `?page=2` returned 0 items,
the offset-cursor scheme did not work reliably, and the first page returned 56 of a
stated 86 items. `/sitemap.xml` plus per-page `?format=json` is the dependable route.

**Magento's `/graphql` is public by default** (it is how the Luma/PWA frontend
works) and enforced no page-size cap — `pageSize: 300` was accepted. `/rest/V1/*`
is 401 and unusable.

**Three platforms are worth naming as off-limits rather than as options.**
BigCommerce and Shopware both publish a storefront token in their own page HTML,
and SFCC leaks its site and organisation identifiers the same way. Lifting such a
token and replaying it from a non-browser client is **not** the same act as calling
an endpoint the operator documents as public — it is walking through a gate the
operator put up, which is precisely the distinction Van Buren draws (§5.4) and
precisely what turns a weak ToS claim into a plausible CFAA one. **Recommendation:
do not do this.** Shopify, WooCommerce, Squarespace and Magento give you a public
endpoint with no credential step and cover the overwhelming majority of stores;
there is no reason to reach for the rest.

**The real ceiling is WAFs, not APIs.** Cloudflare, Akamai and DataDome blocked more
stores in the cross-platform sample than missing endpoints did — especially on
Magento and SFCC, where enterprise tenants are the norm. I saw the same on Shopify
(gymshark 403). Budget more engineering for request hygiene than for protocol
coverage — and note that escalating against a WAF is itself an access-control
question, not a technical one.

**Coverage:** Shopify + WooCommerce + Squarespace together are ~73.5% of all
named-platform stores (HTTP Archive, May 2026 crawl: 27.9% + 39.0% + 6.6%) **[S]**,
and all three answer with no credential step. That is the 90/10.

---

## 4. Scale estimate, and where the bottleneck actually is

### 4.1 The funnel

| Stage | Number | Basis |
|---|---|---|
| Live Shopify stores | 3,077,514 | **[S]** StoreLeads public Shopify report |
| ...that sell apparel | **831,725** (27.0%) | **[S]** same source, category breakdown |
| ...reachable via products.json | **~690,000** | 831,725 x 83% **[E from M]** |
| Products per store | 130 avg all-platform **[S]**; my 25-store sample is far larger | **[E]** |
| **Total products** | **~100M** (order of magnitude) | **[E]** |
| **Total image URLs** | **~650M** | x 6.51 images/product **[M]** |

The 83% reachability rests on two independent samples that converge: my curated
brand list gave 66% direct / **87% including myshopify-origin recovery** (n=38),
and a random sample from a 1.9M-domain list gave **83% direct** (n=60). The random
sample is higher because big DTC brands are exactly the ones that go headless;
the long tail runs stock Liquid themes and is wide open.

The independent free-classifier estimate of the apparel share was **31%** against
StoreLeads' **27%** — two unrelated methods within 4 points, which is reassuring.

**Treat ~650M as a theoretical URL ceiling, not a corpus.** It is not deduplicated,
and a large fraction of the 831,725 apparel stores are dropshippers reselling
identical supplier photography. The count of *distinct, brand-shot, usable* fashion
product photographs is far smaller and I have no measurement of it.

### 4.2 The bottleneck is storage. It is not close.

**Discovery: solved, and free.** A public GitHub repo
(`growthenginenowoslawski/shopify-master-list`) carries **1,902,012 Shopify domains**
in a 53 MB gzip, exported 2026-07-20, verified 83% still live **[M, second sample]**.
The licensed equivalent is **StoreLeads Pro at $250 for one month**, which exports
831,725 pre-classified apparel domains **[S]**. Common Crawl via Athena yields
~380-465K Shopify hosts for **~$0.25** in query cost **[E/S]**. Certificate
Transparency and DNS brute force are **dead ends**, conclusively: `*.myshopify.com`
is a single wildcard certificate (no per-store CT entries since 2017) and the DNS
zone is wildcarded so nonexistent handles resolve identically to real ones **[M,
second sample]**. Details in the enumeration section.

**Rate limits: not the bottleneck.** Harvesting metadata from 690,000 stores at
~3 requests each is ~2.1M requests. At a deliberately polite 50 req/s spread across
many hosts that is **~12 hours** **[E from M]**.

**Metadata: not the bottleneck.** At the measured 1,118 bytes/product trimmed,
**100M products = ~107 GiB**, and 1M products = **1.07 GiB** **[M]**. You can hold
the complete product catalog, with every image URL, without downloading a single
pixel.

**Pixels: this is the whole problem.** From the measured table:

| Storage budget | Images at `width=768` (117 KB) | Images at `width=512` (53 KB) |
|---|---|---|
| **1.6 GiB (actual free local disk)** | **~14,400** | ~31,800 |
| 100 GiB | ~900,000 | ~2.0M |
| 1 TiB | ~9.2M | ~20.4M |
| 650M images (the ceiling) | **~72 TiB** | ~33 TiB |

**The local machine is full.** 1.6 GiB free holds about **14,400 more images** —
roughly *half* of Taylor Stitch alone (29,881), and 0.002% of the ceiling. Local
disk is not a staging area for this; it is not even a staging area for one brand.

Bandwidth tracks storage 1:1 (you must download what you store), so at
`width=768`, 1M images is **111 GiB** of transfer. On a home connection at an
effective 20 Mbit/s that is ~13 hours of pure transfer for 1M images; the fetch is
network-bound, not CPU-bound.

### 4.3 What this actually means for the project

The honest framing is an inversion of the question. The channel's supply is
effectively unbounded; the project's capacity is ~14,400 images. So **"how many
products could we reach" is the wrong question** — the answer is "more than you can
possibly store, by four orders of magnitude." The right questions are:

1. **Harvest metadata first, pixels never-by-default.** 1,118 bytes/product means
   the catalog for *every* fashion Shopify store you can enumerate costs ~107 GiB,
   and a 100,000-store slice costs ~1 GiB. Build the index, then pull pixels
   on-demand for the subset a given experiment needs. This alone removes the
   constraint from the critical path.
2. **`?width=` is a free 6.6x.** Downloading at `width=768` instead of native costs
   117 KB instead of 774 KB per image **[M]**. For SigLIP-2 / DINOv3 at 224-384px
   input, 768px is already generous. The existing corpus is stored at a 172 KB
   average, so this is consistent with what the project already does.
3. **Storage belongs on the Modal volume**, which this repo already uses
   (`modal_app_sync_catalog.py`, `sync_outfits_to_modal.py`, and the 20,545-image
   outfit volume from commit `dfc5866`). Local disk is out of the question.
4. **Diversity, not volume, is the win.** 20 brands -> 1,000 brands changes what the
   retrieval model sees far more than 20,205 -> 200,000 images from the same 20
   brands. A budget of ~900,000 images at `width=768` fits in 100 GiB and could
   cover ~1,000 brands at ~900 images each. That is a **44x corpus increase** and a
   **50x brand-diversity increase**, and it is the largest target that is actually
   sane.

**Defensible bottom line:** the reachable ceiling is ~650M image URLs across
~690,000 stores **[E, from M+S]**, the metadata for all of it is cheap, and the
binding constraint is storage and bandwidth — not discovery, not rate limits, and
not the endpoint. But see §5 before treating any of that as a plan.

---

## 5. Legality and terms of service

**This is not legal advice and I am not a lawyer.** What follows is what the
primary documents actually say, quoted, plus the state of the relevant case law.
Where something is genuinely unsettled I say so rather than resolving it.

The short version, stated plainly: **the technical ease of this channel has nothing
to do with its legal position, and the two get further apart the further down the
pipeline you go.** Fetching the JSON is the weakest claim against you. Displaying
the images in a shipped iOS app is the strongest. This project already has a
licensing problem with 0 of 20,532 photos carrying a licence; this channel would
multiply that problem by whatever factor you scale it, not solve it.

### 5.1 Shopify's own terms — and a real contradiction

**Shopify Terms of Service §1.9** (quoted verbatim):

> "You agree not to access the Services or monitor any material or information from
> the Services using any robot, spider, scraper, or other automated means."

**But §1.1 defines who "you" is:**

> "'you' means the Shopify User (if registering for or using a Shopify Service as an
> individual), or the business employing the Shopify User"

A third party who has never registered for a Shopify account is not obviously a
"Shopify User". Contract claims generally require assent, and this is the classic
browsewrap problem (§5.2). **However**, the default storefront `robots.txt` now
explicitly links `https://www.shopify.com/legal/terms` under a "Terms of Service:"
line addressed to crawling agents **[M]** — which is Shopify holding those terms out
to exactly the reader in question. That strengthens a notice argument. It is not
settled and I would not bet on either reading.

**Shopify API Terms of Service** is much more directly on point, and much worse:

> **§2.3.14** — "not use the Shopify API to conduct any systematic or automated data
> collection activities (including scraping, data mining, data extraction and data
> harvesting) **or to build any commerce or product index**."

> **§2.3.24** — "not use any information derived from the Shopify API ... to create,
> develop, **train, fine tune, or improve any machine learning or artificial
> intelligence systems**, models, or technologies ... except with: (a) Shopify's
> prior written consent, or (b) solely with respect to Merchant Data ... the
> relevant Merchant's consent."

Those two clauses describe this project's plan almost word for word: build a product
index, fine-tune a retrieval model on it.

**§2.1.3** scopes them: "Except in the case of Private API Credentials or
unauthenticated Checkout Kit access, in order to access and use the Shopify API,
Developer must obtain an API Client by registering for a Partner Account." These
terms are framed at registered Partners and app developers.

**This produces a concrete, actionable distinction that cuts against the technically
better channel:**

| Channel | Is it "the Shopify API"? | Position |
|---|---|---|
| `GET /products.json` | Arguably not — an undocumented-in-the-API-reference storefront path, and now described in the default `agents.md` as public read-only browsing | Weakest claim against you |
| `POST /api/2026-07/graphql.json` (tokenless) | **Unambiguously yes** — it is the versioned Storefront API endpoint | §2.3.14 and §2.3.24 apply on their face |

**So the GraphQL fallback in §1.9, which is technically more reachable, is the
legally worse channel.** If any of this is pursued, that ordering matters.

**What the platform now says is allowed.** Cutting the other way, and this is
genuine: the default storefront `robots.txt` states "Public product, collection,
page, blog, policy, cart, and localized HTML is crawlable", contains **no `Disallow`
matching `/products.json`, `*.json` or `/products`** on any of the three stores I
checked, and the default `/agents.md` lists `GET /collections/{handle}/products.json`
and `GET /products/{handle}.json` under a heading reading **"Read-Only Browsing (No
Authentication Required)"** **[M]**. Shopify's own automation prohibition in that
file is aimed squarely at **checkout**, not at catalog reading.

Reading the endpoint with a polite crawler is therefore about as well-positioned as
unlicensed scraping ever gets. **That is a statement about access, and only about
access.** It says nothing about the images.

### 5.2 A typical merchant's terms — and they contradict the platform

I fetched two stores that serve `/products.json` happily.

**Taylor Stitch** — a store whose `robots.txt` permits it and whose `agents.md`
documents it — nonetheless states:

> "You agree not to use any robot, spider or other automatic device, process or
> means to access the Website for any purpose, including monitoring or copying any
> of the material on the Website"

> "You must not reproduce, distribute, modify, create derivative works of, publicly
> display, publicly perform, republish, download, store or transmit any of the
> material on our Website"

> "No right, title or interest in or to the Website or any content on the Website is
> transferred to you, and all rights not expressly granted are reserved"

Acceptance is **browsewrap**: "By using the Website, you accept and agree to be
bound and abide by these Terms of Use."

**Stüssy**:

> "The Content of this Site is for personal use only. Permission is granted to
> electronically copy and to print in hard copy portions of this Site **for the sole
> purpose of placing an order with Stussy**."

> "You may not copy, modify, upload, download, transmit, (re)publish, create
> derivative works from, sell, re-sell or otherwise distribute or exploit any Content
> from the Site except as expressly permitted"

Stüssy has no scraping clause at all; its restriction is purely on the content.

**Three things follow.** First, the merchant's terms and the platform's `agents.md`
**flatly contradict each other** on the same URL — Shopify says the product JSON is
crawlable, the merchant says no robots. Second, these are browsewrap, which US
courts have repeatedly found unenforceable absent actual or inquiry notice — but
"unenforceable against a casual visitor" is a much weaker proposition once a
defendant is shown to have read the site systematically. Third, and most
importantly: **the Taylor Stitch clause forbids "download, store ... publicly
display" independently of any scraping clause.** That restriction tracks copyright,
and copyright does not need a contract to work.

### 5.3 Copyright is the real exposure, and it is stage-dependent

Product photographs are original works fixed in a tangible medium. They are
copyrighted from the moment of creation, owned by the brand or the photographer, and
**no licence is granted by the fact that a server will hand them to you.** Public
accessibility is not a licence. This is the same failure mode the project already
has with its existing corpus.

The four stages have genuinely different risk:

| Stage | Right implicated | Realistic exposure |
|---|---|---|
| **(a) Fetching the JSON** | None — facts, titles, prices, and URLs are largely uncopyrightable data; *Feist* denies protection to unoriginal compilations | **Low.** The main theories are contract (§5.1–5.2) and CFAA (§5.4), both weak here |
| **(b) Downloading and storing the images** | **Reproduction** (17 U.S.C. §106(1)) — a verbatim copy of a protected work | **Real.** This is a *prima facie* infringement the moment the file lands on disk; the defence is fair use, which is fact-bound and unresolved for this purpose |
| **(c) Training/fine-tuning on them** | Reproduction, plus the transformative-use question | **Real and actively litigated.** See §5.4. The acquisition method has turned out to matter more than the training |
| **(d) Displaying them in a shipped iOS app** | **Public display** (§106(5)) and likely **distribution** (§106(3)) | **Highest.** Commercial, non-transformative, visible to the rights-holder and their customers, and trivially discoverable |

**The server test matters here and is a partial mitigation.** *Perfect 10 v.
Amazon.com* (9th Cir. 2007) held that a party that merely in-line-links or frames an
image stored on someone else's server does not itself "display" a copy, because it
does not store the image. Hot-linking `cdn.shopify.com` URLs rather than rehosting
therefore sits on much better ground for stage (d) than serving your own copies —
and conveniently it also eliminates the storage bottleneck in §4. **But**: the
server test is Ninth Circuit law and has been rejected or narrowed elsewhere
(notably in the Southern District of New York in the *Goldman v. Breitbart* line),
(reaffirmed in *Hunley v. Instagram*, 83 F.4th 1075 (9th Cir. 2023)) — but it was
**already rejected in the Second Circuit** (*Goldman v. Breitbart News*, 302 F. Supp.
3d 585 (S.D.N.Y. 2018)).

**And a warning that must be resolved before relying on it.** A second researcher
reports a **Fifth Circuit decision of approximately 27 August 2026** siding with a
news aggregator, which **rejected the server test in favour of a "transmission
test"** and separately examined whether URLs themselves are copyright management
information under § 1202. **They could not name the case** — the legal blogs
covering it (Copyright Lately, Plagiarism Today) bot-blocked retrieval — but list
coverage in the ABA Journal (Aug. 28), Courthouse News (Aug. 27), Law360 (Sept. 3)
and Davis Wright Tremaine (Sept. 10). **[UNVERIFIED — a human should spend five
minutes naming it.]** If it is what it appears to be, hot-linking is now
*affirmatively rejected* in a second circuit, not merely unadopted.

**Three further limits on the mitigation**, all of which narrow it considerably:
it addresses **only the display right** and does nothing for stages (b) and (c),
where you hold a copy; an iOS app ships nationwide and internationally, and the
UK/EU have no server test at all (the CJEU's *Svensson*/*GS Media* line treats
communication to the public quite differently); and *Perfect 10*'s favourable
fair-use treatment of thumbnails rested on Google being a **transformative index
that sends users to the source** — an app whose product *is* the browsable imagery
is a materially weaker posture, and weaker still after *Warhol v. Goldsmith* (2023)
and its emphasis on commercial substitution. Practically, hot-linking at scale is
also trivially blocked by referrer checks, breaks when a brand rotates a URL, and is
itself typically a ToS breach.

**A second exposure, and it is the sleeper: 17 U.S.C. § 1202 (copyright management
information).** § 1202(b) prohibits intentionally removing or altering CMI, which
§ 1202(c) defines broadly — title, author, copyright owner, **terms and conditions
for use**, and identifying numbers or **links** to such information. That reaches
embedded IPTC credit lines, XMP rights statements and visible watermarks.

**The "identicality" doctrine cuts against a scrape-and-rehost pipeline, not for
it** — this is the opposite of what I first assumed. Courts (*Kirk Kara v. Western
Stone*, 2020; *O'Neal v. Sideshow*, 583 F. Supp. 3d 1282, 1287 (C.D. Cal. 2022);
*Advanta-STAR v. Search Optics*, 672 F. Supp. 3d 1035 (S.D. Cal. 2023)) require the
defendant's copy to be **identical** to the original. That requirement is what has
been **killing § 1202 claims against generative-AI defendants** — a diffusion
model's output is a derivative, not a copy, so identicality fails (it is why Judge
Orrick dismissed § 1202 with prejudice in *Andersen*). **But a pipeline that
downloads the source JPEG and redistributes it satisfies identicality.** The doctrine
shielding the AI companies does not reach this architecture.

**Why it is exposure *beyond* § 106, and the second reason is the important one:**

1. **Damages stack.** § 1203(c)(3)(B) provides statutory damages of **"not less than
   $2,500 or more than $25,000" per violation**, on top of § 504.
2. **§ 412 does not gate Chapter 12.** The timely-registration prerequisite applies
   to §§ 504 and 505 only. So **a brand whose photography was never registered — most
   small DTC brands, exactly the population this channel targets, and precisely the
   case where you would assume exposure was minimal — still has a live § 1202(b)
   theory at $2,500 minimum per violation.** This is the one exposure that survives
   where the copyright arithmetic looks survivable.

**The real defences are scienter and absence, not identicality.** *Stevens v.
CoreLogic*, 899 F.3d 666, 674 (9th Cir. 2018) is squarely on point — software that
stripped EXIF metadata from real-estate **photographs**, and the Ninth Circuit
affirmed summary judgment for the defendant because there was no evidence it knew
removal would "induce, enable, facilitate, or conceal" infringement; a "general
possibility" of downstream infringement is insufficient. **Double scienter is
frequently case-dispositive.** And there may simply be no CMI to remove: most
e-commerce pipelines strip EXIF/IPTC on upload and again on CDN re-encode
*[UNVERIFIED for Shopify]*. **The cheapest diligence in this entire document: run
`exiftool` across a sample of the existing corpus.** That single test determines
whether this exposure is real or nil.

**One hard line:** *Bain v. Film Independent* (C.D. Cal. 2018) found **watermark
removal** sufficient to plead § 1202, and *Logan v. Meta*, 636 F. Supp. 3d 1052,
1064 (N.D. Cal. 2022) distinguished deliberately "editing a plaintiff's watermark
out of a photo" from incidental omission. **If any pipeline step crops or inpaints
visible watermarks or brand overlays, it crosses from the defensible side of that
line to the indefensible one** and the scienter defence gets much harder. Do not
build a watermark-removal step. (Note this repo already runs SAM2 cropping and
segmentation over product photos — worth checking that nothing in that path removes
brand overlays.)

**The asymmetry worth internalising:** stages (a) and (b) are invisible to the
rights-holder. Stage (d) is not. A brand discovers this the first time someone
searches their product in your app.

### 5.4 The case law

**Source labels in this section:** *[PRIMARY]* = the opinion or statute text was
pulled this session. *[SECONDARY]* = an authoritative report (law-firm alert, law
review, court docket coverage), not the primary document. *[UNVERIFIED]* = could not
confirm. This section was researched by a second agent with source access; I have
not personally re-read each opinion.

**The access cases — do they let you fetch it?**

- **hiQ Labs v. LinkedIn**, 938 F.3d 985 (9th Cir. 2019), GVR'd after *Van Buren*,
  on remand **31 F.4th 1180 (9th Cir. Apr. 18, 2022)** *[PRIMARY for 2022]*. Held
  that scraping public pages likely is not CFAA "access without authorization":
  "a defining feature of public websites is that their publicly available sections
  lack limitations on access… that computer has erected no gates to lift or lower
  in the first place." **Three limits that matter:** it was a *preliminary
  injunction* on a "serious questions" standard, not a merits ruling; footnote 17
  expressly preserves password-protected areas; and the court listed the theories
  that survive — "copyright infringement, misappropriation, unjust enrichment,
  conversion, breach of contract."
  **hiQ then LOST.** Summary judgment Nov. 4, 2022 that it breached the User
  Agreement, and a **December 2022 consent judgment: $500,000, a permanent
  injunction, and deletion of all source code, data and algorithms derived from
  LinkedIn** *[SECONDARY]*. **That remedy is the one to internalise — a court
  ordered the derived corpus destroyed.** The CFAA win did not save the data.
- **Van Buren v. United States**, 593 U.S. 374 (2021) *[PRIMARY]*. "Gates-up-or-down
  inquiry — one either can or cannot access a computer system." **But footnote 8
  leaves the decisive question open**: "we need not address whether this inquiry
  turns only on technological (or 'code-based') limitations on access, or instead
  also looks to limits contained in contracts or policies." So whether a ToS clause
  can itself be a "gate" is **undecided**. This is still the reasoning behind §3's
  recommendation not to lift storefront tokens — a code-based gate is a gate on
  anyone's reading.
- **Meta v. Bright Data**, No. 3:23-cv-00077-EMC (N.D. Cal. Jan. 23, 2024)
  (Judge **Chen**) *[SECONDARY]*. **Meta lost** on summary judgment on breach of
  contract: Bright Data scraped **logged off**, and "The Facebook and Instagram
  Terms do not bar logged-off scraping of public data." Meta waived appeal.
  **Implication for this project: someone who never held an account is in a
  *stronger* position than Bright Data was, not weaker.**
- **Ryanair v. Booking Holdings**, No. 1:20-cv-01191 (D. Del.) *[SECONDARY]* —
  **I had this wrong in an earlier draft and it is worth stating the correction
  plainly.** The July 25, 2024 jury verdict for Ryanair under § 1030(a)(4) awarded
  exactly $5,000 (the statutory floor), and **was then overturned**: the court
  granted Rule 50(b) judgment as a matter of law in late January 2025 for failure to
  prove $5,000 in "loss" *[exact date UNVERIFIED]*. Appeal noticed Feb. 2025; **no
  Third Circuit decision as of Sept. 2026.** The distinguishing fact also matters:
  the conduct was accessing the **password-protected** "myRyanair" portal — squarely
  inside hiQ's footnote 17, and nothing like an open `/products.json`.

**The copyright cases — do they let you keep it and train on it?**

- **Thomson Reuters v. ROSS Intelligence**, 765 F. Supp. 3d 382 (D. Del. Feb. 11,
  2025) (Bibas, J.) *[SECONDARY — the researcher could not obtain the opinion text;
  every retrieval route 403'd]*. Fair use rejected; factors 1 and 4 controlling.
  **Limits are severe**: non-generative system, the copied material was editorial
  *headnotes*, and ROSS was a direct head-to-head competitor. **On appeal**: Third
  Circuit granted interlocutory review June 2025, **argued June 11, 2026, no
  decision as of Sept. 15, 2026.** It will be the first federal appellate ruling on
  AI training and fair use.
- **Bartz v. Anthropic**, No. 3:24-cv-05417-WHA (N.D. Cal. June 23, 2025) (Alsup),
  Dkt. 231 *[PRIMARY]*. **A three-way split.** Training on *lawfully acquired* books
  was fair use — "the 'purpose and character' of using works to train LLMs was
  transformative — spectacularly so." Print-to-digital format shift, fair use.
  **Pirated library copies, not fair use:**

  > "Such piracy of otherwise available copies is inherently, irredeemably
  > infringing even if the pirated copies are immediately used for the
  > transformative use and immediately discarded."

  And: "Pirating copies to build a research library without paying for it, and to
  retain copies should they prove useful for one thing or another, was its own use —
  and not a transformative one."

  **This is the single most important passage in this document for this project.**
  It describes building a corpus of unlicensed copies and keeping it in case it
  proves useful — which is what `apparel_dataset` is. **A future ruling that image
  training is fair use would not cure a corpus that was unlawfully acquired.** The
  case settled for **$1.5bn, ~$3,000/work across ~500,000 works, final approval
  July 20, 2026** — but note a settlement **prices** the piracy, it is not a holding.
- **Kadrey v. Meta**, No. 3:23-cv-03417-VC (N.D. Cal. June 25, 2025) (Chhabria)
  *[SECONDARY]*. Summary judgment for Meta on fair use — **for the 13 named
  plaintiffs only**, not a class ruling, and the judge said so in terms: "This ruling
  does not stand for the proposition that Meta's use of copyrighted materials to
  train its language models is lawful. It stands only for the proposition that these
  plaintiffs made the wrong arguments and failed to develop a record." He also wrote
  that "in many circumstances it will be illegal to copy copyright-protected works to
  train generative AI models without permission." **Note the split, and do not
  harmonise it**: Alsup treated an "explosion of competing works" as outside the
  Copyright Act's concern; Chhabria, three days later in the same courthouse, called
  that market dilution the most promising theory. Unresolved.
- **Getty Images v. Stability AI (UK)**, **[2025] EWHC 2863 (Ch)** (Smith J.,
  Nov. 4, 2025) *[SECONDARY]*. **Getty abandoned its primary training and output
  copyright claims mid-trial** because it could not prove training occurred in the
  UK — **a jurisdictional and evidentiary failure, not a ruling that training is
  lawful.** Anyone citing this case for "AI training is lawful in the UK" is wrong.
  Stability won on secondary infringement on the facts (weights "are purely the
  product of the patterns and features which they have learnt"; the model "does not
  store any of those Copyright Works"). Getty won narrowly on early watermark
  outputs, described as "historic and extremely limited."
  **US case**: reportedly refiled N.D. Cal. Aug. 2025, No. 3:25-cv-06891, 7,216
  registered images. **No US merits ruling on fair use, no trial date.**
- **Andersen v. Stability AI**, No. 3:23-cv-00201-WHO (N.D. Cal.) *[SECONDARY]* —
  **the image case to watch.** The Aug. 12, 2024 order left **direct infringement,
  induced infringement and Lanham Act trade dress alive**, and **dismissed the DMCA
  § 1202 claim with prejudice**. Orrick accepted as plausible that works "may be
  contained in Stable Diffusion as algorithmic or mathematical representations" —
  which **directly collides** with Smith J.'s contrary finding of fact in the UK.
  No class certified, no summary judgment, **no fair use ruling**. *[Trial date
  UNVERIFIED — three inconsistent dates in circulation; do not state one.]*
- **Doe v. GitHub**, No. 24-7700 (9th Cir.) *[SECONDARY]*. **Argued Feb. 11, 2026;
  no published opinion as of Sept. 15, 2026.** Below, the § 1202(b) claims were
  dismissed on the holding that such claims "require that copies be 'identical'" —
  see §5.3, where that doctrine cuts the *opposite* way for a rehost pipeline.
  *Warning: a summary in circulation claiming the Ninth Circuit "affirmed summary
  judgment" here is a conflation with* Stevens v. CoreLogic. *It has not ruled.*

**On image training specifically — a negative finding, phrased as one.** The
researcher found **no US decision reaching fair use on the merits for training a
text-to-image model**, and the UK's only trial never reached the training question.
Every merits fair-use ruling to date (ROSS, Bartz, Kadrey) concerns **text**.
Additionally, and more safely stated: **no US appellate court has ruled on AI
training and fair use at all.** This is an absence-of-evidence claim from a
systematic search, not a proven negative.

**The honest synthesis:** fair use for AI training on images is **unresolved**,
### 5.5 EU, UK and Japan — NOT verified this session

- **EU DSM Directive 2019/790, Article 3** permits text-and-data mining by research
  organisations and cultural-heritage institutions for scientific research, and
  cannot be contracted out of. **A solo commercial developer does not qualify.**
- **Article 4** provides a broader TDM exception that *does* cover commercial use —
  but it is subject to a **reservation of rights**: it applies only where the
  rights-holder has not expressly reserved use "in an appropriate manner, such as
  machine-readable means for content made publicly available online." **A merchant
  ToS clause forbidding reproduction, or a `robots.txt`/TDM opt-out signal, is
  intended to be exactly such a reservation.** Taylor Stitch's "must not …
  reproduce … download, store" clause (§5.2) is a plausible reservation, though
  whether a human-readable ToS satisfies "machine-readable" is contested.
  Article 4 also requires lawful access to the content in the first place.
- **EU AI Act GPAI copyright obligations** (applicable from August 2025) require
  providers of general-purpose AI models to have a copyright policy and respect
  Article 4 reservations. Whether they reach a solo developer fine-tuning an
  existing retrieval encoder is doubtful — a fine-tuned SigLIP-2 is very unlikely to
  be a "general-purpose AI model" — but the *Article 4 reservation* obligation
  applies regardless of the AI Act.
- **UK**: the TDM exception is **non-commercial research only** (CDPA s.29A), which
  does not cover this project. The 2025 Data (Use and Access) Act process
  conspicuously did **not** deliver the broad commercial TDM exception that was
  consulted on.
- **Japan, Article 30-4**: unusually permissive — copying for the purpose of
  information analysis is broadly allowed, including commercially, subject to a
  carve-out where it would unreasonably prejudice the copyright owner's interests.
  Relevant only if the processing genuinely happens there.

**Practical upshot:** there is no jurisdiction on this list where mass-harvesting
copyrighted product photography, storing it, and **displaying it in a commercial
app** is clearly permitted. The EU/Japan TDM exceptions, at their most generous,
reach the *training* step — not the display step.

### 5.6 The display problem, and what the licensed path looks like

Stage (d) is the one that ends projects, and it is the one this project is heading
toward, because outfit search is the shipped product (`ios/Fashion`, commit
`4956cf7`).

**Apple App Store Review Guideline 5.2**, quoted in `docs/scaling_licensed_channels.md`
§8.3:

> **5.2.1** "Don't use protected third-party material such as trademarks, copyrighted
> works, or patented ideas in your app without permission … Apps should be submitted
> by the person or legal entity that **owns or has licensed** the intellectual
> property and other relevant rights."

> **5.2.2** "If your app uses, accesses, monetizes access to, or displays content
> from a third-party service, ensure that you are specifically permitted to do so
> under the service's terms of use. **Authorization must be provided upon request.**"

"Authorization must be provided upon request" is the operative sentence. It does not
require a brand to sue. **It requires you to produce a document, and a
mass-harvested Shopify corpus has no document to produce.** A single rights-holder
complaint to Apple is enough to trigger the request, and the practical outcome of
failing it is removal, not litigation.

DMCA takedown exposure runs the same way: cheap for the complainant, immediate,
and it does not care whether you would have won a fair-use argument.

**Why a brand can afford to come after this, which is the part people skip.**

- **17 U.S.C. § 504(c)** *[PRIMARY]*: statutory damages run **"not less than $750 or
  more than $30,000" per work infringed**, electable any time before final judgment,
  and a court **may increase to $150,000 for willful** infringement. (The "innocent
  infringer" floor of $200 requires that the infringer "was not aware and had no
  reason to believe" the acts infringed — realistically closed to someone who wrote a
  scraper and commissioned a legal memo.) **Plan against $750–$30,000 per work, with
  $150,000 live on willfulness.**
- **17 U.S.C. § 412** *[PRIMARY]* **is the whole game.** No statutory damages **and no
  attorney's fees** for infringement of a published work commenced before the
  effective date of registration (unless registered within three months of first
  publication). **Without** timely registration a brand is limited to actual damages
  and profits under § 504(b) — against a pre-revenue app, close to nothing, and it
  must fund the case out of pocket. **With** it, statutory damages **plus fees** make
  the case fundable on contingency. **Fees are the mechanism, not the damages.**
- **Group registration** *[SECONDARY]*: Copyright Office Circular 42 — "The group must
  include no more than 750 photographs" — at a **$55** filing fee. **A brand registers
  750 product photos for $55 as routine seasonal hygiene.** That $55-per-750 against
  $30,000-plus-per-work asymmetry is the most useful number in this risk analysis.
- **Contested, and it swings the arithmetic by orders of magnitude:** whether each
  photo in a group registration counts as a separate "work" under § 504(c). The
  prevailing view is that separately-created photographs remain separate works, but
  there is contrary authority in the Second Circuit's *Sohm v. Scholastic* line. Even
  the defendant-favourable reading leaves one award **per registration group**.

**And the DMCA safe harbour is structurally unavailable.** § 512(c) protects storage
"**at the direction of a user**" — scraped images the operator selected are the
operator's own editorial choices; **there is no user.** *Mavrix Photographs v.
LiveJournal*, 873 F.3d 1045 (9th Cir. 2017) denied the harbour where volunteer
moderators merely screened and approved third-party posts; *a fortiori*, selection by
your own scraper is not close. Two further gates fail independently: § 512(c)(2)
requires a **designated agent registered with the Copyright Office** (skipping it
forfeits the harbour categorically), and § 512(i)(1) gates all of § 512 on having a
repeat-infringer policy **and** on not interfering with "standard technical measures"
— which sits in direct tension with bypassing robots.txt, rate limits or bot
defences. § 512(d) covers linking, not hosting copies, so it does not rescue a stored
index either.

**A practical trap:** do **not** file a § 512 counter-notification on scraped
content. § 512(f)(2) creates liability for knowingly misrepresenting that material
was removed by mistake — it manufactures a fee-shifting claim against yourself.

**What the licensed path actually is**, and this is the part that should temper any
enthusiasm: a parallel investigation in this repo,
**`docs/scaling_licensed_channels.md`** (2026-09-15), surveyed affiliate networks
(ShareASale, Rakuten, CJ, Awin, Impact), Google Merchant Center, data resellers,
syndication networks, open stock libraries and academic datasets. Its conclusions,
which I have not re-verified but which are directly on point:

- Affiliate feeds grant "a display licence that is revocable, merchant-scoped,
  modification-barred and ML-prohibited — a marketing permission, not a data
  licence, with a kill switch someone else owns."
- **The single best licensed option found is Amazon Berkeley Objects**: 25,916
  fashion listings, 58,927 images, 165 brands, CC BY 4.0 — and **zero garments**
  (footwear, bags, jewellery only). There is an unresolved CC BY vs CC BY-NC
  contradiction between Amazon's README and the AWS Open Data registry.
- "No channel found here replaces the 20,532 outfit photos."

**Hold those two numbers next to each other.** This channel reaches **~650M image
URLs with no licence**. The best licensed channel reaches **58,927 images, none of
them garments**. That gap is the actual strategic problem, and no amount of
Shopify engineering closes it.

**The one architecture that meaningfully narrows the gap** is the *Perfect 10*
server-test route (§5.3): **store the metadata and the CDN URLs, never the pixels,
and hot-link `cdn.shopify.com` at display time.** It removes stage (b) entirely, it
removes stage (d)'s "distribution" limb and puts stage (d)'s "display" limb on the
best available footing in the Ninth Circuit, and — see §4.2 — it also happens to
delete the storage bottleneck, since 1,118 bytes/product means 100M products fit in
~107 GiB while the pixels would need ~72 TiB. It does **not** fix stage (c): you
still hold copies while training. And it introduces real product risk (dead links,
the brand can cut you off) and a non-trivial legal one outside the Ninth Circuit.

---

## 6. Bottom line

**Can the generic Shopify pattern be a mass-harvest channel?** Technically, yes,
and more easily than the question assumes. It is one HTTP request per 250 products,
no auth, no bot protection on two-thirds of stores directly and ~85% including the
`*.myshopify.com` fallback, a stable 13-key schema identical across every store, a
documented `?width=` resizer, no observed rate limiting, and a discovery problem
that is already solved for between $0 and $250. `champion_scraper.py` is ~90% of
the code; what it needs is a store list and a fallback chain.

**What would it actually yield?**

- **Measured:** one mid-size brand (Taylor Stitch) = **3,813 products / 29,881
  images in 17 requests** — 1.48x the entire current corpus.
- **Measured:** 6.51 images/product, 1,118 bytes/product of trimmed metadata,
  8,997 images/GiB at `width=768`.
- **Estimated ceiling:** ~690,000 reachable apparel stores, ~100M products,
  **~650M image URLs** — not deduplicated, and heavily diluted by dropshippers
  reselling identical supplier photography.

**Where the bottleneck is:** **storage and bandwidth, decisively.** Discovery is
solved and nearly free. Rate limits were not observable. The metadata for
everything costs ~107 GiB. The pixels cost **~72 TiB** — against **1.6 GiB of free
local disk**, which holds about **14,400 images**, roughly half of one brand. The
local machine cannot participate in this at all; anything real goes to the Modal
volume this repo already uses.

**Where the risk sits, honestly.** Reading `/products.json` is about as
well-positioned as unlicensed scraping gets — Shopify's default `robots.txt` now
says product data is crawlable, `agents.md` documents the endpoint as "Read-Only
Browsing (No Authentication Required)", and no `Disallow` covers it. **That is the
end of the good news.** Merchant terms contradict the platform on the same URL
(Taylor Stitch forbids robots *and* forbids "download, store … publicly display").
The Shopify API ToS, which unambiguously governs the tokenless GraphQL fallback,
prohibits building "any commerce or product index" (§2.3.14) and prohibits using
API-derived information to "train, fine tune, or improve any machine learning"
system (§2.3.24) — a description of this project's plan. And underneath all of it,
**every one of those ~650M photographs is somebody's copyrighted work, and the
endpoint being open grants no licence whatsoever.** This channel would take the
project's existing licensing problem — 0 of 20,532 photos licensed — and multiply
it by four orders of magnitude.

Three specific things sharpen that, all from §5:

- ***Bartz*** holds unlawfully acquired copies "inherently, irredeemably infringing
  even if … immediately used for the transformative use and immediately discarded."
  **A future ruling that image training is fair use would not cure the corpus.** The
  acquisition and the training are separate questions, and this project is exposed
  on the first one regardless of how the second resolves.
- **hiQ won the CFAA point and still ended up with a $500,000 consent judgment, a
  permanent injunction, and an order to delete all data and algorithms derived from
  LinkedIn.** "The endpoint was public" is not a defence to the claims that actually
  get brought.
- **§ 1202 is the sleeper.** $2,500–$25,000 per violation, stacking on § 504, and
  **it is not gated by registration** — so even brands that never registered their
  photography have a live claim against a rehost pipeline. One `exiftool` run over
  the existing corpus tells you whether that is real.

**It is not fine.** If it ships images to users, the exposure is real, discoverable,
and does not require anyone to sue: Apple's 5.2.2 only requires a rights-holder to
ask you for authorization you do not have.

**What I would actually recommend, in order:**

1. **Harvest metadata, not pixels, and do it from `/products.json` only** — never the
   GraphQL endpoint, whose governing ToS names this exact use. 1,118 bytes/product
   means the index is essentially free and never touches the storage constraint.
   Product titles, `body_html`, `tags`, vendor and price are also the single
   cheapest fix available for the taxonomy work, independent of any image.
2. **Decide the display question before downloading anything.** If images will be
   shown in the app, hot-link the CDN (§5.6) or licence them. If they are
   training-only and never leave the machine, the risk profile is different — but
   §2.3.24 and the unsettled state of AI-training fair use still sit on it.
3. **Do not harvest storefront tokens** on BigCommerce, Shopware or SFCC (§3). Those
   platforms put up a gate; walking through it converts a weak contract argument
   into a plausible CFAA one, for perhaps 25K additional stores you do not need.
   Note *Van Buren* footnote 8 leaves open whether contract terms alone can be a
   "gate" — but a token is a *code-based* gate on anyone's reading, and every
   scraper-loss in §5.4 (hiQ, Ryanair) involved credentialed or logged-in access.
   Likewise: **do not add a watermark-removal step** anywhere in the pipeline (§5.3),
   and **run `exiftool` over the existing corpus** before anything else — it is a
   five-minute test that determines whether the § 1202 exposure is real or nil.
4. **Size the ask to the actual need.** ~900,000 images at `width=768` fits in
   100 GiB on the Modal volume and could cover ~1,000 brands — a **44x corpus
   increase and a 50x brand-diversity increase**. The brand diversity is what would
   move retrieval metrics; the raw image count is not. Chasing 650M is pointless.
5. **Ask Shopify the UCP question before writing any scraper (§1.10).**
   `catalog.shopify.com/api/ucp/mcp` is Shopify's own cross-merchant product search
   and it returns images. Its 1,000-result pagination depth means it will never build
   a 650M-image corpus, and **registering an agent profile makes §2.3.14/§2.3.24
   unambiguously binding on you** — so it is not a back door to training data. But it
   is the only route found here that could end with an answer to Apple's
   "authorization must be provided upon request." **The question to put to Shopify in
   writing is narrow: does Global Catalog access carry any licence to display
   merchant imagery?** One email, and it is the highest-value unresolved question in
   this document.
6. **Read `docs/scaling_licensed_channels.md` first.** It concludes the licensed
   world offers 58,927 images and zero garments. That is the real constraint this
   project is under, and this document does not relieve it.

---

## Appendix: incidental finding

While measuring the existing corpus I found that **7,180 of the 20,205 image paths
in `apparel_dataset/metadata.json` (35.5%) point into a `shoe_dataset/` directory
that does not exist in the repo** **[M]**. Affected brands: nike (2,280), adidas
(1,357), gap (1,051), newbalance (1,049), skechers (924), pacsun (519). On-disk
image files total 15,974 (2.62 GiB, mean 172 KB, median 133 KB). Unrelated to this
research, but worth knowing before anyone quotes "20,205 images" as an on-disk
figure.
