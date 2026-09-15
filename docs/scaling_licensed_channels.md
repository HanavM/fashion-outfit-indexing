# Scaling the corpus with licensed imagery — what the channels actually are

**Written 2026-09-15.** Companion to `docs/licensing_decision_memo.md` (2026-09-11),
which established that **0 of 20,532 outfit photos carry any licence** and that the
catalog's 20,205 images were scraped brand-by-brand with none either. That memo asked
"what do we do with what we have". This one asks the other question: **where do you get
fashion imagery at volume with rights you can actually point at?**

**I am not a lawyer and none of this is legal advice.** Every licence claim below is a
quote from a document with a URL. Where a document could not be read, that is recorded
as a finding, not smoothed over. Section 8 lists what still needs a lawyer.

**Method.** Four parallel research passes (affiliate networks, merchant feeds and data
resellers, open-licence imagery, academic datasets) plus direct measurement. Numbers
marked **measured** were computed here on 2026-09-15 against the live source. Numbers
marked *estimate* are estimates. The session's 200-call web-search budget was exhausted
partway through; the later work is direct URL fetches only, and a few items are marked
explicitly unverified.

---

## 1. The one-paragraph answer

The market you were hoping for does not exist. **Nobody sells a solo developer a
fashion garment catalogue with images and a commercial display licence, at any price.**
The commercial data industry sells metadata plus image URLs and contractually disclaims
the image rights — Zyte says it outright: *"The Services do not provide any copyright or
permissions to you with regard to use of the Service Data."* The academic world is worse:
almost every fashion dataset was built by scraping retailers, so the group that published
it never held the copyright it would need to grant you anything, and several say so in
their own terms. Affiliate networks are the only channel that grants an image *display*
licence, and every one of them scopes it to "solely for promoting this merchant",
makes it revocable, forbids modification, and none of them permits ML training.

The open stock libraries do not rescue it either, and this is the correction that matters
most to the existing memo. **Pexels, Unsplash and Pixabay each prohibit compiling their
content into a similar or competing image service** — an uncomfortably fair description of
an image-search app. Unsplash goes further and excludes *"People's images if they are
recognizable"* from its licence grant entirely, which for outfit photography removes the
use case. Pexels and Pixabay ban *acquisition* for ML by name while staying silent on
use-for-ML, which is ambiguity of the worst kind rather than permission.

Against that, **one thing is genuinely, unambiguously available and free**: Amazon
Berkeley Objects, CC BY 4.0, image files included, published by the rights holder.
Measured here today: **25,916 fashion listings / 58,927 unique catalog images / 165
brands** — 2.9× the image count of the current scraped catalog, with a licence. It
contains **zero garments**. That is the shape of the entire finding: the rights-clean
options are real but they are the wrong shape, and no amount of budget changes that.

The one strategy that does not rest on an unsettled legal question is a **CC0-only
corpus** — Smithsonian, Europeana's CC0 slice, the Met's CC0 subset, Commons' CC0 slice,
and Pixabay's pre-2019 content — roughly **100k–150k images** (*estimate*). CC0 makes the
two genuinely unanswerable questions in this document (are model weights a derivative work;
is a SAM2 garment crop an Adapted Material) moot instead of a bet.

---

## 2. Ranked comparison

Ranked by (volume × rights clarity) / cost, for *this* project — a solo developer, no
traffic, no live app, near-zero budget. "Train?" means: does the licence permit using
the images to fine-tune an embedding model. "Display?" means: may you show them to users
in a commercial iOS app.

| # | Channel | Fashion images reachable | Cost | Display? | Train? | Host copies? | Open to a solo dev *now*? |
|---:|---|---:|---|---|---|---|---|
| **1** | **Amazon Berkeley Objects** (CC BY 4.0) | **58,927** measured (footwear/bags/jewellery; **0 garments**) | **$0** | **Yes** | **Yes** | **Yes** | **Yes — download today** |
| **2** | **Fashionpedia clean subset** (CC BY 4.0 + stock) | **~8,500 clean / ~15,600 permissive** of 45,632 train | $0 | Yes, filtered | Yes, filtered | Yes | **Yes — S3, no gate** |
| 3 | **Flickr** (CC commercial+mod) | **214,250** "fashion", 23,887 "outfit" (measured); 173.5M CC commercial-use images library-wide | $0 | Yes | **No express position** | **No — "cache … other than for reasonable periods"** (24h) | Yes, but commercial key is separate |
| 4 | **Wikimedia Commons** | **≥271,677** `deepcat:Clothing` (measured lower bound); 71,985 at depth 3 | $0 | Yes | **Yes** | **Yes** | Yes — but see §5.5 |
| 5 | **Smithsonian Open Access** | **27,073 textile / 23,384 costume / 21,460 dress** CC0 (measured; overlapping, est. 40–60k distinct) | $0 | **Yes, CC0** | **Yes** | **Yes** | **Yes — cleanest terms found** |
| 6 | **Europeana** (open + media) | **84,063** "fashion", 110,489 "textile" (measured); 43.1% CC0, 38.3% CC BY | $0 | Yes | Yes | Yes | Yes — but Europeana hosts nothing |
| 7 | **Met Museum Open Access** | ≈8,000–11,000 CC0 (two samples, n=60 → 35%, n=76 → 25%) | $0 | CC0 subset yes | Yes | Yes | Yes |
| 8 | **Pixabay pre-2019 CC0 slice** | subset of 83,514 "fashion" (measured); slice size unmeasured | $0 | **Yes, CC0** | **Yes, CC0** | **Yes** | Yes — filter on published date |
| 9 | **Burst by Shopify** | **~1,429** fashion (measured) of 20,000+ | $0 | Yes | Silent | Yes | Yes |
| 10 | **Pexels API** | **unmeasured** — no counts exposed without a key | $0 | Yes, but see §5.3 | **Acquisition-for-ML banned; use silent** | **No — ToS bars bulk copying** | Yes, free key |
| 11 | **Pixabay API** (post-2019) | **83,514** "fashion", 27,897 "dress" (measured) | $0 | Yes | **Acquisition-for-ML banned; use silent** | **Required** — hotlinking forbidden | Yes, but 500-result/query cap |
| 12 | **Unsplash** | **95k** "fashion", 41k "outfit" (measured) | $0 | **Licence excludes recognisable people** | **No — the *use* is banned by name** | **No — API mandates hotlinking** | Yes (demo tier) |
| — | **Openverse** | **unmeasurable** — results hard-clamped at 240 | $0 | indexes only, warrants nothing | n/a | n/a | Not a pipeline (20/min, 200/day, 1 success in ~29 reqs) |
| 13 | **Sovrn //Commerce** affiliate | merchant catalogues, no bulk feed | $0 | Yes, programme-scoped | **No** | Hotlink | **Yes — best affiliate odds** |
| 14 | **Impact.com** affiliate | Catalogs API + FTP bulk | $0 | Yes, revocable at will | **No** | Programme-scoped | Yes, ~2 days |
| 15 | **Awin** (inc. ex-ShareASale) | genuinely good bulk feeds | $1 US / £5 UK deposit | Yes, **"without modification"** | **No — §10.7 explicit** | Programme-scoped | Probably, network level |
| 16 | **Rakuten Advertising** | SFTP full+delta, ≥600×600, 10 imgs/SKU | $0 | Advertiser's terms | No | Programme-scoped | Needs a live website |
| 17 | **Synthetic generation** | unbounded | ~$5–$67 per 1,000 imgs, or ~$0 local | Yes | Yes | Yes | Yes — see §7 |
| 18 | **Datafeedr** | ~1bn products / 27k merchants aggregated | **$49–249/mo** | **No — you still need per-merchant approval** | No | No | Yes, but solves the wrong problem |
| 19 | **Icecat** (fashion vertical) | ~2M products | **€650/mo** | **Images excluded from the licence** | **Banned by name** | No | Yes, but pointless |
| 20 | **CJ / Skimlinks / FlexOffers / eBay / Amazon** | large | $0 | Yes-ish | No | No | **No — traffic gates** |
| 21 | **Bright Data / Oxylabs / Apify / ScrapeHero** | large | $19–550 | **No rights conveyed** | Often banned | No | Yes — but you're buying labour, not a licence |
| 22 | **Freepik / Kaboompics** | large | varies | varies | **No — banned by name** | No | Yes, and don't |
| 23 | **DeepFashion / DeepFashion2 / ModaNet / Dress Code / VITON / M5Product** | 800k / 491k / 55k / … | $0 | **No — research only** | **No** | No | Academic email required |
| 24 | **LAION / Re-LAION** | billions of URLs | $0 | **No — clearly infringing** | Legally contested | No | Yes, and you shouldn't |
| 25 | **Kaggle H&M / Myntra / Polyvore / Feidegger / Atlas** | 105k / 44k / 365k / 8.7k / 184k | $0 | **No — licensor never owned the images** | No | No | Yes, and you shouldn't |

**The single best channel for a solo developer: Amazon Berkeley Objects.** It is the
only entry that is simultaneously free, large, image-bearing, downloadable without a
gate, published by the actual rights holder, and licensed for both display and training.
Its fatal limitation is that it has no garments. §4.1 has the measured census.

**The single best *strategy*, which is not the same thing: a CC0/public-domain-only corpus.**
Rows 4–8 — Smithsonian, Europeana's CC0/PDM slice, the Met's CC0 subset, Commons' CC0/PD
slice, and Pixabay's pre-2019 CC0 content — total roughly **100k–150k images** (*estimate*;
the component counts overlap and were measured per-source, not deduplicated). CC0 is the only
licence here under which the two genuinely unanswerable questions in this document — *are
fine-tuned model weights a derivative work?* and *is a SAM2 garment crop an Adapted Material?*
— stop being bets and become moot. The subject matter is still museum-heavy, so this is a
**training** corpus, not a product surface.

---

## 3. Affiliate networks — the most promising lead, and why it does not survive contact

The hypothesis was right in shape: affiliates *are* granted the right to display merchant
product images. The problem is what that grant says.

### 3.1 The clause that appears, in substance, at every network

Every affiliate image licence is an **affirmative grant of a narrow right**, scoped with
words like "solely for the purpose of", revocable, and expiring when you leave the
programme. Primary sources:

- **Awin §10.1** — *"for the duration of its participation in the Advertiser Program, a
  **revocable**, non-exclusive, non-transferable, royalty-free, worldwide sublicense to
  publish Advertiser Materials, **without modification**, on the Publisher Service in the
  Promotional Spaces **to the extent necessary to enable the Publisher to market the
  respective Advertiser**"*
  — [Awin Publisher Terms, Aug 2025 (PDF)](https://www.awin.com/docs.awin.com/Legal/Publisher+Terms/2025/EN-(US)_Awin-Ltd-Publisher-terms_August-2025.pdf)
- **Awin §10.7** — *"The Publisher shall use information and data obtained from and in
  connection with participating in the Network **only for the purpose of this Agreement.
  Uses for any other purpose, or disclosure of such information and data are
  prohibited.**"* §9.3 names a breach of 10.7 as an indemnity trigger.
- **Rakuten §4.1** — licence granted *"solely for the Permitted Purpose"*, where Permitted
  Purpose = *"participation as a Publisher Partner in the Network"*; §4.1.5 reserves all
  rights not granted.
  — [Rakuten Publisher Membership Agreement](https://rakutenadvertising.com/legal-notices/publisher-membership-agreement/)
- **Impact §10.2** — *"for the sole purpose of fulfilling Creator's obligations in
  connection with the Program"*, *"without manipulation of any kind"*. Note the asymmetry:
  Impact takes a *"perpetual, irrevocable"* licence in **your** content (§10.4) while
  granting you a revocable one.
- **CJ §4(a)** — *"for the limited purposes of Promoting the Advertiser's Program"*, and
  you must *"refrain from copying or modifying"*.
- **Partnerize §3.1** — *"solely for the purposes of … operating the Campaigns in which
  Partner participates"*. Uniquely includes the word "reproduce", but still campaign-scoped.

**No network in this list grants a licence covering ML training.** That is unanimous, and
it is structural rather than an accident of drafting — these are contracts written by
people who knew how to write a broad data licence and chose not to. **Silence about AI is
not permission.** Two networks say it outright:

- **eBay Partner Network §9(10)** bars using eBay Content to *"train algorithms, conduct
  machine learning … train artificial intelligence systems"*.
- **Amazon Associates IP Licence §1** — *"You will not, and will not allow any third party
  to, use Program Content to, directly or indirectly, **develop or improve large language
  or multimodal models, machine learning models or related technology.**"*
  — [Associates Programme IP Licence](https://affiliate-program.amazon.co.uk/help/operating/license)

### 3.2 Two structural traps that matter more than which network you pick

1. **Network approval gets you an empty account.** Per-merchant approval is required at
   *every* network — contractually confirmed at Awin (§4.1), Rakuten, CJ (§4(a)), Impact,
   Partnerize (§1.1, §1.3) and FlexOffers (§11). That second gate is exercised at the
   merchant's sole discretion and is exactly where a zero-traffic TestFlight app is
   weakest. Two partial exceptions: Awin's feed list exposes feeds from *"Advertisers that
   allow Partners to promote their products before joining"*, and Skimlinks auto-joins you
   to ~48,500 programmes — but Skimlinks will not approve you in the first place.
2. **"Without modification" defeats the pipeline even for display.** Awin §10.1 and §9.2.11
   (*"accurately and faithfully reproduced"*), Impact §10.2 (*"without manipulation of any
   kind"*) and Apple's Partnerize terms (*"manipulated in any way … is not permitted"*) bar
   the resize / crop / normalise steps any embedding pipeline requires — and arguably bar
   thumbnail generation too. This is the same problem the memo's option (a) has, arriving
   from a different direction.

### 3.3 Three things are dead and change the question

- **Amazon PA-API 5.0 is retired.** Deprecated 2026-04-30, retired 2026-05-15; live calls
  return HTTP 403 *"Product Advertising API is deprecated. Please migrate to Creators
  API."* — [deprecation notice](https://affiliate-program.amazon.com/creatorsapi/docs/en-us/paapiv5-deprecation)
- **ShareASale is gone.** Legacy platform ceased 2025-10-06; ~9,500 advertisers and 250,000
  publishers migrated into Awin. There is no separate ShareASale to join.
  — [Awin announcement](https://www.awin.com/us/news-and-events/awin-news/awin-shareasale-new-era)
- **ShopStyle Collective is gone.** Rebranded Collective Voice, wind-down announced Dec 2025;
  `api-doc.shopstyle.com` no longer resolves. The large ShopStyle product API no longer exists.

### 3.4 Amazon specifically — disqualifying on three independent grounds

Operative document is the **Associates Programme IP Licence**, not the Operating Agreement.

- **Images may not be stored at all.** §2(h): *"**You will not store or cache Product
  Advertising Content consisting of an image**, but you may store a link to Product
  Advertising Content consisting of an image for up to 24 hours."*
- **ML training prohibited by name** (quoted in §3.1 above).
- **Every use must link to Amazon**; no *"aggregating, analyzing, extracting, or
  repurposing"* without written approval; on termination you must *"delete or otherwise
  destroy all of the Content."*
- **Access gate:** Associates approval needs **3 qualifying sales in 180 days**; Creators
  API access needs, per Amazon's own error-code docs, *"10 qualified sales in the trailing
  30 days"*. No traffic → no sales → no API.

### 3.5 What is actually open this month

| Network | Verdict |
|---|---|
| **Sovrn //Commerce** | **Best odds.** Issues API keys *before* approval; test clicks may be self-generated; iOS apps explicitly in scope; 100 req/s lookup APIs returning merchant-CDN full-size images. Their docs state publishers *"are free to use this information for their own affiliate marketing efforts"*. Free. |
| **Impact.com** | Website *"recommended, not mandatory"*; 2-business-day review; Catalogs API **and** `FTPLocations` bulk files (20,000-result pagination cap, 3,000 req/hr). Free. |
| **Awin** | Network-level approval in ~24h for a **$1 US / £5 UK** deposit, refunded on first commission. Genuinely good bulk tooling: Create-a-Feed, feed-list CSV, JSONL API (5 req/min). |
| **Best Buy** | API key by email in minutes — but a **72-hour cache ceiling** kills it, and it sells no clothes. |
| **Rakuten** | Network gate passable, but asks for a **live website**, and an iOS app is a fourth gate with a pre-launch compliance test. |
| **CJ** | Content Certification needs **10K+ pageviews** *and* excludes software/apps. Doubtful. |
| **Skimlinks** | **Blocked.** Needs 10+ live articles and a *"visible, active audience"*; **90-day lockout on rejection**. |
| **FlexOffers** | Largest feed found (**650M+ products**, FTP + API, daily) but requires content *"freely accessed without requiring login"*. Blocked. |
| **eBay Partner Network** | Mobile apps need written approval **plus an App Store download link**. Blocked pre-launch, and bans ML anyway. |

### 3.6 What affiliate feeds would and would not buy this project

They map onto the memo's **option (e), the catalog surface** — `apparel_dataset`, which is
**4,079 products / 20,205 images / 20 brands** (measured: uniqlo 374, nike 319, gap 296,
americaneagle 217, then champion / carhartt / vans / dickies / huf / everlane / jcrew at
200 each). Most of those brands are reachable through Awin or Rakuten.

So affiliate feeds could convert option (e) from *"copyrighted brand imagery we scraped"*
into *"brand imagery we are licensed to display"*. That is real value. They do **not** give you:

- anything at all for the 20,532 outfit photos — a different problem entirely;
- a training corpus — the `encoder_finetune_plan.md` work would sit squarely outside every
  licence here;
- **permanence.** The licence is revocable and dies on exit, so a corpus built on it has a
  kill switch owned by someone else. That is a *worse* property than the memo's option (c).

**Affiliate feeds are a display layer, not a training layer. Source the two separately.**

---

## 4. Direct merchant feeds, aggregators, and who sells catalogues outright

### 4.1 Amazon Berkeley Objects — the one genuine exception, measured

**Verified independently today** against the canonical S3 bucket. The README
(Last-Modified 2023-07-18) states:

> *"This work is licensed under the Creative Commons Attribution 4.0 International Public
> License."* … *"Credit for the data, including all images and 3d models, must be given to:
> Amazon.com"*
> — https://amazon-berkeley-objects.s3.amazonaws.com/README.md

**The licence changed and the secondary sources did not follow.** Bucket timestamps:

```
LICENSE-CC-BY-NC-4.0.txt   Last-Modified: 2021-06-14   (still present, HTTP 200)
LICENSE-CC-BY-4.0.txt      Last-Modified: 2023-06-26
README.md                  Last-Modified: 2023-07-18
```

ABO was released in 2021 under **CC BY-NC 4.0** and relicensed to **CC BY 4.0** in mid-2023.
The NC file was never deleted, and both the
[AWS Registry of Open Data entry](https://registry.opendata.aws/amazon-berkeley-objects/)
and the [CVPR 2022 paper](https://arxiv.org/abs/2110.06199) still say NonCommercial. That
stale pair is exactly what an adversary would point at. **Get written confirmation from
Amazon before shipping on it.**

**Full census, measured 2026-09-15** — all 16 listing shards downloaded and counted
(147,702 listings, 398,170 unique image ids, 576 product types):

| segment | listings | unique images |
|---|---:|---:|
| Footwear (SHOES 12,965 / BOOT 2,009 / SANDAL 1,845) | **16,819** | |
| Jewellery (rings, earrings, necklaces) | **5,736** | |
| Bags, hats and accessories (HANDBAG 792, HAT 655, ACCESSORY 1,362, …) | **3,361** | |
| **Fashion total** | **25,916** | **58,927** |
| **Garments (shirts, trousers, dresses, outerwear)** | **37** — SALWAR_SUIT_SET 26, SWEATBAND 18, TRACK_SUIT 2 | ≈0 |
| Largest single type: `CELLULAR_PHONE_CASE` | 64,853 (44% of the dataset) | |

4.90 images per fashion listing; **165 distinct brands** in the fashion subset. This
resolves an open question in the source research, which could not quantify the apparel
share: **the answer is that there is essentially none.** The CVPR paper says why — the
authors *"focused only on rigid objects, removing items such as garments, home linens, and
some accessories."*

**Against the project's own catalog:** 4,079 products / 20,205 images / 20 brands scraped,
versus 25,916 products / 58,927 images / 165 brands licensed. **6.4× the products, 2.9× the
images, 8.3× the brands, and a licence** — for footwear and accessories only. The repo
already has a `shoe_dataset/` path convention, so the shoe half of this is a drop-in.

### 4.2 Google Merchant Center and Shopify — reachable is not licensed

- The [product data specification](https://support.google.com/merchants/answer/7052112)
  is public; `image_link` and `additional_image_link` are standard attributes. **There is no
  public directory of merchant feeds**, and no Google API exposes *other* merchants' data —
  Content API for Shopping and the successor
  [Merchant API](https://developers.google.com/merchant/api) manage your own products, with
  third-party access strictly per-merchant OAuth delegation.
- The Merchant Program Addendum has merchants grant **Google and its affiliates** *"a
  royalty-free, perpetual, irrevocable, non-exclusive, worldwide license to use Item
  Content."* The merchant licenses Google. Nothing in that structure licenses you.
- **Shopify `/products.json` is real and trivially harvestable** — verified live, HTTP 200,
  no auth, full product objects with image URLs, on any of an estimated ~800k active
  Shopify apparel stores (*estimate*; trackers report 3.06M–4.8M total stores, 17–27.5%
  apparel, and their taxonomies disagree). It grants you nothing, on two independent
  grounds: **Shopify ToS §1.9** — *"You agree not to access the Services or monitor any
  material or information from the Services using any robot, spider, scraper, or other
  automated means"* — and copyright in the photographs, which survives every scraping case
  you might cite. *hiQ v. LinkedIn* held only that scraping public data is not a **CFAA**
  violation; hiQ still lost on breach of contract, and none of that line of cases touches
  photographic copyright.
- **Google sued SerpApi on 2025-12-19** for copyright infringement and DMCA §1201
  circumvention, explicitly citing scraping of Google Shopping merchant-supplied images. The
  DMCA claims were dismissed with leave to amend on 2026-07-20 — unresolved, not won.
- **Shopify Collabs is closed:** *"Collabs isn't accepting new creator signups at this
  time"*. Even when open it required 1,000+ followers, US/CA/UK only, and exposed no bulk
  feed. Dead end twice over.

### 4.3 Data resellers — cheap, self-serve, and they sell you labour, not rights

| Vendor | Cheapest real price | Images? | Rights conveyed |
|---|---|---|---|
| Bright Data | $250 one-time / 100K records; API $499/mo | **URLs only** | None; §3.8.I bars competitive use |
| Oxylabs | $49/mo; $0.40/1K results | unconfirmed | None; resale of Datasets barred |
| Apify | $19/mo + $1–10/1K | varies by actor | None |
| Zyte | $0 PAYG | unconfirmed | **None, stated outright** |
| ScrapeHero | $199/mo or $550 one-time per site | unconfirmed | None; **AI training banned** |
| Rainforest / BlueCart (Traject) | $23 / $18 per mo | URLs only | "personal purposes only" |
| Datafeedr | **$49–249/mo** | hotlinked URLs | **None — you still need your own per-merchant approvals** |

The industry position, stated honestly by Zyte
([ToS §3.3](https://www.zyte.com/terms-policies/terms-of-service/)):

> *"The Services do not provide any copyright or permissions to you with regard to use of
> the Service Data or Data Feeds … it is your sole responsibility to determine the lawful
> use."*

ScrapeHero goes further: *"the data may not be redistributed, sold, provided free of
charge, shared or repackaged"* and *"This data cannot be used for training AI/LLM models."*

**AWS Data Exchange, Snowflake Marketplace and Databricks Marketplace all carry the same
Bright Data "Fashion Listings" product.** These are table shares; the "images" column is a
URL pointing at the retailer's own server. There is no mechanism in any of those
marketplaces to receive, let alone licence, image bytes.

### 4.4 Syndication networks — the only ones with real image rights, and you cannot qualify

**Icecat, Syndigo, Salsify and 1WorldSync** are the only players with a genuine image-rights
framework. The model: a brand grants distribution rights to **named retail trading partners
it actually sells to**. GDSN recipients need a **GS1 Global Location Number** (GS1
membership ~$200+/yr) *plus* a real commercial buying relationship with each brand. A solo
iOS developer cannot realistically qualify.

Icecat is the closest to open and fails on two specific clauses:

> *"The images themselves and other manufacturer or 3rd Party Copyrighted Materials are
> **NOT included** in the Icecat Information and thus **not covered by this License**.
> Download and use thereof are at Client's own risk."*
> — and the Fair Use Policy: *"to **not use Icecat data for AI training**."*
> — https://iceclog.com/open-catalog-interface-oci-open-icecat-xml-and-full-icecat-xml-repositories/

Paid Full Icecat: Starter from €375/mo, **Fashion vertical from €650/mo**, Complete from
€3,750/mo. Whether the paid image licence is broader than the open one is not stated publicly.

---

## 5. Openly licensed imagery at scale

**The headline correction to the memo.** All three obvious candidates — Pexels, Unsplash and
Pixabay — **prohibit compiling their content into a similar or competing image service**, which
is an uncomfortably fair description of an image-search app. On top of that, each blocks the
ML-training use in a different way and to a different degree. The order below is by how
usable each actually is, not by how well-known it is.

### 5.1 The measured volumes

Counts read off each site's own result tallies, 2026-09-15. These are the platforms' numbers,
not a sample.

| query | Flickr (CC comm+mod) | Unsplash | Pixabay | Wikimedia Commons (`deepcat`) |
|---|---:|---:|---:|---:|
| fashion | **214,250** | 95,000 | **83,514** | ≥190,096 |
| clothing | 73,858 | 46,000 | 15,921 | **≥271,677** |
| dress | 55,996 | 32,000 | 27,897 | — |
| street style | 28,429 | 22,000 | — | — |
| outfit | 23,887 | 41,000 | 3,000+ | — |
| streetwear | — | 3,400 | 600+ | — |

- **Pexels exposes no counts at all** without an API key — search pages carry no total, and
  `?page=500` returns nothing (depth-capped). **Treat any Pexels fashion figure as
  unmeasured**, including the project's own ~6,000 from 2026-08-03.
- **Commons `deepcat:` figures are lower bounds** — each returns *"Deep category query returned
  too many categories. Only a subset of categories has been applied."* They are consistent with
  the depth-3 walk in §5.5 being an unexhausted tree.
- Discount Commons full-text "dress" (3.27M) and "outfit" (883k): they match *address*,
  *dressed*, *dressing*, and military/expedition usage.
- Flickr library-wide CC counters: **CC BY 2.0 93.6M, CC BY-SA 2.0 49.7M, PDM 18.6M, CC0 6.0M,
  CC BY 4.0 3.1M, CC BY-SA 4.0 2.5M — 173,456,226 commercial-use-and-modification total.**
  NonCommercial is 328,289,293, i.e. **62% of Flickr's CC pool is off-limits.**

### 5.2 Unsplash — excluded by its own licence, twice

This is the sharpest finding in the section, and it is not about rate limits.

> *"Note that the Unsplash License does not include the right to use: Trademarks, logos, or
> brands that appear in Images; **People's images if they are recognizable in the Images**;
> Works of art or authorship that appear in Images."* — [ToS §5](https://unsplash.com/terms)

Outfit photography is *definitionally* recognisable people wearing branded clothing. The
carve-out removes essentially the entire use case. And ToS §8 bans the training use outright:

> *"Use the Images in connection with any machine learning and/or artificial intelligence
> datasets (e.g., training any machine learning and/or artificial intelligence models)"*
> … *"Copy the look and feel of the Site or access, download, copy, modify, distribute,
> perform, or use any Images to create a similar or competitive service"* … *"Except for use
> of the Unsplash API in accordance with our API Terms, access the Service using any bots,
> spiders, scripts, crawlers, scrapers"*

The [API Terms](https://unsplash.com/api-terms) add that you *"must directly use or embed the
related image URLs returned by the API … (generally referred to as **hotlinking**)"* and must
attribute the photographer with a profile link **on every display**. The deployed service
serves bytes off a Modal volume via `/photo?path=…`; that architecture is non-compliant.

**Two corrections to the brief's premises.** Production rate limit is **1,000 req/hr, not
5,000** (*"For applications in demo mode … 50 requests per hour. After approval for production,
this limit is increased to 1000 requests per hour."*), and `per_page` maxes at 30. And the
**dataset polarity is the opposite of what is usually assumed**: per `TERMS.md` §2, **Lite**
(25k) grants a licence to *"internally use the Commercial Licensed Data to train machine
learning models"*, while **Full** (4.8M) is non-commercial and *"Any machine learning models
… trained with … Non-Commercial Licensed Data must not be used for commercial purposes."*
But §3 forbids serving either — *"disclose, deliver, disseminate, or publish any portion of
the Licensed Data in any manner."* So **Lite = train yes, redistribute no.**

### 5.3 Pexels — the memo's option (c), and its terms are worse than the memo assumed

The decision memo rests option (c) on Pexels: *"free API key, 200 req/hr and 20k req/month,
~6,000 fashion photos available, and the Pexels License permits commercial use without
attribution."* The licence part is right. **The Terms of Service part was not checked, and it
materially undercuts the plan.**

The [Pexels License](https://www.pexels.com/license/) is permissive: free to use, attribution
not required, modification allowed. Prohibitions: *"Don't sell unaltered copies"*, *"Don't imply
endorsement"*, *"Don't redistribute or sell the photos and videos on other stock photo or
wallpaper platforms"*, *"Identifiable people may not appear in a bad light."*

But [ToS §8](https://www.pexels.com/terms-of-service/):

> *"Data mining, extraction, scraping and the use of programs or robots for automatic data
> collection and/or extraction of digital data on the Service and/or the content available
> therein is strictly prohibited for all **unauthorised** purposes, **including without
> limitation for machine learning purposes**."*
> *"**Bulk, large-scale or systematic copying of Content is strictly prohibited** unless
> explicit permission has been granted by us."*
> *"use or compile any Content to **replicate a similar or competing service**"*

And ToS §5 disclaims the thing the memo assumed they supplied:

> *"**We do not warrant that any consents or licenses have been obtained in relation to any
> Content**, and we expressly disclaim any and all responsibility and liability in relation to
> such matters."* — with model releases put squarely on you and an indemnity flowing your way.

**State the ML position precisely, because it is worse than a clean "no".** §8 prohibits
*acquisition* for ML; it says nothing about what you may do with content you legitimately
downloaded, and the §5 grant (*"use, copy, modify or adapt … for commercial or non-commercial
purposes"*) is broad enough that a lawyer could argue training is covered. **Acquisition for
ML is expressly banned; use-for-ML is silent.** A broad grant sitting next to an adjacent
explicit prohibition is the worst kind of ambiguity, not a green light — and the word
*"unauthorised"* carrying the whole clause is **defined nowhere**.

**Three consequences the memo does not carry.**

1. A one-off API pull of ~6,000 photos to build a permanent corpus is *"systematic copying of
   Content"* on a plain reading. Using the sanctioned API is a good argument that it is not
   "robots for automatic data collection"; it is an argument, not a settlement. **Email Pexels
   for the explicit permission their own clause contemplates.** That is cheap and should be
   done before anything is rebuilt.
2. Pexels **does not warrant that the uploader had rights.** "Largely resolves risk 2" is too
   strong — it moves risk 2 from "definitely no permission" to "a chain of title Pexels
   explicitly declines to stand behind".
3. Risk 4 is untouched, as the memo correctly says.

Note also: **there is no separate Pexels API ToS document.** It does not exist. The binding
API conditions are the Guidelines block inside
[the API documentation](https://www.pexels.com/api/documentation/) — *"show a prominent link to
Pexels"*, *"You may not copy or replicate core functionality of Pexels"*, 200 req/hr and 20,000
req/month, `per_page` max 80 — plus the site ToS above.

### 5.4 Pixabay — one real escape hatch, and a hard API ceiling

[ToS §8](https://pixabay.com/service/terms/) is **word-for-word identical to Pexels'** on all
three clauses above. Both are Canva-owned and share the boilerplate. Same precise ML position:
acquisition banned, use silent.

**The escape hatch, and it is the single best risk reducer found in this section** — ToS §4:

> *"Some of the Content made available for download on the Service is subject to and licensed
> under the Creative Commons Zero (CC0) license ('CC0 Content'). **CC0 Content on the Service
> is any content which lists a 'Published date' prior to January 9, 2019.**"*

Pre-2019-01-09 Pixabay content is **CC0** and escapes the Content License restrictions
entirely — the Standalone ban, the competing-service ban, the bulk-copying ban. You still
carry model-release and trademark risk yourself. **If Pixabay is used at all, filter on
published date.** The size of that slice is unmeasured.

Three practical blockers on the rest:

- **`totalHits` is capped at 500 results per query.** 100k images would need ≥200 distinct
  query strings — which is exactly the *"Systematic mass downloads are not allowed"* conduct
  the same page prohibits.
- **Pixabay is the only one of the three that affirmatively tells you to host copies**:
  *"permanent hotlinking of images (using Pixabay URLs in your app) is not allowed. If you
  intend to use the images, please download them to your server first."* (`webformatURL` is
  documented as valid for 24 hours, which is what makes that bite.) Rate limit 100 req/60s.
- **Corpus quality.** §7B permits AI-generated uploads — *"You may upload Content you have
  created with generative AI technology"* — so synthetic garments will pollute an embedding
  model. And contributors can opt out of ML training, but **no API field exposes that flag**,
  so per-contributor opt-outs cannot be honoured even if you wanted to.

### 5.5 Wikimedia Commons — large, free, and the wrong century

**Measured 2026-09-15** by walking the category tree via the MediaWiki API:

| depth | cumulative files in `Category:Clothing` | categories seen |
|---:|---:|---:|
| 1 | 513 | 87 |
| 2 | 4,566 | 1,668 |
| 3 | **71,985** | 7,018 |

Growing ~15× per level and not exhausted at depth 3; the `deepcat:` lower bounds in §5.1
(**≥271,677 for Clothing**, ≥190,096 Fashion, ≥175,209 Textiles, ≥130,814 Costumes) are the
better figure. Licence mix across Commons' 147,684,468 files: **ShareAlike ~59.7M (~40%)**,
CC-BY ~11.7M, **CC0 9,809,186**.

**Share-alike, stated precisely, because the usual framing is both too scary and too casual.**
CC BY-SA 4.0 §3(b) fires only *"if You Share Adapted Material You produce"*, and §1(a) defines
Adapted Material as modified *"in a manner requiring permission"*. §2(a)(4) says technical and
format modifications *"never produce Adapted Material"*. So:

- **App code is safe.** Share-alike does not reach it.
- **Hosting and displaying unmodified images is safe** — attribution and licence notice only.
- **SAM2-style garment crops are genuinely at risk.** CC's own FAQ: *"if they were modified by
  the distributor the modified photo would need to be licensed under the same terms."* This
  project already has `outfit_dataset/product_labels.bbox.json` in the working tree.
- **Whether fine-tuned model weights are Adapted Material is unanswered in every
  jurisdiction.** CC's own position paper is advocacy, not law, and is framed around research
  and generative models rather than a commercial retrieval product.

**And one catch nobody mentions** — CC BY-SA 4.0 §2(a)(5)(C):

> *"You may not offer or impose any additional or different terms or conditions on … the
> Licensed Material if doing so restricts exercise of the Licensed Rights by any recipient."*

Boilerplate App Store EULA text — *"you may not copy or redistribute any content in this
application"* — **breaches this for every BY-SA image served.** Trivially fixed with a carve-out,
but only if you know about it. And §6(a) terminates rights **automatically** on non-compliance:
at 100k images, one systematic attribution bug is 100k simultaneous terminations.

**The real problem is relevance, not rights.** Commons' clothing tree is dominated by historical
costume, museum artefacts, military uniforms and national dress. Close to useless as a
contemporary outfit-search surface; moderately useful as garment-shape training data.

### 5.6 Museums and aggregators — the cleanest terms in the whole document

**Smithsonian Open Access** is the best-drafted licence found anywhere in this research:

> *"Yes, you may use Smithsonian Open Access assets designated as CC0 for commercial purposes
> **without any attribution, permission, or fee** paid to the Smithsonian."* — and the
> boundary is equally clear: *"You may not use any assets with usage conditions for commercial
> purposes."* — [si.edu/openaccess/faq](https://www.si.edu/openaccess/faq)

Measured CC0 counts (`media_usage:CC0`): **textile 27,073 · costume 23,384 · dress 21,460 ·
fashion 3,815** — overlapping, so **est. 40–60k distinct**. Rate limit 1,000 req/hr and 1,000
rows per request via api.data.gov, **plus a full dataset dump on S3** that bypasses the API
entirely.

**Europeana**, `reusability=open&media=true`: **fashion 84,063 · textile 110,489 · costume
55,437 · clothing 45,770 · dress 45,752.** The rights mix on "fashion" is unusually clean —
**CC0 43.1%, CC BY 38.3%, PDM 16.1%, BY-SA under 2%.** Two catches: Europeana **hosts nothing**
(images sit on ~4,000 provider servers), and the rights statement is the contributing
institution's assertion, **not a warranty**. Rate limit unverified — pro.europeana.eu is
Cloudflare-gated.

**The Met**, measured two ways: the Costume Institute reports **31,429 objects with images**;
independent samples found **35% `isPublicDomain` (n=60)** and **25% (n=76)**, so call it
**≈8,000–11,000 CC0 objects**. One important refinement: in the n=76 sample the
`isPublicDomain` set and the non-empty `primaryImage` set were **identical** — zero cases of an
image URL on a non-PD object. So `hasImages=true` counts images **on the website**, not images
you can have. Rate limit is documented at 80 req/s, and **`/v1/search` is deprecated and
retires 2026-10-01** — use `/v1.1`.

**Rijksmuseum**: no API key needed; measured with `imageAvailable=true` — clothing 1,426 ·
shoe 329 · textile 255 · dress 99 · costume 55, **est. 2,000–3,500 total**. Current image
rights terms **unverified** (the docs rights pages 404). Too small to be worth the legal spend.

### 5.7 Burst, and the ones that are disqualified outright

**[Burst by Shopify](https://www.shopify.com/stock-photos)**: 20,000+ photos of which
**~1,429 are fashion** (measured), *"free for commercial use with no attribution required"*,
mixed CC0 and a Burst licence. Prohibits selling unaltered files and *"compil[ing] Licensed
Photos to replicate a service similar to or competing with Burst"* — a fashion search app is
plainly not a competing stock photo library. **ML is silent; bulk download is not prohibited** —
the least hostile terms of any stock source surveyed, and the smallest volume. Note
`burst.shopify.com` now 301s to `shopify.com/stock-photos`, `/licensing` 404s, and the terms
were last updated February 2021.

**Disqualified with express AI-training bans in writing:** **Kaboompics** (*"Use images for AI
training, machine learning, biometric tracking, or facial recognition"* under "You may not")
and **Freepik** (bans ML use of content *"including any caption information, title, keywords or
other metadata"*; note `freepik.com/legal/terms-of-use` now 301s to magnific.com, which is its
own ambiguity). **Reshot is dead** — redirects to an Envato retirement notice dated 2026-01-20.

### 5.8 Flickr — the largest pool, with API terms built to stop exactly this

**214,250** licence-filtered "fashion" images and **173.5M** commercial-use-and-modification
images library-wide make Flickr by far the biggest source of *real contemporary people in real
clothes* with a licence. The API terms are the problem, not the licences:

- *"Display more than 30 Flickr user photos per page in your application"* — prohibited.
- *"Cache or store any Flickr user photos other than for reasonable periods"* — the Developer
  Guide pins this at **up to 24 hours**.
- *"Use Flickr APIs for any application that replicates or attempts to replace the essential
  user experience of Flickr.com"*.
- Revenue-generating apps need a **separately granted commercial key** that *"may"* carry fees.
- Rate limit *"under 3600 queries per hour"*; `flickr.photos.search` caps at **4,000 results per
  query**, so 214k "fashion" photos requires sharding.

**Licence-ID correction worth propagating: Flickr licence 11 is CC BY 4.0, not U.S. Government
Work (that is 8).** ID 11 requires attribution; getting this backwards silently drops attribution
from a whole slice.

**Flickr's AI position is "no express position", not permission.** API ToS (May 2018), site ToU
(rev. 2025-07-24), Privacy Policy (rev. 2025-08-26), Help Center and blog were all checked and
none contains an AI-training clause or a corroborable member opt-out. Absence of a prohibition
is not a grant.

**The honest reading: Flickr's *licences* permit what you want and Flickr's *API terms* do not.**
Whether the CC licence on an individual photo survives the API contract that delivered it to you
is a real question and a lawyer's, not an engineer's.

### 5.9 Openverse — not a pipeline, and not measurable

Index size **~820–915M items** (two reads of `/v1/images/stats/` minutes apart gave 822,733,370
and 915,519,784), of which ~88% is Flickr (536M) plus iNaturalist (266M) — and the iNaturalist
half is species observations with zero fashion value. Wikimedia 89.0M, Europeana 13.8M,
Rawpixel 1.3M, Met 498k, Cooper Hewitt 92k, StockSnap 40k, Rijksmuseum 30k.

**Its fashion subset cannot be sized by anyone.** `MAX_RESULT_COUNT` is hard-clamped at **240 for
anonymous and authenticated users alike** (4,800 for selectively-granted "privileged"), so every
apparel query returns `result_count: 240` regardless of the truth. Live anonymous limits measured
from response headers are **20/min and 200/day**, and reliability is disqualifying on its own:
**1 success in ~29 requests** over 45 minutes of HTTP 504s, the one success taking 59.8 seconds.

Structurally it only **indexes** — it aggregates other platforms' licence metadata and **does not
warrant it**, so an upstream mis-tag is your problem. Every fashion-relevant source inside it is
one you can query directly. **Use it to discover sources, never as a pipeline.**

---

## 6. Academic and public fashion datasets — stated plainly

**Almost all research-only. Two exceptions.** The reason is structural, not accidental:
nearly every one of these was built by scraping retailer or user photography, so the group
that published it never held the copyright it would need to grant you anything. Several say
so themselves.

### 6.1 The two that are genuinely commercially usable

| dataset | images | licence | verdict |
|---|---:|---|---|
| **Amazon Berkeley Objects** | 398,212 total / **58,927 fashion** (measured) | CC BY 4.0, attribution to "Amazon.com" | **Commercial use permitted** — display *and* training. Published by the rights holder. Stale NC artefacts must be resolved (§4.1). Zero garments. |
| **Fashionpedia** | 48,825, of which a filterable subset | Annotations + ontology **CC BY 4.0**; images **mixed, per-image** | **Partially usable.** See below. |
| *(Fashion-MNIST)* | 70,000 at 28×28 greyscale | **MIT**, Zalando SE is the rights holder | Genuinely open, genuinely useless. Notable as the **only** case in this survey where a rights-holding company applied a permissive licence to imagery it actually owned. |

**Fashionpedia is the interesting one, because it ships per-image licence provenance** —
the COCO annotation JSON carries a `licenses` array and a per-image `license` id. Counted
from the actual annotation files, train split (45,632 images):

| licence | count | % |
|---|---:|---:|
| CC BY-NC-ND 2.0 | 14,778 | 32.4% |
| CC BY-SA 2.0 | 7,134 | 15.6% |
| CC BY 2.0 | 6,650 | 14.6% |
| CC BY-ND 2.0 | 6,066 | 13.3% |
| CC BY-NC-SA 2.0 | 4,819 | 10.6% |
| CC BY-NC 2.0 | 3,776 | 8.3% |
| Unsplash | 1,275 | 2.8% |
| **all rights reserved / not found** | **544** | **1.2%** |
| Pexels | 456 | 1.0% |
| Burst | 70 | 0.2% |
| Public Domain Mark | 50 | 0.1% |
| Freestocks | 14 | 0.0% |

- **Clean** (commercial OK, derivatives OK, no copyleft): **8,515 images, 18.7%**.
- **Adding CC BY-SA 2.0** (commercial OK, share-alike on your adaptations): **15,649, 34.3%**.
- Val split 28.5% commercial-OK; test split 22.2%, and it contains **313 images (15.3%)
  explicitly tagged "all rights reserved/not found."**

**Independently verified here 2026-09-15** by downloading `instances_attributes_val2020.json`
(14 MB, no gate) and counting: 1,158 val images, of which 256 (22.1%) are clean and 330
(28.5%) are clean-plus-BY-SA — reproducing the val figure exactly. The `licenses` array is
machine-readable and carries a URL per id, so **the filter is a three-line script, not a
research project**:

```
0 CC BY 2.0   1 CC BY-SA 2.0   2 CC BY-ND 2.0   3 CC BY-NC-SA 2.0   4 CC BY-NC-ND 2.0
5 CC BY-NC 2.0   6 Public Domain Mark   7 unsplash   8 pexels   9 freestocks
10 burst   11 all rights reserved/not found   ← keep {0,6,7,8,9,10}, optionally 1
```

One nuance worth having: ids 7–10 are *stock-platform* images that Fashionpedia already
distributes as files. The Unsplash **API terms**' hotlinking mandate (§5.2) attaches to API
use, not to the Unsplash **License**, which does grant *"download, copy, modify, distribute"*.
So the Unsplash-sourced images arriving inside Fashionpedia are not subject to the constraint
that rules Unsplash out as a direct channel. Same logic for the Pexels-sourced ones and the
bulk-copying clause. **This is a reading, not a settled point** — put it on the lawyer list.

Fashionpedia's own [terms](https://fashionpedia.github.io/home/data_license.html) are
honest about it: *"**The Fashionpedia does not own the copyright of the images.** Use of the
images must abide by the Terms of use of Flickr, Unsplash, Burst by Shopify, Freestocks,
Kaboompics, and Pexels."* **Do not ship the whole set.** Filter on the `license` field, drop
ids 2/3/4/5/11, and carry attribution.

**The most valuable thing here may not be an image at all.** Fashionpedia's ontology — 27
apparel categories, 19 garment parts, 294 localized fine-grained attributes — is CC BY 4.0
and fully usable commercially. For scale: `docs/hierarchy.json` is **~48 nodes under 5 top-level
categories** (footwear, tops, bottoms, outerwear, accessories), measured. Fashionpedia's
taxonomy is roughly an order of magnitude richer, expert-authored, published with a real
licence, and costs nothing — **worth evaluating against the repo's own hierarchy regardless
of what happens to the imagery question.**

### 6.2 Research-only, unambiguously

| dataset | size | the clause |
|---|---:|---|
| **DeepFashion** | 800k+ | *"available for **non-commercial** research purposes only"*; *"not to … exploit for any commercial purposes, any portion of the images **and any portion of derived data**"*. Academic-email-gated; the form field is literally `Email (e.g. *.edu)`. |
| **DeepFashion2** | 491,895 | **No licence text exists anywhere.** GitHub licence API 404s; no LICENSE file; README silent; terms live only inside a gated Google Form. *An unfindable licence is not a permissive licence.* |
| **DeepFashion-MultiModal** | 44,096 | *"non-commercial research purposes only … any portion of derived data"* |
| **ModaNet** (eBay) | 55,176 | **CC BY-NC 4.0 on the annotations only.** eBay ships no images; the underlying PaperDoll/Chictopia set is dead (project page redirects, GitHub repo 404s). And those were Chictopia *users'* photos — eBay never held rights to them either. |
| **Dress Code** (YNAP) | — | The most restrictive found: non-commercial **and** *"accredited not-for-profit academic institution"* only, **and** *"any other works you make using the Dress Code Dataset will be subject to the same terms"*. Excludes a solo dev twice over. |
| **VITON** | — | **Withdrawn.** README: *"The dataset is no longer publicly available due to copyright issues. For thoese who have already downloaded the dataset, please note that using or distributing it is illegal!"* The clearest cautionary tale in the survey. |
| **VITON-HD** | — | CC BY-NC 4.0 |
| **M5Product** (Alibaba) | 6M | *"it is prohibited for any business entity to obtain or use the Dataset"* |
| **Clothing1M**, **FashionGen**, **Shopping100k**, **DARN**, **Alibaba iFashion/POG** | 1M / 293k / 101k / — / 1.01M outfits | **No licence text found** for any of them. Email-gated or dead official pages. |

### 6.3 Rotted URL lists — nothing to licence

**Street2Shop** (`tamaraberg.com` **404**, the whole site is gone — no licence text
retrievable), **Fashion-IQ** (URL list to Amazon's CDN; README says only *"Community Data
License Agreement (CDLA)"* **without saying which variant** — Permissive allows commercial
use, Sharing imposes copyleft, and no LICENSE file exists to settle it), **Fashion200K**
(Apache-2.0 on code, images scraped from Lyst in 2016), **iMaterialist 2018** (URL list of
Wish merchants' photos, no LICENSE file), **Amazon Reviews 2023** (Amazon CDN URLs, HF
licence field `None`).

Useful workaround: **iMaterialist 2019 and 2020 *are* Fashionpedia** — obtain them from
CVDF's S3 under Fashionpedia's terms and the Kaggle competition rules never attach. The 2018
edition is a separate, research-only Wish dataset.

### 6.4 The trap: a permissive tag applied by someone who never owned the images

This is the most common failure mode in this space, and it is worth naming because the tags
look reassuring.

- **Myntra "Fashion Product Images"** (Kaggle, ~44k images) — uploaded by an individual, not
  Myntra/Flipkart. He had no authority to apply *any* licence to Myntra's photography.
  Whatever the card says, it grants you nothing. **Kaggle does not verify that uploaders hold
  the rights they claim to grant.**
- **Zalando FEIDEGGER** (8,732 dress product shots) — LICENSE is MIT, but scoped to *"this
  **software** and associated documentation files"*. The data is a **URL list to Zalando's own
  CDN**. Zalando published no image licence at all.
- **Atlas** (Flipkart, 183,996 images) — MIT on *"this project"*, i.e. the code; and the image
  URLs point at `images.voonik.com`, which **does not resolve**. Voonik is defunct.
- **Polyvore Outfits** (365,054 items) — polyvore.com shut down in 2018; the surviving
  HuggingFace mirror is tagged `license: cc-by-4.0` by an academic author, applied to
  user-composed collages and scraped retailer photography from a site they never owned.
- **H&M Personalized Fashion Recommendations** (105k article images) — **the competition rules
  could not be read without a Kaggle account**, which the research constraint forbade. That is
  a finding, not an omission: read Section "A. DATA → Data Access and Use" yourself before
  relying on anything. Independently of the rules text, these are H&M's copyrighted product
  photographs released for a modelling competition. A competition entry form does not convey
  the right to display H&M's catalogue in a rival commercial app.

**The rule to carry:** a `LICENSE` file tells you what the **annotator** permits. It is
silent on what the **photographer** permits, and the photographer is the one who can sue.
Three of these datasets say so in their own words — CUHK (*"All images … are obtained from
the Internet which are not property of MMLAB"*), Fashionpedia (*"does not own the copyright
of the images"*), LAION (*"The images are under their copyright"*).

### 6.5 LAION — not a dataset of images

LAION distributes **no images**: *"LAION datasets are simply **indexes to the internet** …
lists of URLs to the original images together with the ALT texts."* The CC-BY-4.0 applies to
the **metadata parquet files**; *"**The images are under their copyright.**"* LAION's own
recommendation is *"to use the dataset for research purposes"* and they *"do not recommend
using it for creating ready-to-go industrial products."*

LAION-5B was **taken down entirely in December 2023** after Stanford's Internet Observatory
identified 1,008 links to CSAM or likely CSAM, and re-released as **Re-LAION-5B** in August
2024. **This is the one place where training and display genuinely come apart — and it still
does not help.** Training on LAION-derived images is a contested grey zone in active
litigation. *Displaying* those images in a commercial app is not grey at all: every image
belongs to a third party, ~99.99% under no CC licence, with no attribution data and an
opt-out regime you would have to honour continuously. Plus you inherit the contamination
history in your provenance story. Do not build a product corpus from LAION.

### 6.6 Two items in the brief that do not exist

- **"ACHR"** — not identifiable as a fashion dataset. Searches return only hemispheric radar
  and acetylcholine receptor antibodies. It is almost certainly a **method** name from a
  related-work table, not a dataset. Nothing to licence.
- **"KITS"** — resolves to **KiTS, the Kidney Tumor Segmentation Challenge**. No fashion
  content whatsoever. Any source listing it among fashion datasets is in error.

Also: **no Farfetch public dataset was found.** The lead is the fashionXrecsys workshop,
which Farfetch researchers co-organise; its "Datasets" section merely lists other people's
datasets.

---

## 7. Synthetic imagery as a fallback

### 7.1 What it costs

API pricing, September 2026 (*from vendor comparison pages, not measured here*): roughly
**$0.005/image** (GPT Image 1 Mini) to **$0.067/image** (Nano Banana 2 1K), with FLUX.2 Pro
at $0.055 and Imagen 4 Fast/Standard/Ultra at $0.02/$0.04/$0.06. Batch APIs from OpenAI and
Google cut those roughly in half. So **20,000 outfit images costs $50–$1,340 via API**, which
against a $30/month ceiling is not viable at the top of that range and marginal at the bottom.

**Local generation is the actually-affordable route.** `FLUX.1-schnell` is **Apache-2.0** —
personal, scientific and commercial use, no fee. (`FLUX.1-dev` is **non-commercial** and has
required a paid BFL licence for business use since June 2025 — do not mix them up.) SDXL is
CreativeML Open RAIL++-M with no revenue cap. On Modal's A10G at $1.10/hr, generation is
bounded by the same GPU budget as everything else.

### 7.2 What the rights position actually is

Two separate questions, and both matter:

1. **Do you own the output?** In the US, **no, not on prompts alone.** The Copyright Office's
   January 2025 guidance concluded that works produced entirely by AI without meaningful human
   creative input are not copyrightable, and that prompts alone do not make the user an author.
   *Thaler v. Perlmutter* (D.C. Cir., March 2025) confirmed the human-authorship requirement
   nationwide. Practical effect: you can *use* generated images, you likely cannot *stop
   anyone else* from using them. For this product that is a tolerable trade.
2. **Can you be sued over them?** Yes, on two theories the model licence does not touch.
   **Right of publicity** now explicitly covers AI-generated likenesses in several states
   (Tennessee's ELVIS Act, Arkansas HB 1071, Washington SSB 5886 effective 2026-06-10, with
   the federal NO FAKES Act introduced but not enacted as of August 2026). A synthetic face
   that resembles a real identifiable person is actionable, and an "AI Generated" label does
   not cure it — New York GBL §396-b has required such disclosure since 2026-06-09 and it is
   a disclosure rule, not a safe harbour. And **training-data provenance** for the open-weight
   models is itself the contested question in §6.5.

So synthetic **reduces risk 4 without eliminating it** — the same verdict the memo reaches
for Pexels, arriving by a different road.

### 7.3 The measured argument against it for training

[Geng et al., *The Unmet Promise of Synthetic Training Images*](https://arxiv.org/abs/2406.05184)
found that synthetic data generated by Stable Diffusion is *"universally matched or
outperformed by real data from the simple retrieval baseline"* — i.e. retrieving real
task-relevant images from the generator's own training set beats generating new ones, due to
generator artifacts and inaccurate task-relevant detail. They argue targeted retrieval is
*"a critical baseline … that current methods do not yet surpass."*

This lines up with what `SESSION_HANDOFF.md` already says about this project: **the encoder is
the binding constraint, not corpus size.** Synthetic images would grow the corpus while making
the distribution *less* like the deployment distribution — and `docs/eval_log.md`'s Control 3
already measured what that does, finding AUROC 0.9670 for separating catalog photography from
real outfit photos on confidence alone. A synthetic corpus would install a third distribution.

And the memo's own objection stands and is the strongest one: **the stated product is "real
outfit photos of real people."** Generated outfits are a different product.

---

## 8. What this changes, and what a lawyer still has to answer

### 8.1 Corrections to `licensing_decision_memo.md`

1. **Option (c) is weaker than stated.** Pexels' ToS prohibits *"bulk, large-scale or
   systematic copying of Content"*, bans *"use or compile any Content to replicate a similar
   or competing service"*, bans data collection *"including without limitation for machine
   learning purposes"*, and **expressly disclaims warranting that consents or licences were
   obtained**. "Largely resolves risk 2" overstates it. The cheap fix is to ask Pexels in
   writing for the explicit permission their own clause contemplates — do that before
   rebuilding on it, not after.
2. **A new option (f) exists: the licensed catalog.** ABO's 58,927 CC BY 4.0 images make
   option (e) — ship the catalog surface — achievable *with a licence* rather than with
   scraped brand imagery, for footwear and accessories. That is a materially better version
   of the memo's stated fallback.
3. **The budget premise may be wrong.** Modal's Starter plan includes **$30/month in free
   compute, recurring**, not a one-time grant, plus **1 TiB of free volume storage**. If the
   `$28.13 of $30` figure is the free monthly credit rather than a self-imposed cap, it
   **resets**, and the "cannot afford another GPU pass" conclusion is a one-month statement
   rather than a permanent one. Worth checking on the billing page before deciding anything
   on cost grounds.
4. **A new option (g): a CC0-only training corpus.** ~100k–150k images across Smithsonian,
   Europeana, the Met, Commons and pre-2019 Pixabay (*estimate*, components overlap and were
   not deduplicated). It is the wrong *subject matter* for a product surface — museum-heavy —
   but it is the only corpus in this document under which training is not a legal bet. It
   pairs naturally with option (e)/(f): licensed catalog on the surface, CC0 for the encoder.
5. **Unsplash is out, and not for the reason anyone expects.** Its licence *"does not include
   the right to use … People's images if they are recognizable in the Images"*. For a corpus
   of outfit photos that is not a caveat, it is a disqualification.

### 8.2 Unit economics, from the repo's own measured throughput

At the repo's measured **26 img/s** for a SigLIP2-384 forward on an idle A10G at $1.10/hr:

| images | A10G-hours | cost |
|---:|---:|---:|
| 10,000 | 0.11 | $0.12 |
| 100,000 | 1.07 | $1.18 |
| 500,000 | 5.34 | $5.88 |
| 1,000,000 | 10.68 | $11.75 |

Storage, at measured 368 KB/image (outfit) and 213 KB/image (catalog): 500k images ≈ 134 GiB
against a **1 TiB free** volume. **Neither compute nor storage is the binding constraint at
these volumes.** The binding constraints are rights and relevance — which is the whole point
of this document.

### 8.3 The App Store hook, quoted

The memo's lawyer-question 2 asked what Apple actually requires. The text, verbatim:

> **5.2.1 Generally:** *"Don't use protected third-party material such as trademarks,
> copyrighted works, or patented ideas in your app without permission … Apps should be
> submitted by the person or legal entity that **owns or has licensed** the intellectual
> property and other relevant rights."*
>
> **5.2.2 Third-Party Sites/Services:** *"If your app uses, accesses, monetizes access to, or
> displays content from a third-party service, ensure that you are specifically permitted to
> do so under the service's terms of use. **Authorization must be provided upon request.**"*
>
> — [App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)

*"Authorization must be provided upon request"* is the operative sentence. It converts every
licence question in this document into a document you may be asked to produce. ABO's answer
to that request is a README URL. The scraped corpus has no answer.

### 8.4 Still needs a lawyer

In addition to the six questions in `licensing_decision_memo.md` §6:

7. **ABO's licence contradiction.** README and official site say CC BY 4.0 (since mid-2023);
   the AWS Open Data registry and the CVPR paper still say CC BY-NC 4.0, and the NC licence
   file is still in the bucket. Which governs? Ask Amazon in writing, cite the timestamps,
   and ask them to fix the registry.
8. **Does an API-sanctioned bulk pull satisfy or violate Pexels' ToS clause** on systematic
   copying? And does their no-warranty-of-consents disclaimer change the risk calculus for
   option (c)?
9. **CC BY-SA scope, split into the two parts that actually differ.** (a) Confirm the reading
   in §5.5 that unmodified display is not Adapted Material but a **SAM2 garment crop is** —
   this project already has `outfit_dataset/product_labels.bbox.json` in the working tree, so
   it is a live question, not a hypothetical. (b) Are **fine-tuned model weights** Adapted
   Material? Unanswered in every jurisdiction; CC's own position paper is advocacy, not law.
10. **Does an affiliate licence's "without modification" clause bar resizing for display
    and for embedding?** If yes, affiliate feeds are a *hotlink-only* channel and the memo's
    §5 option (a) architecture problem returns unchanged.
11. **Do CC BY 4.0 images of identifiable people cure risk 4?** They do not, on the same
    reasoning the memo applies to Pexels — but ABO's imagery is largely product-on-white,
    which sidesteps it in practice for most of the 58,927. Worth confirming what fraction
    contains people.
12. **The App Store EULA vs CC BY-SA §2(a)(5)(C).** Boilerplate store-EULA text — *"you may
    not copy or redistribute any content in this application"* — is an *"additional or
    different term … that restricts exercise of the Licensed Rights by any recipient"*, which
    the licence forbids. This is cheap to fix with a carve-out and expensive to discover late,
    because §6(a) terminates rights **automatically** on non-compliance: at 100k images, one
    systematic bug is 100k simultaneous terminations. **Fix this before any BY-SA image
    ships.**
13. **Does a CC licence on a Flickr photo survive the API contract that delivered it?** Flickr's
    *licences* permit what this project wants; Flickr's *API terms* (30 photos/page, 24-hour
    cache ceiling, no replicating the Flickr experience, separate commercial key) do not. Which
    governs the copy in your hands is a genuine question and the gate on the single largest
    pool of real people in real clothes.
14. **Is an API pull within documented rate limits "systematic copying"?** The same question as
    #8, but it recurs verbatim at Pixabay (*"Systematic mass downloads are not allowed"*, with
    "systematic" undefined) and matters for any of the stock channels.

### 8.5 Unverified — do not treat as findings

- **Fashion advertiser and SKU counts at every affiliate network.** No primary source existed
  and nothing was estimated.
- **Whether a TestFlight link satisfies any network's "promotional space" field.** This is the
  single highest-value unknown in the whole document, and **one support email to Sovrn or
  Impact would resolve it.**
- **Pexels fashion volume.** Pexels exposes **no counts at all** without an API key — search
  pages carry no total and are depth-capped. The memo's ~6,000 (2026-08-03, measured with a
  key) is the only figure that exists, and it was **not re-measured here.**
- **Openverse's fashion subset is unmeasurable by anyone**, not merely unmeasured: results are
  hard-clamped at 240 for anonymous and authenticated users alike. Index size read twice
  minutes apart gave 822,733,370 and 915,519,784 — call it ~820–915M.
- **The size of Pixabay's pre-2019 CC0 slice** — the one genuinely useful escape hatch found in
  §5.4, and its volume is unknown.
- **Europeana's rate limit** — pro.europeana.eu is Cloudflare-gated.
- **Rijksmuseum's current image-rights terms** — the docs rights pages 404 and the new Linked
  Art API exposes no machine-readable rights field.
- **Flickr's AI-training position** — checked across API ToS, site ToU, Privacy Policy, Help
  Center and blog; **no clause found.** Record that as "no express position", **not** as
  permission.
- **CJ's current publisher agreement** — not published; quotes are from a 2007 SEC-archived copy.
- **Walmart, Target and Etsy affiliate terms** — behind login. Undocumented is worse than
  restrictive, not better.
- **Kaggle H&M and iMaterialist competition rules** — login-walled, deliberately not read.
- **Whether Icecat's paid image licence is broader than its free one** — not stated publicly.

---

## 9. Bottom line

**For a solo developer with no traffic and no money, there is exactly one channel that is
open today, free, large, and carries a licence from the party that actually owns the
images: Amazon Berkeley Objects.** 25,916 fashion listings, 58,927 unique catalog images,
165 brands, CC BY 4.0, downloadable this afternoon — and **zero garments**. It is 2.9× the
current catalog's image count with a licence attached, for footwear, bags and jewellery only.
The right next step is to resolve the CC BY / CC BY-NC contradiction in writing with Amazon
before building on it.

**The best *strategy* is a different thing from the best *channel*: a CC0-only corpus.**
Smithsonian (~40–60k), Europeana's CC0/PDM slice (~50k), the Met's CC0 subset (~8–11k),
Commons' CC0 slice, and Pixabay's pre-2019 content together reach roughly **100k–150k
images** (*estimate* — components overlap and were not deduplicated). It is museum-heavy and
therefore a **training** corpus rather than a product surface. What it buys is that the two
questions nobody can currently answer — *are fine-tuned model weights a derivative work?* and
*is a SAM2 garment crop an Adapted Material?* — stop being bets and become irrelevant.

**Third: Fashionpedia's clean subset**, ~8,500 images filtered on its own per-image licence
field, plus an ontology under CC BY 4.0 that is roughly an order of magnitude richer than
`docs/hierarchy.json` and may be worth more than the pictures.

**Everything else is one of five disappointments.** The **affiliate networks** grant a display
licence that is revocable, merchant-scoped, modification-barred and ML-prohibited — a marketing
permission, not a data licence, with a kill switch someone else owns. The **data resellers**
sell collection labour and contractually disclaim the rights you need. The **academic datasets**
are research-only, and the ones with reassuring MIT/Apache/CC-BY tags are mostly cases of
someone licensing photographs they never owned. The **open stock libraries** all forbid
compiling their content into a similar or competing service, and Unsplash — the largest and
most permissive-sounding — excludes recognisable people from its grant entirely, which is the
whole use case. And **Flickr**, which has the imagery and the licences, has API terms written
to stop exactly this: 30 photos per page, a 24-hour cache ceiling, and a separate commercial key.

**The uncomfortable summary: no channel found here replaces the 20,532 outfit photos.** The
rights-clean sources are product photography and museum artefacts. Real photographs of real
people wearing real clothes, at volume, with a licence that also survives the API contract
that delivers them, is the thing that does not exist for sale — which is why the corpus was
scraped in the first place. This document does not change the memo's decision. It removes
several of the escape hatches someone might have hoped were behind it, weakens option (c)
rather than strengthening it, and adds two real options: a **licensed catalog** for the product
surface and a **CC0 corpus** for the encoder.

**The two cheapest next actions, neither of which needs a lawyer or a dollar:**
one email to Pexels asking for the explicit bulk permission their own ToS contemplates, and
one support email to Sovrn or Impact asking whether a TestFlight link satisfies their
"promotional space" requirement. Both are single unknowns that gate whole branches of this table.
