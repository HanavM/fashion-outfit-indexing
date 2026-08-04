"""Reddit real-outfit scraper -> `outfit_dataset/` (NOT apparel_dataset).

Collects photos of real people wearing real outfits from fashion
communities. These records are DELIBERATELY UNLABELED: images + provenance
only, no garment/category/brand fields, not even placeholders. Labeling
happens later and unsupervised, from this project's own models. See
SCRAPING_PROCESS.md, "Real-outfit scraping (`outfit_dataset/`)".

Source notes:
  - **Reddit's own endpoints are hard-blocked from this machine.** Every
    documented public route returns HTTP 403 with an HTML block page:
    `www.reddit.com/r/<sub>/top.json`, `old.reddit.com`, `api.reddit.com`,
    `oauth.reddit.com`, with a browser User-Agent, with a descriptive
    custom User-Agent, and with full browser header sets. The
    "public .json endpoints work without OAuth at modest rates" assumption
    in the task brief no longer holds -- Reddit now gates the JSON API
    behind OAuth app credentials, and no Reddit credentials exist in this
    repo's `.env`. Public redlib/libreddit mirrors are 403 too.
  - **Workaround actually used: Arctic Shift**
    (`https://arctic-shift.photon-reddit.com/api/posts/search`), a public
    Reddit archive API (the post-Pushshift successor). No auth, no
    blocking, and it returns the *complete original Reddit post JSON* --
    `id`, `permalink`, `author`, `title`, `created_utc`, `link_flair_text`,
    `is_gallery`, `gallery_data`, `media_metadata`, `preview` -- so the
    record schema and gallery handling are identical to what the native
    API would have given. Images themselves come straight from Reddit's
    own CDNs (`i.redd.it` / `preview.redd.it`), which are NOT blocked.
  - **Score is NOT a usable quality signal here, and flair is.** Arctic
    Shift snapshots a post shortly after creation, so archived `score`
    reflects when the snapshot happened, not how the post did. Measured
    2026-08-03: median score is 1 even on 2.5-year-old r/streetwear posts.
    An earlier version of this scraper gated on `--min-score` and added a
    SCORE_MATURITY_DATE cutoff to compensate; neither worked, and together
    they discarded most of the corpus (including the 5 most recent months)
    on a signal that was noise. Both are gone.
    The replacement is a per-subreddit **flair allowlist**. `WDYWT` /
    `WIWT` / `Fit Check` posts are, by the subreddits' own rules, one
    person wearing one outfit -- exactly the target. Measured precision on
    r/streetwear: 45 of 57 image posts (79%) before any model filtering.
    See docs/outfit_sourcing_plan.md for the full measurement.
  - Pagination is by time: `sort=desc` plus a moving `before` cursor set
    to the oldest `created_utc` seen so far (`limit` caps at 100).
  - Dedup is PERCEPTUAL (imagehash.phash, Hamming distance <=
    PHASH_DISTANCE), not by url/id -- reposts across these subs are
    endemic and the same photo reappears resized under new post ids.
    A repost is caught even when its post id, host and pixel size differ.
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from dataset_utils import load_outfit_records, outfit_key, save_outfit_records_safe
from outfit_scrape_common import (
    HEADERS, MIN_IMAGE_BYTES, MIN_IMAGE_SIDE, PHASH_DISTANCE, USER_AGENT,
    download_images, format_stats, is_duplicate, load_seen_hashes, new_stats,
)

API = "https://arctic-shift.photon-reddit.com/api/posts/search"

DATASET_DIR = Path("outfit_dataset/reddit")

OLDEST_DATE = "2023-01-01"
# No upper date bound: the old SCORE_MATURITY_DATE cutoff existed only to
# protect a score filter that no longer exists. Scraping the full window
# re-opens the most recent months, which are the best-quality posts.
NEWEST_DATE = None


def norm_flair(value) -> str:
    """Normalize a flair for matching.

    Several subs publish flairs with trailing spaces and mixed case
    (r/fashion's `'Advice Wanted Please! '`, r/femalefashion's
    `'[WIWT]'`), so exact-string matching silently matches nothing.
    """
    return (value or "").strip().casefold()


# (subreddit, image target, allowed flairs, score floor).
#
# Targets are in IMAGES, not posts; gallery posts yield several images each.
#
# `flairs` is the PRIMARY precision filter. A set means "accept only these
# flairs" (values already normalized by norm_flair). `None` means the sub
# does not use flair usefully -- every image post is unflaired or the flair
# carries no outfit signal -- so accept any image post and rely on the
# subreddit's own topic for precision. Verified per-sub against live flair
# distributions on 2026-08-03; do not add a sub here without checking its
# actual flairs first, since a guessed flair string yields exactly zero.
#
# Score floors are 0 (disabled) by default -- see the module docstring.
# Kept as a knob because score IS live on the largest subs (r/streetwear's
# median is 34 among image posts), just not reliable enough to gate on.
SUBREDDITS = [
    ("streetwear", 300, {"wdywt", "inspo"}, 0),
    ("fashion", 250, {"outfit of the day", "advice wanted please!",
                      "feedback wanted!", "label my style",
                      "special occasion outfit",
                      "help me coordinate my outfit!", "🤩showcase 🤩"}, 0),
    ("mensfashion", 250, {"fit check", "ootd / wiwt"}, 0),
    ("OUTFITS", 250, {"outfit of the day 👗", "felt cute ✨", "my work fit 🔥",
                      "advice ❔ women's fashion :dress:",
                      "advice ❔ men's fashion :tie:"}, 0),
    # Unflaired subs: every image post is a fit by the sub's own topic.
    # r/streetwearfits caveat: the sub degraded in 2026 and its recent
    # posts are ~100% NSFW/TikTok-link spam (measured 2026-08-03: 100/100
    # posts in 2026-05..07 flagged over_18, vs 0/100 in 2025-06, which
    # passed 91/100). The over_18 filter rejects all of it correctly, so
    # this just costs a few wasted API pages before the walk-back reaches
    # the clean 2023-2025 content -- which is some of the best here.
    ("streetwearfits", 200, None, 0),
    ("femalefashion", 200, None, 0),
    ("femalefashionadvice", 100, None, 0),
    ("malefashion", 150, {"wiwt", "inspo"}, 0),
    ("rawdenim", 120, {"fit pic"}, 0),
    # Kept but tightly flair-gated: r/japanesestreetwear is mostly
    # DISCUSSION and PICK-UP (flat-lay purchase hauls -- exactly the
    # product-shot look this dataset exists to contrast against), and
    # r/techwearclothing is almost entirely unflaired. Small targets
    # because WDYWT is rare on both; a shortfall NOTE here is expected.
    ("japanesestreetwear", 60, {"wdywt"}, 0),
    ("techwearclothing", 30, {"wdywt"}, 0),
    # Added 2026-08-03 after probing 14 further candidate subs against live
    # flair distributions (100-post samples, 2023-01..2025-09). Only the ones
    # that actually measured well are here; the rejects are listed below so
    # they are not re-probed.
    #
    # r/altfashion is the best single addition found: 70 of 100 posts are
    # usable image posts and ALL of them carry an outfit flair. It also adds
    # subcultural range (goth/punk/emo/kawaii) that the mainstream
    # streetwear subs do not cover at all.
    ("altfashion", 300, {"outfit (goth, punk, emo, scene, etc.)",
                         "outfit (other/general alt)",
                         "outfit (pastel, kawaii, decora etc)",
                         "masculine fit 🤍", "feminine fit 🩷",
                         "non-binary fit 🤍"}, 0),
    # r/OOTD: 81 usable image posts per 100 and the sub uses no flair at all
    # -- the name is the filter. Highest raw density measured anywhere.
    ("OOTD", 300, None, 0),
    # Body-diverse advice subs, all unflaired-accept: their flair vocabularies
    # are height/region buckets rather than content types (r/PetiteFashionAdvice
    # tags by height, r/BigMenFashionAdvice by country), so flair carries no
    # outfit signal and the subreddit's own topic is the precision. ~45 usable
    # image posts per 100 each. These matter beyond volume: every other source
    # here skews slim and young, and Phase 3 is meant to work on real bodies.
    ("BigMenFashionAdvice", 200, None, 0),
    ("PetiteFashionAdvice", 200, None, 0),
    ("curvyfashion", 100, None, 0),
    # r/vintagefashion needs its flair gate: 'sweet find!' is 18% of image
    # posts and is flat-lay thrift hauls -- the product-shot look this dataset
    # exists to contrast against -- while ootd/menswear/inspo/advice are worn.
    ("vintagefashion", 150, {"ootd", "menswear", "inspo", "advice plz"}, 0),
]

# Deliberately NOT in the default list: r/malefashionadvice. Its image
# posts are flaired Question/Discussion/Guide -- no worn-outfit flair
# exists, and "Question" mixes fit checks with product and sizing queries.
# Add it back with an explicit --subreddit if that mix proves acceptable.
#
# Also probed 2026-08-03 and rejected, so nobody re-probes them:
#   r/goodyearwelt      63 image posts/100, but they are boot close-ups
#                       flaired review/questions -- footwear, not outfits.
#   r/FashionReps       42/100, but 24 are 'w2c' product shots of replicas.
#   r/AusFemaleFashion  42/100, dominated by 'recommendations wanted'
#                       (product links); 'look of the day' was 1 in 100.
#   r/workwear          45/100 but the mix is workwear PRODUCT questions.
#   r/OutfitIdeas, r/koreanfashion, r/genderqueerfashion, r/streetweardaily
#                       returned zero posts from the archive -- private,
#                       renamed, or too small to have been indexed.

MAX_IMAGES_PER_POST = 4
# PHASH_DISTANCE / MIN_IMAGE_BYTES / MIN_IMAGE_SIDE / IMAGE_SLEEP now live in
# outfit_scrape_common so every outfit source dedups and validates
# identically -- re-exported above for callers that import them from here.
API_SLEEP = 1.2
CHECKPOINT_EVERY = 15

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
SKIP_AUTHORS = {"[deleted]", "AutoModerator", "None", None}


def get_json(params, attempt=0):
    """GET with backoff. Reddit-adjacent hosts rate-limit hard; 429/5xx get
    exponential backoff rather than being treated as a dead end."""
    try:
        resp = requests.get(API, headers=HEADERS, params=params, timeout=60)
    except requests.RequestException as error:
        if attempt >= 4:
            raise
        time.sleep(5 * (2 ** attempt))
        return get_json(params, attempt + 1)
    if resp.status_code in (429, 500, 502, 503, 504):
        if attempt >= 4:
            print(f"  giving up after {attempt} retries (HTTP {resp.status_code})")
            return []
        wait = 10 * (2 ** attempt)
        print(f"  HTTP {resp.status_code}, backing off {wait}s")
        time.sleep(wait)
        return get_json(params, attempt + 1)
    resp.raise_for_status()
    return resp.json().get("data") or []


def iter_posts(subreddit):
    """Walk a subreddit backwards in time, 100 posts at a time."""
    before = NEWEST_DATE
    while True:
        params = {
            "subreddit": subreddit,
            "limit": 100,
            "sort": "desc",
            "after": OLDEST_DATE,
        }
        if before is not None:
            params["before"] = before
        posts = get_json(params)
        if not posts:
            return
        for post in posts:
            yield post
        oldest = min(p.get("created_utc", 0) for p in posts)
        if not oldest:
            return
        before = str(int(oldest))
        time.sleep(API_SLEEP)


def image_urls_for(post):
    """Full-resolution image URLs for a post, in published order.

    Handles the three shapes that matter: reddit galleries (multiple images
    under `gallery_data` + `media_metadata`), single i.redd.it image posts,
    and posts whose only image lives in the `preview` blob.
    """
    urls = []
    if post.get("is_gallery") and post.get("media_metadata"):
        meta = post["media_metadata"]
        items = (post.get("gallery_data") or {}).get("items") or []
        media_ids = [i.get("media_id") for i in items] or list(meta)
        for media_id in media_ids:
            entry = meta.get(media_id) or {}
            if entry.get("e") != "Image":
                continue
            source = entry.get("s") or {}
            if source.get("u"):
                urls.append(source["u"])
            else:
                ext = (entry.get("m") or "image/jpg").split("/")[-1]
                urls.append(f"https://i.redd.it/{media_id}.{ext}")
        return urls

    url = post.get("url") or ""
    if post.get("post_hint") == "image" or url.lower().endswith(IMAGE_EXTS):
        return [url]

    previews = (post.get("preview") or {}).get("images") or []
    for preview in previews:
        source = (preview.get("source") or {}).get("url")
        if source:
            urls.append(source)
    return urls


def is_candidate(post, allowed_flairs, min_score):
    if post.get("author") in SKIP_AUTHORS:
        return False
    if post.get("stickied"):
        return False
    # Only USER-deleted posts are excluded. `removed_by_category` is set on
    # 53-97% of posts depending on sub, but those are overwhelmingly
    # `automod_filtered` / `moderator` -- subreddit rule violations (missing
    # ID comment, wrong day thread), not bad images -- and the images are
    # usually still live (6 of 8 sampled fetched HTTP 200 on 2026-08-03).
    # Dropping them all discarded most of the corpus for no quality gain.
    # `deleted` is different: the user removed it and the image is gone.
    if post.get("removed_by_category") == "deleted":
        return False
    if post.get("over_18") or post.get("is_video"):
        return False
    if (post.get("score") or 0) < min_score:
        return False
    if allowed_flairs is not None and norm_flair(post.get("link_flair_text")) not in allowed_flairs:
        return False
    return bool(image_urls_for(post))


def download_post_images(post_id, urls, seen_hashes, stats):
    """Download up to MAX_IMAGES_PER_POST of a post's images."""
    return download_images(DATASET_DIR / post_id, urls, seen_hashes, stats,
                           max_images=MAX_IMAGES_PER_POST)


def build_record(post, subreddit, paths, urls, hashes):
    flair = post.get("link_flair_text")
    return {
        "source": "reddit",
        "source_id": post["id"],
        "post_url": "https://www.reddit.com" + post.get("permalink", f"/comments/{post['id']}/"),
        "author": post.get("author", ""),
        "title": post.get("title", ""),
        "section": f"r/{subreddit}",
        "created_utc": int(post.get("created_utc", 0)),
        "source_tags": [flair] if flair else [],
        "image_count": len(paths),
        "images": paths,
        "image_urls": urls,
        "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phash": hashes,
    }


def scrape_subreddit(subreddit, target_images, allowed_flairs, min_score,
                     existing_ids, seen_hashes, stats):
    gate = "any image post" if allowed_flairs is None else f"flairs {sorted(allowed_flairs)}"
    print(f"\n=== r/{subreddit} (target {target_images} images, {gate}"
          f"{f', min score {min_score}' if min_score else ''}) ===")
    touched, images_here, posts_here, scanned = {}, 0, 0, 0

    for post in iter_posts(subreddit):
        scanned += 1
        if images_here >= target_images:
            break
        if post.get("id") in existing_ids:
            continue
        if not is_candidate(post, allowed_flairs, min_score):
            continue

        paths, urls, hashes = download_post_images(
            post["id"], image_urls_for(post), seen_hashes, stats
        )
        if not paths:
            continue

        record = build_record(post, subreddit, paths, urls, hashes)
        touched[outfit_key(record)] = record
        existing_ids.add(post["id"])
        images_here += len(paths)
        posts_here += 1
        print(f"  [{images_here}/{target_images}] {post['id']} "
              f"({len(paths)} img, score {post.get('score')}) {post.get('title', '')[:60]}")

        if len(touched) >= CHECKPOINT_EVERY:
            save_outfit_records_safe(touched)
            print(f"  checkpointed {len(touched)} records")
            touched = {}

    if touched:
        save_outfit_records_safe(touched)
        print(f"  final checkpoint: {len(touched)} records")

    if images_here < target_images:
        print(f"  NOTE: r/{subreddit} yielded {images_here}/{target_images} images "
              f"from {scanned} posts scanned -- archive exhausted or filters too tight.")
    return posts_here, images_here, scanned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subreddit", action="append",
                        help="limit to these subreddits (repeatable)")
    parser.add_argument("--target", type=int,
                        help="override per-subreddit image target")
    parser.add_argument("--min-score", type=int,
                        help="override per-subreddit score floor (default 0 -- "
                             "archived scores are unreliable, see module docstring)")
    parser.add_argument("--ignore-flair", action="store_true",
                        help="accept any image post, bypassing the per-subreddit "
                             "flair allowlist (lower precision, higher yield)")
    args = parser.parse_args()

    plan = SUBREDDITS
    if args.subreddit:
        wanted = {s.lower() for s in args.subreddit}
        plan = [row for row in plan if row[0].lower() in wanted]
        for name in args.subreddit:
            if name.lower() not in {row[0].lower() for row in SUBREDDITS}:
                # Unknown sub: no verified flair list exists, so accept any
                # image post rather than guess a flair string that matches
                # nothing and silently yields zero.
                plan.append((name, args.target or 100, None, args.min_score or 0))

    records = load_outfit_records()
    existing_ids = {r["source_id"] for r in records if r.get("source") == "reddit"}
    seen_hashes = load_seen_hashes(records)
    print(f"Existing reddit outfit records: {len(existing_ids)} "
          f"({len(seen_hashes)} known image hashes)")

    stats = new_stats()
    summary = []
    for subreddit, target, allowed_flairs, min_score in plan:
        posts, images, scanned = scrape_subreddit(
            subreddit,
            args.target or target,
            None if args.ignore_flair else allowed_flairs,
            args.min_score if args.min_score is not None else min_score,
            existing_ids, seen_hashes, stats,
        )
        summary.append((subreddit, posts, images, scanned))

    final = load_outfit_records()
    total_images = sum(r.get("image_count", 0) for r in final)
    print("\n=== summary ===")
    for subreddit, posts, images, scanned in summary:
        print(f"  r/{subreddit:22s} {posts:4d} posts  {images:4d} images  ({scanned} scanned)")
    print(f"  skipped: {format_stats(stats)}")
    print(f"Total outfit_dataset records: {len(final)}, images: {total_images}")


if __name__ == "__main__":
    main()
