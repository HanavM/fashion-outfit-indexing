"""Canonicalize the messy, LLM-generated `structured_caption.taxonomy_path`
values in apparel_dataset/metadata.json into one consistent hierarchy tree,
and write it out as HIERARCHY (a nested dict) for use by any zero-shot HSC
climbing code (replaces the notebook's old hand-written shoe-only dict --
see docs/roadmap.md Phase 0/1 deferred item).

Why this exists: raw taxonomy_path values disagree with themselves across
records for the same real-world category -- e.g. jeans shows up as
('apparel','bottoms','pants','jeans'), ('apparel','pants','jeans','baggy
jeans'), and ('apparel','bottoms','pants','jeans','straight jeans') all in
the same dataset, and hoodie shows up under three different second-level
roots ('hoodies and pullovers' / 'outerwear' / 'top'). None of that is
usable as a hierarchy as-is.

Design principle (per user direction): the *category* level (what a
scraper targets and what a SigLIP-style classifier's coarse label is) must
be a visually distinct, non-overlapping silhouette bucket. Fine-grained
distinctions that look nearly identical on a garment (jeans vs. chinos vs.
cargo pants -- all "pants" silhouette) are demoted to *attributes* / leaf
labels under one category node, not separate category nodes. This still
satisfies the spec's broad-semantic-retrieval requirement ("blue jeans"
must be findable, per docs/project_spec_v1.md 8.1) because the leaf label
is still present in the tree and in positive_texts -- it just isn't a
*scrape target* or a *training class* competing with visually-identical
siblings.

Root/group level (2026-08-01 restructuring, per user direction to
"finalize the categories intelligently"): was 3 groups (footwear,
apparel, accessory) with "apparel" flatly lumping tops, bottoms, AND
outerwear into one bucket -- no body-region distinction at all. Split
into 5 groups instead: footwear, tops, bottoms, outerwear, accessories.
Two reasons this matters, not just tidiness:
  1. HSC category gating (hierarchical_retrieval_pipeline.py) backs off
     to "all categories under this group" when confidence is real but not
     leaf-specific -- under the old 3-group tree, backing off from a
     low-confidence jacket call landed on "apparel" and included every
     t-shirt/pants/short in the gate too. Under 5 groups, the same
     backoff lands on "outerwear" only, a real precision improvement for
     free (same climbing algorithm, tighter groups).
  2. This is also the target ontology multi-item outfit-photo detection
     (docs/roadmap.md's Tier 3 gap, spec section 4.2 -- not built yet)
     will need once it exists: a detector's classes have to be spatially/
     visually separable regions of a photo, and "top" vs. "bottom" vs.
     "outerwear" vs. "footwear" vs. "accessory" is exactly that kind of
     class set (a jacket worn open over a t-shirt is two simultaneously-
     visible, separately-locatable items -- they need to be two different
     target classes, not one shared "apparel" bucket). Choosing this
     grouping now means the eventual detector's output vocabulary already
     matches what hierarchical_retrieval_pipeline.py expects downstream,
     no separate taxonomy to reconcile later.
Outerwear = jacket-family items only (bomber/field/puffer/track/shirt
jacket, coat, windbreaker, denim jacket) -- hoodies and sweaters/
sweatshirts stayed in "tops" rather than "outerwear" despite sometimes
being worn as an outer layer, matching the standard convention (also
DeepFashion2's) that "outerwear" means garments specifically cut/sold as
an outer layer over other clothing, not mid-layers that can incidentally
be worn last.

Writes:
  - docs/hierarchy.json           the canonical HIERARCHY tree
  - Adds `structured_caption.canonical_taxonomy_path` to every record in
    apparel_dataset/metadata.json (new field, non-destructive -- the
    original `taxonomy_path` is left untouched, same convention as
    caption vs. structured_caption).
"""

import json
from pathlib import Path

import dataset_utils

METADATA_PATH = Path("apparel_dataset/metadata.json")
HIERARCHY_OUT = Path("docs/hierarchy.json")

# ============================================================
# Canonicalization rules
# ============================================================
# Maps every distinct raw taxonomy_path leaf token (case-insensitive) to
# (canonical_root, canonical_category, leaf_or_None).
# canonical_category is the visually-distinct scrape/classifier bucket.
# leaf (if not None) is kept as a fine-grained retrieval label, not a
# separate category.

CANONICAL_CATEGORY_OF_LEAF = {
    # footwear
    "sneaker": ("footwear", "sneaker", None),
    "low-top sneaker": ("footwear", "sneaker", "low-top sneaker"),
    "golf sneaker": ("footwear", "sneaker", "golf sneaker"),
    "indoor sneaker": ("footwear", "sneaker", "indoor sneaker"),
    # loafers were mis-tagged under the sneaker root by the captioning LLM --
    # a loafer is not a sneaker silhouette. Correct it here rather than
    # propagate the error.
    "loafer": ("footwear", "loafer", None),
    "loafers": ("footwear", "loafer", None),
    "loafer sneaker": ("footwear", "loafer", None),

    # tops
    "t-shirt": ("tops", "t-shirt", None),
    "graphic tee": ("tops", "t-shirt", "graphic tee"),
    "graphic t-shirt": ("tops", "t-shirt", "graphic tee"),
    "crop t-shirt": ("tops", "t-shirt", "crop t-shirt"),
    "tank top": ("tops", "tank top", None),
    "muscle tank top": ("tops", "tank top", "muscle tank top"),
    "shirt": ("tops", "shirt", None),
    "button-down shirt": ("tops", "shirt", "button-down shirt"),
    "sweatshirt": ("tops", "sweatshirt", None),
    "hoodie": ("tops", "hoodie", None),
    "pullover hoodie": ("tops", "hoodie", "pullover hoodie"),
    "sweater": ("tops", "sweater", None),
    "jersey": ("tops", "sweater", "jersey"),
    "polo sweater": ("tops", "sweater", "polo sweater"),
    "sweater pant": ("bottoms", "pants", "sweater pant"),  # miscategorized leaf, it's a bottom

    # bottoms -- every denim/fit/material variant folds into "pants" as a
    # leaf label, per the no-overlapping-visual-categories rule.
    "pants": ("bottoms", "pants", None),
    "baggy pants": ("bottoms", "pants", "baggy pants"),
    "straight pants": ("bottoms", "pants", "straight pants"),
    "relaxed taper pants": ("bottoms", "pants", "relaxed taper pants"),
    "cargo pants": ("bottoms", "pants", "cargo pants"),
    "khakis": ("bottoms", "pants", "khakis"),
    "trousers": ("bottoms", "pants", "trousers"),
    "track pants": ("bottoms", "pants", "track pants"),
    "sweatpants": ("bottoms", "pants", "sweatpants"),
    "joggers": ("bottoms", "pants", "joggers"),
    "tights": ("bottoms", "pants", "tights"),
    "leggings": ("bottoms", "pants", "leggings"),
    "briefs": ("bottoms", "pants", "briefs"),
    "jeans": ("bottoms", "pants", "jeans"),
    "baggy jeans": ("bottoms", "pants", "baggy jeans"),
    "straight jeans": ("bottoms", "pants", "straight jeans"),
    "cargo jeans": ("bottoms", "pants", "cargo jeans"),

    "shorts": ("bottoms", "shorts", None),
    "biker shorts": ("bottoms", "shorts", "biker shorts"),
    "cargo shorts": ("bottoms", "shorts", "cargo shorts"),
    "denim shorts": ("bottoms", "shorts", "denim shorts"),
    "jean shorts": ("bottoms", "shorts", "jean shorts"),
    "volley shorts": ("bottoms", "shorts", "volley shorts"),

    # New Gap categories (Jackets/Hats/Socks) -- structured_caption hasn't
    # been generated for these yet (caption_apparel.py runs after the
    # scrape finishes), so these leaf names are anticipated, not confirmed
    # against real LLM output. Re-run this script after captioning and
    # check the "no mapping found" report for anything actually generated
    # that isn't covered here.
    "jacket": ("outerwear", "jacket", None),
    "coat": ("outerwear", "jacket", "coat"),
    "denim jacket": ("outerwear", "jacket", "denim jacket"),
    "bomber jacket": ("outerwear", "jacket", "bomber jacket"),
    "puffer jacket": ("outerwear", "jacket", "puffer jacket"),
    "track jacket": ("outerwear", "jacket", "track jacket"),
    "field jacket": ("outerwear", "jacket", "field jacket"),
    "shirt jacket": ("outerwear", "jacket", "shirt jacket"),
    "windbreaker": ("outerwear", "jacket", "windbreaker"),
    "outerwear": ("outerwear", "jacket", None),

    "hat": ("accessories", "hat", None),
    "cap": ("accessories", "hat", "cap"),
    "baseball cap": ("accessories", "hat", "baseball cap"),
    "baseball hat": ("accessories", "hat", "baseball cap"),
    "beanie": ("accessories", "hat", "beanie"),

    "sock": ("accessories", "socks", None),
    "socks": ("accessories", "socks", None),
    "crew socks": ("accessories", "socks", "crew socks"),
    "athletic socks": ("accessories", "socks", "athletic socks"),
}


def canonicalize(taxonomy_path):
    """Returns (canonical_root, canonical_category, leaf_label_or_None)."""

    if not taxonomy_path:
        return None

    # Prefer the most specific (last) leaf token that we have a mapping
    # for -- walk from the end backwards so e.g. "baggy jeans" wins over
    # "jeans" wins over "pants" if all three happen to be present.
    for token in reversed(taxonomy_path):
        key = str(token).strip().lower()
        if key in CANONICAL_CATEGORY_OF_LEAF:
            return CANONICAL_CATEGORY_OF_LEAF[key]

    return None


def main():
    # dataset_utils.load_records(), not a bare METADATA_PATH.read_text() --
    # this file is written to concurrently by long-running scraper/caption/
    # segmentation scripts (see dataset_utils.py's own docstring for the
    # real incident that made this mandatory). A blind read-modify-write
    # here caused exactly that failure mode for real on 2026-08-01: running
    # this script while two scraper forks were mid-write reverted
    # metadata.json to this script's stale read-time snapshot, discarding
    # dozens of records the forks had appended in the meantime. Recovery
    # was re-running the affected scrapers (idempotent, only backfills
    # missing product_codes) -- but the real fix is this script never
    # blindly overwriting the whole file again.
    metadata = dataset_utils.load_records()

    hierarchy = {}
    unmapped = set()
    touched = {}

    for product in metadata:
        sc = product.get("structured_caption")
        if not sc:
            continue

        raw_path = sc.get("taxonomy_path") or []
        result = canonicalize(raw_path)

        if result is None:
            if raw_path:
                unmapped.add(tuple(raw_path))
            continue

        root, category, leaf = result
        canonical_path = [root, category] + ([leaf] if leaf else [])
        sc["canonical_taxonomy_path"] = canonical_path
        touched[product["product_code"]] = product

        node = hierarchy.setdefault(root, {}).setdefault(category, set())
        if leaf:
            node.add(leaf)

    # sets -> sorted lists for JSON
    hierarchy_json = {
        root: {category: sorted(leaves) for category, leaves in categories.items()}
        for root, categories in hierarchy.items()
    }

    HIERARCHY_OUT.parent.mkdir(parents=True, exist_ok=True)
    HIERARCHY_OUT.write_text(json.dumps(hierarchy_json, indent=2), encoding="utf-8")

    dataset_utils.save_records_safe(touched)

    print(f"Updated {len(touched)} records with canonical_taxonomy_path")
    print(f"Wrote hierarchy tree to {HIERARCHY_OUT}")
    print(f"\nCanonical categories found:")
    for root, categories in hierarchy_json.items():
        for category, leaves in categories.items():
            print(f"  {root}/{category}  ({len(leaves)} fine-grained leaf labels)")

    if unmapped:
        print(f"\n{len(unmapped)} distinct raw taxonomy_path values had no mapping (left untouched):")
        for tp in sorted(unmapped):
            print(f"  {tp}")


if __name__ == "__main__":
    main()
