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

Writes:
  - docs/hierarchy.json           the canonical HIERARCHY tree
  - Adds `structured_caption.canonical_taxonomy_path` to every record in
    apparel_dataset/metadata.json (new field, non-destructive -- the
    original `taxonomy_path` is left untouched, same convention as
    caption vs. structured_caption).
"""

import json
from pathlib import Path

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
    "t-shirt": ("apparel", "t-shirt", None),
    "graphic tee": ("apparel", "t-shirt", "graphic tee"),
    "graphic t-shirt": ("apparel", "t-shirt", "graphic tee"),
    "crop t-shirt": ("apparel", "t-shirt", "crop t-shirt"),
    "tank top": ("apparel", "tank top", None),
    "muscle tank top": ("apparel", "tank top", "muscle tank top"),
    "shirt": ("apparel", "shirt", None),
    "button-down shirt": ("apparel", "shirt", "button-down shirt"),
    "sweatshirt": ("apparel", "sweatshirt", None),
    "hoodie": ("apparel", "hoodie", None),
    "pullover hoodie": ("apparel", "hoodie", "pullover hoodie"),
    "sweater": ("apparel", "sweater", None),
    "jersey": ("apparel", "sweater", "jersey"),
    "polo sweater": ("apparel", "sweater", "polo sweater"),
    "sweater pant": ("apparel", "pants", "sweater pant"),  # miscategorized leaf, it's a bottom

    # bottoms -- every denim/fit/material variant folds into "pants" as a
    # leaf label, per the no-overlapping-visual-categories rule.
    "pants": ("apparel", "pants", None),
    "baggy pants": ("apparel", "pants", "baggy pants"),
    "straight pants": ("apparel", "pants", "straight pants"),
    "relaxed taper pants": ("apparel", "pants", "relaxed taper pants"),
    "cargo pants": ("apparel", "pants", "cargo pants"),
    "khakis": ("apparel", "pants", "khakis"),
    "trousers": ("apparel", "pants", "trousers"),
    "track pants": ("apparel", "pants", "track pants"),
    "sweatpants": ("apparel", "pants", "sweatpants"),
    "joggers": ("apparel", "pants", "joggers"),
    "tights": ("apparel", "pants", "tights"),
    "leggings": ("apparel", "pants", "leggings"),
    "briefs": ("apparel", "pants", "briefs"),
    "jeans": ("apparel", "pants", "jeans"),
    "baggy jeans": ("apparel", "pants", "baggy jeans"),
    "straight jeans": ("apparel", "pants", "straight jeans"),
    "cargo jeans": ("apparel", "pants", "cargo jeans"),

    "shorts": ("apparel", "shorts", None),
    "biker shorts": ("apparel", "shorts", "biker shorts"),
    "cargo shorts": ("apparel", "shorts", "cargo shorts"),
    "denim shorts": ("apparel", "shorts", "denim shorts"),
    "jean shorts": ("apparel", "shorts", "jean shorts"),
    "volley shorts": ("apparel", "shorts", "volley shorts"),

    # New Gap categories (Jackets/Hats/Socks) -- structured_caption hasn't
    # been generated for these yet (caption_apparel.py runs after the
    # scrape finishes), so these leaf names are anticipated, not confirmed
    # against real LLM output. Re-run this script after captioning and
    # check the "no mapping found" report for anything actually generated
    # that isn't covered here.
    "jacket": ("apparel", "jacket", None),
    "coat": ("apparel", "jacket", "coat"),
    "denim jacket": ("apparel", "jacket", "denim jacket"),
    "bomber jacket": ("apparel", "jacket", "bomber jacket"),
    "puffer jacket": ("apparel", "jacket", "puffer jacket"),
    "track jacket": ("apparel", "jacket", "track jacket"),
    "field jacket": ("apparel", "jacket", "field jacket"),
    "shirt jacket": ("apparel", "jacket", "shirt jacket"),
    "windbreaker": ("apparel", "jacket", "windbreaker"),
    "outerwear": ("apparel", "jacket", None),

    "hat": ("accessory", "hat", None),
    "cap": ("accessory", "hat", "cap"),
    "baseball cap": ("accessory", "hat", "baseball cap"),
    "baseball hat": ("accessory", "hat", "baseball cap"),
    "beanie": ("accessory", "hat", "beanie"),

    "sock": ("accessory", "socks", None),
    "socks": ("accessory", "socks", None),
    "crew socks": ("accessory", "socks", "crew socks"),
    "athletic socks": ("accessory", "socks", "athletic socks"),
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
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    hierarchy = {}
    unmapped = set()
    updated = 0

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
        updated += 1

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

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Updated {updated} records with canonical_taxonomy_path")
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
