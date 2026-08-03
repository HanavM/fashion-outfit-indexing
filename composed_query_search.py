"""Scoped-down v1 of spec section 4.8's "image + text query" flow (e.g.
"this shoe with cargo jorts"), built against a real data constraint: the
spec's own worked example assumes an "outfit record" containing multiple
items that co-occurred in one real photo, which this catalog does not
have. `apparel_dataset` is single-product catalog photos (one garment per
image) across 6 brands -- there is no dataset yet of real outfit photos
with linked/co-occurring items. `segment_outfit.py` is a first-pass
multi-item detector for that future data, but it's only been smoke-tested
on 2 catalog photos, not benchmarked, and nothing has actually built the
"outfit record" the spec's step 3 needs (spec Phase 5 / this repo's
docs/roadmap.md Phase 3, still missing).

**What this script actually does, honestly**: given (1) an image of one
item and (2) a text fragment describing a SECOND, separately-desired item
or attribute, it runs TWO INDEPENDENT SEARCHES and returns both side by
side:
  1. `hierarchical_retrieval_pipeline.py`'s exact-image pipeline
     (SigLIP2 category/identity shortlist -> DINOv3 rerank) identifies the
     closest-matching catalog product for the query image.
  2. A lightweight facet parser splits the text fragment into a target
     category (matched against docs/hierarchy.json's real taxonomy, with a
     small slang/synonym table -- e.g. "jorts" -> "denim shorts") and any
     attribute keywords (matched against the real attribute vocabulary
     actually present in structured_caption.attributes across the
     catalog -- color/material/pattern/fit/closure/pocket_type/
     distressing/defining_features). The parsed category (if any) is used
     to filter `catalog_query_search.py`'s existing text search.

The two result sets are NOT claimed to have been worn together, found in
the same real photo, or confirmed compatible in any way -- that claim
would require spec Phase 5's outfit-conjunction retrieval, which itself
depends on Phase 3's still-missing multi-item real-photo data (see
docs/roadmap.md). This is "identify item A, separately find items matching
description B," clearly labeled as such in every output.

**Validation status, stated plainly, not overclaimed**:
- The facet parser (`parse_text_fragment`) and the catalog_query_search.py
  half (`composed_search`'s second_item_matches) were run and validated
  locally against real hand-written examples and the real catalog. Informal
  parser check, 15 hand-written fragments against my own hand-judged
  expected category (NOT a rigorous benchmark, no held-out set, one
  person's own examples/judgment): 15/15 category-parse matches after two
  real fixes made while testing -- (1) plural nouns ("loafers") weren't
  matching their singular taxonomy term ("loafer") until word-boundary
  matching was changed to allow an optional trailing "s"; (2) "khaki
  pants"/"striped polo" needed two more synonym-table entries
  (khaki->khakis, polo->polo sweater) since the bare adjective/noun alone
  didn't reach the actual taxonomy leaf. `second_item_matches` was spot-
  checked against real catalog output too -- e.g. "with cargo jorts"
  correctly surfaced a real product literally named "...Baggy Jorts..."
  and several other real denim-shorts products; "with khaki pants" and
  "with a bomber jacket" both returned real, correctly-categorized
  products. Original testing pass ran with an unactivated environment
  (system python, torch 2.0.0) that made catalog_query_search.py's
  semantic fallback look unavailable -- **corrected same day**: this
  repo's own `.venv` (torch 2.12.1/transformers 5.12.1) works fine.
  Re-tested with `.venv` active: a query with real canonical coverage
  ("with a distressed denim jacket") stayed on the lexical path as
  expected, and a genuinely novel phrasing with zero canonical match
  ("with embroidered lettering stitched across the back panel")
  correctly fell through to the semantic engine and returned real scored
  results. The semantic fallback path is confirmed working locally.
- **Update, 2026-08-02, later same day**: `identified_item` now HAS been
  run end to end locally, for the first time -- the HF_TOKEN blocker
  above is resolved (added to this repo's `.env`, auto-loaded by
  `hierarchical_retrieval_pipeline.py`). Ran `--image <real product
  photo> --text "with white sneakers"`: `identified_item` correctly
  returned the query photo's own product back, and `second_item_matches`
  correctly surfaced real white sneakers for the parsed "sneaker"
  category. **Important caveat this result does NOT resolve**: no
  fine-tuned DINOv3 checkpoint exists in this dev environment (only
  reachable from Colab Drive) -- this ran against the FROZEN base
  DINOv3 model, and the query image was almost certainly already inside
  the gallery it searched (no held-out split applied for this ad-hoc
  smoke test), so "found itself" is a real confirmation the CODE PATH
  works end to end without crashing, not a discrimination-accuracy
  result. The real fine-tuned-checkpoint accuracy of `identified_item`
  still needs validation on Colab/Modal wherever the real DINOv3
  checkpoint is reachable -- what changed today is "never ran, unknown
  if it even works" becoming "runs correctly, mechanism confirmed,
  accuracy still unmeasured."

Usage:
    python3 composed_query_search.py --image path/to/query.jpg --text "with cargo jorts" --top-k 10
    python3 composed_query_search.py --text "with cargo jorts" --top-k 10   # text-only half, no image stage
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# Facet parser -- pure string/data matching, no model weights, safe to
# import and test in isolation (deliberately does NOT import
# hierarchical_retrieval_pipeline.py at module load time, since that pulls
# in torch/transformers and starts verifying the whole catalog's images).
# ============================================================

_HIERARCHY_CANDIDATES = [
    Path(__file__).parent / "hierarchy.json",
    Path(__file__).parent / "docs" / "hierarchy.json",
]
HIERARCHY_PATH = next((p for p in _HIERARCHY_CANDIDATES if p.is_file()), _HIERARCHY_CANDIDATES[0])

STOPWORDS = {
    "a", "an", "the", "with", "and", "in", "for", "of", "on", "some", "this",
    "that", "these", "those", "wearing", "worn", "plus", "also",
}

# Small, deliberately non-exhaustive slang/synonym table -- keyword
# matching against the real taxonomy, not a full NLP system. Each key maps
# to a real leaf or category string that exists in docs/hierarchy.json.
CATEGORY_SYNONYMS = {
    "jorts": "denim shorts",
    "jort": "denim shorts",
    "cargos": "cargo pants",
    "cargo": "cargo pants",  # only used as a category cue if no more specific term wins; "cargo" is also
                              # an attribute keyword below (cargo pockets) -- category match is tried first,
                              # so a bare "cargo" alone maps to cargo pants rather than staying unmatched.
    "kicks": "sneaker",
    "sneakers": "sneaker",
    "trainers": "sneaker",
    "tee": "t-shirt",
    "tees": "t-shirt",
    "tshirt": "t-shirt",
    "t-shirts": "t-shirt",
    "hoodies": "hoodie",
    "sweats": "sweatpants",
    "sweatpants": "sweatpants",
    "joggers": "joggers",
    "cap": "baseball cap",
    "caps": "baseball cap",
    "hats": "hat",
    "socks": "socks",
    "puffer": "puffer jacket",
    "windbreaker": "track jacket",
    "coat": "jacket",
    "denim jacket": "jacket",  # "denim jacket" leaf doesn't exist in the taxonomy (no denim-specific
                                # jacket leaf) -- backs off to the "jacket" category, "denim" surfaces
                                # separately as a material attribute keyword instead.
    "khaki": "khakis",
    "polo": "polo sweater",
    "polo shirt": "polo sweater",
}


def _load_hierarchy():
    with HIERARCHY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_category_vocab(hierarchy):
    """term (lowercase) -> {"leaf": str or None, "category": str, "group": str}.
    Both leaf-level and category-level terms are indexed -- a leaf is more
    specific and preferred when both match."""
    vocab = {}
    for group, categories in hierarchy.items():
        for category, leaves in categories.items():
            vocab[category.lower()] = {"leaf": None, "category": category, "group": group}
            for leaf in leaves:
                vocab[leaf.lower()] = {"leaf": leaf, "category": category, "group": group}
    return vocab


ATTRIBUTE_FIELDS = [
    "color", "material", "pattern", "fit", "length", "silhouette",
    "closure", "pocket_type", "distressing",
]


def _build_attribute_vocab(metadata):
    """token (lowercase word, len>=3) -> Counter[(field, canonical_value)] --
    built from the REAL structured_caption.attributes values actually
    present in the catalog, not an assumed/hand-written list. Includes
    defining_features' free-text `feature` strings, which is where
    construction details like "cargo pockets" actually live (confirmed by
    inspecting the real data before writing this -- plain `pocket_type` is
    empty across the whole catalog as of this writing, defining_features
    is where that signal actually shows up)."""
    token_index = defaultdict(Counter)

    def index_value(field, value):
        value = str(value).strip().lower()
        if not value:
            return
        for word in re.findall(r"[a-z][a-z\-']+", value):
            if len(word) >= 3 and word not in STOPWORDS:
                token_index[word][(field, value)] += 1

    for product in metadata:
        attributes = ((product.get("structured_caption") or {}).get("attributes") or {})
        for field in ATTRIBUTE_FIELDS:
            for value in attributes.get(field) or []:
                index_value(field, value)
        for entry in attributes.get("defining_features") or []:
            feature = entry.get("feature") if isinstance(entry, dict) else entry
            if feature:
                index_value("defining_features", feature)

    return token_index


_CATEGORY_VOCAB_CACHE = None
_ATTRIBUTE_VOCAB_CACHE = None


def get_category_vocab():
    global _CATEGORY_VOCAB_CACHE
    if _CATEGORY_VOCAB_CACHE is None:
        _CATEGORY_VOCAB_CACHE = _build_category_vocab(_load_hierarchy())
    return _CATEGORY_VOCAB_CACHE


def get_attribute_vocab(metadata_path):
    global _ATTRIBUTE_VOCAB_CACHE
    if _ATTRIBUTE_VOCAB_CACHE is None:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        _ATTRIBUTE_VOCAB_CACHE = _build_attribute_vocab(metadata)
    return _ATTRIBUTE_VOCAB_CACHE


def _tokenize(text):
    return re.findall(r"[a-z][a-z\-']+", text.lower())


def _term_pattern(term):
    """Word-boundary match with an optional trailing 's' on the term's
    last word, so a singular taxonomy term ("loafer") still matches the
    plural a user actually types ("loafers") without needing a dedicated
    synonym-table entry for every plural."""
    return re.compile(r"\b" + re.escape(term) + r"s?\b")


def _find_category_match(fragment, category_vocab):
    """Scans the normalized fragment for the most specific real taxonomy
    term (leaf beats category, multi-word beats single-word), checking
    both real vocab terms and the synonym table. Returns
    (matched_term, leaf_or_none, category, group, consumed_words) or None."""
    normalized = re.sub(r"[^a-z0-9\s]", " ", fragment.lower())

    candidates = []  # (specificity_rank, term_word_count, term, info)
    for term, info in category_vocab.items():
        if _term_pattern(term).search(normalized):
            specificity = 2 if info["leaf"] else 1
            candidates.append((specificity, len(term.split()), term, info))
    for synonym, target_term in CATEGORY_SYNONYMS.items():
        if _term_pattern(synonym).search(normalized) and target_term.lower() in category_vocab:
            info = category_vocab[target_term.lower()]
            specificity = 2 if info["leaf"] else 1
            candidates.append((specificity, len(synonym.split()), synonym, info))

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    _, _, matched_term, info = candidates[0]
    consumed_words = set(_tokenize(matched_term))
    return {
        "matched_term": matched_term,
        "leaf": info["leaf"],
        "category": info["category"],
        "group": info["group"],
        "consumed_words": consumed_words,
    }


def _find_attribute_matches(remaining_words, attribute_vocab, max_matches=5):
    """For each leftover word (after the category match's words are
    removed), looks up the real attribute vocabulary and reports the
    field(s) + example real catalog value(s) it came from. Returns a list
    of {"keyword", "fields", "example_values"} dicts, most-evidenced
    first."""
    matches = []
    seen_keywords = set()
    for word in remaining_words:
        if word in STOPWORDS or word in seen_keywords or word not in attribute_vocab:
            continue
        seen_keywords.add(word)
        field_counter = Counter()
        value_counter = Counter()
        for (field, value), count in attribute_vocab[word].items():
            field_counter[field] += count
            value_counter[value] += count
        matches.append({
            "keyword": word,
            "fields": [f for f, _ in field_counter.most_common()],
            "example_values": [v for v, _ in value_counter.most_common(3)],
            "evidence_count": sum(field_counter.values()),
        })
    matches.sort(key=lambda m: m["evidence_count"], reverse=True)
    return matches[:max_matches]


def parse_text_fragment(text_fragment, metadata_path):
    """The lightweight facet parser: splits a fragment like "with cargo
    jorts" into a target category (real taxonomy leaf/category/group, or
    None if nothing matched) and attribute keywords (matched against the
    real catalog vocabulary). Deliberately simple substring/token matching
    -- not a full NLP parser, per the task's own instruction not to
    overbuild this."""
    attribute_vocab = get_attribute_vocab(metadata_path)
    category_vocab = get_category_vocab()

    category_match = _find_category_match(text_fragment, category_vocab)
    all_words = _tokenize(text_fragment)
    consumed = category_match["consumed_words"] if category_match else set()
    remaining_words = [w for w in all_words if w not in consumed]

    attribute_matches = _find_attribute_matches(remaining_words, attribute_vocab)

    return {
        "raw_text": text_fragment,
        "category": {
            "leaf": category_match["leaf"] if category_match else None,
            "category": category_match["category"] if category_match else None,
            "group": category_match["group"] if category_match else None,
            "matched_term": category_match["matched_term"] if category_match else None,
        } if category_match else None,
        "attributes": attribute_matches,
    }


# ============================================================
# Combined search -- pulls in the heavy modules lazily, only when actually
# invoked, so parse_text_fragment stays cheaply testable on its own.
# ============================================================

def composed_search(image_path, text_fragment, top_k=10, metadata_path=None, retriever=None, canonical_only=False):
    """Runs two independent searches and returns both, clearly labeled.
    `retriever`: an already-constructed HierarchicalRetriever, so callers
    doing multiple queries don't pay model-load cost per call. If omitted
    and image_path is given, one is constructed here (slow: loads SigLIP2
    + DINOv3 and builds/reads the catalog indexes).
    `canonical_only`: skip catalog_query_search.py's semantic-embedding
    fallback (no model weights loaded at all for the text side) -- useful
    on a machine where the model stack isn't usable, or just to force the
    cheap/fast lexical-only path. (Confirmed working with this repo's own
    `.venv` as of 2026-08-02 -- an earlier note here about torch 2.0.0
    making this unavailable was an unactivated-environment issue, not a
    real limitation.)"""
    from catalog_query_search import CatalogQuerySearch, METADATA_PATH as CATALOG_METADATA_PATH

    metadata_path = metadata_path or CATALOG_METADATA_PATH
    parsed = parse_text_fragment(text_fragment, metadata_path)

    identified_item = None
    if image_path is not None:
        if retriever is None:
            from hierarchical_retrieval_pipeline import HierarchicalRetriever
            retriever = HierarchicalRetriever()
        raw_result = retriever.retrieve(str(image_path), final_top_k=1)
        identified_item = raw_result["results"][0] if raw_result["results"] else None

    engine = CatalogQuerySearch()
    # Query catalog_query_search.py with the PARSED terms, not the raw
    # fragment verbatim -- slang the synonym table resolved (e.g. "jorts")
    # generally isn't itself a substring of any real canonical label, only
    # the taxonomy term it resolved to is ("denim shorts"). Falls back to
    # the raw fragment if nothing parsed at all (parser found no category
    # or attribute signal, so there's nothing better to search with).
    parsed_category = parsed["category"] or {}
    category_term = parsed_category.get("leaf") or parsed_category.get("category")
    attribute_keywords = [a["keyword"] for a in parsed["attributes"]]
    if category_term or attribute_keywords:
        query_for_second_item = " ".join(attribute_keywords + ([category_term] if category_term else []))
    else:
        query_for_second_item = text_fragment
    try:
        raw_hits = engine.search(query_for_second_item, top_k=top_k * 3, canonical_only=canonical_only)
    except Exception as error:
        # Defensive fallback, not silent: the semantic-embedding stage
        # needs a working local model stack (torch/transformers), which
        # isn't guaranteed on every machine this script runs on. Degrade to
        # canonical-only rather than crashing the whole composed query over
        # a fallback stage that was never the primary signal anyway.
        print(f"  (semantic fallback unavailable ({error!r}) -- falling back to canonical-only text match)")
        raw_hits = engine.search(query_for_second_item, top_k=top_k * 3, canonical_only=True)

    target_category = (parsed["category"] or {}).get("category")
    if target_category:
        code_to_product = engine.code_to_product
        filtered = []
        for hit in raw_hits:
            product = code_to_product.get(hit["product_code"], {})
            structured = product.get("structured_caption") or {}
            canonical_path = structured.get("canonical_taxonomy_path") or []
            product_category = None
            for group, categories in _load_hierarchy().items():
                for category, leaves in categories.items():
                    leaf_names = {category.lower()} | {leaf.lower() for leaf in leaves}
                    if any(str(p).lower() in leaf_names for p in canonical_path):
                        product_category = category
            if product_category and product_category.lower() == target_category.lower():
                filtered.append(hit)
        second_item_matches = filtered[:top_k] if filtered else raw_hits[:top_k]
        category_filter_applied = bool(filtered)
    else:
        second_item_matches = raw_hits[:top_k]
        category_filter_applied = False

    return {
        "identified_item": identified_item,
        "parsed_text_query": parsed,
        "second_item_matches": second_item_matches,
        "category_filter_applied": category_filter_applied,
        "note": (
            "identified_item and second_item_matches are TWO INDEPENDENT "
            "SEARCHES, not a confirmed outfit -- this system has no data "
            "showing these items were ever worn together in a real photo "
            "(that needs multi-item outfit-photo detection, not built yet; "
            "see docs/roadmap.md Phase 3)."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to the query image of item A.")
    parser.add_argument("--text", type=str, required=True, help='Text fragment describing item B, e.g. "with cargo jorts".')
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--metadata", type=str, default=None, help="Override path to metadata.json (defaults to catalog_query_search.py's own default).")
    parser.add_argument("--canonical-only", action="store_true",
                         help="Skip catalog_query_search.py's semantic-embedding fallback for the text side "
                              "(no model weights loaded for that stage). Also engaged automatically if the "
                              "semantic fallback errors out (e.g. no usable local torch/transformers stack).")
    args = parser.parse_args()

    result = composed_search(args.image, args.text, top_k=args.top_k, metadata_path=args.metadata, canonical_only=args.canonical_only)

    print(f"\nText fragment: '{args.text}'")
    parsed = result["parsed_text_query"]
    if parsed["category"]:
        c = parsed["category"]
        print(f"  Parsed category: leaf={c['leaf']!r} category={c['category']!r} group={c['group']!r} (matched on {c['matched_term']!r})")
    else:
        print("  Parsed category: none matched")
    if parsed["attributes"]:
        print("  Parsed attribute keywords:")
        for a in parsed["attributes"]:
            print(f"    '{a['keyword']}'  fields={a['fields']}  example real values={a['example_values']}")
    else:
        print("  Parsed attribute keywords: none")

    if args.image:
        print(f"\nidentified_item (from {args.image}):")
        print(f"  {result['identified_item']}")
    else:
        print("\n(no --image given -- identified_item skipped)")

    print(f"\nsecond_item_matches (category filter applied: {result['category_filter_applied']}):")
    for rank, hit in enumerate(result["second_item_matches"], start=1):
        label = hit.get("matched_label") or f"semantic score={hit.get('score', 0):.4f}"
        print(f"  #{rank}  [{hit['match_type']}: {label}]  {hit['brand']} {hit['name']}  [{hit['product_code']}]")

    print(f"\nNOTE: {result['note']}")
