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

**Update, 2026-08-04 -- the "two independent searches" caveat is now
conditional, because there is finally real co-occurrence evidence.**
Roadmap 11.1 landed: `index_outfits.py` ran multi-item detection over
`outfit_dataset/`'s real outfit photos and `build_outfit_cooccurrence.py`
aggregated the result into `outfit_cooccurrence.json` -- an index of which
garment categories were actually detected together on the same real
person. Two changes here follow from it:

  1. When the query image's category and the requested category BOTH appear
     in that index, the response now carries an `outfit_evidence` block
     (how many outfits showed them together, p(b|a), lift) and the note
     stops saying nothing shows these worn together, because something
     does. Colour evidence, where present, re-ranks (never filters)
     `second_item_matches`.
  2. `--text` is now optional. With an image alone, companions are
     PROPOSED from co-occurrence -- "what actually gets worn with this" --
     rather than requiring the user to already know what they want. That
     is the outfit-level result the spec's architecture diagram asks for.

The old independent-search path is untouched and is still what runs when
the index is missing, when the anchor category is unknown, or when the
corpus never showed the requested pairing. That fallback is deliberate:
thin evidence must degrade to the honest old answer, not to a confident
wrong one.

**The evidence is model-derived and has no ground truth.** The outfit
corpus is deliberately unlabelled (SCRAPING_PROCESS.md, 2026-08-03); the
categories behind every count are SAM2 masks scored zero-shot by
FashionCLIP, with precision and recall unmeasured. `outfit_evidence`
therefore reports what the DETECTOR saw, and says so, in the payload.

Usage:
    python3 composed_query_search.py --image path/to/query.jpg --text "with cargo jorts" --top-k 10
    python3 composed_query_search.py --text "with cargo jorts" --top-k 10   # text-only half, no image stage
    python3 composed_query_search.py --image path/to/query.jpg              # companions proposed from co-occurrence
    python3 composed_query_search.py --anchor-category jacket              # co-occurrence only, no model load
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
    # heel_type/sole_type/toe_shape were missing here despite being added
    # to newLLMprompt.py's schema the same day this file was written
    # (commit eb4fcb0) and to both by-facet eval scripts (commit 02a5a9a)
    # -- found via code review, 2026-08-02. Without these, a query like
    # "with a chunky rubber sole" or "with a round toe" silently never
    # matched as an attribute keyword, even once footwear records have
    # these fields populated -- no crash, just a quiet vocabulary gap.
    "heel_type", "sole_type", "toe_shape",
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
# Catalog product -> taxonomy category.
#
# Lifted out of composed_search's inner loop (where it used to be an inline
# double loop over the whole hierarchy per hit) because the co-occurrence
# join needs the SAME mapping for the query image's own identified product.
# Both sides must agree on what "jacket" means or the anchor silently never
# matches the index.
# ============================================================

def product_category(product):
    """The hierarchy CATEGORY (not leaf) a catalog product sits in, or None.
    Category-level on purpose: that is the granularity the outfit detector
    emits, so it is the only granularity the two indexes can be joined on."""
    structured = product.get("structured_caption") or {}
    canonical_path = structured.get("canonical_taxonomy_path") or []
    if not canonical_path:
        return None
    # Reverse lookup built once. The version this replaced re-read and
    # re-walked the whole hierarchy for every hit of every query.
    lookup = _taxonomy_term_to_category()
    found = None
    for entry in canonical_path:
        category = lookup.get(str(entry).lower())
        if category:
            found = category
    return found


_TERM_TO_CATEGORY_CACHE = None


def _taxonomy_term_to_category():
    """Every taxonomy term (category name or leaf) -> its category."""
    global _TERM_TO_CATEGORY_CACHE
    if _TERM_TO_CATEGORY_CACHE is None:
        lookup = {}
        for _group, categories in _load_hierarchy().items():
            for category, leaves in categories.items():
                lookup[category.lower()] = category
                for leaf in leaves:
                    lookup[leaf.lower()] = category
        _TERM_TO_CATEGORY_CACHE = lookup
    return _TERM_TO_CATEGORY_CACHE


def product_colors(product):
    structured = product.get("structured_caption") or {}
    attributes = structured.get("attributes") or {}
    return [str(c).strip().lower() for c in (attributes.get("color") or []) if str(c).strip()]


# ============================================================
# Combined search -- pulls in the heavy modules lazily, only when actually
# invoked, so parse_text_fragment stays cheaply testable on its own.
# ============================================================

COOCCURRENCE_PATH = Path(__file__).parent / "outfit_cooccurrence.json"

# A pairing seen in fewer outfits than this is not reported as evidence.
# Under it, one or two detector mistakes are the entire signal.
MIN_EVIDENCE_OUTFITS = 5

INDEPENDENT_SEARCH_NOTE = (
    "identified_item and second_item_matches are TWO INDEPENDENT "
    "SEARCHES, not a confirmed outfit -- no co-occurrence evidence was "
    "available for this query, so nothing here shows these items were "
    "ever worn together in a real photo."
)

def composed_search(image_path, text_fragment, top_k=10, metadata_path=None, retriever=None,
                    canonical_only=False, use_cooccurrence=True,
                    cooccurrence_path=COOCCURRENCE_PATH, anchor_category=None):
    """Identify the item in `image_path`, then find a companion item.

    The companion comes from real co-occurrence when the outfit index has
    evidence for it, and from the old independent text search when it does
    not. `retriever`: an already-constructed HierarchicalRetriever, so
    callers doing multiple queries don't pay model-load cost per call. If
    omitted and image_path is given, one is constructed here (slow: loads
    SigLIP2 + DINOv3 and builds/reads the catalog indexes).
    `canonical_only`: skip catalog_query_search.py's semantic-embedding
    fallback (no model weights loaded at all for the text side) -- useful
    on a machine where the model stack isn't usable, or just to force the
    cheap/fast lexical-only path. (Confirmed working with this repo's own
    `.venv` as of 2026-08-02 -- an earlier note here about torch 2.0.0
    making this unavailable was an unactivated-environment issue, not a
    real limitation.)
    `anchor_category`: override the category the co-occurrence lookup keys
    on, so the outfit half can be exercised without loading the image
    stack at all.
    `use_cooccurrence`: set False to force the pre-11.1 behaviour, which is
    what makes the two paths comparable rather than just replaced."""
    from catalog_query_search import CatalogQuerySearch, METADATA_PATH as CATALOG_METADATA_PATH

    metadata_path = metadata_path or CATALOG_METADATA_PATH
    parsed = parse_text_fragment(text_fragment, metadata_path) if text_fragment else None

    identified_item = None
    if image_path is not None:
        if retriever is None:
            from hierarchical_retrieval_pipeline import HierarchicalRetriever
            retriever = HierarchicalRetriever()
        raw_result = retriever.retrieve(str(image_path), final_top_k=1)
        identified_item = raw_result["results"][0] if raw_result["results"] else None

    engine = CatalogQuerySearch()

    # ---- anchor: what the query image actually is, in the taxonomy the
    # outfit detector speaks. Without this there is nothing to condition on.
    anchor_colors = []
    if anchor_category is None and identified_item is not None:
        anchor_product = engine.code_to_product.get(identified_item.get("product_code"), {})
        anchor_category = product_category(anchor_product)
        anchor_colors = product_colors(anchor_product)

    cooccurrence = None
    if use_cooccurrence:
        from build_outfit_cooccurrence import OutfitCooccurrence
        cooccurrence = OutfitCooccurrence.load_if_available(cooccurrence_path)

    companion_suggestions = []
    if cooccurrence is not None and anchor_category and cooccurrence.known_category(anchor_category):
        companion_suggestions = cooccurrence.companions(
            anchor_category, top_k=8, min_count=MIN_EVIDENCE_OUTFITS)
    # ---- what to search for.
    # Normally the user says ("with cargo jorts"). When they don't, the
    # co-occurrence index proposes it -- that proposal is the whole point of
    # 11.1, and it is the one case where the companion category is chosen
    # from observed outfits rather than from the user's own words.
    parsed_category = (parsed or {}).get("category") or {}
    category_term = parsed_category.get("leaf") or parsed_category.get("category")
    attribute_keywords = [a["keyword"] for a in (parsed or {}).get("attributes", [])]
    companion_source = "user_text"

    if not text_fragment:
        if not companion_suggestions:
            return {
                "identified_item": identified_item,
                "anchor_category": anchor_category,
                "parsed_text_query": None,
                "second_item_matches": [],
                "category_filter_applied": False,
                "companion_suggestions": [],
                "outfit_evidence": None,
                "cooccurrence_available": cooccurrence is not None,
                "note": (
                    "No text fragment was given and the outfit co-occurrence index "
                    "has no evidence for this item's category, so there is nothing "
                    "to suggest a companion from. Provide --text to use the "
                    "independent text-search path."
                ),
            }
        category_term = companion_suggestions[0]["category"]
        companion_source = "cooccurrence"

    # Query catalog_query_search.py with the PARSED terms, not the raw
    # fragment verbatim -- slang the synonym table resolved (e.g. "jorts")
    # generally isn't itself a substring of any real canonical label, only
    # the taxonomy term it resolved to is ("denim shorts"). Falls back to
    # the raw fragment if nothing parsed at all (parser found no category
    # or attribute signal, so there's nothing better to search with).
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

    target_category = parsed_category.get("category") or (
        category_term if companion_source == "cooccurrence" else None)
    code_to_product = engine.code_to_product
    if target_category:
        filtered = [hit for hit in raw_hits
                    if (product_category(code_to_product.get(hit["product_code"], {})) or "").lower()
                    == target_category.lower()]
        candidate_hits = filtered if filtered else raw_hits
        category_filter_applied = bool(filtered)
    else:
        candidate_hits = raw_hits
        category_filter_applied = False

    # ---- the actual outfit evidence, if any.
    outfit_evidence = None
    if cooccurrence is not None and anchor_category and target_category:
        raw_evidence = cooccurrence.evidence_for(anchor_category, target_category)
        if raw_evidence and raw_evidence["cooccurrence_count"] >= MIN_EVIDENCE_OUTFITS:
            outfit_evidence = dict(raw_evidence)
            outfit_evidence.update({
                "anchor_category": anchor_category,
                "companion_category": target_category,
                "basis": "co-occurrence detected in real outfit photos "
                         "(outfit_cooccurrence.json)",
                "labels_are_ground_truth": False,
                "caveat": "These counts are what an UNVALIDATED detector reported "
                          "(SAM2 masks + FashionCLIP zero-shot) over an unlabelled "
                          "corpus. They are evidence of what the model saw, not a "
                          "measured fact about what people wear.",
            })

    # ---- colour bias. Applied only on top of a category the evidence
    # already supports, and only as a re-rank: the colour signal is a
    # centre-of-bbox palette vote, far too crude to remove results with.
    color_evidence = []
    if outfit_evidence and anchor_colors:
        for anchor_color in anchor_colors:
            color_evidence.extend(
                cooccurrence.companion_colors(anchor_category, anchor_color, target_category))
    if color_evidence:
        preferred = {}
        for entry in color_evidence:
            preferred[entry["color"]] = preferred.get(entry["color"], 0) + entry["count"]
        ordered = list(candidate_hits)
        for position, hit in enumerate(ordered):
            colors = product_colors(code_to_product.get(hit["product_code"], {}))
            support = max((preferred.get(color, 0) for color in colors), default=0)
            hit["cooccurrence_color_support"] = support
            hit["_original_rank"] = position
        # Stable: co-occurring colours float up, everything else keeps its
        # relative text-search order rather than being reshuffled.
        ordered.sort(key=lambda h: (-h["cooccurrence_color_support"], h["_original_rank"]))
        for hit in ordered:
            hit.pop("_original_rank", None)
        candidate_hits = ordered

    second_item_matches = candidate_hits[:top_k]

    if outfit_evidence:
        share = outfit_evidence["share_of_outfits_with_anchor"]
        note = (
            f"Grounded in observed co-occurrence: in {outfit_evidence['cooccurrence_count']} "
            f"of {outfit_evidence['anchor_outfits']} real outfit photos where the detector "
            f"found a '{anchor_category}', it also found a '{target_category}' "
            f"({share*100:.0f}%, lift {outfit_evidence['lift']:.2f}). These are DETECTED "
            f"labels over an unlabelled corpus, with no measured precision -- evidence of "
            f"what the model saw worn together, not a verified fact."
        )
    elif cooccurrence is None:
        note = (INDEPENDENT_SEARCH_NOTE +
                " (outfit_cooccurrence.json is not present -- run "
                "build_outfit_cooccurrence.py to enable the evidence path.)")
    elif not anchor_category:
        note = (INDEPENDENT_SEARCH_NOTE +
                " (the query item's category could not be resolved, so the outfit "
                "index could not be consulted.)")
    else:
        note = (INDEPENDENT_SEARCH_NOTE +
                f" (the outfit corpus did not show '{anchor_category}' together with "
                f"'{target_category}' in at least {MIN_EVIDENCE_OUTFITS} photos.)")

    return {
        "identified_item": identified_item,
        "anchor_category": anchor_category,
        "parsed_text_query": parsed,
        "second_item_matches": second_item_matches,
        "category_filter_applied": category_filter_applied,
        "companion_source": companion_source,
        "companion_suggestions": companion_suggestions,
        "outfit_evidence": outfit_evidence,
        "cooccurrence_available": cooccurrence is not None,
        "note": note,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to the query image of item A.")
    parser.add_argument("--text", type=str, default=None,
                        help='Text fragment describing item B, e.g. "with cargo jorts". '
                             'Optional since 11.1: omit it and companions are proposed from '
                             'observed outfit co-occurrence instead.')
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--metadata", type=str, default=None, help="Override path to metadata.json (defaults to catalog_query_search.py's own default).")
    parser.add_argument("--anchor-category", type=str, default=None,
                        help="Skip the image stage and key the co-occurrence lookup on this "
                             "taxonomy category directly (e.g. 'jacket'). Loads no model weights.")
    parser.add_argument("--no-cooccurrence", action="store_true",
                        help="Force the pre-11.1 two-independent-searches behaviour.")
    parser.add_argument("--canonical-only", action="store_true",
                         help="Skip catalog_query_search.py's semantic-embedding fallback for the text side "
                              "(no model weights loaded for that stage). Also engaged automatically if the "
                              "semantic fallback errors out (e.g. no usable local torch/transformers stack).")
    args = parser.parse_args()

    if not args.text and not (args.image or args.anchor_category):
        parser.error("give --text, or --image/--anchor-category to propose companions from co-occurrence")

    result = composed_search(args.image, args.text, top_k=args.top_k, metadata_path=args.metadata,
                             canonical_only=args.canonical_only,
                             use_cooccurrence=not args.no_cooccurrence,
                             anchor_category=args.anchor_category)

    parsed = result["parsed_text_query"]
    if parsed:
        print(f"\nText fragment: '{args.text}'")
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
    else:
        print("\n(no --text given -- companion category taken from outfit co-occurrence)")

    if args.image:
        print(f"\nidentified_item (from {args.image}):")
        print(f"  {result['identified_item']}")
    else:
        print("\n(no --image given -- identified_item skipped)")
    print(f"anchor_category: {result['anchor_category']}")

    if result["companion_suggestions"]:
        print("\nObserved companions for the anchor (DETECTED, not verified; ranked by lift):")
        for suggestion in result["companion_suggestions"]:
            print(f"  {suggestion['category']:<12} n={suggestion['cooccurrence_count']:<5} "
                  f"p(companion|anchor)={suggestion['share_of_outfits_with_anchor']:.2f}  "
                  f"lift={suggestion['lift']:.2f}")
    elif result["cooccurrence_available"]:
        print("\n(no co-occurrence evidence for this anchor category)")

    if result["outfit_evidence"]:
        evidence = result["outfit_evidence"]
        print(f"\noutfit_evidence: {evidence['anchor_category']} + {evidence['companion_category']} "
              f"seen together in {evidence['cooccurrence_count']} of "
              f"{evidence['anchor_outfits']} outfits (lift {evidence['lift']:.2f})")
        for url in evidence["example_post_urls"][:3]:
            print(f"    example: {url}")

    print(f"\nsecond_item_matches (category filter applied: {result['category_filter_applied']}, "
          f"companion source: {result['companion_source']}):")
    for rank, hit in enumerate(result["second_item_matches"], start=1):
        label = hit.get("matched_label") or f"semantic score={hit.get('score', 0):.4f}"
        boost = hit.get("cooccurrence_color_support")
        boost_note = f"  [colour co-occurrence support={boost}]" if boost else ""
        print(f"  #{rank}  [{hit['match_type']}: {label}]  {hit['brand']} {hit['name']}  [{hit['product_code']}]{boost_note}")

    print(f"\nNOTE: {result['note']}")
