"""Any-specificity text query search -- spec sections 2.5 ("specificity
must depend on confidence... never force an exact product prediction")
and 4.6/4.8 ("canonical label generation" + "text-only query" flow),
neither of which had ever been built as an actual artifact before this:
`free_text_visual_search.py` only ever did pure embedding-similarity
search, with no lexical/metadata matching stage at all.

The concrete problem this solves: a user should get the SAME product
back whether they say "white sneaker" (broad) or "Air Force 1" (exact),
because both descriptions are true of the same real product. This isn't
a new capability that needs new training -- it's already latent in data
this project already generated and already paid for. Every product's
`structured_caption.positive_texts` was written by caption_apparel.py
specifically to span specificity levels for the SAME image, e.g. a real
record in this catalog has positive_texts = ["a sneaker", "a white
sneaker", "a white Nike sneaker", "a white Nike Air Force 1", "a Nike
Air Force 1 '07 with a leather upper", "Nike Air Force 1 '07 SKU
CW2288-111"] -- literally the spec 4.6 canonical-label chain, generated
months before this script existed, just never indexed for lookup.

Two-stage query, spec 4.8's actual flow ("search structured metadata...
search SigLIP semantic vectors... merge and rank"):
  1. **Canonical/lexical match** (fast, exact, high-precision): every
     product's positive_texts + a few deterministic combinations (color+
     category, brand+category -- covering combinations the LLM caption
     prompt didn't happen to generate verbatim) are indexed as canonical
     labels -> product_code. A query matches a label via substring
     containment either direction ("air force 1" is a substring of "a
     white nike air force 1"; "sneaker" is a substring of a query like
     "show me a sneaker"). This needs no model at all -- pure string
     matching, same pragmatic style as the rest of this codebase's
     structured-data logic (build_hierarchy.py, dataset_utils.py).
  2. **Semantic embedding fallback** (`free_text_visual_search.py`,
     reused not duplicated): for queries with no/weak canonical-label
     hits -- genuinely novel phrasing, typos, synonyms the LLM captions
     never used. Canonical matches, when present, are ranked above
     semantic-only matches (a lexical hit is stronger evidence than
     cosine similarity), and merged with them per spec 4.8's "merge and
     rank... include exact branded products that inherit broader facts."

Usage:
    python3 catalog_query_search.py --query "white sneaker" --top-k 15
    python3 catalog_query_search.py --query "Air Force 1" --top-k 15
    python3 catalog_query_search.py --query "white sneaker" --canonical-only  # skip the model entirely, fast
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from free_text_visual_search import DATASET_ROOT, METADATA_PATH, FreeTextVisualSearch

ARTICLE_PREFIX = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def normalize_label(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = ARTICLE_PREFIX.sub("", text)
    return text.strip(" ,.")


def display_brand(brand):
    brand = str(brand or "").strip()
    if not brand:
        return ""
    known = {
        "adidas": "Adidas", "nike": "Nike", "puma": "Puma", "reebok": "Reebok",
        "asics": "ASICS", "vans": "Vans", "converse": "Converse", "salomon": "Salomon",
        "saucony": "Saucony", "newbalance": "New Balance", "new balance": "New Balance",
        "gap": "Gap", "pacsun": "PacSun", "skechers": "Skechers", "levis": "Levi's",
        "champion": "Champion",
    }
    return known.get(brand.lower(), brand.title())


def build_canonical_index():
    """Returns (label_to_codes: dict[str, set[str]], code_to_product: dict[str, dict])."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    label_to_codes = defaultdict(set)
    code_to_product = {}

    for product in metadata:
        code = str(product.get("product_code", "")).strip()
        if not code:
            continue
        code_to_product[code] = product

        structured = product.get("structured_caption") or {}
        labels = set()

        # 1. The LLM-generated specificity chain -- the primary source,
        # since it already spans generic -> exact FOR THIS SPECIFIC
        # IMAGE, not a generic template.
        for text in structured.get("positive_texts", []) or []:
            normalized = normalize_label(text)
            if normalized:
                labels.add(normalized)

        # 2. Deterministic combinations, to cover cases the LLM prompt's
        # positive_texts cap (5-10 entries) didn't happen to include --
        # e.g. a product with 2 colors might only get one color phrased
        # in positive_texts.
        taxonomy_path = (structured.get("canonical_taxonomy_path")
                          or structured.get("taxonomy_path") or [])
        leaf_category = taxonomy_path[-1] if taxonomy_path else product.get("category", "")
        leaf_category = normalize_label(leaf_category)
        brand = display_brand(product.get("brand", ""))
        attributes = structured.get("attributes", {}) or {}

        if leaf_category:
            labels.add(leaf_category)
        for color in attributes.get("color", []) or []:
            if leaf_category:
                labels.add(normalize_label(f"{color} {leaf_category}"))
        for material in attributes.get("material", []) or []:
            if leaf_category:
                labels.add(normalize_label(f"{material} {leaf_category}"))
        if brand and leaf_category:
            labels.add(normalize_label(f"{brand} {leaf_category}"))
            for color in attributes.get("color", []) or []:
                labels.add(normalize_label(f"{brand} {color} {leaf_category}"))

        # 3. The raw scraped product name -- often contains the exact
        # phrase a user would type (e.g. "Air Force 1") in wording the
        # LLM caption prompt may have paraphrased slightly differently.
        name = normalize_label(product.get("name", ""))
        if name:
            labels.add(name)

        for label in labels:
            label_to_codes[label].add(code)

    return dict(label_to_codes), code_to_product


def search_canonical(query, label_to_codes, top_k=15):
    """Substring containment either direction, scored by specificity
    (longer matched label = more specific = ranked higher). Returns
    [(product_code, best_matched_label, specificity_score), ...]."""
    normalized_query = normalize_label(query)
    if not normalized_query:
        return []

    best_score_by_code = {}
    best_label_by_code = {}
    for label, codes in label_to_codes.items():
        if normalized_query in label or label in normalized_query:
            score = len(label.split())  # crude specificity proxy: more words = more specific
            for code in codes:
                if score > best_score_by_code.get(code, -1):
                    best_score_by_code[code] = score
                    best_label_by_code[code] = label

    ranked = sorted(best_score_by_code.items(), key=lambda item: item[1], reverse=True)
    return [(code, best_label_by_code[code], score) for code, score in ranked[:top_k]]


class CatalogQuerySearch:
    """Combines canonical/lexical matching with the existing semantic
    embedding search -- lazy-loads the semantic engine (real model
    weights, real cost) only on first use, so --canonical-only queries
    never pay that cost at all."""

    def __init__(self):
        print("Building canonical label index...")
        self.label_to_codes, self.code_to_product = build_canonical_index()
        print(f"Indexed {len(self.label_to_codes):,} canonical labels across {len(self.code_to_product):,} products.")
        self._semantic_engine = None

    @property
    def semantic_engine(self):
        if self._semantic_engine is None:
            self._semantic_engine = FreeTextVisualSearch()
        return self._semantic_engine

    def search(self, query, top_k=15, canonical_only=False):
        canonical_hits = search_canonical(query, self.label_to_codes, top_k=top_k)
        canonical_codes = {code for code, _, _ in canonical_hits}

        results = []
        for code, label, score in canonical_hits:
            product = self.code_to_product.get(code, {})
            results.append({
                "product_code": code, "brand": product.get("brand", ""), "name": product.get("name", ""),
                "match_type": "canonical", "matched_label": label, "score": score,
            })

        if canonical_only:
            return results[:top_k]

        remaining_slots = top_k - len(results)
        if remaining_slots > 0:
            # Semantic fallback fills in the rest -- covers genuinely
            # novel phrasing the canonical index has no label for at all.
            # A canonical-then-semantic merge, not semantic-then-filter:
            # the semantic engine's own top_k is padded so there's enough
            # to backfill after dropping anything already covered above.
            semantic_hits = self.semantic_engine.search(query, top_k=top_k + len(canonical_codes))
            for hit in semantic_hits:
                if hit["product_code"] in canonical_codes:
                    continue
                results.append({
                    "product_code": hit["product_code"], "brand": hit["brand"], "name": hit["name"],
                    "match_type": "semantic", "matched_label": None, "score": hit["score"],
                })
                if len(results) >= top_k:
                    break

        return results[:top_k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--canonical-only", action="store_true",
                         help="Skip the semantic-embedding fallback entirely -- no model weights loaded, "
                              "fast, but only returns results with an actual lexical match.")
    args = parser.parse_args()

    engine = CatalogQuerySearch()
    results = engine.search(args.query, top_k=args.top_k, canonical_only=args.canonical_only)

    print(f"\nQuery: '{args.query}'" + ("  [canonical-only]" if args.canonical_only else ""))
    for rank, r in enumerate(results, start=1):
        if r["match_type"] == "canonical":
            print(f"  #{rank}  [canonical: '{r['matched_label']}']  {r['brand']} {r['name']}  [{r['product_code']}]")
        else:
            print(f"  #{rank}  [semantic, score={r['score']:.4f}]  {r['brand']} {r['name']}  [{r['product_code']}]")
