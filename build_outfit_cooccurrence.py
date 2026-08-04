"""Build the outfit co-occurrence index -- roadmap item 11.1, the final box
of the spec's section-3 architecture diagram ("outfit-level results").

WHAT PROBLEM THIS SOLVES
------------------------
`composed_query_search.py` answers "this jacket with cargo pants" by running
TWO INDEPENDENT SEARCHES and then admitting, in its own response payload,
that nothing in the system has ever shown those two things worn together.
That admission was correct: the catalog is single-product studio photos,
one garment per image, so it contains no evidence about what goes with
what. Co-occurrence is not derivable from it at all.

`outfit_dataset/` changes that. It is 6,860 photos of real people wearing
real outfits, and `index_outfits.py` runs multi-item detection over them.
Two garments detected in ONE photo is direct observational evidence that
those two garments were worn together by a real person. This script turns
those per-photo detections into an aggregate index of which garment
categories (and colours) actually appear together, so `/compose` can rank
companions by observed evidence instead of by an unrelated text search.

THE UNIT OF OBSERVATION IS THE OUTFIT, NOT THE IMAGE
----------------------------------------------------
`index_outfits.py` processes up to 2 images per scraped post, and a
multi-image post is usually the SAME outfit from several angles. Counting
per image would therefore double-count one person's single outfit as two
independent observations, inflating every count and every confidence
interval derived from it. So detections are collapsed per RECORD: within a
record, at most one item per category survives (the highest-confidence
one), and that de-duplicated set is one outfit observation. This is a real
statistical choice, not a formatting one -- it roughly halves the counts
and it is the honest number.

WHAT THESE NUMBERS ARE, AND EMPHATICALLY ARE NOT
------------------------------------------------
Every count here is MODEL-DERIVED WITH NO GROUND TRUTH:

  - The garment labels come from SAM2 mask proposals scored by FashionCLIP
    zero-shot. Nobody has ever labelled these photos. There is no
    validation set, no measured precision, no measured recall. Per the
    2026-08-03 decision in SCRAPING_PROCESS.md the corpus is deliberately
    unlabelled.
  - Therefore a pair count of N does NOT mean "N real people wore these
    together." It means "the detector reported both categories in N
    photos." Detector bias is indistinguishable from fashion truth in this
    data. If FashionCLIP over-calls "jacket", jacket will look popular.
  - MISSING items are invisible and systematically so. A photo cropped at
    the waist can never contribute a footwear co-occurrence. Absence of a
    pair is weak evidence of anything.
  - Colours are a crude centre-of-bbox palette vote (see
    index_outfits.dominant_color), not a colour model.

Every consumer of this artifact is expected to surface that. The index
carries the caveats inline, in `caveats`, so they travel with the data
rather than living only in this docstring.

MEASURES
--------
For an unordered pair (a, b) over N outfit observations:
  count          co-occurring observations
  p_b_given_a    count / count(a)   -- "of outfits with a, this share had b"
  lift           p(a,b) / (p(a)p(b)) -- >1 means more often than independent
  npmi           normalized pointwise mutual information, in [-1, 1]

`lift`/`npmi` matter because raw counts just rank by popularity: pants
co-occur with everything, so counts alone would recommend pants for every
query. Lift asks whether the pairing is more common than chance.

Usage:
    python3 build_outfit_cooccurrence.py
    python3 build_outfit_cooccurrence.py --metadata outfit_dataset/metadata.json \
        --out outfit_cooccurrence.json --min-confidence 0.5
"""

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_METADATA = Path("outfit_dataset/metadata.json")
DEFAULT_OUT = Path("outfit_cooccurrence.json")

SCHEMA_VERSION = 1

CAVEATS = [
    "MODEL-DERIVED, NO GROUND TRUTH. Garment labels are SAM2 mask proposals "
    "scored zero-shot by FashionCLIP over an unlabelled corpus. Precision and "
    "recall have never been measured because no labelled outfit set exists.",
    "A pair count is 'the detector reported both categories in N photos', NOT "
    "'N people wore these together'. Detector bias is indistinguishable from "
    "fashion truth here.",
    "Misses are systematic, so absence of a pair is weak evidence. A photo "
    "framed above the knee cannot contribute a footwear co-occurrence no "
    "matter what the person was wearing.",
    "One outfit observation = one scraped post, de-duplicated to at most one "
    "item per category. Multi-image posts are usually the same outfit from "
    "several angles and are NOT counted twice.",
    "Colours are a crude centre-of-bbox palette vote over the un-masked crop "
    "(index_outfits.dominant_color), not a colour model. Navy/black and "
    "beige/white are expected to confuse.",
    "The corpus is Reddit/Pinterest/wear-site street style. It is not a "
    "representative sample of how anyone dresses; it is a sample of what gets "
    "posted.",
]


def outfit_observations(records, min_confidence=0.0, sources=None):
    """One de-duplicated observation per record. Yields
    (record, {category: item}) for records that produced any detection."""
    for record in records:
        if sources and record.get("source") not in sources:
            continue
        items = record.get("detected_items") or []
        best = {}
        for item in items:
            if item.get("confidence", 0.0) < min_confidence:
                continue
            category = item.get("category")
            if not category:
                continue
            if category not in best or item["confidence"] > best[category]["confidence"]:
                best[category] = item
        if best:
            yield record, best


def _npmi(joint_p, p_a, p_b):
    if joint_p <= 0 or p_a <= 0 or p_b <= 0:
        return 0.0
    pmi = math.log(joint_p / (p_a * p_b))
    denominator = -math.log(joint_p)
    if denominator == 0:
        return 0.0
    return pmi / denominator


def build_index(records, min_confidence=0.0, min_pair_count=3, sources=None):
    category_counts = Counter()
    category_groups = {}
    pair_counts = Counter()
    # (category, colour) -> Counter[(other_category, other_colour)]
    color_pair_counts = defaultdict(Counter)
    color_counts = defaultdict(Counter)  # category -> Counter[colour]
    context_pairs = defaultdict(Counter)  # context -> Counter[(a, b)]
    context_totals = Counter()

    total_outfits = 0
    multi_item_outfits = 0
    item_count_hist = Counter()
    examples = defaultdict(list)  # (a, b) -> [post_url, ...]

    for record, best in outfit_observations(records, min_confidence, sources):
        total_outfits += 1
        categories = sorted(best)
        item_count_hist[len(categories)] += 1
        for category in categories:
            category_counts[category] += 1
            category_groups[category] = best[category].get("category_group")
            color = (best[category].get("color") or {}).get("name")
            if color:
                color_counts[category][color] += 1

        if len(categories) < 2:
            continue
        multi_item_outfits += 1

        # `section` is scraped provenance (subreddit / board), the closest
        # thing this corpus has to a style label. It is a context key, never
        # presented as a garment attribute.
        context = record.get("section") or record.get("source")
        context_totals[context] += 1

        for i, a in enumerate(categories):
            for b in categories[i + 1:]:
                pair = (a, b)
                pair_counts[pair] += 1
                context_pairs[context][pair] += 1
                if len(examples[pair]) < 5 and record.get("post_url"):
                    examples[pair].append(record["post_url"])
                color_a = (best[a].get("color") or {}).get("name")
                color_b = (best[b].get("color") or {}).get("name")
                if color_a and color_b:
                    color_pair_counts[(a, color_a)][(b, color_b)] += 1
                    color_pair_counts[(b, color_b)][(a, color_a)] += 1

    denominator = max(total_outfits, 1)
    pairs = []
    for (a, b), count in pair_counts.most_common():
        if count < min_pair_count:
            continue
        p_a = category_counts[a] / denominator
        p_b = category_counts[b] / denominator
        joint = count / denominator
        pairs.append({
            "a": a,
            "b": b,
            "count": count,
            "p_b_given_a": count / category_counts[a],
            "p_a_given_b": count / category_counts[b],
            "lift": joint / (p_a * p_b) if p_a and p_b else 0.0,
            "npmi": _npmi(joint, p_a, p_b),
            "example_post_urls": examples[(a, b)],
        })

    color_pairs = {}
    for (category, color), counter in color_pair_counts.items():
        entries = [
            {"category": other_category, "color": other_color, "count": count}
            for (other_category, other_color), count in counter.most_common(40)
            if count >= min_pair_count
        ]
        if entries:
            color_pairs[f"{category}|{color}"] = entries

    by_context = {}
    for context, counter in context_pairs.items():
        if context_totals[context] < 20:
            continue  # too few outfits for a per-context rate to mean anything
        top = [
            {"a": a, "b": b, "count": count,
             "share_of_context_outfits": count / context_totals[context]}
            for (a, b), count in counter.most_common(15)
            if count >= min_pair_count
        ]
        if top:
            by_context[context] = {"outfits": context_totals[context], "top_pairs": top}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "source_corpus": "outfit_dataset/metadata.json",
            "detector": "index_outfits.py -> segment_outfit.detect_outfit_items "
                        "(SAM2 automatic masks + FashionCLIP zero-shot categories)",
            "labels_are_ground_truth": False,
            "min_confidence": min_confidence,
            "min_pair_count": min_pair_count,
            "sources": sorted(sources) if sources else "all",
            "observation_unit": "one scraped post, de-duplicated to at most one item per category",
        },
        "caveats": CAVEATS,
        "totals": {
            "outfit_observations": total_outfits,
            "outfits_with_2plus_items": multi_item_outfits,
            "multi_item_rate": multi_item_outfits / denominator,
            "items_per_outfit_histogram": {str(k): v for k, v in sorted(item_count_hist.items())},
            "distinct_pairs_kept": len(pairs),
        },
        "categories": {
            category: {
                "outfits": count,
                "group": category_groups.get(category),
                "share_of_outfits": count / denominator,
                "top_colors": [
                    {"color": color, "count": n}
                    for color, n in color_counts[category].most_common(6)
                ],
            }
            for category, count in category_counts.most_common()
        },
        "pairs": pairs,
        "color_pairs": color_pairs,
        "by_context": by_context,
        "schema_notes": {
            "pairs[].count": "outfit observations containing BOTH a and b",
            "pairs[].p_b_given_a": "count / categories[a].outfits",
            "pairs[].lift": "p(a,b) / (p(a) p(b)); >1 = co-occurs more than chance",
            "pairs[].npmi": "normalized PMI in [-1,1]; 0 = independent",
            "color_pairs": "key is '<category>|<color>'; values are the companion "
                           "(category, color) combinations observed alongside it",
            "by_context": "keyed by scraped `section` (subreddit / board), which is "
                          "provenance, not a style label the system assigned",
        },
    }


# ------------------------------------------------------------------
# Query side -- imported by composed_query_search.py.
# ------------------------------------------------------------------

class OutfitCooccurrence:
    """Read-only accessor over the built index. Deliberately dependency-free
    (stdlib json only) so composed_query_search.py can consult it without
    loading any model weights."""

    def __init__(self, path=DEFAULT_OUT):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self._by_category = defaultdict(list)
        for pair in self.data.get("pairs", []):
            self._by_category[pair["a"]].append((pair["b"], pair, "p_b_given_a"))
            self._by_category[pair["b"]].append((pair["a"], pair, "p_a_given_b"))

    @classmethod
    def load_if_available(cls, path=DEFAULT_OUT):
        """Returns None instead of raising when the index has not been built.
        Callers fall back to the old independent-search path in that case --
        the co-occurrence path is an upgrade, not a hard dependency."""
        try:
            return cls(path)
        except (OSError, ValueError):
            return None

    @property
    def caveats(self):
        return self.data.get("caveats", [])

    def known_category(self, category):
        return category in self.data.get("categories", {})

    def companions(self, category, top_k=8, min_count=3, rank_by="lift"):
        """Categories observed alongside `category`, ranked by `lift`
        (surprise) or `count` (popularity). Empty list when there is no
        evidence -- that emptiness is what the caller uses to decide to fall
        back."""
        results = []
        for other, pair, conditional_key in self._by_category.get(category, []):
            if pair["count"] < min_count:
                continue
            results.append({
                "category": other,
                "group": self.data["categories"].get(other, {}).get("group"),
                "cooccurrence_count": pair["count"],
                "share_of_outfits_with_anchor": pair[conditional_key],
                "lift": pair["lift"],
                "npmi": pair["npmi"],
                "example_post_urls": pair.get("example_post_urls", []),
            })
        key = (lambda r: (r["lift"], r["cooccurrence_count"])) if rank_by == "lift" \
            else (lambda r: (r["cooccurrence_count"], r["lift"]))
        results.sort(key=key, reverse=True)
        return results[:top_k]

    def evidence_for(self, category_a, category_b):
        """The observed evidence that these two specific categories go
        together, or None if the corpus never showed them together often
        enough to be kept."""
        for other, pair, conditional_key in self._by_category.get(category_a, []):
            if other == category_b:
                return {
                    "cooccurrence_count": pair["count"],
                    "share_of_outfits_with_anchor": pair[conditional_key],
                    "lift": pair["lift"],
                    "npmi": pair["npmi"],
                    "anchor_outfits": self.data["categories"][category_a]["outfits"],
                    "example_post_urls": pair.get("example_post_urls", []),
                }
        return None

    def companion_colors(self, category_a, color_a, category_b, top_k=5):
        """Which colours of `category_b` were seen with a `color_a`
        `category_a`. Used to bias, never to filter -- the colour signal is
        the weakest thing in this index."""
        entries = self.data.get("color_pairs", {}).get(f"{category_a}|{color_a}", [])
        matching = [e for e in entries if e["category"] == category_b]
        matching.sort(key=lambda e: e["count"], reverse=True)
        return matching[:top_k]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Drop detections below this FashionCLIP confidence before counting.")
    parser.add_argument("--min-pair-count", type=int, default=3,
                        help="Pairs seen fewer times than this are dropped as noise (default 3).")
    parser.add_argument("--source", action="append", default=None,
                        help="Restrict to these sources (repeatable).")
    args = parser.parse_args()

    records = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    detected = [r for r in records if r.get("detection_meta")]
    print(f"{len(records):,} records on disk, {len(detected):,} carry detections.")
    if not detected:
        print("Nothing to build from. Run index_outfits.py (or modal_app_index_outfits.py) first.")
        return

    index = build_index(records,
                        min_confidence=args.min_confidence,
                        min_pair_count=args.min_pair_count,
                        sources=set(args.source) if args.source else None)

    Path(args.out).write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    totals = index["totals"]
    print(f"\nWrote {args.out}")
    print(f"  outfit observations   : {totals['outfit_observations']:,}")
    print(f"  with 2+ items         : {totals['outfits_with_2plus_items']:,} "
          f"({totals['multi_item_rate']*100:.1f}%)")
    print(f"  distinct pairs kept   : {totals['distinct_pairs_kept']:,}")
    print(f"  items/outfit histogram: {totals['items_per_outfit_histogram']}")
    print("\n  top pairs by count:")
    for pair in index["pairs"][:12]:
        print(f"    {pair['a']:<12} + {pair['b']:<12} n={pair['count']:<5} "
              f"p(b|a)={pair['p_b_given_a']:.2f}  lift={pair['lift']:.2f}")
    print("\nThese are UNVALIDATED model outputs, not measured facts about "
          "what people wear. See the index's own `caveats` array.")


if __name__ == "__main__":
    main()
