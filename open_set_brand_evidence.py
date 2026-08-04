"""Brand evidence as an OPEN-SET signal, not a ranking one.

Gap analysis item 11.3c. `brand_evidence.py` measured brand-as-ranking and
it was worth +0.10pt -- one query in 1,043 -- because DINOv3 had already
learned brand implicitly (products of one brand look alike). This asks the
other question, where the same signal is not redundant:

    Does a wordmark that matches NO catalog brand tell us the product is
    off-catalog?

Why this is worth trying when brand-as-ranking failed: DINOv3's score
answers "how close is the nearest catalog product", and open-set rejection
built on it reaches AUROC 0.769 with no usable operating point -- 1%
false-reject costs 68% false-accept (docs/eval_log.md). A brand read is
ORTHOGONAL evidence: it does not ask how close anything looks, it reads
what the garment says it is. Reading "PATAGONIA" on a catalog that carries
no Patagonia is direct evidence, not a similarity heuristic.

The hard part is not matching, it is distinguishing three cases that
`brand_evidence.py` collapses into one `no_brand_match`:

    1. no text at all                     -> no evidence either way
    2. text that is not a brand           -> no evidence ("ADJUSTABLE",
       ("XL", a care label)                  size digits)
    3. text matching a KNOWN brand we do  -> POSITIVE evidence of
       not carry                             off-catalog

Only (3) is a signal, and separating it from (2) needs a brand gazetteer
wider than the catalog's own vocabulary. That is what OFF_CATALOG_BRANDS
is: real apparel brands the catalog does not stock.

## Honest bound on this

`brand_evidence.py` measured that only 11.13% of catalog product photos
yield ANY readable brand, and 51.01% contain no legible text at all. So
this can only ever fire on a minority of inputs. It is a high-precision
partial signal, not a replacement for the DINOv3 score -- the useful shape
is an OVERRIDE (trust it when it fires, fall back otherwise), not a blend.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

import brand_evidence as be

# Real apparel brands the 6-brand deployment does not stock. Deliberately
# far wider than the six local-only brands used to evaluate this, so the
# gazetteer is not simply the answer key -- a production system would carry
# thousands of these, and the question under test is whether OCR can READ
# them, not whether a list can contain them.
OFF_CATALOG_BRANDS = (
    "carhartt", "champion", "dickies", "levis", "stussy", "vans",
    "patagonia", "north face", "columbia", "under armour", "puma", "reebok",
    "asics", "converse", "new era", "supreme", "obey", "volcom", "quiksilver",
    "billabong", "rip curl", "element", "thrasher", "hurley", "oakley",
    "timberland", "wrangler", "lee", "diesel", "guess", "calvin klein",
    "tommy hilfiger", "ralph lauren", "polo", "lacoste", "fila", "kappa",
    "ellesse", "umbro", "brooks", "saucony", "hoka", "salomon", "merrell",
    "keen", "crocs", "birkenstock", "dr martens", "clarks", "uniqlo",
    "zara", "h and m", "forever 21", "american eagle", "hollister",
    "abercrombie", "urban outfitters", "brandy melville", "shein",
    "lululemon", "gymshark", "alo yoga", "fabletics", "outdoor voices",
    "arcteryx", "canada goose", "moncler", "burberry", "gucci", "prada",
    "balenciaga", "off white", "palace", "bape", "kith", "aime leon dore",
    "carharttwip", "jordan", "yeezy", "on running", "veja", "allbirds",
)

# Aliases for gazetteer entries whose printed wordmark differs from the
# canonical token. Same mechanism brand_evidence.py uses for the catalog.
OFF_CATALOG_ALIASES = {
    "levis": ["levis", "levi s", "levi strauss"],
    "north face": ["north face", "the north face", "tnf"],
    "dr martens": ["dr martens", "doc martens", "docmartens"],
    "arcteryx": ["arcteryx", "arc teryx"],
    "off white": ["off white", "offwhite"],
    "h and m": ["h and m", "hm"],
    "carharttwip": ["carhartt wip", "wip"],
    "stussy": ["stussy", "stussy"],
}

# Sub-brand -> parent. A gazetteer hit whose PARENT is stocked is not
# evidence of anything off-catalog.
#
# Added after the only false positive in the 2026-08-04 evaluation: a Nike
# shoe printed "AIR JORDAN", the gazetteer listed `jordan` as an unstocked
# brand, and it was confidently called off-catalog. OCR was right; the
# gazetteer was wrong to treat a sub-brand as independent. The
# in-catalog-wins-ties rule did not save it either, because the shoe never
# printed "Nike" anywhere legible -- so the only fix is knowing the
# relationship.
#
# This is the failure mode that erodes trust fastest: telling someone the
# product in their hand is not stocked, confidently, when it is.
SUB_BRAND_PARENTS = {
    "jordan": "nike",
    "air jordan": "nike",
    "yeezy": "adidas",
    "y3": "adidas",
    "carharttwip": "carhartt",
    "polo": "ralph lauren",
}

# A gazetteer hit needs to be STRICTER than a catalog hit. A false
# "off-catalog" verdict tells the user their in-catalog product is not
# stocked, which is worse than staying silent -- and the gazetteer is much
# larger than the catalog vocabulary, so it has correspondingly more
# opportunity to collide with unrelated text.
MIN_SIM_SHORT_OFF = float(os.environ.get("OFFSET_BRAND_MIN_SIM_SHORT", "95"))
MIN_SIM_LONG_OFF = float(os.environ.get("OFFSET_BRAND_MIN_SIM_LONG", "88"))


@lru_cache(maxsize=1)
def off_catalog_alias_table() -> tuple[tuple[str, str], ...]:
    """(normalized_alias, canonical_brand) for brands NOT in the catalog.

    Anything the catalog actually stocks is removed, so the same brand can
    never be both in-catalog and off-catalog evidence. That subtraction is
    what makes this deployment-specific: the 6-brand Volume and the
    12-brand local catalog produce different gazetteers from the same list.
    """
    stocked = {be._normalize(b) for b in be.catalog_brands()}
    for brand in be.catalog_brands():
        for alias in be.BRAND_ALIASES.get(brand, [brand]):
            stocked.add(be._normalize(alias))

    pairs: list[tuple[str, str]] = []
    for brand in OFF_CATALOG_BRANDS:
        for alias in OFF_CATALOG_ALIASES.get(brand, [brand]):
            norm = be._normalize(alias)
            if norm and norm not in stocked:
                pairs.append((norm, brand))
    return tuple(sorted(set(pairs)))


@dataclass
class OpenSetBrandEvidence:
    verdict: str            # in_catalog | off_catalog | no_evidence
    brand: str | None
    score: float
    reason: str
    matched_text: str | None = None
    similarity: float = 0.0
    ocr_conf: float = 0.0
    all_text: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def classify_text(lines: list[tuple[str, float]]) -> OpenSetBrandEvidence:
    """Three-way verdict from OCR lines. Pure -- no model, unit-testable.

    In-catalog is checked FIRST and wins ties. If a garment says both
    "Nike" (stocked) and something gazetteer-ish, the stocked brand is the
    stronger claim -- collaborations and retailer tags routinely put a
    second brand on a genuinely in-catalog product.
    """
    catalog_hit = be.match_brand_in_text(lines)
    if catalog_hit.reason == "match":
        return OpenSetBrandEvidence(
            verdict="in_catalog", brand=catalog_hit.brand, score=catalog_hit.score,
            reason="catalog_brand_read", matched_text=catalog_hit.matched_text,
            similarity=catalog_hit.similarity, ocr_conf=catalog_hit.ocr_conf,
            all_text=catalog_hit.all_text,
        )

    kept = [(t, c) for t, c in lines if c >= be.MIN_OCR_CONF and be._normalize(t)]
    if not kept:
        return OpenSetBrandEvidence(
            verdict="no_evidence", brand=None, score=0.0, reason="no_text",
            all_text=[t for t, _ in lines],
        )

    from rapidfuzz import fuzz

    best = None
    for raw, conf in kept:
        for cand in be._candidates(be._normalize(raw)):
            if cand in be.CANDIDATE_BLOCKLIST:
                continue
            for alias, brand in off_catalog_alias_table():
                floor = (MIN_SIM_SHORT_OFF if len(alias) <= be.SHORT_ALIAS_LEN
                         else MIN_SIM_LONG_OFF)
                sim = fuzz.ratio(alias, cand)
                if sim < floor:
                    continue
                score = (sim / 100.0) * conf
                if best is None or score > best[0]:
                    best = (score, brand, sim / 100.0, conf, raw)

    if best is None:
        # Text present, but nothing brand-like in either vocabulary. This is
        # NOT evidence of off-catalog -- most such text is sizes, care
        # labels and page chrome. Reporting it as off-catalog would be the
        # category gate's mistake: an unrecoverable exclusion from a signal
        # that was never about membership.
        return OpenSetBrandEvidence(
            verdict="no_evidence", brand=None, score=0.0, reason="text_but_no_brand",
            all_text=[t for t, _ in kept],
        )

    score, brand, sim, conf, raw = best

    # A sub-brand whose parent is stocked is not off-catalog evidence. See
    # SUB_BRAND_PARENTS: this is the Nike/"AIR JORDAN" case.
    parent = SUB_BRAND_PARENTS.get(brand)
    if parent:
        stocked = {be._normalize(b) for b in be.catalog_brands()}
        if be._normalize(parent) in stocked:
            return OpenSetBrandEvidence(
                verdict="no_evidence", brand=None, score=0.0,
                reason="sub_brand_of_stocked_parent", matched_text=raw,
                similarity=sim, ocr_conf=conf, all_text=[t for t, _ in kept],
            )

    return OpenSetBrandEvidence(
        verdict="off_catalog", brand=brand, score=score,
        reason="known_brand_not_stocked", matched_text=raw,
        similarity=sim, ocr_conf=conf, all_text=[t for t, _ in kept],
    )


def classify_image(image_path, reader=None) -> OpenSetBrandEvidence:
    reader = reader or be.BrandDetector()
    return classify_text(reader.read_text(image_path))
