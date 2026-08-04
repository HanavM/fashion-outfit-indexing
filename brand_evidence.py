"""Brand evidence from an image: OCR + fuzzy match against the catalog's
own brand vocabulary.

This is spec section 4.5's "Brand evidence" path, which had zero references
in the codebase before this file (see `docs/product_gap_analysis.md` item
11.3). The spec lists five evidence sources -- visible logo, OCR, trademark
pattern, distinctive construction, catalog-candidate agreement -- and is
explicit that brand must be nullable and must never be assigned from weak
style resemblance. This module implements the OCR source only, and it
abstains rather than guessing.

## Why easyocr and not paddleocr

Both were considered. `paddleocr` needs `paddlepaddle`, which has **no wheel
at all** for this environment (macOS arm64 / CPython 3.14) -- `pip install
--dry-run paddlepaddle` resolves to "no matching distribution", so it is not
a preference, it is unavailable without building Paddle from source.
`easyocr` installs cleanly, is pure PyTorch, and therefore reuses the torch
build this repo already depends on and runs on the same MPS/CUDA device the
rest of the pipeline uses. On this machine full-resolution OCR is ~2.5-4s per
image on CPU-with-8-threads but ~2.5s on MPS and well under a second on a
CUDA GPU, so the device reuse is worth real wall-clock.

## Resolution matters more than anything else

Measured on a real Carhartt jacket photo whose chest wordmark OCR *can*
read: at 768px the detector finds nothing, at 1024px it finds garbage
("connatt", conf 0.03), at 1536px+ it reads "carbartt" at conf 0.76-0.91.
Wordmarks are small relative to the frame, so downscaling to save time
destroys the only signal this module exists to read. `MAX_IMAGE_DIM` is
therefore deliberately large; do not lower it to make an eval finish sooner.
This is the same failure mode `docs/product_gap_analysis.md` item 11.2
records for `segment_outfit.py`.

## Matching

OCR output is noisy in a specific, exploitable way: it reads "carbartt" for
CARHARTT, "stussv" for STUSSY. So the match is fuzzy (rapidfuzz ratio), not
exact. Short brand names are the danger case -- "gap", "vans", "nike" are 3-4
characters and sit a single edit away from ordinary garment words ("cap",
"jeans", "bike") -- so aliases at or below `SHORT_ALIAS_LEN` require a much
higher similarity than long ones. Every threshold here is exposed so the
eval can sweep it rather than trust it.

Usage as a library:

    from brand_evidence import BrandDetector
    det = BrandDetector()                 # vocabulary read from metadata.json
    ev = det.detect("path/to/image.jpg")
    ev.brand    # canonical brand key, or None
    ev.score    # 0..1, or 0.0 when brand is None
    ev.reason   # "match" | "no_text" | "no_brand_match"

CLI:

    .venv/bin/python brand_evidence.py --images a.jpg b.jpg
    .venv/bin/python brand_evidence.py --brand carhartt --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
METADATA_PATH = Path(
    os.environ.get("APPAREL_METADATA", REPO_ROOT / "apparel_dataset" / "metadata.json")
)

# Image roots tried in order when resolving a metadata `images` entry. The
# metadata stores paths like "shoe_dataset/nike/..." or
# "apparel_dataset/gap/..." which are relative to the repo root for the
# brands scraped in place, but the older shoe brands only exist under
# apparel_dataset_full/<brand>/... -- so that root gets the first path
# component stripped. Read-only: nothing in this module writes to either.
IMAGE_ROOTS = [
    (REPO_ROOT, 0),
    (REPO_ROOT / "apparel_dataset_full", 1),
]

# See the module docstring -- lowering this silently destroys the signal.
MAX_IMAGE_DIM = int(os.environ.get("BRAND_OCR_MAX_DIM", "2048"))

# easyocr's own per-line confidence. Below this the text is usually a
# hallucinated fragment of texture, not writing.
MIN_OCR_CONF = float(os.environ.get("BRAND_OCR_MIN_CONF", "0.10"))

# CRAFT detector sensitivity. easyocr's defaults (0.7 / 0.4) are tuned for
# documents; garment wordmarks are low-contrast tonal embroidery as often as
# they are printed text, so these are loosened. Measured on a 120-image pilot
# before the full run -- see docs/eval_log.md; the loosened setting is what
# the reported numbers use.
OCR_TEXT_THRESHOLD = float(os.environ.get("BRAND_OCR_TEXT_THRESHOLD", "0.5"))
OCR_LOW_TEXT = float(os.environ.get("BRAND_OCR_LOW_TEXT", "0.3"))
# >1 upscales before detection, which is how small wordmarks become legible.
OCR_MAG_RATIO = float(os.environ.get("BRAND_OCR_MAG_RATIO", "1.0"))

# Fuzzy-match similarity floors (rapidfuzz ratio, 0-100).
SHORT_ALIAS_LEN = 5
MIN_SIM_SHORT = float(os.environ.get("BRAND_MIN_SIM_SHORT", "90"))
MIN_SIM_LONG = float(os.environ.get("BRAND_MIN_SIM_LONG", "80"))

# Default accept threshold on the combined score. Chosen from the sweep in
# `brand_evidence_eval.py`, not guessed; see docs/eval_log.md.
DEFAULT_ACCEPT_SCORE = float(os.environ.get("BRAND_ACCEPT_SCORE", "0.55"))


# --------------------------------------------------------------------------
# Brand vocabulary
# --------------------------------------------------------------------------

# Extra surface forms per canonical brand key. The keys must match the raw
# lowercase `brand` field in metadata.json. Anything not listed here falls
# back to [brand_key] alone. These are the strings that actually appear as
# wordmarks on garments or in page chrome -- not marketing taglines, which
# would be a different (and much weaker) evidence source.
BRAND_ALIASES: dict[str, list[str]] = {
    "adidas": ["adidas", "adidas originals"],
    "carhartt": ["carhartt", "carhartt wip"],
    "champion": ["champion"],
    "dickies": ["dickies", "dickies 1922"],
    "gap": ["gap"],
    "levis": ["levis", "levi", "levi strauss", "levis strauss"],
    "newbalance": ["new balance", "newbalance"],
    "nike": ["nike"],
    "pacsun": ["pacsun", "pacific sunwear"],
    "skechers": ["skechers"],
    "stussy": ["stussy", "stuessy"],
    "vans": ["vans", "vans off the wall"],
}

# Words that appear constantly on apparel and are close enough to a short
# brand alias to trip it. Checked BEFORE matching, so a candidate token
# equal to one of these can never produce brand evidence. Kept short and
# justified: each entry is a real word seen in catalog OCR output.
CANDIDATE_BLOCKLIST = {
    "cap", "caps", "gaps", "bag", "bags", "tag", "tags",
    "jeans", "van", "bike", "like", "nice", "size", "sale",
    "made", "wash", "care", "cotton", "fit", "new", "balance",
}


def _normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    Stussy's real wordmark is "Stussy" with a diaeresis; OCR may or may not
    produce it, so both sides go through NFKD accent stripping.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def catalog_brands() -> tuple[str, ...]:
    """The brand vocabulary, read from the catalog itself (read-only)."""
    with open(METADATA_PATH) as fh:
        records = json.load(fh)
    return tuple(sorted({r["brand"] for r in records if r.get("brand")}))


@lru_cache(maxsize=1)
def alias_table() -> tuple[tuple[str, str], ...]:
    """(normalized_alias, canonical_brand) pairs for the catalog's brands."""
    pairs: list[tuple[str, str]] = []
    for brand in catalog_brands():
        for alias in BRAND_ALIASES.get(brand, [brand]):
            norm = _normalize(alias)
            if norm:
                pairs.append((norm, brand))
    return tuple(sorted(set(pairs)))


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass
class BrandEvidence:
    brand: str | None
    score: float
    reason: str                       # match | no_text | no_brand_match
    matched_text: str | None = None   # the raw OCR line the match came from
    similarity: float = 0.0           # 0-1 fuzzy similarity
    ocr_conf: float = 0.0             # easyocr confidence of that line
    runner_up: str | None = None      # second-best distinct brand, if any
    runner_up_score: float = 0.0
    n_text_regions: int = 0           # OCR lines surviving MIN_OCR_CONF
    all_text: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Matching (pure -- no OCR, unit-testable without a model)
# --------------------------------------------------------------------------


def _candidates(normalized_line: str) -> list[str]:
    """Whole line, single tokens, and adjacent bigrams.

    Bigrams exist for the multi-word aliases ("new balance",
    "vans off the wall" is caught by the whole line instead).
    """
    tokens = normalized_line.split()
    out = [normalized_line]
    out += tokens
    out += [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
    return [c for c in dict.fromkeys(out) if c]


def match_brand_in_text(
    lines: list[tuple[str, float]],
    min_sim_short: float = MIN_SIM_SHORT,
    min_sim_long: float = MIN_SIM_LONG,
) -> BrandEvidence:
    """Fuzzy-match OCR lines against the brand vocabulary.

    `lines` is [(raw_text, ocr_confidence)]. Score is
    `similarity * ocr_confidence`, both in 0..1 -- deliberately NOT collapsed
    into a single tuned number, because the eval sweeps the accept threshold
    over this score and a hand-picked blend would bake in an assumption the
    measurement is supposed to test.
    """
    kept = [(t, c) for t, c in lines if c >= MIN_OCR_CONF and _normalize(t)]
    if not kept:
        return BrandEvidence(
            brand=None, score=0.0, reason="no_text", n_text_regions=0,
            all_text=[t for t, _ in lines],
        )

    from rapidfuzz import fuzz

    best_per_brand: dict[str, tuple[float, float, float, str]] = {}
    for raw, conf in kept:
        norm_line = _normalize(raw)
        for cand in _candidates(norm_line):
            if cand in CANDIDATE_BLOCKLIST:
                continue
            for alias, brand in alias_table():
                floor = min_sim_short if len(alias) <= SHORT_ALIAS_LEN else min_sim_long
                sim = fuzz.ratio(alias, cand)
                if sim < floor:
                    continue
                score = (sim / 100.0) * conf
                prev = best_per_brand.get(brand)
                if prev is None or score > prev[0]:
                    best_per_brand[brand] = (score, sim / 100.0, conf, raw)

    if not best_per_brand:
        return BrandEvidence(
            brand=None, score=0.0, reason="no_brand_match",
            n_text_regions=len(kept), all_text=[t for t, _ in kept],
        )

    ranked = sorted(best_per_brand.items(), key=lambda kv: -kv[1][0])
    brand, (score, sim, conf, raw) = ranked[0]
    runner_up, runner_up_score = (ranked[1][0], ranked[1][1][0]) if len(ranked) > 1 else (None, 0.0)
    return BrandEvidence(
        brand=brand, score=score, reason="match", matched_text=raw,
        similarity=sim, ocr_conf=conf, runner_up=runner_up,
        runner_up_score=runner_up_score, n_text_regions=len(kept),
        all_text=[t for t, _ in kept],
    )


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------


def resolve_image_path(rel: str) -> Path | None:
    """Map a metadata `images` entry onto a real local file, or None."""
    for root, strip in IMAGE_ROOTS:
        parts = rel.split("/")
        p = root.joinpath(*parts[strip:])
        if p.exists():
            return p
    return None


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BrandDetector:
    """Lazy-loads easyocr on first use so importing this module is cheap."""

    def __init__(self, device: str | None = None, max_dim: int = MAX_IMAGE_DIM,
                 text_threshold: float = OCR_TEXT_THRESHOLD,
                 low_text: float = OCR_LOW_TEXT,
                 mag_ratio: float = OCR_MAG_RATIO):
        self.device = _pick_device(device)
        self.max_dim = max_dim
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.mag_ratio = mag_ratio
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                ["en"], gpu=(self.device != "cpu"), verbose=False,
            )
        return self._reader

    def read_text(self, image_path: str | Path) -> list[tuple[str, float]]:
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        if max(img.size) > self.max_dim:
            img.thumbnail((self.max_dim, self.max_dim))
        raw = self.reader.readtext(
            np.array(img),
            text_threshold=self.text_threshold,
            low_text=self.low_text,
            mag_ratio=self.mag_ratio,
            canvas_size=max(self.max_dim, 2560),
        )
        return [(t, float(c)) for _box, t, c in raw]

    def detect(self, image_path: str | Path) -> BrandEvidence:
        return match_brand_in_text(self.read_text(image_path))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _self_test() -> int:
    """Matching-layer checks that need no model and no images."""
    cases = [
        # (ocr lines, expected brand)
        ([("CARBARTT", 0.9)], "carhartt"),          # real OCR misread, must still hit
        ([("CARHARTT WIP", 0.95)], "carhartt"),
        ([("new balance", 0.8)], "newbalance"),
        ([("Levi's", 0.7)], "levis"),
        ([("STUSSY", 0.9)], "stussy"),
        ([("GAP", 0.9)], "gap"),
        ([("cap", 0.9)], None),                      # blocklisted near-miss
        ([("jeans", 0.9)], None),
        ([("100% COTTON MADE IN VIETNAM", 0.9)], None),
        ([], None),
        ([("wwww", 0.02)], None),                    # below MIN_OCR_CONF
    ]
    failures = 0
    for lines, expected in cases:
        got = match_brand_in_text(lines).brand
        ok = got == expected
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {lines!r:50s} -> {got!r} (expected {expected!r})")
    print(f"\n{len(cases) - failures}/{len(cases)} matching cases pass")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="*", help="image paths to run detection on")
    ap.add_argument("--brand", help="sample images from this catalog brand instead")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="run the matching-layer checks (no model, no images)")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    paths: list[Path] = []
    if args.images:
        paths = [Path(p) for p in args.images]
    elif args.brand:
        with open(METADATA_PATH) as fh:
            records = json.load(fh)
        for rec in records:
            if rec.get("brand") != args.brand:
                continue
            for rel in rec.get("images", [])[:1]:
                p = resolve_image_path(rel)
                if p:
                    paths.append(p)
                    break
            if len(paths) >= args.limit:
                break
    else:
        print("pass --images, --brand or --self-test", file=sys.stderr)
        return 2

    det = BrandDetector(device=args.device)
    print(f"device={det.device} max_dim={det.max_dim} brands={len(catalog_brands())}")
    for p in paths:
        ev = det.detect(p)
        print(f"\n{p}")
        print(f"  brand={ev.brand} score={ev.score:.3f} reason={ev.reason} "
              f"sim={ev.similarity:.2f} conf={ev.ocr_conf:.2f} matched={ev.matched_text!r}")
        print(f"  text({ev.n_text_regions}): {ev.all_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
