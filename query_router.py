"""Rules-based intent routing for the unified `POST /query` endpoint.

Design of record: `docs/unified_query_design.md`. The short version of that
argument, because it constrains this file:

  Unifying the INTERFACE and unifying the REPRESENTATION are different
  things. The second one was measured and lost -- `--score-fusion` blended
  the SigLIP2 semantic score into the DINOv3 identity score and cost
  **-6.22pt R@1** (50.76% -> 44.54%) with shortlist miss *identical* in
  both arms, i.e. a pure ranking regression. The mechanism is understood:
  SigLIP2 scores are identity-level, so every colorway sibling gets the
  same score, which flattens exactly the sibling distinctions DINOv3's
  fine-tune bought (+31.3pt).

So this module does **no retrieval and no scoring**. It decides which of
the three already-measured paths a request should take, and nothing else.
`/query` then calls the same internal helpers `/identify`, `/compose` and
`/search` call, so parity is true by construction rather than by
re-implementation.

Deliberately non-semantic and cheap. Four signals, all lexical:

  1. is an image present
  2. does the text ask a property question ("what brand is this")
  3. does the text name a COMPANION item ("with cargo jorts")
  4. does the text carry an exactness cue ("this exact one")

An LLM router is the expensive version of this. It buys robustness on
phrasings the vocabulary misses -- not new capability -- so it is deferred
until this version's failure modes are measured rather than guessed.

## The one non-obvious rule

Bare "with" is ambiguous in exactly the way that matters here:

    "this shoe with cargo jorts"   -> companion request  (two searches)
    "a shirt with stripes"         -> attribute of the SAME item (one)

Keying on the token alone gets the second case wrong every time. So bare
"with" only counts as a companion cue when the text that FOLLOWS it parses
to a real taxonomy category. "jorts" resolves (via the existing slang table)
to `denim shorts`; "stripes" resolves to nothing. Multi-word forms
("goes with", "to wear with", ...) are unambiguous and need no such check.

That check reuses `composed_query_search.parse_text_fragment`, which is the
same parser `/compose` itself uses -- so the router cannot decide a
companion exists that the composer then fails to parse.
"""

import re

# ---------------------------------------------------------------
# Lexical cue tables.
#
# These are vocabulary, not logic. Keep them here (not inline in the
# matcher) so that the routing failure modes are auditable as data --
# "which phrasings does v1 miss" should be answerable by reading a list.
# ---------------------------------------------------------------

# Unambiguous companion constructions. Each already contains its own verb,
# so unlike bare "with" they cannot be an attribute of the anchor item.
#
# Regexes rather than a literal substring table on purpose: a flat list
# has to enumerate "pair with" / "pairs with" / "paired with" / "pair it
# with" / "pair this with" separately and *will* keep missing one (it
# missed "pair this with" on the first run). The optional pronoun slot
# below covers the whole family in one rule.
COMPANION_PATTERNS = (
    # pair/wear/style/match/combine [it|this|that|these|them] with
    r"\b(?:pair|wear|style|match|combine)(?:s|ed|ing)?"
    r"(?:\s+(?:it|this|that|these|them|one|ones))?\s+with\b",
    # go / goes / going / goes well with
    r"\bgo(?:es|ing)?\s+(?:well\s+)?with\b",
    # worn with
    r"\bworn\s+with\b",
    # along with / together with
    r"\b(?:along|together)\s+with\b",
)

# Bare prepositions that MIGHT introduce a companion. Only honoured when
# the trailing text parses to a taxonomy category -- see module docstring.
AMBIGUOUS_COMPANION_TOKENS = ("with",)

# "What is this thing's property" -- answerable from the identified
# catalog record, not from a separate retrieval path.
PROPERTY_QUESTION_PATTERNS = (
    r"\bwhat brand\b",
    r"\bwhich brand\b",
    r"\bwhose brand\b",
    r"\bwho makes?\b",
    r"\bwho made\b",
    r"\bwhat make\b",
    r"\bwhat label\b",
    r"\bbrand (?:is|are|of)\b",
)

# Caller wants the *specific* product, not the visual neighbourhood.
EXACTNESS_CUES = (
    "this exact",
    "the exact",
    "exactly this",
    "exact same",
    "this specific",
    "this particular",
    "exact product",
    "exact model",
    "exact one",
)

# Caller explicitly wants the neighbourhood, not the specific product.
SIMILARITY_CUES = (
    "like this",
    "similar to this",
    "similar to these",
    "something like",
    "looks like",
    "in the style of",
    "same vibe",
)

INTENT_IDENTIFY = "identify"
INTENT_COMPOSE = "compose"
INTENT_SEARCH = "search"
INTENT_BRAND = "brand"
INTENT_UNROUTABLE = "unroutable"

# Which already-existing, already-measured path each intent executes.
# INTENT_BRAND deliberately maps to /identify: see `_brand_note`.
INTENT_TO_PATH = {
    INTENT_IDENTIFY: "/identify",
    INTENT_COMPOSE: "/compose",
    INTENT_SEARCH: "/search",
    INTENT_BRAND: "/identify",
}


def _normalise(text):
    """Lowercase, collapse whitespace, strip terminal punctuation. Keeps
    internal punctuation (hyphens in "low-top") because the taxonomy has
    hyphenated terms."""
    if not text:
        return ""
    lowered = str(text).lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip(" .!?,;:")


def _find_property_question(text):
    for pattern in PROPERTY_QUESTION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def _find_cue(text, cues):
    for cue in cues:
        if cue in text:
            return cue
    return None


def _split_on_companion(text):
    """Return (cue, trailing_fragment) for the first companion cue found,
    else (None, None).

    Multi-word phrases are checked before the bare prepositions so that
    "wear it with jeans" reports the specific phrase rather than the
    ambiguous token -- the caller uses that distinction to decide whether
    a taxonomy confirmation is required."""
    for pattern in COMPANION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0), text[match.end():].strip()
    for token in AMBIGUOUS_COMPANION_TOKENS:
        match = re.search(rf"\b{re.escape(token)}\b", text)
        if match:
            return token, text[match.end():].strip()
    return None, None


def _parses_to_category(fragment, parse_fragment):
    """True when `fragment` names something the real taxonomy knows.

    `parse_fragment` is injected rather than imported so this module stays
    unit-testable without loading the catalog metadata (which the real
    parser reads to build its attribute vocabulary). In production it is
    `composed_query_search.parse_text_fragment` bound to the live
    metadata path."""
    if not fragment or parse_fragment is None:
        return False, None
    try:
        parsed = parse_fragment(fragment)
    except Exception:
        # A parser failure must not take down routing; fall back to the
        # conservative answer (not a companion), which routes to
        # /identify -- the path that ignores the text rather than one
        # that acts on a misparse.
        return False, None
    category = (parsed or {}).get("category")
    if category and category.get("matched_term"):
        return True, category
    return False, None


def route(has_image, text, parse_fragment=None):
    """Decide which measured path answers this request.

    Args:
        has_image: whether the request carried a decodable image.
        text: the caller's free text, possibly empty.
        parse_fragment: callable(str) -> parsed dict, normally
            `composed_query_search.parse_text_fragment` bound to the
            catalog metadata path. Optional; without it, bare-"with"
            companion detection is disabled (the conservative direction).

    Returns a dict with `intent`, `path`, `reason`, `signals`, and
    `companion_text` (the fragment /compose should be given, which is the
    caller's own text -- /compose parses it itself, and handing it a
    pre-sliced fragment would diverge from what /compose does today).
    """
    normalised = _normalise(text)
    has_text = bool(normalised)

    signals = {
        "has_image": bool(has_image),
        "has_text": has_text,
        "property_question": None,
        "companion_cue": None,
        "companion_category": None,
        "exactness_cue": None,
        "similarity_cue": None,
    }

    if not has_image and not has_text:
        return {
            "intent": INTENT_UNROUTABLE,
            "path": None,
            "reason": "neither an image nor text was supplied",
            "signals": signals,
            "companion_text": None,
        }

    if not has_image:
        return {
            "intent": INTENT_SEARCH,
            "path": INTENT_TO_PATH[INTENT_SEARCH],
            "reason": "text only, so there is nothing to identify -- catalog browse",
            "signals": signals,
            "companion_text": None,
        }

    if not has_text:
        return {
            "intent": INTENT_IDENTIFY,
            "path": INTENT_TO_PATH[INTENT_IDENTIFY],
            "reason": "image only -- instance-seeking",
            "signals": signals,
            "companion_text": None,
        }

    # --- image AND text, the only genuinely ambiguous case ---

    signals["exactness_cue"] = _find_cue(normalised, EXACTNESS_CUES)
    signals["similarity_cue"] = _find_cue(normalised, SIMILARITY_CUES)

    # Property questions first: "what brand goes with this" is vanishingly
    # rare, while "what brand is this" is a headline query, and the
    # property reading is the safer of the two to be wrong about (it still
    # runs identify, just with a brand-shaped spoken answer).
    property_question = _find_property_question(normalised)
    if property_question:
        signals["property_question"] = property_question
        return {
            "intent": INTENT_BRAND,
            "path": INTENT_TO_PATH[INTENT_BRAND],
            "reason": (f"property question ({property_question!r}) about the pictured item; "
                       "answered from the identified catalog record"),
            "signals": signals,
            "companion_text": None,
        }

    cue, trailing = _split_on_companion(normalised)
    if cue:
        # `cue` is now matched TEXT, not a table key, so "is this the
        # ambiguous bare preposition" is the membership test that stays
        # correct as the pattern list grows.
        unambiguous = cue not in AMBIGUOUS_COMPANION_TOKENS
        confirmed, category = (True, None) if unambiguous else _parses_to_category(
            trailing, parse_fragment)
        if unambiguous:
            # Still parse it when we can, purely to report what the
            # companion resolved to; routing does not depend on it.
            _, category = _parses_to_category(trailing, parse_fragment)
        if confirmed:
            signals["companion_cue"] = cue
            signals["companion_category"] = category
            reason = (f"companion phrase {cue!r}" if unambiguous else
                      f"'{cue}' followed by the taxonomy term "
                      f"{(category or {}).get('matched_term')!r}")
            return {
                "intent": INTENT_COMPOSE,
                "path": INTENT_TO_PATH[INTENT_COMPOSE],
                "reason": f"{reason} -- anchor item plus a separately-searched companion",
                "signals": signals,
                "companion_text": text,
            }

    # Everything else with an image is an identify. The text is reported
    # back parsed, but it does NOT filter the candidate set: category
    # gating measured net-negative on seven independent runs
    # (docs/eval_log.md), so acting on it here would be a known
    # regression dressed up as a feature.
    if signals["exactness_cue"]:
        reason = f"exactness cue {signals['exactness_cue']!r} -- identity match wanted"
    elif signals["similarity_cue"]:
        reason = (f"similarity cue {signals['similarity_cue']!r} with an image -- "
                  "ranked neighbours of the pictured item")
    else:
        reason = "image with descriptive text naming no companion -- instance-seeking"
    return {
        "intent": INTENT_IDENTIFY,
        "path": INTENT_TO_PATH[INTENT_IDENTIFY],
        "reason": reason,
        "signals": signals,
        "companion_text": None,
    }


def bind_parser(metadata_path):
    """Build the `parse_fragment` callable for production use.

    Imported lazily: `composed_query_search` reads catalog metadata to
    build its attribute vocabulary, and the router is imported in contexts
    (tests, the CLI below) that should not pay for that."""
    import composed_query_search

    def parse_fragment(fragment):
        return composed_query_search.parse_text_fragment(fragment, str(metadata_path))

    return parse_fragment


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Explain how /query would route an input.")
    ap.add_argument("text", nargs="*", help="the caller's text")
    ap.add_argument("--image", action="store_true", help="pretend an image was supplied")
    ap.add_argument("--metadata", default="apparel_dataset/metadata.json",
                    help="catalog metadata, for real taxonomy matching")
    args = ap.parse_args()

    parser = None
    try:
        parser = bind_parser(args.metadata)
    except Exception as error:  # noqa: BLE001 -- CLI convenience only
        print(f"(taxonomy parser unavailable: {error}; bare-'with' detection off)")

    print(json.dumps(route(args.image, " ".join(args.text), parser), indent=2))
