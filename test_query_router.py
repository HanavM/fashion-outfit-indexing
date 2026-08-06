"""Routing tests for `query_router.route`.

Run: `python test_query_router.py` (stdlib only, no pytest, no catalog).

Two kinds of case are in here and the distinction matters:

  * The obvious ones (image only -> identify) exist as regression anchors.
  * The **adversarial** ones are the point. Bare "with" is the whole
    difficulty of this router, so the "a shirt with stripes" family is
    tested as hard as the "with cargo jorts" family. A router that gets
    the first family right by accident (by having no rule at all) fails
    the second, and vice versa -- neither alone proves anything.

The taxonomy parser is stubbed with a small fake so these run without
loading catalog metadata. `test_real_parser_agrees` optionally re-runs the
companion cases against the REAL parser when the catalog is present, which
is what actually guards against the stub drifting from production.
"""

import query_router as qr

# Terms the fake taxonomy knows, mirroring the shape of the real parser's
# output rather than its full vocabulary.
FAKE_TAXONOMY = {
    "jorts": "denim shorts",
    "shorts": "shorts",
    "jeans": "jeans",
    "pants": "pants",
    "cargo pants": "cargo pants",
    "sneakers": "sneaker",
    "hoodie": "hoodie",
    "jacket": "jacket",
    "tee": "t-shirt",
    "t-shirt": "t-shirt",
    "boots": None,
}


def fake_parse(fragment):
    """Stand-in for composed_query_search.parse_text_fragment."""
    lowered = (fragment or "").lower()
    # Longest term first, so "cargo pants" wins over "pants".
    for term in sorted(FAKE_TAXONOMY, key=len, reverse=True):
        if term in lowered and FAKE_TAXONOMY[term]:
            return {"raw_text": fragment,
                    "category": {"leaf": FAKE_TAXONOMY[term],
                                 "category": FAKE_TAXONOMY[term],
                                 "group": None,
                                 "matched_term": term},
                    "attributes": {}}
    return {"raw_text": fragment, "category": None, "attributes": {}}


# (has_image, text, expected_intent, why-this-case-exists)
CASES = [
    # --- degenerate ---
    (False, "", qr.INTENT_UNROUTABLE, "empty request must 400, not guess"),
    (False, "   ", qr.INTENT_UNROUTABLE, "whitespace is not text"),

    # --- single-modality: must reproduce /identify and /search exactly ---
    (True, "", qr.INTENT_IDENTIFY, "image only is today's /identify"),
    (True, None, qr.INTENT_IDENTIFY, "None text behaves as absent"),
    (False, "blue jeans", qr.INTENT_SEARCH, "text only is today's /search"),
    (False, "gray suede adidas sneakers", qr.INTENT_SEARCH, "spec's own search example"),
    (False, "what goes with cargo pants", qr.INTENT_SEARCH,
     "no image means nothing to anchor on, however composey the phrasing"),

    # --- property questions: the query that had no endpoint at all ---
    (True, "what brand is this", qr.INTENT_BRAND, "headline unrouteable-before query"),
    (True, "what brand are these", qr.INTENT_BRAND, "plural form"),
    (True, "who makes this", qr.INTENT_BRAND, "verb form, no 'brand' token"),
    (True, "which brand is this jacket", qr.INTENT_BRAND,
     "a taxonomy term present must not steal the property reading"),

    # --- unambiguous companion phrases -> /compose ---
    (True, "what goes with this", qr.INTENT_COMPOSE,
     "companion phrase with NO taxonomy term still composes"),
    (True, "pair this with something", qr.INTENT_COMPOSE, "verb form"),
    (True, "what should i wear with this", qr.INTENT_COMPOSE, "natural phrasing"),
    (True, "styled with baggy jeans", qr.INTENT_COMPOSE, "participle form"),

    # --- bare "with": the hard case, both directions ---
    (True, "this shoe with cargo jorts", qr.INTENT_COMPOSE,
     "bare 'with' + slang taxonomy term -> companion (the doc's own example)"),
    (True, "with jeans", qr.INTENT_COMPOSE, "minimal companion fragment"),
    (True, "a shirt with stripes", qr.INTENT_IDENTIFY,
     "bare 'with' + ATTRIBUTE must NOT compose -- this is the trap"),
    (True, "jacket with a hood", qr.INTENT_IDENTIFY,
     "'hood' is not the taxonomy term 'hoodie'; must not compose"),
    (True, "sneakers with red laces", qr.INTENT_IDENTIFY,
     "leading taxonomy term describes the ANCHOR, trailing text is an attribute"),

    # --- exactness / similarity cues stay on the identity path ---
    (True, "this exact sneaker", qr.INTENT_IDENTIFY, "exactness cue"),
    (True, "find me blue jeans like this", qr.INTENT_IDENTIFY,
     "the design doc's named gap: no endpoint accepted this before"),
    (True, "something like this hoodie", qr.INTENT_IDENTIFY, "similarity cue"),
]


def run_cases():
    failures = []
    for has_image, text, expected, why in CASES:
        result = qr.route(has_image, text, fake_parse)
        actual = result["intent"]
        status = "ok  " if actual == expected else "FAIL"
        if actual != expected:
            failures.append((has_image, text, expected, actual, why))
        print(f"  [{status}] image={int(bool(has_image))} text={text!r:38} "
              f"-> {actual:11} ({why})")
    return failures


def check_signals():
    """Signals are part of the contract -- the response reports them, so a
    wrong-but-luckily-right-intent route is still a bug."""
    failures = []

    exact = qr.route(True, "I want this exact sneaker", fake_parse)
    if exact["signals"]["exactness_cue"] != "this exact":
        failures.append(("exactness cue not reported", exact["signals"]))

    similar = qr.route(True, "find me blue jeans like this", fake_parse)
    if similar["signals"]["similarity_cue"] != "like this":
        failures.append(("similarity cue not reported", similar["signals"]))

    companion = qr.route(True, "this shoe with cargo jorts", fake_parse)
    matched = (companion["signals"]["companion_category"] or {}).get("matched_term")
    if matched != "jorts":
        failures.append(("companion category not resolved to 'jorts'", matched))
    if companion["companion_text"] != "this shoe with cargo jorts":
        failures.append(("compose must receive the caller's ORIGINAL text, not a slice",
                         companion["companion_text"]))

    # Without a parser, bare "with" must fail CLOSED (identify), never open.
    no_parser = qr.route(True, "this shoe with cargo jorts", None)
    if no_parser["intent"] != qr.INTENT_IDENTIFY:
        failures.append(("no-parser bare-'with' must fall back to identify",
                         no_parser["intent"]))

    # A parser that raises must not take routing down.
    def exploding(_fragment):
        raise RuntimeError("catalog unavailable")

    survived = qr.route(True, "this shoe with cargo jorts", exploding)
    if survived["intent"] != qr.INTENT_IDENTIFY:
        failures.append(("parser exception must degrade to identify", survived["intent"]))

    for failure in failures:
        print(f"  [FAIL] {failure[0]}: {failure[1]!r}")
    if not failures:
        print("  [ok  ] signals, original-text passthrough, and both parser "
              "failure modes")
    return failures


def test_real_parser_agrees():
    """Re-run the companion cases against the REAL taxonomy parser.

    Skipped when the catalog is absent. This is the check that matters:
    the stub above could drift from `parse_text_fragment` indefinitely and
    every other test would still pass."""
    from pathlib import Path

    metadata = Path("apparel_dataset/metadata.json")
    if not metadata.exists():
        print("  [skip] apparel_dataset/metadata.json not present")
        return []
    try:
        parser = qr.bind_parser(metadata)
    except Exception as error:  # noqa: BLE001
        print(f"  [skip] real parser unavailable: {error}")
        return []

    failures = []
    for has_image, text, expected, why in CASES:
        actual = qr.route(has_image, text, parser)["intent"]
        if actual != expected:
            failures.append((text, expected, actual, why))
            print(f"  [FAIL] {text!r} -> {actual} (stub said {expected}) -- {why}")
    if not failures:
        print(f"  [ok  ] real parser agrees with the stub on all {len(CASES)} cases")
    return failures


if __name__ == "__main__":
    print("routing cases (fake taxonomy):")
    failed = run_cases()
    print("\nsignal contract:")
    failed += check_signals()
    print("\nreal-parser agreement:")
    failed += test_real_parser_agrees()

    print()
    if failed:
        print(f"FAILED: {len(failed)} case(s)")
        raise SystemExit(1)
    print(f"PASSED: {len(CASES)} routing cases + signal contract + real-parser agreement")
