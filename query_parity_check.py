"""Prove `/query` reproduces `/identify`, `/compose` and `/search` exactly.

`docs/unified_query_design.md` states the bar this script enforces:

    This is a routing change, not a retrieval change. It should be
    possible to ship `/query` with the three existing endpoints untouched
    behind it, and verify that `/query` reproduces each of them exactly on
    the same inputs. If it does not, the routing is wrong and that is
    measurable rather than a matter of taste.

So: for each case, POST the same body to the legacy endpoint and to
`/query`, and deep-diff the two responses.

**Fields excluded from the diff, and why each is legitimate:**

  * `latency_ms` -- wall-clock, never equal twice.
  * `route`      -- `/query` only; it is the whole point of the endpoint.
  * `spoken` on the BRAND cases only -- "what brand is this" deliberately
    gets a brand-shaped spoken answer while the structured results stay
    byte-identical. Excluding it anywhere else would hide a real
    divergence, so the exclusion is per-case, not global.

Nothing else is excluded. In particular `results`, `confidence`,
`garment_gate`, `predicted_category` and `companions` are compared in
full, element by element, because a routing bug that silently changed
`top_k` or dropped the category gate would show up in exactly those.

Usage:
    python query_parity_check.py                      # live service
    python query_parity_check.py --base-url http://localhost:8000
    python query_parity_check.py --image path/to.jpg

Needs `FASHION_API_KEY` in the environment or `.env`. The key is never
printed, including on auth failure.
"""

import argparse
import base64
import json
import os
import random
import sys
from pathlib import Path

import requests

DEFAULT_BASE_URL = "https://hanavm--fashion-serve-fashionservice-api.modal.run"
METADATA = Path("apparel_dataset/metadata.json")

# Compared everywhere except where noted in the module docstring.
ALWAYS_IGNORED = {"latency_ms", "route"}


def load_api_key():
    key = os.environ.get("FASHION_API_KEY")
    if key:
        return key
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("FASHION_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("FASHION_API_KEY not found in environment or .env")


def pick_catalog_image(seed=11):
    """A real catalog photo, so the identity path has something to find.

    Deterministic by seed: a parity run that picks a different image every
    time cannot distinguish 'the endpoints diverge' from 'the input
    changed'."""
    if not METADATA.exists():
        raise SystemExit(f"{METADATA} not found; pass --image explicitly")
    records = json.loads(METADATA.read_text())
    with_images = [r for r in records if r.get("images")]
    if not with_images:
        raise SystemExit("no catalog record has images; pass --image explicitly")
    rng = random.Random(seed)
    for _ in range(200):
        record = rng.choice(with_images)
        path = Path(record["images"][0])
        if path.exists():
            return path
    raise SystemExit("no catalog image path resolved on disk; pass --image explicitly")


def diff(left, right, ignored, path="", out=None):
    """Recursive deep diff. Returns a list of human-readable differences.

    Written out rather than using `==` on the whole payload so a failure
    names the exact field -- 'responses differ' is not an actionable
    result when the payload has a hundred nested keys."""
    out = [] if out is None else out
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key in ignored:
                continue
            child = f"{path}.{key}" if path else key
            if key not in left:
                out.append(f"{child}: missing from legacy response")
            elif key not in right:
                out.append(f"{child}: missing from /query response")
            else:
                diff(left[key], right[key], ignored, child, out)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append(f"{path}: length {len(left)} vs {len(right)}")
        else:
            for index, (a, b) in enumerate(zip(left, right)):
                diff(a, b, ignored, f"{path}[{index}]", out)
    elif left != right:
        out.append(f"{path}: {left!r} != {right!r}")
    return out


def post(base_url, endpoint, body, key, timeout):
    response = requests.post(
        f"{base_url.rstrip('/')}{endpoint}",
        json=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=timeout,
    )
    if response.status_code == 401:
        # Never echo the key, not even a prefix.
        raise SystemExit(f"401 from {endpoint}: FASHION_API_KEY rejected. "
                         "Note that `modal deploy` does NOT cycle a warm container, "
                         "so a rotated key needs `modal app stop` first.")
    return response.status_code, response.json()


def build_cases(image_b64):
    """Each case: (name, legacy endpoint, legacy body, /query body, extra ignores).

    The /query body carries NO endpoint hint -- if it needed one, the
    routing would not be doing its job."""
    return [
        ("identify: image only",
         "/identify", {"image_base64": image_b64},
         {"image_base64": image_b64},
         set()),

        ("identify: image + top_k passthrough",
         "/identify", {"image_base64": image_b64, "top_k": 3},
         {"image_base64": image_b64, "top_k": 3},
         set()),

        ("identify: image + descriptive text (must NOT compose)",
         "/identify", {"image_base64": image_b64},
         {"image_base64": image_b64, "text": "a shirt with stripes"},
         set()),

        ("identify: image + similarity cue",
         "/identify", {"image_base64": image_b64},
         {"image_base64": image_b64, "text": "find me something like this"},
         set()),

        ("brand: property question routes to identify",
         "/identify", {"image_base64": image_b64},
         {"image_base64": image_b64, "text": "what brand is this"},
         # Structured results must match exactly; only the spoken line
         # differs, by design.
         {"spoken"}),

        ("compose: bare 'with' + taxonomy term",
         "/compose", {"image_base64": image_b64, "text": "with cargo pants"},
         {"image_base64": image_b64, "text": "with cargo pants"},
         set()),

        ("compose: unambiguous companion phrase",
         "/compose", {"image_base64": image_b64, "text": "what goes with this"},
         {"image_base64": image_b64, "text": "what goes with this"},
         set()),

        ("search: text only",
         "/search", {"query": "blue jeans"},
         {"text": "blue jeans"},
         set()),

        ("search: text only, top_k passthrough",
         "/search", {"query": "gray suede sneakers", "top_k": 5},
         {"text": "gray suede sneakers", "top_k": 5},
         set()),

        ("search: composey phrasing but no image stays a search",
         "/search", {"query": "what goes with cargo pants"},
         {"text": "what goes with cargo pants"},
         set()),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("FASHION_API_URL", DEFAULT_BASE_URL))
    ap.add_argument("--image", type=Path, help="query image (default: a seeded catalog photo)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="generous by default: a cold Modal container has taken 353s")
    args = ap.parse_args()

    key = load_api_key()
    image_path = args.image or pick_catalog_image(args.seed)
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()

    print(f"base url : {args.base_url}")
    print(f"image    : {image_path}")
    print()

    failures = []
    for name, endpoint, legacy_body, query_body, extra_ignored in build_cases(image_b64):
        try:
            legacy_status, legacy = post(args.base_url, endpoint, legacy_body, key, args.timeout)
            query_status, unified = post(args.base_url, "/query", query_body, key, args.timeout)
        except requests.RequestException as error:
            failures.append((name, [f"request failed: {error}"]))
            print(f"  [ERR ] {name}\n         request failed: {error}")
            continue

        problems = []
        if legacy_status != query_status:
            problems.append(f"status {legacy_status} ({endpoint}) != {query_status} (/query)")

        route = (unified.get("route") or {})
        expected_path = endpoint
        if route.get("equivalent_endpoint") != expected_path:
            problems.append(f"routed to {route.get('equivalent_endpoint')!r}, "
                            f"expected {expected_path!r} -- {route.get('reason')}")

        problems += diff(legacy, unified, ALWAYS_IGNORED | extra_ignored)

        if problems:
            failures.append((name, problems))
            print(f"  [FAIL] {name}")
            for problem in problems[:8]:
                print(f"         {problem}")
            if len(problems) > 8:
                print(f"         ... and {len(problems) - 8} more")
        else:
            print(f"  [ok  ] {name}  -> {route.get('intent')} ({route.get('reason')})")

    print()
    if failures:
        print(f"PARITY FAILED: {len(failures)} of {len(build_cases(image_b64))} cases")
        return 1
    print(f"PARITY HELD: /query reproduced all {len(build_cases(image_b64))} cases exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
