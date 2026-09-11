#!/usr/bin/env python
"""Smoke-test the outfit-search API *the way the iOS client uses it*.

Why this exists
---------------
On 2026-09-11 the iOS app was launched for the first time and outfit search
could not work at all. Three faults, each fatal on its own:

1. `/outfit_search` was sent to the identifier host, where the route does
   not exist.
2. `image_url` comes back as ``/photo?path=<rel>`` and the client wrapped it
   a second time, producing ``/photo?path=/photo%3Fpath%3D...``.
3. `/photo` is authenticated and `AsyncImage` cannot send a header, so every
   photo 401'd behind a search that had already succeeded.

Every one of them is a *seam*: the services were up, the Swift compiled, and
the existing evaluation stack covers neither the client nor the transport
between them. `outfit_search_eval.py` measures ranking quality and would have
been perfectly happy while the app showed a grid of grey boxes.

So this asserts the things a retrieval metric structurally cannot:

* the product route is reachable **on the host the client actually uses**
* a result carries a usable image reference
* that reference resolves to **bytes that decode as an image**

Run it against the deployment before believing the app works::

    .venv/bin/python client_smoke_test.py
    .venv/bin/python client_smoke_test.py --text "a green jacket" --top-k 5

Exits non-zero on the first failure, so it is usable as a gate.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from urllib.parse import urljoin

import requests

# Defaults mirror ios/Config/Shared.xcconfig. Two services, not one — that
# confusion is fault 1 above, so the test names them separately on purpose.
OUTFIT_BASE = os.environ.get(
    "FASHION_OUTFIT_BASE_URL",
    "https://hanavm--outfit-serve-outfitservice-api.modal.run")
IDENTIFIER_BASE = os.environ.get(
    "FASHION_API_BASE_URL",
    "https://hanavm--fashion-serve-fashionservice-api.modal.run")

# Cold start is ~25 s on the CPU container; a tight timeout here reads as an
# outage when it is really just a scaled-to-zero app waking up.
TIMEOUT = 120


class SmokeFailure(AssertionError):
    """A check that must hold for the client to work at all."""


def _key() -> str:
    key = os.environ.get("FASHION_API_KEY")
    if not key:
        # Fall back to the repo .env, which is gitignored. Never printed.
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as handle:
                for line in handle:
                    name, _, value = line.partition("=")
                    if name.strip() == "FASHION_API_KEY":
                        key = value.strip().strip('"').strip("'")
                        break
    if not key:
        raise SmokeFailure(
            "No FASHION_API_KEY in the environment or .env — cannot test an "
            "authenticated API.")
    return key


def resolve_image_url(reference: str, outfit_base: str) -> str:
    """Reproduce `LiveFashionAPI.imageURL(for:)` exactly.

    Keeping this in lockstep with the Swift is the entire point: a drift
    between them is fault 2, and it is invisible from either side alone.
    """
    if not reference:
        raise SmokeFailure("result carried an empty image reference")
    if reference.startswith(("http://", "https://")):
        return reference                      # already absolute
    if reference.startswith("/"):
        return urljoin(outfit_base, reference)  # '/photo?path=<rel>'
    return f"{outfit_base.rstrip('/')}/photo?path={reference}"  # bare corpus path


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
        return
    print(f"  FAIL  {name}{': ' + detail if detail else ''}")
    raise SmokeFailure(f"{name}{': ' + detail if detail else ''}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", action="append", default=None,
                        help="A query clause. Repeat for multi-garment queries.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--outfit-base", default=OUTFIT_BASE)
    parser.add_argument("--identifier-base", default=IDENTIFIER_BASE)
    parser.add_argument("--skip-identifier", action="store_true",
                        help="Only test the product path.")
    args = parser.parse_args()

    texts = args.text or ["a black leather jacket", "baggy jeans"]
    key = _key()
    headers = {"Authorization": f"Bearer {key}"}

    print(f"outfit service: {args.outfit_base}")
    print(f"query: {texts}\n")

    # 1. The product route, on the host the client actually uses. A 404 here
    #    is fault 1 and means the base URL is pointed at the wrong service.
    started = time.time()
    response = requests.post(f"{args.outfit_base}/outfit_search", headers=headers,
                             json={"texts": texts, "top_k": args.top_k},
                             timeout=TIMEOUT)
    check("/outfit_search is reachable on the outfit host",
          response.status_code != 404,
          "404 — this host does not serve outfit search. Base URL points at "
          "the identifier?")
    check("/outfit_search returns 200", response.status_code == 200,
          f"HTTP {response.status_code}: {response.text[:200]}")

    payload = response.json()
    results = payload.get("results") or []
    check("the query returns results", bool(results), "empty result list")
    print(f"        {len(results)} results in {time.time() - started:.2f}s "
          f"(server reported {payload.get('latency_ms')} ms)")

    # 2. A usable image reference, resolved the way the client resolves it.
    first = results[0]
    reference = first.get("image_url", "")
    check("a result carries an image reference", bool(reference))
    photo_url = resolve_image_url(reference, args.outfit_base)
    check("the resolved photo URL is not double-wrapped",
          photo_url.count("/photo") == 1,
          f"wrapped twice: {photo_url}")

    # 3. Authenticated, and the bytes are actually an image. Fault 3 lived
    #    here: unauthenticated this is a 401, which the UI rendered as
    #    "Photo unavailable" on every tile.
    anonymous = requests.get(photo_url, timeout=TIMEOUT)
    check("/photo rejects an unauthenticated request",
          anonymous.status_code == 401,
          f"expected 401, got {anonymous.status_code} — is the corpus public?")

    photo = requests.get(photo_url, headers=headers, timeout=TIMEOUT)
    check("/photo returns 200 with credentials", photo.status_code == 200,
          f"HTTP {photo.status_code}: {photo.text[:200]}")
    check("the response is image bytes, not an error page",
          photo.headers.get("Content-Type", "").startswith("image/"),
          f"Content-Type: {photo.headers.get('Content-Type')}")

    try:
        from PIL import Image
        image = Image.open(io.BytesIO(photo.content))
        image.verify()
        check(f"the bytes decode as an image ({image.size[0]}x{image.size[1]})", True)
    except ImportError:
        # A JPEG/PNG magic-number check is a weaker assertion than a decode,
        # but it still catches an HTML error page served with a 200.
        magic = photo.content[:4]
        check("the bytes start with an image magic number",
              magic.startswith((b"\xff\xd8\xff", b"\x89PNG")),
              f"leading bytes {magic!r}")

    # 4. Provenance. Every displayed photo has to link back to its source —
    #    this is a licensing commitment, not a nicety. See
    #    docs/licensing_decision_memo.md.
    with_provenance = sum(1 for r in results if r.get("post_url"))
    check("every result carries a post_url back to its source",
          with_provenance == len(results),
          f"only {with_provenance} of {len(results)}")

    if not args.skip_identifier:
        print("\nidentifier service (the 'what is this item' feature):")
        health = requests.get(f"{args.identifier_base}/health", headers=headers,
                              timeout=TIMEOUT)
        check("/health returns 200", health.status_code == 200,
              f"HTTP {health.status_code}")
        index = health.json().get("index", {})
        print(f"        {index.get('catalog_products')} products, "
              f"{len(index.get('brands') or [])} brands")

    print("\nAll checks passed. The client path works end to end.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeFailure as failure:
        print(f"\nSMOKE TEST FAILED: {failure}", file=sys.stderr)
        sys.exit(1)
