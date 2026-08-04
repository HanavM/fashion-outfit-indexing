"""Phase 7's client half: exercise the serving API end to end from a laptop.

WHY THIS EXISTS
---------------
The product surface is a Siri Shortcut ("what does this jacket look like
with cargo pants?"), and a Shortcut is not code you can run, diff, test or
debug -- it is a list of taps in an iPhone app. So the contract gets a
second client that IS code. This script does exactly what the Shortcut in
`siri/README.md` does, in the same order, against the same endpoints, with
the same failure handling:

    read an image -> (optionally) crop the garment out of it
    -> base64 -> POST /identify or /compose -> speak one sentence
    -> show the ranked alternatives

That makes it two things at once: the way the flow gets tested without an
iPhone in the loop, and the reference the Shortcut recipe is transcribed
from. If the two ever disagree, this file is the one that was run.

THE CONTRACT IT SPEAKS
----------------------
Defined in `siri/README.md` (single source of truth; `modal_app_serve.py`
is owned by the serving work and is NOT edited from here). Summary:

    GET  /health   -> {"status": "ok", ...}
    POST /identify {"image_base64": str, "top_k": int}
    POST /compose  {"image_base64": str, "text": str, "top_k": int}

    both POSTs return: {"spoken": str, "confidence": float,
                        "rejected": bool, ...results...}

Field extraction here is deliberately tolerant (`results` or `matches`,
`rejected` or `rejected_open_set`) because the pipeline's own internal
name is `rejected_open_set` (hierarchical_retrieval_pipeline.py's
`retrieve()`), and a client that hard-crashes on a synonym is a worse
client than one that finds the field. It is NOT tolerant about the field
being absent entirely -- see WHEN IN DOUBT below.

WHEN IN DOUBT, SAY SO (the whole point of the degradation ladder)
----------------------------------------------------------------
A voice assistant that confidently names the wrong jacket is worse than
one that says "I'm not sure", because the user cannot see the ranked list
to catch the error -- they only hear one sentence. So:

  * The server's own `spoken` line is used verbatim ONLY when the answer
    is not rejected and clears --min-confidence. Otherwise this client
    writes its own hedged sentence and ignores the server's, on the
    principle that the hedge must survive a server that forgot to hedge.
  * A MISSING `rejected`/`confidence` field is treated as unsafe, not as
    "fine" -- absence of a rejection signal is not evidence of a match.
  * Network failure, auth failure, HTTP error, malformed JSON and empty
    results all have their own spoken line, and all exit non-zero.

Exit codes (the Shortcut has no equivalent, but CI and shell loops do):
    0  confident answer spoken
    2  ran fine, but the honest answer was "I'm not sure"
    3  could not reach / could not use the server

THE SCREENSHOT PROBLEM
----------------------
See `siri/README.md`'s section of the same name for the full argument.
Short version: "seen on screen" means the input is a screenshot of
arbitrary UI -- nav bars, price text, four other products -- and every
accuracy number this project has is catalog-photo-to-catalog-photo.
`--segment` runs `segment_outfit.py` locally first and sends the single
best garment crop instead of the whole screen. It is off by default
because it loads SAM2 (slow on CPU) and because its thresholds are not
validated for screenshots either -- it is a lever to measure, not a fix
to assume.

Usage:
    export FASHION_API_URL=https://...modal.run
    export FASHION_API_TOKEN=...
    python3 siri_client.py --health
    python3 siri_client.py --image shot.png
    python3 siri_client.py --image shot.png --text "with cargo pants"
    python3 siri_client.py --image shot.png --segment --show-request
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_URL = os.environ.get("FASHION_API_URL", "http://127.0.0.1:8000")
DEFAULT_TOKEN_ENV = "FASHION_API_TOKEN"

# Below this, the client refuses to assert and hedges instead. Not tuned
# against a labeled screenshot set -- there isn't one (see docstring) --
# so it is a conservative default and an explicit flag, not a claim.
DEFAULT_MIN_CONFIDENCE = 0.35

# 20s was too short for the case that matters most: the FIRST call.
#
# Measured 2026-08-04 against the live service. The serving container
# scales to zero after 20 minutes idle, and a cold start is ~17s of model
# and index loading before it can answer at all -- so the first request of
# any session reliably blew a 20s budget and the client reported "The
# fashion service isn't responding right now." Every warm call after it
# answered in 1-2.4s. The one request a user is most likely to make is the
# one that failed.
#
# 90s is not a latency target, it is a ceiling for the cold path. If the
# service is genuinely down, connection errors surface immediately and do
# not wait this out.
DEFAULT_TIMEOUT = 90.0
DEFAULT_TOP_K = 5

# Payload guard. iOS Shortcuts will happily base64 a 12MB screenshot and
# then time out on a cellular connection; the server has a body limit too.
# Downscale before that happens rather than after.
MAX_UPLOAD_DIM = 1280
MAX_UPLOAD_BYTES = 4 * 1024 * 1024

EXIT_OK = 0
EXIT_UNSURE = 2
EXIT_UNAVAILABLE = 3


# --------------------------------------------------------------------------
# image preparation
# --------------------------------------------------------------------------

def load_and_prepare_image(path, max_dim=MAX_UPLOAD_DIM):
    """Returns (jpeg_bytes, note). Screenshots arrive as PNG with an alpha
    channel and at 3x retina resolution; both are wasted bytes on the wire
    and neither helps the encoder, which resizes to 224/384 anyway."""
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    original = image.size
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    note = f"{original[0]}x{original[1]} -> {image.size[0]}x{image.size[1]}, {len(buffer.getvalue())/1024:.0f} KB JPEG"
    return buffer.getvalue(), note


def segment_best_garment(path, category_hint=None):
    """Crop the most plausible single garment out of a screenshot using
    segment_outfit.py, so the encoder sees a garment instead of a web page.

    Returns (jpeg_bytes_or_None, note). Imports are local and lazy: SAM2 +
    FashionCLIP are ~1GB of weights and most invocations of this script
    don't want them loaded at all.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    from segment_outfit import (
        detect_outfit_items, SAM2_CONFIG, SAM2_CHECKPOINT, MASK_GENERATOR_KWARGS,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"  # no MPS, see segment_outfit.py
    processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model = AutoModelForZeroShotImageClassification.from_pretrained(
        "patrickjohncyh/fashion-clip").to(device)
    sam2 = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2, **MASK_GENERATOR_KWARGS)

    items = detect_outfit_items(path, mask_generator, processor, clip_model, device)
    if not items:
        return None, "segmentation found no garment in this image"

    chosen = items[0]
    if category_hint:
        hint = category_hint.lower()
        matching = [i for i in items
                    if i["category"].lower() in hint or i["category_group"].lower() in hint]
        if matching:
            chosen = matching[0]

    buffer = io.BytesIO()
    chosen["crop"].convert("RGB").save(buffer, format="JPEG", quality=90)
    others = ", ".join(f"{i['category']}({i['confidence']:.2f})" for i in items[1:])
    note = (f"segmented {len(items)} item(s), sending "
            f"{chosen['category_group']}/{chosen['category']} "
            f"(confidence {chosen['confidence']:.3f}, area {chosen['area_fraction']:.3f})"
            + (f"; also saw {others}" if others else ""))
    return buffer.getvalue(), note


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

class ServerUnavailable(Exception):
    """Anything that means 'no usable answer came back': DNS, connection
    refused, timeout, 4xx/5xx, non-JSON body. Collapsed into one type
    because the spoken degradation is identical for all of them -- the
    detail goes to stderr for the human, not to the speaker."""


def post_json(url, path, payload, token, timeout):
    import requests

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.post(url.rstrip("/") + path, json=payload,
                                 headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        raise ServerUnavailable(f"timed out after {timeout:.0f}s")
    except requests.exceptions.RequestException as error:
        raise ServerUnavailable(f"could not reach {url}: {error}")

    if response.status_code == 401 or response.status_code == 403:
        raise ServerUnavailable(f"auth rejected (HTTP {response.status_code}) -- "
                                f"check ${DEFAULT_TOKEN_ENV}")
    if response.status_code >= 400:
        raise ServerUnavailable(f"HTTP {response.status_code}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError:
        raise ServerUnavailable(f"response was not JSON: {response.text[:300]}")


def get_health(url, token, timeout):
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = requests.get(url.rstrip("/") + "/health", headers=headers, timeout=timeout)
    except requests.exceptions.RequestException as error:
        raise ServerUnavailable(f"could not reach {url}: {error}")
    if response.status_code >= 400:
        raise ServerUnavailable(f"HTTP {response.status_code}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError:
        raise ServerUnavailable(f"response was not JSON: {response.text[:300]}")


# --------------------------------------------------------------------------
# response interpretation
# --------------------------------------------------------------------------

def first_present(payload, *names, default=None):
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return default


def identity_block(payload):
    """The part of the response that describes WHAT THE IMAGE IS, with its
    own results/confidence/rejection.

    /identify returns that block at the top level. /compose nests it under
    `primary` and puts the text-search companions beside it, so the
    rejection flag and the confidence live one level down -- and a client
    that only looked at the top level of a /compose response would find no
    rejection field at all and hedge on every single composed query.
    Finding it is the difference between a usable feature and a permanent
    'I'm not sure'."""
    if isinstance(payload.get("results"), list):
        return payload
    nested = first_present(payload, "primary", "primary_item")
    if isinstance(nested, dict) and isinstance(nested.get("results"), list):
        return nested
    if isinstance(nested, dict):
        return nested  # server sent a bare product as `primary`
    return payload


def extract_results(payload):
    """(primary_product, companions, flat_results) -- normalised across
    both endpoint shapes so the printing code doesn't branch."""
    block = identity_block(payload)
    flat = first_present(block, "results", "matches", "products", default=[]) or []
    companions = first_present(payload, "companions", "second_item_matches", default=[]) or []
    if block is payload:
        primary = None
    else:
        # /compose: the primary item is the top hit of the nested block,
        # or the nested dict itself if the server sent a bare product.
        primary = flat[0] if flat else (block if "product_code" in block else None)
    return primary, list(companions), list(flat)


def describe_product(product):
    """One human-readable line for a product dict, tolerant of which of the
    catalog's naming fields the server chose to surface."""
    if not isinstance(product, dict):
        return str(product)
    brand = first_present(product, "brand", "brand_name", default="")
    name = first_present(product, "name", "product_name", "title", default="(unnamed)")
    code = first_present(product, "product_code", "code", "id", default="")
    score = first_present(product, "score", "similarity", "distance")
    parts = [p for p in [str(brand).strip(), str(name).strip()] if p]
    line = " ".join(parts) or str(code)
    if code:
        line += f"  [{code}]"
    if isinstance(score, (int, float)):
        line += f"  score={score:.3f}"
    return line


def is_rejected(payload):
    """True = the server said this is not in the catalog. None = the server
    did not say, which this client treats as unsafe rather than as a pass:
    the open-set rejection already exists in the pipeline
    (`rejected_open_set`) and must not be silently dropped at the API edge
    (roadmap Phase 6.4). If it's missing, something dropped it."""
    block = identity_block(payload)
    value = first_present(block, "rejected", "rejected_open_set")
    if value is None:
        return None
    return bool(value)


def rejection_is_trustworthy(payload):
    """False when the server itself says its rejection is uncalibrated
    (`reject_threshold_calibrated: false`, or no threshold set at all --
    the pipeline's REJECT_SIMILARITY_THRESHOLD is None by default, which
    makes `rejected_open_set` always false and therefore meaningless).

    This does NOT force a hedge -- doing that would make every answer
    "I'm not sure" and the feature pointless. It means the confidence gate
    below is the ONLY thing standing between the user and a confident
    wrong answer, and the reason line says so out loud so nobody reads a
    `rejected: false` as "the system checked"."""
    block = identity_block(payload)
    if block.get("reject_threshold_calibrated") is False:
        return False
    if "reject_threshold" in block and block.get("reject_threshold") is None:
        return False
    return True


def confidence_of(payload):
    block = identity_block(payload)
    value = first_present(block, "confidence", "hsc_confidence", "score")
    return float(value) if isinstance(value, (int, float)) else None


def decide_spoken(payload, min_confidence, segmentation_note=None):
    """Returns (spoken_line, confident: bool, reason: str).

    The server's own `spoken` is used verbatim only for a confident,
    non-rejected answer. Everything else gets a locally-written hedge, so
    that a server which forgets to hedge cannot make this client assert."""
    rejected = is_rejected(payload)
    confidence = confidence_of(payload)
    primary, companions, flat = extract_results(payload)
    top = primary or (flat[0] if flat else None)

    if rejected is True:
        name = product_phrase(top) if isinstance(top, dict) else None
        if name:
            return (f"I'm not sure. The closest thing I found was {name}, "
                    f"but it isn't a confident match.", False, "server rejected (open-set)")
        return ("I'm not sure -- I couldn't match that to anything in the catalog.",
                False, "server rejected (open-set)")

    if rejected is None:
        return ("I'm not sure -- the service didn't tell me whether that was a real match.",
                False, "response carried no rejection field")

    if not top and not companions:
        return ("I'm not sure -- I couldn't find anything that looks like that.",
                False, "empty result set")

    if confidence is None:
        return (f"I think that's {product_phrase(top)}, but I'm not certain.",
                False, "response carried no confidence field")

    if confidence < min_confidence:
        return (f"I'm not sure. It might be {product_phrase(top)}, "
                f"but the match is weak.", False,
                f"confidence {confidence:.3f} < {min_confidence:.2f}")

    caveat = "" if rejection_is_trustworthy(payload) else \
        " (server says its open-set rejection is UNCALIBRATED -- the confidence gate is the only guard)"

    server_line = payload.get("spoken")
    if isinstance(server_line, str) and server_line.strip():
        return (server_line.strip(), True, f"server spoken line, confidence {confidence:.3f}{caveat}")

    # Server met the bar but sent no spoken line -- build one rather than
    # going silent, since the Shortcut has nothing else to say.
    line = f"That looks like {product_phrase(top)}."
    if companions:
        line += f" It goes with {product_phrase(companions[0])}."
    return (line, True, f"locally composed (server sent no spoken line){caveat}")


def product_phrase(product):
    if not isinstance(product, dict):
        return "something I can't name"
    brand = str(first_present(product, "brand", "brand_name", default="")).strip()
    name = str(first_present(product, "name", "product_name", "title", default="")).strip()
    phrase = " ".join(p for p in [brand, name] if p)
    return phrase or "an unnamed catalog item"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run(args):
    url = args.url
    token = args.token or os.environ.get(DEFAULT_TOKEN_ENV)

    if args.health:
        try:
            payload = get_health(url, token, args.timeout)
        except ServerUnavailable as error:
            print(f"UNAVAILABLE: {error}", file=sys.stderr)
            print("SPOKEN: The fashion service isn't responding right now.")
            return EXIT_UNAVAILABLE
        print(json.dumps(payload, indent=2))
        return EXIT_OK

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"no such image: {image_path}", file=sys.stderr)
        print("SPOKEN: I couldn't find an image to look at.")
        return EXIT_UNAVAILABLE

    segmentation_note = None
    image_bytes = None
    if args.segment:
        try:
            image_bytes, segmentation_note = segment_best_garment(str(image_path), args.text)
        except Exception as error:
            # Segmentation is an optimisation, never a hard dependency:
            # a screenshot sent whole is a worse query, not no query.
            segmentation_note = f"segmentation failed ({error}); sending the whole image"
            image_bytes = None
        if image_bytes is None and segmentation_note and "no garment" in segmentation_note:
            if args.strict_segment:
                print(f"NOTE: {segmentation_note}", file=sys.stderr)
                print("SPOKEN: I don't see any clothing in that image.")
                return EXIT_UNSURE

    if image_bytes is None:
        image_bytes, prepare_note = load_and_prepare_image(str(image_path))
    else:
        # Re-run the size guard over the crop too.
        temp = io.BytesIO(image_bytes)
        from PIL import Image
        with Image.open(temp) as crop:
            crop = crop.convert("RGB")
            if max(crop.size) > MAX_UPLOAD_DIM:
                crop.thumbnail((MAX_UPLOAD_DIM, MAX_UPLOAD_DIM))
            out = io.BytesIO()
            crop.save(out, format="JPEG", quality=90)
            image_bytes = out.getvalue()
        prepare_note = f"crop {len(image_bytes)/1024:.0f} KB JPEG"

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        print(f"image still {len(image_bytes)/1e6:.1f} MB after downscale; refusing to upload",
              file=sys.stderr)
        return EXIT_UNAVAILABLE

    encoded = base64.b64encode(image_bytes).decode("ascii")
    if segmentation_note:
        print(f"[segment] {segmentation_note}", file=sys.stderr)
    print(f"[image] {prepare_note}, {len(encoded)/1024:.0f} KB base64", file=sys.stderr)

    endpoint = "/compose" if args.text else "/identify"
    payload = {"image_base64": encoded, "top_k": args.top_k}
    if args.text:
        payload["text"] = args.text

    if args.show_request:
        redacted = dict(payload, image_base64=f"<{len(encoded)} base64 chars>")
        print(f"[request] POST {url.rstrip('/')}{endpoint}\n"
              f"{json.dumps(redacted, indent=2)}", file=sys.stderr)

    started = time.time()
    try:
        response = post_json(url, endpoint, payload, token, args.timeout)
    except ServerUnavailable as error:
        print(f"UNAVAILABLE: {error}", file=sys.stderr)
        print("SPOKEN: I can't reach the fashion service right now.")
        return EXIT_UNAVAILABLE
    elapsed = time.time() - started

    if args.raw:
        print(json.dumps(response, indent=2))

    spoken, confident, reason = decide_spoken(response, args.min_confidence, segmentation_note)
    print(f"SPOKEN: {spoken}")
    print(f"[{elapsed*1000:.0f} ms | {reason}]", file=sys.stderr)

    primary, companions, flat = extract_results(response)
    if primary:
        # /compose: primary is flat[0], so the rest of the identity block
        # is "other things the image might be", not a second result list.
        print(f"\nprimary: {describe_product(primary)}")
        for rank, item in enumerate(flat[1:], start=2):
            print(f"  or #{rank}: {describe_product(item)}")
    elif flat:
        print("\nresults:")
        for rank, item in enumerate(flat, start=1):
            print(f"  {rank}. {describe_product(item)}")
    if companions:
        print("\ncompanions:")
        for rank, item in enumerate(companions, start=1):
            print(f"  {rank}. {describe_product(item)}")

    # The server attaches its own standing caveats (e.g. /compose's "these
    # are two independent searches, nothing shows these were worn together
    # -- that needs Phase 8"). Print them: a caveat the API bothered to
    # send and the client swallows is a caveat that doesn't exist.
    note = response.get("note")
    if note:
        print(f"\nnote: {note}")

    return EXIT_OK if confident else EXIT_UNSURE


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="Path to a screenshot or photo.")
    parser.add_argument("--text", default=None,
                        help="Second-item phrase, e.g. 'with cargo pants'. "
                             "Present -> POST /compose; absent -> POST /identify.")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"Base URL of the serving app (default ${{FASHION_API_URL}} or {DEFAULT_URL}).")
    parser.add_argument("--token", default=None,
                        help=f"Bearer token (default ${DEFAULT_TOKEN_ENV}).")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                        help="Below this the client hedges instead of asserting "
                             f"(default {DEFAULT_MIN_CONFIDENCE}).")
    parser.add_argument("--segment", action="store_true",
                        help="Run segment_outfit.py locally first and send the best garment "
                             "crop instead of the whole screenshot. Loads SAM2 -- slow on CPU.")
    parser.add_argument("--strict-segment", action="store_true",
                        help="With --segment, refuse to fall back to the whole image when no "
                             "garment is found; say 'I don't see any clothing' instead.")
    parser.add_argument("--health", action="store_true", help="GET /health and exit.")
    parser.add_argument("--raw", action="store_true", help="Print the full JSON response.")
    parser.add_argument("--show-request", action="store_true",
                        help="Print the request body (base64 redacted) -- use this to build "
                             "the Shortcut's 'Get Contents of URL' step.")
    args = parser.parse_args()

    if not args.health and not args.image:
        parser.error("--image is required (or use --health)")

    sys.exit(run(args))


if __name__ == "__main__":
    main()
