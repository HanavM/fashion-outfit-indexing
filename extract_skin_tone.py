"""Estimate the wearer's skin tone in each outfit photo, for a styling filter.

    .venv/bin/python extract_skin_tone.py --device mps
    .venv/bin/python extract_skin_tone.py --report

## Why

How a colour reads against your own skin is a real thing people choose
clothes on, and "show me this on someone with my skin tone" is a normal
fashion-app filter. This estimates a tone per photo so outfit search can
offer it.

## What it is, and what it is NOT

**It is a pixel estimate of how the skin appears IN THAT PHOTOGRAPH.** It
is not a demographic label, it is not an attribute of the person, and it
should never be presented as one. The same person photographed in warm
indoor light, in shade, and under a phone flash will land on different
tones — that is a property of photography, not of them.

Reported on the **Monk Skin Tone scale** (10 tones), which was designed
for exactly this kind of inclusive product use and is broader in the
darker range than Fitzpatrick, which was built to describe sunburn risk
and compresses most non-white tones into two categories.

**Known unreliability, stated up front:**
- No white-balance correction. Uncontrolled lighting is the dominant error
  source and it is large.
- Skin in shadow reads darker; blown highlights read lighter.
- The parser's Face/arm/leg classes are model output, not ground truth.
- Nothing here is validated against any labelled set, because none exists.

`confidence` is the fraction of the frame that produced usable skin pixels
plus the parser's own posterior over them. Treat a low value as "no
estimate" rather than as a dark or light reading.

## MEASURED: absolute tone binning does not work, use the LAB value

The Monk swatches sit at L* 78-94 for tones 1-5, while real photographed
skin in this corpus measures **L* 11-61**. Nearest-swatch matching in full
CIELAB therefore could not return a tone below 6 at all, and the first 12
photos all landed in 6-10. Down-weighting lightness 4x helped but 26 of 40
still collapsed onto a single tone. That is an exposure artifact, not a
property of the people photographed.

**So `monk_tone` is a convenience label carrying `monk_reliable: False`,
and the search filter is RELATIVE** -- "outfits on people whose skin reads
similarly to this reference photo" -- comparing two values measured the
same way, where the shared bias largely cancels. An absolute
classification would need controlled lighting or a labelled calibration
set, and neither exists here.

## How

ATR-18 (the same parser `garment_proposer` uses) has Face, Left/Right-arm
and Left/Right-leg classes. Face is weighted highest: it is the largest
reliably-exposed skin region and least likely to be a shadowed forearm or
a sock mistaken for a leg. Pixels are converted to CIELAB and reduced with
a **trimmed median** (middle 60%), which discards specular highlights and
deep shadow without being thrown by either.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("APPAREL_DATASET_ROOT", str(REPO_ROOT / "apparel_dataset"))
OUTFIT_METADATA = REPO_ROOT / "outfit_dataset" / "metadata.json"

# ATR-18 indices for skin. Hair (2) is deliberately excluded, and so are
# Sunglasses (3) which sit on the face.
SKIN_CLASSES = {11: ("face", 3.0), 14: ("arm", 1.0), 15: ("arm", 1.0),
                12: ("leg", 0.7), 13: ("leg", 0.7)}

# Monk Skin Tone scale, the published 10 swatches.
MONK = [
    (1, "#f6ede4"), (2, "#f3e7db"), (3, "#f7ead0"), (4, "#eadaba"), (5, "#d7bd96"),
    (6, "#a07e56"), (7, "#825c43"), (8, "#604134"), (9, "#3a312a"), (10, "#292420"),
]

MIN_SKIN_PIXELS = 500


def srgb_to_lab(rgb):
    import numpy as np

    srgb = np.asarray(rgb, dtype=float) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array([[0.4124, 0.3576, 0.1805],
                       [0.2126, 0.7152, 0.0722],
                       [0.0193, 0.1192, 0.9505]])
    xyz = linear @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883])
    scaled = xyz / white
    epsilon, kappa = 216 / 24389, 24389 / 27
    f = np.where(scaled > epsilon, np.cbrt(scaled), (kappa * scaled + 16) / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def monk_lab():
    import numpy as np

    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for _, h in MONK], dtype=float)
    return srgb_to_lab(rgb)


def estimate(image, label_map, probs, monk_reference):
    """-> dict or None. `label_map` is HxW ATR ids, `probs` CxHxW posteriors."""
    import numpy as np

    pixels = np.asarray(image, dtype=float)
    height, width = label_map.shape
    if pixels.shape[:2] != (height, width):
        from PIL import Image as PILImage

        pixels = np.asarray(PILImage.fromarray(pixels.astype("uint8"))
                            .resize((width, height)), dtype=float)

    samples, weights, posteriors = [], [], []
    for class_id, (_, weight) in SKIN_CLASSES.items():
        mask = label_map == class_id
        count = int(mask.sum())
        if count < 50:
            continue
        samples.append(pixels[mask])
        weights.append(np.full(count, weight))
        posteriors.append(probs[class_id][mask])
    if not samples:
        return None

    values = np.concatenate(samples, axis=0)
    weight_vector = np.concatenate(weights, axis=0)
    posterior = np.concatenate(posteriors, axis=0)
    if len(values) < MIN_SKIN_PIXELS:
        return None

    lab = srgb_to_lab(values)

    # Drop pixels that cannot be skin regardless of how dark the photo is.
    # Skin is always warm -- positive a* (red) and positive b* (yellow) --
    # while shadow and dark clothing are near-neutral. Without this gate,
    # photos came back with a*=1.1, b*=-0.3, which is grey: the parser had
    # labelled trousers or shadow as "leg" and those pixels dominated the
    # median. Threshold is deliberately loose; it is a plausibility gate,
    # not a skin classifier.
    plausible = (lab[:, 1] > 4.0) & (lab[:, 2] > 4.0)
    if plausible.sum() < MIN_SKIN_PIXELS:
        return None
    lab = lab[plausible]
    posterior = posterior[plausible]
    weight_vector = weight_vector[plausible]

    # Trimmed median on lightness: drops specular highlights and the
    # darkest shadow, which are the two things that most distort a mean.
    order = np.argsort(lab[:, 0])
    low, high = int(len(order) * 0.20), int(len(order) * 0.80)
    keep = order[low:high] if high > low else order
    reduced = np.median(lab[keep], axis=0)

    # Match on HUE and CHROMA, with lightness down-weighted 4x.
    #
    # Absolute lightness cannot carry this. The Monk swatches sit at
    # L* 78-94 for tones 1-5, while real photographed skin in this corpus
    # measures L* 11-61 -- so nearest-swatch in full CIELAB could never
    # return a tone below 6, and the first 12 photos all landed in 6-10.
    # That is an exposure and lighting artifact, not a property of the
    # people. Down-weighting L* keeps the axis that survives bad lighting
    # (skin hue) and discounts the one that does not.
    scale = np.array([0.25, 1.0, 1.0])
    distances = np.linalg.norm((monk_reference - reduced) * scale, axis=1)
    tone = int(MONK[int(distances.argmin())][0])

    coverage = len(values) / float(height * width)
    confidence = float(min(1.0, coverage * 8) * np.average(posterior, weights=weight_vector))
    return {
        # Reported, but see the module docstring: absolute binning is NOT
        # reliable here and should not be the primary interface. The LAB
        # value is the trustworthy output; the bin is a convenience label.
        "monk_tone": tone,
        "monk_reliable": False,
        "lab": [round(float(v), 2) for v in reduced],
        "skin_pixels": int(len(values)),
        "coverage": round(float(coverage), 4),
        "confidence": round(confidence, 3),
        "method": ("ATR face/arm/leg pixels, trimmed-median CIELAB, nearest Monk "
                   "swatch. NO white-balance correction -- lighting is the "
                   "dominant error and this is an estimate of the PHOTO, not "
                   "of the person."),
    }


def run(args):
    import numpy as np
    from PIL import Image, ImageOps

    import garment_proposer
    from dataset_utils import load_outfit_records, outfit_key, save_outfit_records_safe

    records = load_outfit_records()
    todo = []
    for record in records:
        if record.get("skin_tone") and not args.force:
            continue
        images = [p for p in (record.get("images") or [])[:1] if (REPO_ROOT / p).exists()]
        if images:
            todo.append((record, images[0]))
    todo = todo[:args.limit] if args.limit else todo
    print(f"  {len(todo):,} photos to estimate")
    if not todo:
        return

    print(f"  loading parser on {args.device}…")
    processor, model = garment_proposer.load_human_parser(args.device)
    reference = monk_lab()

    touched, done, skipped = {}, 0, 0
    started = last_save = time.time()
    for index, (record, rel) in enumerate(todo, 1):
        try:
            with Image.open(REPO_ROOT / rel) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
            label_map, probs = garment_proposer.parse_person(
                image, processor, model, args.device)
            result = estimate(image, np.asarray(label_map), np.asarray(probs), reference)
        except Exception as error:  # noqa: BLE001
            skipped += 1
            if skipped <= 3:
                print(f"  [warn] {rel}: {error}")
            continue

        if result is None:
            # Recorded explicitly: "we looked and found too little skin" is
            # different from "not processed yet", and only the first should
            # stop a re-run from trying again.
            result = {"monk_tone": None, "confidence": 0.0,
                      "method": "insufficient skin pixels"}
        record["skin_tone"] = result
        touched[outfit_key(record)] = record
        done += 1

        if time.time() - last_save > args.checkpoint_seconds:
            save_outfit_records_safe(touched)
            touched, last_save = {}, time.time()
            rate = index / (time.time() - started)
            print(f"  {index:6,}/{len(todo):,}  {rate:.2f}/s  "
                  f"~{(len(todo)-index)/rate/60:.0f} min left")
    if touched:
        save_outfit_records_safe(touched)
    print(f"\n  estimated {done:,} ({skipped} failed) in "
          f"{(time.time()-started)/60:.1f} min")


def report(args):
    import collections

    records = json.loads(OUTFIT_METADATA.read_text())
    have = [r for r in records if r.get("skin_tone")]
    tones = collections.Counter(
        (r["skin_tone"] or {}).get("monk_tone") for r in have)
    usable = [r for r in have
              if (r["skin_tone"] or {}).get("monk_tone")
              and r["skin_tone"].get("confidence", 0) >= 0.15]

    print(f"\n  {len(have):,} of {len(records):,} photos estimated")
    print(f"  {len(usable):,} with a tone and confidence >= 0.15\n")
    print(f"  {'Monk tone':>10}  {'n':>6}")
    for tone in range(1, 11):
        print(f"  {tone:>10}  {tones.get(tone, 0):6,}")
    print(f"  {'no estimate':>10}  {tones.get(None, 0):6,}")
    print("\n  Lighting is uncorrected and is the dominant error. This is an\n"
          "  estimate of how skin appears in a PHOTOGRAPH, not an attribute of\n"
          "  a person, and nothing here is validated against a labelled set.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true", help="re-estimate existing")
    ap.add_argument("--checkpoint-seconds", type=float, default=120.0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report(args) if args.report else run(args)


if __name__ == "__main__":
    main()
