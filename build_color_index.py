"""Perceptual color index: pixel-level dominant color extraction + a
continuous LAB-space similarity index, deliberately separate from the
SigLIP2 color facet (which the by-facet eval found mediocre -- 31.57%
R@1, docs/eval_log.md 2026-07-29). Built for two concrete future product
features per user direction: (1) a "filter by color" facet that has to be
reliable, and (2) "show me clothing in this exact color" from a photo --
neither is well served by a text-embedding model, which is the wrong tool
for precise, continuous perceptual color matching. This is a deterministic
feature-extraction + distance pipeline instead, no model training involved.

Design validated against real published practice before building (not just
asserted): fashion color-extraction papers/blog posts consistently use
segment -> CIELAB -> k-means -> filter-by-area/contrast -> dedupe via
CIEDE2000 (e.g. the "Color Feature Based Dominant Color Extraction" IEEE
paper's pipeline, and a Medium writeup implementing the same shape:
Faster R-CNN + Graph Cut segmentation, CIELAB k-means with a bilateral
pre-filter, colors ranked by area/contrast/saturation, near-duplicates
merged via CIEDE2000). CIELAB is used specifically because it's
perceptually uniform -- equal numeric distance ~= equal perceived
difference, which plain RGB Euclidean distance does not give you (it
overstates differences where human vision is dull and understates them
where vision is sharp). Practical recommendation from that same research:
LAB Euclidean (CIE76) is cheap and good enough for bulk similarity
ranking/filtering; CIEDE2000 is reserved for higher-precision final
re-ranking of a short candidate list, since it's much more expensive to
compute and its extra accuracy mostly matters in specific hue regions
(notably blues).

Pipeline, per product:
1. Prefer SAM2 crops (`cropped_images`, currently Nike-only -- 552/1234
   products, see segment_apparel.py) to isolate the garment from
   background/skin before extracting color. For products without a crop
   (the other 682), fall back to a border-margin heuristic: discard the
   outer ~15% ring of the image, since these are consistent studio product
   photos where the garment is roughly centered and the border reliably
   contains background continuation. This is a real, known limitation
   (not silently papered over) -- flagged in the coverage report this
   script prints, same convention as build_hierarchy.py's unmapped report.
2. Downsample to a fixed small size (128x128) for k-means speed, convert
   sRGB -> CIELAB via the standard D65 formulas (implemented directly in
   numpy -- no scikit-image/colormath dependency, so this runs unmodified
   on Colab, Modal, and a bare local venv alike).
3. K-means cluster pixel LAB values into K=5 candidates (3-6 is the
   typical dominant-color count found in the fashion color-extraction
   literature), a small pure-numpy implementation for portability.
4. Filter tiny clusters (<3% of pixels -- almost certainly residual
   background/noise, not a real garment color), rank the rest by pixel
   area (dominant = largest surviving cluster).
5. Map each surviving color to the nearest of 21 canonical colors (the
   same vocabulary build_color_hierarchy.py already canonicalized the
   scraped text `color` attribute into) via LAB Euclidean distance against
   a reference swatch table -- this is what makes "filter by color"
   reliable: it's the *same* canonical bucket the text pipeline uses, just
   assigned from actual pixels instead of brand marketing copy.

Writes:
  - structured_caption.attributes.pixel_color on every product (new field,
    non-destructive): {"primary": {hex, lab, canonical, area_fraction},
    "secondary": [...]}
  - docs/color_similarity_index.json: product_code -> primary LAB vector +
    hex, the flat index color_similarity_search.py's query function loads
    for the "find similar colors" feature.

Usage:
    python3 build_color_index.py
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Needs real full-size images (unlike build_hierarchy.py/build_color_hierarchy.py,
# which only touch metadata.json text), so it defaults to the Colab Drive
# layout like the model-training scripts, not a bare repo-relative path.
# APPAREL_DATASET_ROOT overrides for local test runs (e.g. against a full
# local rsync copy) without touching the Colab default.
DATASET_ROOT = Path(os.environ.get("APPAREL_DATASET_ROOT", "/content/drive/MyDrive/apparel_dataset"))
METADATA_PATH = DATASET_ROOT / "metadata.json"
SIMILARITY_INDEX_PATH = Path(__file__).parent / "docs" / "color_similarity_index.json"

DOWNSAMPLE_SIZE = 128           # k-means operates on a 128x128 pixel grid
# LAB distance (CIE76) beyond which a pixel counts as "not background",
# for products with no SAM2 crop. ~10-15 is a reasonable "clearly a
# different color" threshold in LAB (a delta-E of 2-3 is a just-noticeable
# difference; product photography lighting/shadow gradients on a seamless
# backdrop commonly drift a few units on their own, so this needs to clear
# that noise floor without being so loose it eats real garment color).
BACKGROUND_DISTANCE_THRESHOLD = 12.0
MIN_FOREGROUND_FRACTION = 0.05  # below this, trust unmasked pixels instead (see load_pixels_lab)
NUM_CLUSTERS = 5
MIN_CLUSTER_AREA_FRACTION = 0.03
MAX_SECONDARY_COLORS = 2        # kept in addition to the primary
IMAGES_PER_PRODUCT = 2          # representative images averaged per product
KMEANS_ITERATIONS = 15
KMEANS_SEED = 42

# Representative sRGB swatches for the 21 canonical colors from
# docs/color_hierarchy.json -- chosen as reasonably central/typical
# examples of each family, not tuned per-dataset.
CANONICAL_COLOR_SWATCHES_HEX = {
    "black": "#1a1a1a", "cream": "#f2ead8", "gray": "#8a8a8a", "silver": "#c4c4c4",
    "navy": "#1b2a4a", "blue": "#2a5bd6", "teal": "#157a6e", "green": "#2e8b45",
    "olive": "#6b6b28", "yellow": "#e8d426", "gold": "#c9a227", "orange": "#e07b1f",
    "red": "#c8272a", "maroon": "#7a1f2b", "pink": "#e79ac8", "purple": "#7a3fa0",
    "brown": "#6b4226", "tan": "#c9a06a", "khaki": "#a99461", "gum": "#b98a5a",
    "multi": None,  # never matched against -- "multi" isn't a single point in color space
}


# ============================================================
# Color space conversion (sRGB -> CIELAB, D65 illuminant, standard formulas)
# ============================================================

def srgb_to_linear(channel):
    channel = channel / 255.0
    return np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)


def rgb_array_to_lab(rgb):
    """rgb: (..., 3) uint8 or float array in [0, 255]. Returns (..., 3) LAB."""
    linear = srgb_to_linear(rgb.astype(np.float64))
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # D65 reference white
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x, y, z = x / xn, y / yn, z / zn

    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ = 200 * (fy - fz)
    return np.stack([L, a, b_], axis=-1)


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip("#")
    return np.array([int(hex_code[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


CANONICAL_LAB = {
    name: rgb_array_to_lab(hex_to_rgb(hex_code))
    for name, hex_code in CANONICAL_COLOR_SWATCHES_HEX.items() if hex_code is not None
}


def nearest_canonical_color(lab):
    best_name, best_distance = None, float("inf")
    for name, canonical_lab in CANONICAL_LAB.items():
        distance = float(np.linalg.norm(lab - canonical_lab))
        if distance < best_distance:
            best_name, best_distance = name, distance
    return best_name, best_distance


def lab_to_hex(lab):
    """Approximate round-trip LAB -> sRGB, for display purposes only (not
    used in any distance computation, which always stays in LAB)."""
    L, a, b_ = lab
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b_ / 200

    def finv(t):
        delta = 6.0 / 29.0
        return np.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4.0 / 29.0))

    xn, yn, zn = 0.95047, 1.0, 1.08883
    x, y, z = finv(fx) * xn, finv(fy) * yn, finv(fz) * zn

    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    def to_srgb(channel):
        channel = np.clip(channel, 0, 1)
        return np.where(channel <= 0.0031308, channel * 12.92, 1.055 * channel ** (1 / 2.4) - 0.055)

    rgb = np.clip(np.round(to_srgb(np.array([r, g, b])) * 255), 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ============================================================
# Pure-numpy k-means on LAB pixel values (no sklearn dependency, so this
# runs unmodified on Colab/Modal/local without any pip install changes)
# ============================================================

def kmeans_lab(pixels_lab, k, iterations, seed):
    rng = np.random.default_rng(seed)
    n = pixels_lab.shape[0]
    if n <= k:
        return pixels_lab.copy(), np.arange(n)

    centroid_indices = rng.choice(n, size=k, replace=False)
    centroids = pixels_lab[centroid_indices].copy()

    for _ in range(iterations):
        distances = np.linalg.norm(pixels_lab[:, None, :] - centroids[None, :, :], axis=-1)
        assignments = distances.argmin(axis=1)
        new_centroids = centroids.copy()
        for cluster_index in range(k):
            members = pixels_lab[assignments == cluster_index]
            if len(members) > 0:
                new_centroids[cluster_index] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids, atol=1e-3):
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids, assignments


def resolve_image_path(raw_path):
    raw_path = Path(raw_path)
    candidates = [raw_path, DATASET_ROOT / raw_path, DATASET_ROOT.parent / raw_path]
    if raw_path.parts and raw_path.parts[0] == DATASET_ROOT.name:
        candidates.append(DATASET_ROOT.joinpath(*raw_path.parts[1:]))
    if len(raw_path.parts) >= 4:
        candidates.append(DATASET_ROOT.joinpath(*raw_path.parts[-4:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def estimate_background_lab(lab_grid):
    """Median color of the full border ring (all four edges, not just a
    fixed-fraction crop) -- these catalog photos use a consistent
    near-uniform seamless backdrop, so the border ring's median is a good
    estimate of the background color regardless of how much of the frame
    it actually occupies (validated: a first version that just trimmed a
    fixed 15% border still left the background dominating the k-means
    clusters for the vast majority of uncropped products, since real
    padding varies a lot more than a fixed fraction assumes)."""
    border = np.concatenate([lab_grid[0, :, :], lab_grid[-1, :, :], lab_grid[:, 0, :], lab_grid[:, -1, :]], axis=0)
    return np.median(border, axis=0)


def load_pixels_lab(image_path, has_crop):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE), Image.BILINEAR)
    rgb = np.asarray(image, dtype=np.float64)
    lab_grid = rgb_array_to_lab(rgb)

    if not has_crop:
        # No SAM2 crop available -- estimate the studio background color
        # from the border ring, then mask out every pixel in the WHOLE
        # image close to it (not just a border crop, since background
        # padding varies a lot and can extend well past any fixed margin).
        background_lab = estimate_background_lab(lab_grid)
        distance_from_background = np.linalg.norm(lab_grid - background_lab[None, None, :], axis=-1)
        foreground_mask = distance_from_background > BACKGROUND_DISTANCE_THRESHOLD
        # Safety fallback: if almost everything got masked out (garment is
        # itself near the background color -- true white-on-white product
        # shots exist in this catalog), trust the unmasked pixels instead
        # of returning near-nothing; clustering on the full image still
        # correctly reports "white" in that case, which is the right answer.
        if foreground_mask.mean() < MIN_FOREGROUND_FRACTION:
            return lab_grid.reshape(-1, 3)
        return lab_grid[foreground_mask]

    return lab_grid.reshape(-1, 3)


def extract_dominant_colors(image_paths, has_crop):
    all_pixels = []
    for path in image_paths[:IMAGES_PER_PRODUCT]:
        try:
            all_pixels.append(load_pixels_lab(path, has_crop))
        except Exception:
            continue
    if not all_pixels:
        return None
    pixels_lab = np.concatenate(all_pixels, axis=0)

    centroids, assignments = kmeans_lab(pixels_lab, NUM_CLUSTERS, KMEANS_ITERATIONS, KMEANS_SEED)

    total = len(assignments)
    clusters = []
    for cluster_index in range(len(centroids)):
        count = int((assignments == cluster_index).sum())
        area_fraction = count / total if total else 0.0
        if area_fraction < MIN_CLUSTER_AREA_FRACTION:
            continue
        clusters.append({"lab": centroids[cluster_index], "area_fraction": area_fraction})

    if not clusters:
        return None
    clusters.sort(key=lambda c: -c["area_fraction"])
    return clusters


def describe_color(lab, area_fraction):
    canonical, distance = nearest_canonical_color(lab)
    return {
        "hex": lab_to_hex(lab),
        "lab": [round(float(v), 2) for v in lab],
        "canonical": canonical,
        "canonical_distance": round(distance, 2),
        "area_fraction": round(area_fraction, 3),
    }


def main():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    similarity_index = {}
    products_with_crops = 0
    products_without_crops = 0
    products_skipped = 0

    for product in metadata:
        product_code = str(product.get("product_code", "")).strip()
        if not product_code:
            continue

        cropped_images = product.get("cropped_images") or []
        raw_images = product.get("images") or []
        use_crops = bool(cropped_images)
        source_images = cropped_images if use_crops else raw_images

        resolved_paths = [p for p in (resolve_image_path(r) for r in source_images) if p is not None]
        if not resolved_paths:
            products_skipped += 1
            continue

        clusters = extract_dominant_colors(resolved_paths, has_crop=use_crops)
        if not clusters:
            products_skipped += 1
            continue

        if use_crops:
            products_with_crops += 1
        else:
            products_without_crops += 1

        primary = describe_color(clusters[0]["lab"], clusters[0]["area_fraction"])
        secondary = [describe_color(c["lab"], c["area_fraction"]) for c in clusters[1:1 + MAX_SECONDARY_COLORS]]

        structured = product.setdefault("structured_caption", {})
        attributes = structured.setdefault("attributes", {})
        attributes["pixel_color"] = {
            "primary": primary, "secondary": secondary,
            "source": "sam2_crop" if use_crops else "border_margin_heuristic",
        }

        similarity_index[product_code] = {
            "lab": primary["lab"], "hex": primary["hex"], "canonical": primary["canonical"],
        }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    SIMILARITY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIMILARITY_INDEX_PATH.write_text(json.dumps(similarity_index, indent=2), encoding="utf-8")

    total = products_with_crops + products_without_crops
    print(f"Extracted pixel colors for {total} products "
          f"({products_with_crops} via SAM2 crop, {products_without_crops} via border-margin fallback)")
    print(f"Skipped (no resolvable/decodable images): {products_skipped}")
    print(f"Wrote {SIMILARITY_INDEX_PATH} ({len(similarity_index)} entries)")


if __name__ == "__main__":
    main()
