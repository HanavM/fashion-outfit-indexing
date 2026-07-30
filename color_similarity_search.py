"""Query side of the perceptual color pipeline (see build_color_index.py's
docstring for the full design rationale). Given a photo -- a product photo
or a real photo of someone wearing something -- extracts its dominant
color the same way build_color_index.py does, then ranks the catalog by
LAB distance.

Two-tier distance, per the validated research (see build_color_index.py):
LAB Euclidean (CIE76) for the initial full-catalog ranking (cheap), then
CIEDE2000 for re-ranking just the top candidates (more accurate,
especially in blue hues, but too expensive to run against every product on
every query). Imports directly from build_color_index.py rather than
duplicating the color-space math, unlike the Colab training scripts'
duplicate-everything convention -- these two scripts are a build/query
pair meant to run together in the same environment, not standalone
Colab-cell drops, so keeping the LAB conversion in one place (and
therefore guaranteed consistent between build and query) matters more here
than standalone-portability does.

Usage:
    python3 color_similarity_search.py --image path/to/query.jpg --top-k 10
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from build_color_index import (
    DATASET_ROOT,
    METADATA_PATH,
    SIMILARITY_INDEX_PATH,
    extract_dominant_colors,
    resolve_image_path,
)

CIEDE2000_RERANK_POOL = 50  # how many CIE76-ranked candidates get the more expensive CIEDE2000 pass


def ciede2000(lab1, lab2):
    """Standard CIEDE2000 implementation (Sharma et al. 2005 reference
    formula). Only run on a short candidate list -- see module docstring."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    C_bar = (C1 + C2) / 2

    G = 0.5 * (1 - math.sqrt(C_bar ** 7 / (C_bar ** 7 + 25 ** 7))) if C_bar > 0 else 0
    a1p, a2p = a1 * (1 + G), a2 * (1 + G)
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)

    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0

    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        diff = h2p - h1p
        if abs(diff) <= 180:
            dhp = diff
        elif diff > 180:
            dhp = diff - 360
        else:
            dhp = diff + 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2)

    Lp_bar = (L1 + L2) / 2
    Cp_bar = (C1p + C2p) / 2
    if C1p * C2p == 0:
        hp_bar = h1p + h2p
    else:
        diff = abs(h1p - h2p)
        if diff <= 180:
            hp_bar = (h1p + h2p) / 2
        elif (h1p + h2p) < 360:
            hp_bar = (h1p + h2p + 360) / 2
        else:
            hp_bar = (h1p + h2p - 360) / 2

    T = (1 - 0.17 * math.cos(math.radians(hp_bar - 30))
         + 0.24 * math.cos(math.radians(2 * hp_bar))
         + 0.32 * math.cos(math.radians(3 * hp_bar + 6))
         - 0.20 * math.cos(math.radians(4 * hp_bar - 63)))

    d_theta = 30 * math.exp(-(((hp_bar - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cp_bar ** 7 / (Cp_bar ** 7 + 25 ** 7)) if Cp_bar > 0 else 0
    Sl = 1 + (0.015 * (Lp_bar - 50) ** 2) / math.sqrt(20 + (Lp_bar - 50) ** 2)
    Sc = 1 + 0.045 * Cp_bar
    Sh = 1 + 0.015 * Cp_bar * T
    Rt = -math.sin(math.radians(2 * d_theta)) * Rc

    kL = kC = kH = 1.0
    return math.sqrt(
        (dLp / (kL * Sl)) ** 2 + (dCp / (kC * Sc)) ** 2 + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )


def load_catalog_lookup():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    lookup = {}
    for product in metadata:
        code = str(product.get("product_code", "")).strip()
        if code:
            lookup[code] = {"brand": product.get("brand", ""), "name": product.get("name", "")}
    return lookup


def search(image_path, top_k=10):
    if not SIMILARITY_INDEX_PATH.is_file():
        raise FileNotFoundError(f"{SIMILARITY_INDEX_PATH} not found -- run build_color_index.py first.")
    similarity_index = json.loads(SIMILARITY_INDEX_PATH.read_text(encoding="utf-8"))
    catalog_lookup = load_catalog_lookup()

    clusters = extract_dominant_colors([Path(image_path)], has_crop=False)
    if not clusters:
        raise RuntimeError(f"Could not extract a dominant color from {image_path}.")
    query_lab = np.array(clusters[0]["lab"])

    codes = list(similarity_index.keys())
    catalog_lab = np.array([similarity_index[c]["lab"] for c in codes])

    cie76_distances = np.linalg.norm(catalog_lab - query_lab[None, :], axis=1)
    order = np.argsort(cie76_distances)[:CIEDE2000_RERANK_POOL]

    reranked = []
    for index in order:
        code = codes[index]
        distance = ciede2000(query_lab, catalog_lab[index])
        reranked.append((code, distance))
    reranked.sort(key=lambda item: item[1])

    results = []
    for rank, (code, distance) in enumerate(reranked[:top_k], start=1):
        entry = catalog_lookup.get(code, {})
        results.append({
            "rank": rank, "product_code": code, "brand": entry.get("brand", ""), "name": entry.get("name", ""),
            "hex": similarity_index[code]["hex"], "canonical_color": similarity_index[code]["canonical"],
            "ciede2000_distance": round(distance, 2),
        })
    return {"query_hex": clusters[0].get("lab"), "query_canonical": None, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    output = search(args.image, top_k=args.top_k)
    print(f"\nQuery dominant color (LAB): {output['query_hex']}")
    for entry in output["results"]:
        print(f"  #{entry['rank']}  {entry['brand']} {entry['name']}  [{entry['product_code']}]  "
              f"{entry['hex']} ({entry['canonical_color']})  ΔE2000={entry['ciede2000_distance']}")
