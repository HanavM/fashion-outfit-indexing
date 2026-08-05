"""
logo_detector.py -- brand recognition that does NOT depend on reading text.

WHY THIS EXISTS
---------------
Spec §4.5 lists "Visible logo" and "OCR" as two *separate* brand-evidence
sources. Only OCR was ever built (`brand_evidence.py`), and two independent
measurements on 2026-08-04 showed exactly where it stops:

    brand-as-ranking   : reads a brand on 11.13% of catalog photos,
                         51.01% of photos contain no legible text at all.
    brand-as-open-set  : 100% precision, 9.17% recall.

Both split the same way, and the split is not photo quality -- it is
**wordmark vs logo**: adidas 40% recall (prints its name in a legible
sans-serif), carhartt/dickies ~17-21% (wordmark labels), levis 5% (a small
red tab), champion 2.5% (an embroidered "C"), stussy 0.0% (handwritten
script). OCR cannot read those *by construction*; no threshold or engine
swap fixes it. What they need is a classifier over the visual mark.

APPROACH, AND WHY
-----------------
Frozen backbone + linear probe, trained on the catalog's own free
supervision (every product record carries a ground-truth `brand`).

  * Frozen, not fine-tuned. Both SigLIP2-base-384 and DINOv3-ViT-B/16 are
    already in this repo's HF cache -- no new download, no GPU rental, and
    a linear probe on frozen features is the standard way to ask "is this
    information already in the representation?" before spending anything on
    training. The 2026-08-04 attribute-head row established the same
    protocol for attributes and it answered the question honestly.

  * Multi-crop, because the mark is small. A logo is often <2% of the
    frame; a whole-image 384px embedding averages it away. So each image is
    also embedded as a set of crops (centre + quadrants), the probe is
    trained on crop-level features with the image's brand label (weak /
    multiple-instance supervision), and at inference the crop scores are
    pooled. `--pool max` is the MIL-style read ("some region says adidas"),
    `--pool mean` is the whole-garment read. Both are reported; the
    `--crops full` arm is the honest control that says whether the crops
    bought anything at all.

  * Held out BY PRODUCT IDENTITY, never by image. Two images of one product
    on opposite sides of the split measures memorisation, not recognition.
    The group key is (brand, slug) so colourway siblings -- which are the
    *same* garment in a different colour -- travel together.

THE CONFOUND, STATED UP FRONT
-----------------------------
A brand classifier trained on catalog photography can score very well for
entirely the wrong reason: brands shoot on different backgrounds, at
different crops, with different lighting and model styling. That is photo
*style*, not a logo, and it transfers nothing to a user's phone photo.
`--eval-cropped` tests exactly this: it re-scores the held-out products
using the background-removed garment crops in `cropped_images`. A probe
that is really reading marks should survive; a probe reading studio
background should collapse. Read the two numbers together or not at all.

WHAT THE CONTROLS ACTUALLY SAID (2026-08-05, eval_log)
------------------------------------------------------
The confound won. Held-out accuracy is 99.08% and macro per-brand recall
98.44% against OCR's 11.07% -- but the full-image arm BEATS the crops
(99.08% > 98.34%), accuracy at 32x32 where no mark is legible is still
83.95%, and on 300 real outfit photos confidence falls 0.852 -> 0.406,
with confidence alone separating real-photo from catalog-photo at AUROC
0.967. **This is a brand-photography-style classifier, not a logo
detector.** It is a good free brand labeller for catalog-side imagery and
a bad query-side signal, and it is deliberately not wired into retrieval.

Image-level brand labels turned out to be solvable without ever looking
at the mark, so the optimiser never looked. A real logo detector needs
mark-level supervision -- boxes on logos, or a mark-crop dataset.

USAGE
-----
    .venv/bin/python logo_detector.py --embed          # cache features
    .venv/bin/python logo_detector.py --evaluate       # 12-brand P/R + confusion
    .venv/bin/python logo_detector.py --eval-cropped   # background-removal control
    .venv/bin/python logo_detector.py --evaluate --crops full --degrade 32 --pool full
                                                       # logo-legibility control
    .venv/bin/python logo_detector.py --nn-baseline    # brand already in the embedding?
    .venv/bin/python logo_detector.py --domain-check   # does it survive real photos?
    .venv/bin/python logo_detector.py --open-set       # 6-in / 6-out rejection

Each control needs its own cached feature set (the cache tag encodes
--crops / --degrade / cropped-vs-as-shot), so run the matching --embed
first or let the eval build it.

Nothing here writes to apparel_dataset/ -- it is read-only over the catalog,
and every cached feature goes to --cache-dir (default: a scratch path).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DB_FILE = Path("apparel_dataset/metadata.json")

# Image paths in metadata.json were written by scrapers that ran from
# different working roots over the project's life. Two conventions exist and
# both are live: the path as stored, and the same path with its first
# component replaced by apparel_dataset_full/ (the 8GB full-resolution tree
# the shoe brands live in). Resolve, never rewrite -- metadata.json is
# read-only here.
FULL_TREE = Path("apparel_dataset_full")

# The six brands the deployed 6-brand service stocks, per the 2026-08-04
# open-set row. Kept identical so the numbers here are directly comparable
# to that row's 100%-precision / 9.17%-recall OCR result.
IN_CATALOG_BRANDS = ["adidas", "gap", "newbalance", "nike", "pacsun", "skechers"]

# Per-brand OCR recall from the 2026-08-04 brand_evidence_eval row
# (1,186 images, 50 products/brand x 2 images). This is the number the logo
# detector has to beat, brand by brand -- the whole point is covering the
# brands OCR cannot read.
OCR_RECALL = {
    "adidas": 0.400, "carhartt": 0.210, "skechers": 0.160, "levis": 0.140,
    "dickies": 0.120, "vans": 0.091, "nike": 0.071, "gap": 0.067,
    "champion": 0.040, "pacsun": 0.020, "newbalance": 0.010, "stussy": 0.000,
}

SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-384"
DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


# --------------------------------------------------------------------------
# Catalog sampling
# --------------------------------------------------------------------------

def resolve_image(path_str: str) -> Path | None:
    p = Path(path_str)
    if p.exists():
        return p
    parts = p.parts
    if len(parts) > 1:
        alt = FULL_TREE.joinpath(*parts[1:])
        if alt.exists():
            return alt
    return None


def identity_key(record: dict) -> str:
    """Group key for the held-out split.

    slug is the garment; product_code is the colourway. Splitting on
    product_code would put "Gazelle Bold / white" in train and
    "Gazelle Bold / black" in test, which is the same garment photographed
    twice and would inflate every number below.
    """
    return f"{record['brand']}::{record.get('slug') or record['product_code']}"


def sample_dataset(records, products_per_brand, images_per_product, seed, use_cropped=False):
    """-> list of (brand, identity, image_path). Deterministic in `seed`."""
    by_identity = defaultdict(list)
    for r in records:
        by_identity[identity_key(r)].append(r)

    by_brand = defaultdict(list)
    for ident, recs in by_identity.items():
        by_brand[recs[0]["brand"]].append(ident)

    rng = random.Random(seed)
    rows = []
    for brand in sorted(by_brand):
        idents = sorted(by_brand[brand])
        rng.shuffle(idents)
        taken = 0
        for ident in idents:
            if taken >= products_per_brand:
                break
            paths = []
            for r in by_identity[ident]:
                field = r.get("cropped_images") if use_cropped else r.get("images")
                for raw in (field or []):
                    # `cropped_images` interleaves real crops (*_cropped.jpg)
                    # with the ORIGINAL file wherever segmentation declined to
                    # crop. Keeping those would silently put as-shot photos
                    # back into the control that exists to remove them.
                    if use_cropped and "_cropped" not in Path(raw).name:
                        continue
                    resolved = resolve_image(raw)
                    if resolved is not None:
                        paths.append(resolved)
            if not paths:
                continue
            rng.shuffle(paths)
            for p in paths[:images_per_product]:
                rows.append((brand, ident, str(p)))
            taken += 1
    return rows


def split_identities(identities, test_fraction):
    """Deterministic hash split, so train/test membership does not move when
    the sample size changes."""
    train, test = set(), set()
    for ident in identities:
        h = int(hashlib.sha1(ident.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (test if h < test_fraction else train).add(ident)
    return train, test


# --------------------------------------------------------------------------
# Crops
# --------------------------------------------------------------------------

def crop_boxes(mode: str):
    """Normalised (l, t, r, b) boxes. The whole image is always crop 0 so
    `--pool mean` degrades gracefully to the full-image arm."""
    full = [(0.0, 0.0, 1.0, 1.0)]
    if mode == "full":
        return full
    if mode == "grid":
        return full + [
            (0.0, 0.0, 0.55, 0.55), (0.45, 0.0, 1.0, 0.55),
            (0.0, 0.45, 0.55, 1.0), (0.45, 0.45, 1.0, 1.0),
            (0.25, 0.25, 0.75, 0.75),
        ]
    if mode == "grid9":
        boxes = list(full)
        for i in range(3):
            for j in range(3):
                boxes.append((j / 3 * 0.9, i / 3 * 0.9, j / 3 * 0.9 + 0.4, i / 3 * 0.9 + 0.4))
        return boxes
    raise SystemExit(f"unknown crop mode {mode}")


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

def load_backbone(name: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if name == "siglip2":
        model = AutoModel.from_pretrained(SIGLIP2_MODEL_ID, torch_dtype=torch.float32).to(device).eval()
        processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)

        def encode(pil_batch):
            inputs = processor(images=pil_batch, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
            # transformers has changed this return type across versions --
            # some return the tensor, some a ...WithPooling object.
            if not hasattr(feats, "float"):
                feats = feats.pooler_output
            return feats.float().cpu().numpy()

        return encode, device
    if name == "dinov3":
        token = os.environ.get("HF_TOKEN")
        model = AutoModel.from_pretrained(DINOV3_MODEL_ID, torch_dtype=torch.float32, token=token).to(device).eval()
        processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID, token=token)

        def encode(pil_batch):
            inputs = processor(images=pil_batch, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            return out.pooler_output.float().cpu().numpy()

        return encode, device
    raise SystemExit(f"unknown backbone {name}")


def embed_rows(rows, backbone, crops_mode, batch_size, progress_every=50, degrade=0):
    from PIL import Image

    encode, device = load_backbone(backbone)
    boxes = crop_boxes(crops_mode)
    print(f"backbone={backbone} device={device} crops={len(boxes)} images={len(rows)}", flush=True)

    feats = []
    pending, pending_idx = [], []
    out = [None] * (len(rows) * len(boxes))

    def flush():
        if not pending:
            return
        vecs = encode(pending)
        for slot, vec in zip(pending_idx, vecs):
            out[slot] = vec
        pending.clear()
        pending_idx.clear()

    for i, (_brand, _ident, path) in enumerate(rows):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = None
        w, h = (img.size if img else (1, 1))
        for c, (l, t, r, b) in enumerate(boxes):
            slot = i * len(boxes) + c
            if img is None:
                out[slot] = None
                continue
            piece = img if c == 0 else img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
            if degrade:
                # Destroy every detail smaller than the garment while leaving
                # colour, silhouette, layout and studio style intact. A logo
                # is unreadable at 32px; a brand's photography is not. If
                # accuracy survives this, the probe is not reading marks.
                piece = piece.resize((degrade, degrade)).resize((384, 384))
            pending.append(piece)
            pending_idx.append(slot)
            if len(pending) >= batch_size:
                flush()
        if (i + 1) % progress_every == 0:
            flush()
            print(f"  embedded {i + 1}/{len(rows)} images", flush=True)
    flush()

    dim = next(v.shape[0] for v in out if v is not None)
    keep_rows, matrix = [], []
    for i in range(len(rows)):
        chunk = out[i * len(boxes):(i + 1) * len(boxes)]
        if any(v is None for v in chunk):
            continue
        keep_rows.append(i)
        matrix.append(np.stack(chunk))
    feats = np.stack(matrix).astype(np.float32)  # (n_images, n_crops, dim)
    assert feats.shape[2] == dim
    return keep_rows, feats


def cache_path(cache_dir: Path, tag: str) -> Path:
    return cache_dir / f"{tag}.npz"


def build_or_load_features(args, use_cropped=False, tag_suffix=""):
    records = json.loads(DB_FILE.read_text())
    rows = sample_dataset(records, args.products_per_brand, args.images_per_product,
                          args.seed, use_cropped=use_cropped)
    degrade = getattr(args, "degrade", 0)
    tag = (f"{args.backbone}_{args.crops}_p{args.products_per_brand}"
           f"_i{args.images_per_product}_s{args.seed}"
           f"{f'_d{degrade}' if degrade else ''}{tag_suffix}")
    path = cache_path(args.cache_dir, tag)
    if path.exists() and not args.refresh:
        blob = np.load(path, allow_pickle=True)
        print(f"loaded cached features {path} {blob['feats'].shape}")
        return list(blob["brands"]), list(blob["idents"]), blob["feats"]

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    keep, feats = embed_rows(rows, args.backbone, args.crops, args.batch_size, degrade=degrade)
    brands = [rows[i][0] for i in keep]
    idents = [rows[i][1] for i in keep]
    np.savez_compressed(path, feats=feats, brands=np.array(brands), idents=np.array(idents))
    print(f"cached features -> {path} {feats.shape}")
    return brands, idents, feats


# --------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------

def l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def fit_probe(train_feats, train_labels, C):
    from sklearn.linear_model import LogisticRegression

    n, k, d = train_feats.shape
    X = l2(train_feats.reshape(n * k, d))
    y = np.repeat(train_labels, k)
    clf = LogisticRegression(max_iter=3000, C=C)
    clf.fit(X, y)
    return clf


def image_probs(clf, feats, pool):
    n, k, d = feats.shape
    P = clf.predict_proba(l2(feats.reshape(n * k, d))).reshape(n, k, -1)
    if pool == "mean":
        return P.mean(axis=1)
    if pool == "max":
        M = P.max(axis=1)
        return M / M.sum(axis=1, keepdims=True)
    if pool == "full":
        return P[:, 0, :]
    raise SystemExit(f"unknown pool {pool}")


def per_class_report(y_true, y_pred, classes):
    rows = []
    for i, c in enumerate(classes):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((c, prec, rec, f1, tp + fn))
    return rows


def confusion(y_true, y_pred, classes):
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def print_confusion(m, classes):
    w = max(len(c) for c in classes)
    head = " " * (w + 2) + " ".join(f"{c[:5]:>5}" for c in classes)
    print(head)
    for i, c in enumerate(classes):
        print(f"{c:<{w}}  " + " ".join(f"{m[i, j]:>5}" for j in range(len(classes))))


def auroc(pos_scores, neg_scores):
    """Rank-based AUROC with tie handling. pos = should score HIGH."""
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_evaluate(args, use_cropped_test=False):
    brands, idents, feats = build_or_load_features(args)
    classes = sorted(set(brands))
    cls_index = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_index[b] for b in brands])
    idents = np.array(idents)

    train_ids, test_ids = split_identities(sorted(set(idents)), args.test_fraction)
    tr = np.array([i in train_ids for i in idents])
    te = ~tr
    print(f"\nsplit BY PRODUCT IDENTITY: {len(train_ids)} train identities / "
          f"{len(test_ids)} test identities; {int(tr.sum())} train images / {int(te.sum())} test images")
    assert not (set(idents[tr]) & set(idents[te])), "identity leaked across split"

    clf = fit_probe(feats[tr], y[tr], args.C)

    if use_cropped_test:
        cb, ci, cf = build_or_load_features(args, use_cropped=True, tag_suffix="_cropped")
        ci = np.array(ci)
        mask = np.array([i in test_ids for i in ci])
        if mask.sum() == 0:
            print("no cropped test images available")
            return
        # only classes the probe knows
        keep = mask & np.array([b in cls_index for b in cb])
        eval_feats = cf[keep]
        y_true = np.array([cls_index[b] for b in np.array(cb)[keep]])
        label = "CROPPED (background-removed) test images"
    else:
        eval_feats = feats[te]
        y_true = y[te]
        label = "held-out test images (as-shot catalog photos)"

    print(f"\n=== {label}: n={len(y_true)} ===")
    for pool in ("full", "mean", "max"):
        if args.crops == "full" and pool != "full":
            continue
        probs = image_probs(clf, eval_feats, pool)
        y_pred = probs.argmax(axis=1)
        acc = float((y_pred == y_true).mean())
        print(f"\n-- pool={pool}  accuracy={acc:.2%}  (chance={1 / len(classes):.2%})")
        if pool == args.report_pool or args.crops == "full":
            rows = per_class_report(y_true, y_pred, classes)
            print(f"{'brand':<12} {'prec':>7} {'recall':>7} {'F1':>7} {'n':>5} {'OCR rec':>8} {'delta':>8}")
            for c, p, r, f1, n in rows:
                o = OCR_RECALL.get(c)
                print(f"{c:<12} {p:>7.1%} {r:>7.1%} {f1:>7.1%} {n:>5} "
                      f"{(f'{o:.1%}' if o is not None else '-'):>8} "
                      f"{(f'{100 * (r - o):+.1f}pt' if o is not None else '-'):>8}")
            # Macro over classes that are actually PRESENT. The cropped
            # control only covers 7 brands; averaging a 0% recall over
            # brands with no test images would understate it by ~40pt.
            present = [(c, r) for c, _, r, _, n in rows if n > 0]
            macro_r = np.mean([r for _, r in present])
            print(f"macro recall over the {len(present)} brands present: {macro_r:.2%} "
                  f"vs OCR macro recall on the same brands "
                  f"{np.mean([OCR_RECALL[c] for c, _ in present if c in OCR_RECALL]):.2%}")
            print("\nconfusion matrix (rows = truth, cols = predicted):")
            print_confusion(confusion(y_true, y_pred, classes), classes)

            # High-precision operating point: abstain below a confidence
            # threshold. This is the shape OCR had (100% precision, abstain
            # 88.87%) and the only shape safe to wire into retrieval.
            conf = probs.max(axis=1)
            print("\nabstain sweep (accept only above threshold):")
            print(f"{'thresh':>7} {'fire rate':>10} {'precision':>10} {'recall':>10}")
            for t in (0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99):
                fire = conf >= t
                if fire.sum() == 0:
                    continue
                prec = float((y_pred[fire] == y_true[fire]).mean())
                print(f"{t:>7.2f} {fire.mean():>10.2%} {prec:>10.2%} "
                      f"{float(((y_pred == y_true) & fire).sum() / len(y_true)):>10.2%}")


def cmd_nn_baseline(args):
    """How much brand does the IDENTITY-retrieval path already know?

    This is the control the 2026-08-04 `--brand-boost` row needed and did
    not have. That row's finding was that brand evidence bought +0.10pt
    because "products of one brand look alike, so DINOv3 had already ranked
    a same-brand product first". Here that claim is measured directly:
    take the held-out image, find its nearest neighbour among the training
    images by cosine on the SAME frozen embedding, and ask whether the
    neighbour's brand is right. If that number is already close to the
    probe's accuracy, a brand classifier adds no ranking information no
    matter how good it is -- which is the expected negative result.
    """
    brands, idents, feats = build_or_load_features(args)
    idents = np.array(idents)
    brands_arr = np.array(brands)
    train_ids, test_ids = split_identities(sorted(set(idents)), args.test_fraction)
    tr = np.array([i in train_ids for i in idents])
    te = ~tr

    G = l2(feats[tr][:, 0, :])          # full-image embedding only
    Q = l2(feats[te][:, 0, :])
    sims = Q @ G.T
    nn = sims.argmax(axis=1)
    nn_brand = brands_arr[tr][nn]
    true_brand = brands_arr[te]
    acc = float((nn_brand == true_brand).mean())
    print(f"\nnearest-neighbour brand accuracy on {args.backbone} (identity-style retrieval): {acc:.2%}")
    print(f"{'brand':<12} {'NN brand acc':>13} {'n':>5}")
    for b in sorted(set(true_brand)):
        m = true_brand == b
        print(f"{b:<12} {float((nn_brand[m] == b).mean()):>13.2%} {int(m.sum()):>5}")


def cmd_domain_check(args):
    """Does the probe survive leaving catalog photography?

    Every number in --evaluate and --open-set is measured on catalog
    product shots, which is also what the probe trained on. The deployment
    case is a phone photo of a person wearing the garment. `outfit_dataset`
    is exactly that -- real Reddit/Pinterest outfit photos -- and although
    it carries no brand labels, it does not need them for this question:
    if the probe's confidence on real photos collapses to the same range as
    its confidence on off-catalog *brands*, then what --open-set measured
    is a catalog-photography detector wearing a brand detector's clothes,
    and it would reject a real user's in-catalog garment just as eagerly.
    """
    from PIL import Image

    brands, idents, feats = build_or_load_features(args)
    idents = np.array(idents)
    classes = sorted(set(brands))
    cls_index = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_index[b] for b in brands])
    train_ids, test_ids = split_identities(sorted(set(idents)), args.test_fraction)
    tr = np.array([i in train_ids for i in idents])
    clf = fit_probe(feats[tr], y[tr], args.C)

    catalog_conf = image_probs(clf, feats[~tr], args.report_pool).max(axis=1)

    outfits = json.loads(Path("outfit_dataset/metadata.json").read_text())
    rng = random.Random(args.seed)
    rng.shuffle(outfits)
    paths = []
    for rec in outfits:
        for raw in rec.get("images") or []:
            if Path(raw).exists():
                paths.append(raw)
                break
        if len(paths) >= args.domain_samples:
            break

    encode, _ = load_backbone(args.backbone)
    boxes = crop_boxes(args.crops)
    vecs = []
    for i in range(0, len(paths), 8):
        batch_imgs, owners = [], []
        for p in paths[i:i + 8]:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            w, h = img.size
            for c, (l, t, r, b) in enumerate(boxes):
                piece = img if c == 0 else img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
                if args.degrade:
                    piece = piece.resize((args.degrade, args.degrade)).resize((384, 384))
                batch_imgs.append(piece)
            owners.append(p)
        if batch_imgs:
            vecs.append(encode(batch_imgs).reshape(len(owners), len(boxes), -1))
    real = np.concatenate(vecs).astype(np.float32)
    real_conf = image_probs(clf, real, args.report_pool).max(axis=1)

    print(f"\nreal outfit photos scored: {len(real_conf)}   catalog held-out: {len(catalog_conf)}")
    for name, arr in (("catalog held-out", catalog_conf), ("real outfit photos", real_conf)):
        qs = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"{name:<20} mean {arr.mean():.3f}  median {qs[2]:.3f}  "
              f"p10 {qs[0]:.3f}  p90 {qs[4]:.3f}  frac>0.5 {float((arr > 0.5).mean()):.2%}")
    print(f"AUROC separating real-outfit from catalog by confidence alone: "
          f"{auroc(-real_conf, -catalog_conf):.4f}  "
          f"(1.0 = confidence is purely a 'is this a catalog photo' detector)")


def cmd_open_set(args):
    """Brand mark as OFF-CATALOG evidence.

    The probe is trained on the six brands the deployed service stocks. A
    query whose mark the probe cannot confidently place in those six is
    evidence the product is off-catalog. This is the question the
    2026-08-04 open-set row asked of OCR (100% precision, 9.17% recall) and
    that the DINOv3 distance path answers badly (AUROC 0.769).
    """
    brands, idents, feats = build_or_load_features(args)
    brands_arr = np.array(brands)
    idents = np.array(idents)

    known = [b for b in IN_CATALOG_BRANDS if b in set(brands)]
    cls_index = {c: i for i, c in enumerate(known)}
    is_known = np.array([b in cls_index for b in brands])
    train_ids, test_ids = split_identities(sorted(set(idents)), args.test_fraction)

    tr = is_known & np.array([i in train_ids for i in idents])
    in_te = is_known & np.array([i in test_ids for i in idents])
    off_te = ~is_known

    print(f"\nopen set: known brands {known}")
    print(f"off-catalog brands {sorted(set(brands_arr[off_te]))}")
    print(f"train {int(tr.sum())} imgs / in-catalog test {int(in_te.sum())} / off-catalog {int(off_te.sum())}")

    y = np.array([cls_index.get(b, -1) for b in brands])
    clf = fit_probe(feats[tr], y[tr], args.C)

    for pool in (("full", "mean", "max") if args.crops != "full" else ("full",)):
        p_in = image_probs(clf, feats[in_te], pool).max(axis=1)
        p_off = image_probs(clf, feats[off_te], pool).max(axis=1)
        # score should be HIGH for off-catalog -> use negated confidence
        a = auroc(-p_off, -p_in)
        print(f"\n-- pool={pool}  AUROC (off-catalog detection) = {a:.4f}   "
              f"[DINOv3 distance path: 0.769]")
        print(f"{'reject below':>13} {'false-reject':>13} {'false-accept':>13} {'youden J':>9}")
        best = None
        grid = np.unique(np.concatenate([p_in, p_off]))
        for t in grid:
            fr = float((p_in < t).mean())      # in-catalog wrongly rejected
            fa = float((p_off >= t).mean())    # off-catalog wrongly accepted
            j = (1 - fr) + (1 - fa) - 1
            if best is None or j > best[0]:
                best = (j, t, fr, fa)
        for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, best[1]):
            fr = float((p_in < t).mean())
            fa = float((p_off >= t).mean())
            tag = "  <- best balanced" if abs(t - best[1]) < 1e-9 else ""
            print(f"{t:>13.3f} {fr:>13.2%} {fa:>13.2%} {(1 - fr) + (1 - fa) - 1:>9.3f}{tag}")

        # OCR-comparable operating point: what off-catalog recall do we get
        # at ZERO false rejects (the "never wrong when it fires" regime OCR
        # occupied at 100% precision / 9.17% recall)?
        safe_t = float(p_in.min())
        rec_at_zero_fr = float((p_off < safe_t).mean())
        print(f"at zero false-reject (t={safe_t:.3f}): off-catalog recall "
              f"{rec_at_zero_fr:.2%}   [OCR: 9.17% at 100% precision]")
        # and a 1%-false-reject point, which is what the DINOv3 row quoted
        t1 = float(np.quantile(p_in, 0.01))
        print(f"at 1% false-reject (t={t1:.3f}): off-catalog recall {float((p_off < t1).mean()):.2%}"
              f"   [DINOv3 path: 32% at 1% false-reject]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed", action="store_true", help="build/cache features only")
    ap.add_argument("--evaluate", action="store_true", help="12-brand held-out evaluation")
    ap.add_argument("--eval-cropped", action="store_true",
                    help="style-confound control: test on background-removed garment crops")
    ap.add_argument("--open-set", action="store_true", help="6-in / 6-out rejection evaluation")
    ap.add_argument("--domain-check", action="store_true",
                    help="score real outfit photos (outfit_dataset) with the catalog-trained probe")
    ap.add_argument("--domain-samples", type=int, default=300)
    ap.add_argument("--nn-baseline", action="store_true",
                    help="how much brand the identity-retrieval path already knows (orthogonality control)")
    ap.add_argument("--backbone", default="siglip2", choices=["siglip2", "dinov3"])
    ap.add_argument("--crops", default="grid", choices=["full", "grid", "grid9"])
    ap.add_argument("--pool", dest="report_pool", default="max", choices=["full", "mean", "max"])
    ap.add_argument("--products-per-brand", type=int, default=100)
    ap.add_argument("--images-per-product", type=int, default=3)
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--degrade", type=int, default=0,
                    help="logo-legibility control: downsample each crop to NxN before "
                         "embedding, destroying any mark while keeping colour/silhouette/"
                         "studio style. Accuracy that survives this is not logo reading.")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(os.environ.get("LOGO_CACHE_DIR", "checkpoints/logo_detector_cache")))
    args = ap.parse_args()

    if not any([args.embed, args.evaluate, args.eval_cropped, args.open_set, args.nn_baseline, args.domain_check]):
        ap.error("pick one of --embed / --evaluate / --eval-cropped / --open-set / "
                 "--nn-baseline / --domain-check")

    if args.embed:
        build_or_load_features(args)
    if args.evaluate:
        cmd_evaluate(args)
    if args.eval_cropped:
        cmd_evaluate(args, use_cropped_test=True)
    if args.nn_baseline:
        cmd_nn_baseline(args)
    if args.open_set:
        cmd_open_set(args)
    if args.domain_check:
        cmd_domain_check(args)


if __name__ == "__main__":
    main()
