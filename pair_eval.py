"""The first honest measurement of the condition this product actually ships in.

    python pair_eval.py build --n 60        # crop real photos, query the live API
    python pair_eval.py label               # label them in a browser
    python pair_eval.py score               # the numbers

## Why this exists

**Every accuracy number in this project is catalog-photo → catalog-photo.**
R@1 47.65%, 58.58% at fixed gallery, the +31.3pt identity fine-tune, all of
it: a clean studio photo of a known-category garment, matched against other
clean studio photos. The real condition -- someone points a phone at what
they are wearing -- has never been measured once.

`outfit_dataset` is 6,860 real worn photos, already collected. Run the
garment proposer over a sample, hand the crops to the live `/query`
endpoint, and have a human say whether the answer is right. That is the
whole idea. `docs/production_plan.md` calls it the only item on the list
that can change what we believe about the system.

## What it measures, and the honest limits of each number

**1. Retrieval accuracy on consumer photos (R@1 / R@5 / R@10).**
Computed only over items a human confirmed ARE in the catalog. That
denominator will be small -- most people in these photos are wearing
brands we do not carry -- so the confidence interval will be wide and the
scorer prints it rather than hiding it. A wide interval on the real
condition still beats a tight one on the wrong condition.

**2. Open-set behaviour, on the deployment distribution.** This is the
part we cannot get any other way. `reject_threshold` is uncalibrated
(AUROC 0.769 on the proxy set, no usable operating point: 1% false-reject
costs 68% false-accept), so `rejected_open_set` never fires by default and
the API says so. Every out-of-catalog item labelled here is a real
negative drawn from the actual input distribution, and its top-1 score is
exactly the quantity a threshold would have to separate. Even 40 of these
is the first calibration data of the right kind.

**3. What the failures look like.** Whether wrong answers are near-misses
(right category, wrong product) or nonsense decides what to fix next, and
that is a judgement no metric makes for you.

## Sampling discipline

Uniform over outfit records, seeded, **not** filtered by whether the
pipeline was confident about them. Sampling on confidence would measure
the system on the inputs it already handles, which is how a benchmark
comes out flattering and useless. The proposer's own
`MIN_PARSER_SCORE=0.5` floor is kept because that is what runs in
production -- matching deployment, not curating.

Note what this does NOT establish: the crops come from
`garment_proposer.py`, whose precision was inspected by eye at ~91% on 40
photos, on the same 40 images the threshold was chosen on. So a bad answer
here can mean bad retrieval OR a bad crop, and the labeller is given an
explicit "the crop is not a usable garment" option to separate them.
"""

import argparse
import base64
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "pair_eval"
CROP_DIR = OUT_DIR / "crops"
MANIFEST = OUT_DIR / "manifest.json"
LABELS = OUT_DIR / "labels.json"
OUTFIT_METADATA = REPO_ROOT / "outfit_dataset" / "metadata.json"
CATALOG_METADATA = REPO_ROOT / "apparel_dataset" / "metadata.json"
DEFAULT_URL = "https://hanavm--fashion-serve-fashionservice-api.modal.run"

# Label verdicts. Kept as constants because `score` and the browser UI must
# agree on the exact strings, and a typo would silently drop a whole class
# of item from the denominator.
CORRECT = "correct"            # a shown candidate is the right product
IN_CATALOG_MISSED = "missed"   # right product IS in the catalog, not shown
NOT_IN_CATALOG = "absent"      # the real item is not a product we carry
BAD_CROP = "badcrop"           # the crop is not a usable garment
SKIP = "skip"


def load_api_key():
    key = os.environ.get("FASHION_API_KEY")
    if key:
        return key
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("FASHION_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("FASHION_API_KEY not found in environment or .env")


# ----------------------------------------------------------------------
# build
# ----------------------------------------------------------------------

NON_US_SECTIONS = ("wear.jp", "korean")
WOMENS_SECTIONS = ("femalefashion", "femalefashionadvice", "petitefashionadvice",
                   "womens%20outfit", "womens outfit")


def matchable(record):
    """Can this post's garments plausibly be in the catalog at all?

    The catalog is 20 US brands and 100% men's. Measured 2026-08-06, 22% of
    the corpus is Japan/Korea-sourced and 20% is women's fashion, so ~42%
    could never match whatever the model does.

    Labelling those is what made the first attempt useless -- the owner's
    report was that "the exact thing is never really in the catalog", and
    they were right, because nearly half the sample was structurally
    incapable of matching. Sampling from the matchable half is not
    cherry-picking the easy cases; it is removing items whose answer is
    known in advance and which therefore measure nothing."""
    section = str(record.get("section") or "").lower()
    if record.get("source") == "wear" or any(k in section for k in NON_US_SECTIONS):
        return False
    return not any(k in section for k in WOMENS_SECTIONS)


def sample_photos(count, seed, us_mens_only=True):
    """Seeded uniform sample of outfit photos, one image per post.

    One image per POST, not per image: the same outfit shot from two
    angles is not two independent trials, and counting it twice would
    quietly inflate whatever the sample says."""
    records = json.loads(OUTFIT_METADATA.read_text())
    usable = []
    skipped = 0
    for record in records:
        if us_mens_only and not matchable(record):
            skipped += 1
            continue
        images = [p for p in (record.get("images") or []) if (REPO_ROOT / p).exists()]
        if images:
            usable.append((record, images[0]))
    if skipped:
        print(f"  skipped {skipped:,} posts that cannot match a US men's catalog")
    if not usable:
        raise SystemExit(
            "no outfit images found on disk. The corpus was re-detected on Modal, "
            "so crops and possibly images may live on the volume rather than here.")
    rng = random.Random(seed)
    rng.shuffle(usable)
    return usable[:count]


def build(args):
    import requests
    from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

    sys.path.insert(0, str(REPO_ROOT))
    import garment_proposer

    CROP_DIR.mkdir(parents=True, exist_ok=True)
    key = load_api_key()
    photos = sample_photos(args.n, args.seed, not args.all_sources)
    print(f"  sampled {len(photos)} outfit photos (seed {args.seed})")

    print("  loading the human parser + FashionCLIP (first run downloads weights)...")
    processor, model = garment_proposer.load_human_parser(device=args.device)
    clip_processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    clip_model = AutoModelForZeroShotImageClassification.from_pretrained(
        "patrickjohncyh/fashion-clip").to(args.device)

    items = []
    for index, (record, image_path) in enumerate(photos, 1):
        source = REPO_ROOT / image_path
        try:
            proposals = garment_proposer.propose_garment_items(
                str(source), processor, model, clip_processor, clip_model,
                device=args.device)
        except Exception as error:  # noqa: BLE001
            print(f"  [{index}/{len(photos)}] {image_path}: proposer failed: {error}")
            continue

        if not proposals:
            print(f"  [{index}/{len(photos)}] {image_path}: no garments proposed")
            continue

        for order, proposal in enumerate(proposals[:args.max_items_per_photo]):
            # Use the proposer's OWN crop, not a re-crop from the bbox: it
            # blanks everything outside the garment inside that box, which
            # is the whole reason its precision beat SAM2's. Re-cropping
            # the raw bbox here would hand the encoder back the background
            # the proposer just removed, and quietly measure a different
            # pipeline than the one that ships.
            crop_name = f"{record['source']}_{record['source_id']}_{order}.jpg"
            crop_path = CROP_DIR / crop_name
            proposal["crop"].convert("RGB").save(crop_path, "JPEG", quality=92)

            payload = {
                "image_base64": base64.b64encode(crop_path.read_bytes()).decode(),
                "top_k": args.top_k,
            }
            try:
                response = requests.post(
                    f"{args.url.rstrip('/')}/query", json=payload,
                    headers={"Authorization": f"Bearer {key}"}, timeout=300)
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as error:
                print(f"  [{index}/{len(photos)}] query failed: {error}")
                continue

            items.append({
                "id": f"{record['source']}_{record['source_id']}_{order}",
                "crop": str(crop_path.relative_to(REPO_ROOT)),
                "source_image": image_path,
                # Needed to draw the region back onto the full photo. A
                # masked crop alone is genuinely hard to judge -- it blanks
                # everything outside the garment, so a labeller loses the
                # context that tells them what they are looking at. The UI
                # shows both.
                "bbox": proposal.get("bbox"),
                "post_url": record.get("post_url"),
                "source": record.get("source"),
                # The proposer's own view, kept so a labeller can see when
                # the detector and the retriever disagree about what the
                # thing even is.
                "proposed_category": proposal.get("category"),
                "proposer_confidence": proposal.get("confidence"),
                "candidates": [
                    {"rank": c["rank"], "product_code": c["product_code"],
                     "brand": c["brand"], "name": c["name"],
                     "category": c.get("category"), "score": c["score"]}
                    for c in result.get("results", [])
                ],
                "top1_score": result.get("confidence"),
                "garment_gate": result.get("garment_gate"),
                "rejected_open_set": result.get("rejected_open_set"),
            })
        print(f"  [{index}/{len(photos)}] {image_path}: "
              f"{min(len(proposals), args.max_items_per_photo)} item(s), "
              f"{len(items)} total")

    OUT_DIR.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "seed": args.seed, "photos": len(photos), "items": items,
        "url": args.url, "top_k": args.top_k,
    }, indent=2))
    print(f"\n  wrote {len(items)} items to {MANIFEST}")
    print(f"  next: python pair_eval.py label")


# ----------------------------------------------------------------------
# label
# ----------------------------------------------------------------------

LABEL_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>pair eval — labelling</title>
<style>
 :root{color-scheme:light dark;--bg:#fff;--fg:#16161a;--muted:#6b6b76;--line:#e3e3e8;
       --card:#fafafb;--accent:#2f6f4f;--warn:#8a5a00}
 @media(prefers-color-scheme:dark){:root{--bg:#131316;--fg:#ececf1;--muted:#9a9aa6;
       --line:#2c2c33;--card:#1b1b20;--accent:#7fd1a6;--warn:#e0b25f}}
 *{box-sizing:border-box}
 body{margin:0;padding:20px 16px 80px;background:var(--bg);color:var(--fg);
      font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 main{max-width:940px;margin:0 auto}
 h1{font-size:18px;margin:0 0 2px}
 .sub{color:var(--muted);font-size:13px;margin-bottom:18px}
 .bar{height:4px;background:var(--line);border-radius:99px;margin-bottom:20px}
 .bar div{height:100%;background:var(--accent);border-radius:99px;transition:.2s}
 .split{display:flex;gap:22px;flex-wrap:wrap}
 .left{flex:0 0 300px}
 .left img{width:100%;border-radius:10px;background:var(--card)}
 .ctx{position:relative;display:block;margin-bottom:8px}
 .ctx .box{position:absolute;border:2px solid var(--accent);border-radius:3px;
           box-shadow:0 0 0 9999px rgba(0,0,0,.42);pointer-events:none}
 .cropimg{max-height:150px;width:auto !important;display:block;margin:0 auto}
 .right{flex:1 1 380px;min-width:280px}
 .meta{font-size:12px;color:var(--muted);margin-top:8px;word-break:break-word}
 .cand{display:flex;align-items:center;gap:10px;padding:7px 9px;border:1px solid var(--line);
       border-radius:9px;margin-bottom:6px;cursor:pointer;background:var(--card)}
 .cand:hover{border-color:var(--accent)}
 .cand img{width:46px;height:46px;object-fit:cover;border-radius:6px;flex:0 0 46px;
           background:var(--bg)}
 .cand .n{flex:1;min-width:0;font-size:14px}
 .cand .n b{display:block;font-size:12px;color:var(--muted);font-weight:500}
 .cand .s{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
 .verdicts{margin-top:16px;display:flex;gap:8px;flex-wrap:wrap}
 .verdicts button{padding:9px 14px;font-size:13px;border:1px solid var(--line);
        border-radius:9px;background:var(--card);color:var(--fg);cursor:pointer;font-weight:500}
 .verdicts button:hover{border-color:var(--warn)}
 kbd{font:11px ui-monospace,Menlo,monospace;border:1px solid var(--line);
     border-radius:4px;padding:0 4px;color:var(--muted)}
 .done{text-align:center;padding:60px 0;font-size:17px}
</style></head><body><main>
 <h1>pair eval — is the answer right?</h1>
 <div class="sub">Click the candidate that IS the pictured item. If none is,
   use the buttons below. <kbd>1</kbd>–<kbd>9</kbd> pick a candidate,
   <kbd>n</kbd> not in catalog, <kbd>m</kbd> in catalog but missed,
   <kbd>x</kbd> bad crop, <kbd>s</kbd> skip.</div>
 <div class="bar"><div id="bar"></div></div>
 <div id="body"></div>
</main>
<script>
let items = [], labels = {}, i = 0;

async function boot() {
  const res = await fetch('/data');
  const d = await res.json();
  items = d.items; labels = d.labels || {};
  i = items.findIndex(it => !labels[it.id]);
  if (i < 0) i = items.length;
  render();
}

function render() {
  document.getElementById('bar').style.width =
    (Object.keys(labels).length / items.length * 100) + '%';
  if (i >= items.length) {
    document.getElementById('body').innerHTML =
      '<div class="done">All ' + items.length + ' labelled.<br>' +
      '<span class="sub">Run <code>python pair_eval.py score</code></span></div>';
    return;
  }
  const it = items[i];
  // Both views: the masked crop is what the encoder actually saw, and the
  // full photo with the region boxed is what a human needs to judge it.
  // Showing only the crop makes labelling near-impossible -- everything
  // outside the garment is blanked, so there is no context at all.
  let h = '<div class="split"><div class="left">' +
    '<div class="ctx"><img src="/file?path=' + encodeURIComponent(it.source_image) + '"' +
      ' onload="drawBox(this)" data-bbox="' + (it.bbox||[]).join(',') + '">' +
      '<div class="box" style="display:none"></div></div>' +
    '<img class="cropimg" src="/file?path=' + encodeURIComponent(it.crop) + '">' +
    '<div class="meta">item ' + (i+1) + ' of ' + items.length +
    '<br>proposer: ' + esc(it.proposed_category || '?') +
    ' (' + (it.proposer_confidence != null ? it.proposer_confidence.toFixed(2) : '?') + ')' +
    '<br>top-1 score: ' + (it.top1_score != null ? it.top1_score.toFixed(3) : '?') +
    (it.post_url ? '<br><a href="' + esc(it.post_url) + '" target="_blank" rel="noopener">source photo</a>' : '') +
    '</div></div><div class="right">';

  it.candidates.forEach((c, n) => {
    h += '<div class="cand" onclick="pick(' + c.rank + ')">' +
      (c.image ? '<img loading="lazy" src="/file?path=' + encodeURIComponent(c.image) + '">'
               : '<div style="width:46px;height:46px;flex:0 0 46px"></div>') +
      '<div class="n">' + esc(c.name || '') + '<b>' + esc(c.brand || '') +
      (c.category ? ' · ' + esc(c.category) : '') + '</b></div>' +
      '<div class="s">' + (n < 9 ? '<kbd>' + (n+1) + '</kbd> ' : '') +
      (c.score != null ? c.score.toFixed(3) : '') + '</div></div>';
  });

  h += '<div class="verdicts">' +
    '<button onclick="verdict(\'absent\')">Not in catalog <kbd>n</kbd></button>' +
    '<button onclick="verdict(\'missed\')">In catalog, not shown <kbd>m</kbd></button>' +
    '<button onclick="verdict(\'badcrop\')">Bad crop <kbd>x</kbd></button>' +
    '<button onclick="verdict(\'skip\')">Skip <kbd>s</kbd></button>' +
    '</div></div></div>';
  document.getElementById('body').innerHTML = h;
}

function save(label) {
  labels[items[i].id] = label;
  fetch('/label', {method:'POST', headers:{'Content-Type':'application/json'},
                   body: JSON.stringify({id: items[i].id, label})});
  i++; render();
}
function pick(rank){ save({verdict:'correct', rank}); }
function verdict(v){ save({verdict:v}); }

document.addEventListener('keydown', e => {
  if (i >= items.length) return;
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= 9 && items[i].candidates[n-1]) return pick(items[i].candidates[n-1].rank);
  if (e.key === 'n') verdict('absent');
  if (e.key === 'm') verdict('missed');
  if (e.key === 'x') verdict('badcrop');
  if (e.key === 's') verdict('skip');
});

// Position the highlight over the region the crop came from. bbox is in
// ORIGINAL pixel coordinates, so it has to be scaled by the rendered size
// -- naturalWidth/Height give the original, which is why this runs onload
// rather than at render time.
function drawBox(img){
  const raw=(img.dataset.bbox||'').split(',').filter(Boolean).map(Number);
  const box=img.parentElement.querySelector('.box');
  if(raw.length!==4||!img.naturalWidth){ box.style.display='none'; return; }
  const sx=img.clientWidth/img.naturalWidth, sy=img.clientHeight/img.naturalHeight;
  const [l,t,r,b]=raw;
  box.style.left=(l*sx)+'px'; box.style.top=(t*sy)+'px';
  box.style.width=((r-l)*sx)+'px'; box.style.height=((b-t)*sy)+'px';
  box.style.display='block';
}

function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
boot();
</script></body></html>
"""


def label(args):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, unquote, urlparse

    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found — run `pair_eval.py build` first")
    manifest = json.loads(MANIFEST.read_text())
    catalog = {r["product_code"]: r for r in json.loads(CATALOG_METADATA.read_text())}

    # Join thumbnails in at serve time rather than baking them into the
    # manifest: several brands have no images in this dev environment, and
    # that set changes as scrapes land.
    for item in manifest["items"]:
        for candidate in item["candidates"]:
            record = catalog.get(candidate["product_code"])
            images = (record or {}).get("images") or []
            if images:
                candidate["image"] = images[0]

    allowed_roots = [(REPO_ROOT / "pair_eval").resolve(),
                     (REPO_ROOT / "apparel_dataset").resolve(),
                     (REPO_ROOT / "outfit_dataset").resolve()]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, code, body, ctype="application/json"):
            payload = body if isinstance(body, bytes) else str(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, LABEL_PAGE.encode(), "text/html; charset=utf-8")
            if self.path.startswith("/data"):
                existing = json.loads(LABELS.read_text()) if LABELS.exists() else {}
                return self._send(200, json.dumps(
                    {"items": manifest["items"], "labels": existing}).encode())
            if self.path.startswith("/file"):
                raw = parse_qs(urlparse(self.path).query).get("path", [""])[0]
                target = (REPO_ROOT / unquote(raw)).resolve()
                if not any(str(target).startswith(str(root)) for root in allowed_roots):
                    return self._send(403, b'{"detail":"forbidden"}')
                if not target.is_file():
                    return self._send(404, b'{"detail":"not on disk"}')
                return self._send(200, target.read_bytes(), "image/jpeg")
            self._send(404, b'{"detail":"not found"}')

        def do_POST(self):
            if self.path != "/label":
                return self._send(404, b'{"detail":"not found"}')
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            existing = json.loads(LABELS.read_text()) if LABELS.exists() else {}
            existing[body["id"]] = body["label"]
            # Written on every single label, deliberately: a labelling pass
            # is human time, and losing it to a closed laptop would be
            # unrecoverable in a way a re-run of `build` is not.
            LABELS.write_text(json.dumps(existing, indent=2))
            self._send(200, b'{"ok":true}')

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    done = len(json.loads(LABELS.read_text())) if LABELS.exists() else 0
    print(f"\n  {len(manifest['items'])} items, {done} already labelled")
    print(f"  open: http://localhost:{args.port}\n  ctrl-c to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped — progress saved after every label")


# ----------------------------------------------------------------------
# score
# ----------------------------------------------------------------------

def wilson(hits, total, z=1.96):
    """Wilson score interval. Used rather than the normal approximation
    because the in-catalog denominator here will be small, and the normal
    interval is badly wrong (and can leave [0,1]) at small n."""
    if not total:
        return (0.0, 0.0)
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def auroc(positives, negatives):
    """Probability a random positive outscores a random negative.
    Rank-based, so ties are handled by averaging rather than dropped."""
    if not positives or not negatives:
        return None
    combined = sorted([(s, 1) for s in positives] + [(s, 0) for s in negatives])
    ranks, index = {}, 0
    while index < len(combined):
        end = index
        while end + 1 < len(combined) and combined[end + 1][0] == combined[index][0]:
            end += 1
        average = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[position] = average
        index = end + 1
    positive_rank_sum = sum(ranks[p] for p, (_, y) in enumerate(combined) if y == 1)
    n_pos, n_neg = len(positives), len(negatives)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def score(args):
    if not (MANIFEST.exists() and LABELS.exists()):
        raise SystemExit("need both manifest.json and labels.json — build and label first")
    manifest = json.loads(MANIFEST.read_text())
    labels = json.loads(LABELS.read_text())
    by_id = {item["id"]: item for item in manifest["items"]}

    ranks, in_catalog = [], 0
    absent_scores, correct_scores, wrong_scores = [], [], []
    counts = {CORRECT: 0, IN_CATALOG_MISSED: 0, NOT_IN_CATALOG: 0, BAD_CROP: 0, SKIP: 0}

    for item_id, label in labels.items():
        item = by_id.get(item_id)
        if not item:
            continue
        verdict = label.get("verdict")
        counts[verdict] = counts.get(verdict, 0) + 1
        top1 = item.get("top1_score")
        if verdict == CORRECT:
            in_catalog += 1
            ranks.append(label.get("rank"))
            if top1 is not None:
                (correct_scores if label.get("rank") == 1 else wrong_scores).append(top1)
        elif verdict == IN_CATALOG_MISSED:
            in_catalog += 1
            if top1 is not None:
                wrong_scores.append(top1)
        elif verdict == NOT_IN_CATALOG and top1 is not None:
            absent_scores.append(top1)

    total_labelled = sum(counts.values())
    print(f"\n  labelled {total_labelled} of {len(manifest['items'])} items "
          f"from {manifest['photos']} real outfit photos\n")
    print(f"  {counts[CORRECT]:4d}  a shown candidate was correct")
    print(f"  {counts[IN_CATALOG_MISSED]:4d}  in the catalog but not in the top-{manifest['top_k']}")
    print(f"  {counts[NOT_IN_CATALOG]:4d}  not a product we carry")
    print(f"  {counts[BAD_CROP]:4d}  crop was not a usable garment")
    print(f"  {counts[SKIP]:4d}  skipped")

    print("\n  --- RETRIEVAL, consumer photo -> catalog "
          "(the number this project has never had) ---")
    if not in_catalog:
        print("    No item was confirmed to be in the catalog, so R@k is undefined.")
        print("    That is itself a finding: label more, or grow brand coverage first.")
    else:
        for k in (1, 5, 10):
            hits = sum(1 for r in ranks if r and r <= k)
            low, high = wilson(hits, in_catalog)
            print(f"    R@{k:<3} {hits/in_catalog:6.2%}   ({hits}/{in_catalog}, "
                  f"95% CI {low:.1%}–{high:.1%})")
        if in_catalog < 30:
            print(f"    n={in_catalog} is small. The interval is the honest read, "
                  "not the point estimate.")

    print("\n  --- OPEN SET: can a threshold separate in-catalog from not? ---")
    if not absent_scores or not correct_scores:
        print("    Need both correct and not-in-catalog items to say anything.")
    else:
        area = auroc(correct_scores, absent_scores)
        print(f"    AUROC {area:.3f}  ({len(correct_scores)} correct vs "
              f"{len(absent_scores)} out-of-catalog)")
        print(f"    correct  top-1 score: median {median(correct_scores):.3f}")
        print(f"    absent   top-1 score: median {median(absent_scores):.3f}")
        if wrong_scores:
            print(f"    wrong    top-1 score: median {median(wrong_scores):.3f}")
        print("\n    Operating points (threshold on top-1 score):")
        print(f"    {'thresh':>8} {'false-reject':>14} {'false-accept':>14}")
        for threshold in sorted({round(s, 2) for s in correct_scores + absent_scores}):
            false_reject = sum(1 for s in correct_scores if s < threshold) / len(correct_scores)
            false_accept = sum(1 for s in absent_scores if s >= threshold) / len(absent_scores)
            print(f"    {threshold:8.2f} {false_reject:13.1%} {false_accept:13.1%}")
        print("\n    Compare: the proxy calibration gave AUROC 0.769 with no usable"
              "\n    operating point (1% false-reject cost 68% false-accept). These"
              "\n    negatives are from the real input distribution, which the proxy"
              "\n    set was not.")

    print("\n  Caveat that travels with every number above: crops come from"
          "\n  garment_proposer.py, whose ~91% precision was eyeballed on the same"
          "\n  40 images its threshold was chosen on. 'Bad crop' labels are broken"
          "\n  out above so retrieval failure and detection failure stay separable.\n")


def median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="crop real photos and query the live API")
    b.add_argument("--n", type=int, default=60, help="outfit photos to sample")
    b.add_argument("--seed", type=int, default=17)
    b.add_argument("--all-sources", action="store_true",
                   help="include Japan/Korea and women's posts, which cannot "
                        "match a US men's catalog (default: exclude them)")
    b.add_argument("--max-items-per-photo", type=int, default=3)
    b.add_argument("--top-k", type=int, default=10)
    b.add_argument("--device", default="cpu")
    b.add_argument("--url", default=os.environ.get("FASHION_API_URL", DEFAULT_URL))
    b.set_defaults(func=build)

    l = sub.add_parser("label", help="label them in a browser")
    l.add_argument("--port", type=int, default=7870)
    l.set_defaults(func=label)

    s = sub.add_parser("score", help="compute the numbers")
    s.set_defaults(func=score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
