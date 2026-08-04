"""Run index_outfits.py on a Modal GPU: multi-item detection over the
6,860-record real-outfit corpus (roadmap Phase 8, step 1).

Why this exists, and why it is not optional: `index_outfits.py` runs SAM2's
automatic mask generator over ~9,300 photos. The same generator, measured
locally on 2026-08-03 (modal_app_segment_apparel.py's docstring), managed
~1.3 crops/min on an 8-core M-series and drove the machine to a load
average of 68. At that rate this corpus is measured in DAYS. On an A10G it
is hours and costs a few dollars.

Design, following modal_app_segment_apparel.py -- which is the working
template here, so the same rules apply:

  - **Ships the real scripts, unchanged.** `index_outfits.py` and
    `segment_outfit.py` are uploaded as-is and invoked as a subprocess. No
    fork, no reimplementation, no "Modal version" of the detector. A
    forked copy is how two different detection policies end up in one
    index, and detection_meta.params would then be recording a lie.
  - **Runs with cwd=/data**, because index_outfits.py addresses the corpus
    by the relative path `outfit_dataset/metadata.json`. Nothing in it is
    Modal-aware.
  - **Its own Volume (`outfit-index`).** Never `fashion-dataset`, and
    never anything that holds `apparel_dataset/` -- this job must not be
    able to write to the catalog even by accident.
  - **Resumable by construction.** index_outfits.py skips records whose
    `detection_meta.params` already match the current params, and
    checkpoints every 10 records through a read-merge-write, so a timeout
    loses at most 10 records and a rerun continues where it stopped.

  - **Python 3.11 on purpose, and it caught a real bug.** index_outfits.py
    originally ended with a nested same-quote f-string that only parses on
    3.12+. The local venv is 3.14, so it looked fine; on this image it
    would have been a SyntaxError raised at IMPORT time -- i.e. after the
    3.5 GB upload, after container start, before a single photo was
    processed. Fixed in index_outfits.py. Keep this pin, or keep
    compiling that file under an older interpreter before a run.

ONE CONTAINER AT A TIME. Do not fan this out across containers against the
same Volume. index_outfits.py's read-merge-write makes it safe against a
*local* concurrent writer, but two Modal containers each hold their own
view of the Volume and only reconcile at commit(), so parallel shards
would silently drop each other's detections. If this needs to go faster,
shard by `--source` into SEPARATE volumes and merge the manifests locally.

Upload (3.5 GB; do this once):
    modal volume create outfit-index
    modal volume put outfit-index outfit_dataset outfit_dataset

Smoke test first -- ALWAYS. It is ~2 minutes and it proves the image,
the checkpoint path, the volume layout and the write-back all work
before committing hours of GPU:
    python3 -m modal run modal_app_index_outfits.py --limit 5

Full run:
    python3 -m modal run modal_app_index_outfits.py
    python3 -m modal run modal_app_index_outfits.py --source reddit

Fetch results back:
    modal volume get outfit-index outfit_dataset/metadata.json <local>
    modal volume get outfit-index 'outfit_dataset/reddit/**/crops/*' <local>

What comes back is UNVALIDATED model output. These photos are unlabeled by
deliberate decision (SCRAPING_PROCESS.md, 2026-08-03) and there is no
ground truth for what garments are in them, so the run's numbers are
coverage, not accuracy -- and they land in the namespaced `detected_items`
/ `detection_meta` keys, never mixed with scraped fact.
"""

from pathlib import Path

import modal

app = modal.App("fashion-index-outfits")

SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", "tqdm", "numpy",
        "accelerate", "safetensors", "hydra-core", "iopath",
    )
    # SAM2 has no PyPI release; same source install as the apparel job.
    .run_commands("pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git")
    # Public 176 MB file: baked at build time rather than pushed over a
    # home connection, and it keeps the Volume to just the corpus.
    .run_commands(
        "mkdir -p /root/checkpoints",
        f"wget -q -O /root/checkpoints/sam2.1_hiera_small.pt {SAM2_CHECKPOINT_URL}",
    )
    .add_local_file(str(Path(__file__).parent / "index_outfits.py"), "/root/index_outfits.py")
    .add_local_file(str(Path(__file__).parent / "segment_outfit.py"), "/root/segment_outfit.py")
)

volume = modal.Volume.from_name("outfit-index", create_if_missing=True)

# 6h, not the apparel job's 4h: ~9,300 images at a per-image cost that has
# never been measured on GPU for THIS corpus (the local measurement was
# catalog photos, which are smaller and simpler). Resumability means an
# overrun costs a rerun, not the work.
TIMEOUT_SECONDS = 6 * 60 * 60


@app.function(image=image, gpu="A10G", volumes={"/data": volume}, timeout=TIMEOUT_SECONDS)
def index_outfits(source: str = None, limit: int = None,
                  max_images_per_record: int = 2, force: bool = False):
    import os
    import subprocess
    import sys
    import time

    import torch
    print(f"cuda: {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}",
          flush=True)

    if not os.path.isdir("/data/outfit_dataset"):
        raise RuntimeError(
            "/data/outfit_dataset is missing. Upload it first:\n"
            "  modal volume create outfit-index\n"
            "  modal volume put outfit-index outfit_dataset outfit_dataset")
    if os.path.exists("/data/apparel_dataset"):
        # Hard stop rather than a warning: this job has no business near
        # the catalog, and a volume that contains both is a volume where a
        # relative-path mistake can overwrite it.
        raise RuntimeError("apparel_dataset is present on this volume -- refusing to run. "
                           "This job must never be able to write to the catalog.")

    # segment_outfit.py resolves the SAM2 checkpoint by the relative path
    # checkpoints/sam2.1_hiera_small.pt, and everything runs from /data.
    os.makedirs("/data/checkpoints", exist_ok=True)
    if not os.path.exists("/data/checkpoints/sam2.1_hiera_small.pt"):
        subprocess.run(["cp", "/root/checkpoints/sam2.1_hiera_small.pt",
                        "/data/checkpoints/sam2.1_hiera_small.pt"], check=True)

    args = [sys.executable, "/root/index_outfits.py",
            "--max-images-per-record", str(max_images_per_record)]
    if source:
        args += ["--source", source]
    if limit:
        args += ["--limit", str(limit)]
    if force:
        args += ["--force"]

    started = time.time()
    # PYTHONUNBUFFERED so per-record progress actually streams -- block
    # buffering to a pipe is why the local apparel run's logs looked frozen
    # at zero records for hours while it was in fact working.
    # PYTHONPATH so `import segment_outfit` resolves from /root while the
    # process runs in /data.
    result = subprocess.run(args, cwd="/data",
                            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root"})
    elapsed = time.time() - started
    print(f"\nindex_outfits exit={result.returncode} in {elapsed / 60:.1f} min", flush=True)

    # Commit even on failure: a partial run's checkpoints are real work and
    # the next run resumes from them.
    volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"index_outfits.py failed ({result.returncode})")


@app.local_entrypoint()
def main(source: str = None, limit: int = None,
         max_images_per_record: int = 2, force: bool = False):
    index_outfits.remote(source=source, limit=limit,
                         max_images_per_record=max_images_per_record, force=force)
