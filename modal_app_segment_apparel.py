"""Run segment_apparel.py on a Modal GPU instead of the local CPU.

Why this exists: SAM2's automatic mask generator is the most expensive
thing in this repo and it was running on a laptop CPU. Measured
2026-08-03 on an 8-core M-series with 4 per-category workers, it managed
~1.3 crops/min and degraded to ~0.7 under any other load, putting the
remaining Dickies work ~9 hours out and driving the machine to a load
average of 68. On a GPU the same model runs the mask generator in a
fraction of the time, and it costs cents.

Design notes:
  - **Identical parameters to the local run.** MASK_GENERATOR_KWARGS,
    MAX_IMAGE_DIM, MIN_CONFIDENCE and the area bounds all come from the
    shipped segment_apparel.py unchanged. This job finishes a brand that
    was PARTLY segmented locally, so any parameter drift would leave one
    brand with two different crop policies in the same index -- worse
    than either policy on its own.
  - **Runs with cwd=/data**, because segment_apparel.py and
    dataset_utils.py address the dataset by the relative path
    `apparel_dataset/metadata.json`. Nothing is hardcoded for Modal.
  - **Its own Volume (`dickies-segment`), not `fashion-dataset`.** The
    latter carries a stale 6-brand metadata.json; pointing this at it
    would either read the wrong catalog or overwrite the current one.
    This volume gets the CURRENT local metadata.json plus only the images
    for records that still need cropping.
  - **Resumable by construction.** segment_apparel.py skips any record
    that already has `cropped_images`, and checkpoints through
    dataset_utils.save_records_safe, so a timeout or preemption loses at
    most CHECKPOINT_EVERY records and a rerun continues where it stopped.

Upload (already done for the current run):
    modal volume create dickies-segment
    modal volume put dickies-segment <staged>/apparel_dataset apparel_dataset

Run:
    python3 -m modal run modal_app_segment_apparel.py
    python3 -m modal run modal_app_segment_apparel.py --brand dickies --category Pants

Fetch results back:
    modal volume get dickies-segment apparel_dataset/metadata.json <local>
    modal volume get dickies-segment 'apparel_dataset/dickies/**' <local>
"""

from pathlib import Path

import modal

app = modal.App("fashion-segment-apparel")

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
    # SAM2 has no PyPI release; the local env has it as an editable install
    # from a source checkout that no longer exists, so pull it from source.
    .run_commands("pip install --no-build-isolation git+https://github.com/facebookresearch/sam2.git")
    # Baked into the image rather than uploaded to the Volume: it is a
    # public 176MB file, so downloading it once at build time is faster
    # than pushing it over a home connection and keeps the Volume to just
    # the dataset.
    .run_commands(
        "mkdir -p /root/checkpoints",
        f"wget -q -O /root/checkpoints/sam2.1_hiera_small.pt {SAM2_CHECKPOINT_URL}",
    )
    .add_local_file(
        str(Path(__file__).parent / "segment_apparel.py"), "/root/segment_apparel.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "dataset_utils.py"), "/root/dataset_utils.py",
    )
)

volume = modal.Volume.from_name("dickies-segment", create_if_missing=True)


@app.function(image=image, gpu="A10G", volumes={"/data": volume}, timeout=4 * 60 * 60)
def segment(brand: str = "dickies", category: str = None, limit: int = None):
    import os
    import subprocess
    import sys
    import time

    import torch
    print(f"cuda: {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}",
          flush=True)

    # segment_apparel.py resolves both the dataset and the SAM2 config by
    # relative path, so run from /data with the checkpoint symlinked in.
    os.makedirs("/data/checkpoints", exist_ok=True)
    if not os.path.exists("/data/checkpoints/sam2.1_hiera_small.pt"):
        subprocess.run(["cp", "/root/checkpoints/sam2.1_hiera_small.pt",
                        "/data/checkpoints/sam2.1_hiera_small.pt"], check=True)

    args = [sys.executable, "/root/segment_apparel.py", "--brand", brand]
    if category:
        args += ["--category", category]
    if limit:
        args += ["--limit", str(limit)]

    started = time.time()
    # PYTHONUNBUFFERED so the per-record progress lines actually stream --
    # block buffering to a pipe is why the local run's logs looked frozen
    # at zero records for hours while it was in fact working.
    result = subprocess.run(args, cwd="/data",
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    elapsed = time.time() - started
    print(f"\nsegment_apparel exit={result.returncode} in {elapsed / 60:.1f} min", flush=True)

    volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"segment_apparel.py failed ({result.returncode})")


@app.local_entrypoint()
def main(brand: str = "dickies", category: str = None, limit: int = None):
    segment.remote(brand=brand, category=category, limit=limit)
