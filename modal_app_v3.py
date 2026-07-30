"""Modal app for the SigLIP2 v3 fine-tune (finetune_siglip2_v3_modal_body.py).

Separate from modal_app.py (the v2 app) rather than a parameterized rerun of
it, for the same reason the training scripts themselves are separate files:
a Modal run's exact executed source should stay pinned per version. Reuses
the same `fashion-dataset` Volume as v2 -- already populated (1234-product
metadata.json + all 6 brands' images confirmed present via `modal volume ls`
before this was written) so no re-upload needed.

GPU bumped from v2's T4 to A10G: v3's batch is 2x bigger (P16xK4=64 vs
P16xK2=32), stage 2 unfreezes 4 transformer blocks instead of 1, and each
training image now pairs against 3 sampled texts instead of 1 -- T4's 16GB
would be tight for all three increases stacked together, and OOMing partway
through a multi-hour run is more expensive than the GPU-tier price bump.

Usage:
    modal run modal_app_v3.py
"""

import subprocess
import threading
from pathlib import Path

import modal

app = modal.App("fashion-siglip2-v3-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", "tqdm", "pandas", "numpy",
        "accelerate", "safetensors", "sentencepiece",
    )
    .add_local_file(
        str(Path(__file__).parent / "finetune_siglip2_v3_modal_body.py"),
        "/root/finetune_siglip2_v3_modal_body.py",
    )
)

volume = modal.Volume.from_name("fashion-dataset", create_if_missing=False)


@app.function(image=image, gpu="A10G", volumes={"/data": volume}, timeout=10 * 60 * 60)
def train():
    stop_event = threading.Event()

    def periodic_commit():
        while not stop_event.wait(180):
            volume.commit()
            print("[modal_app_v3] volume committed (periodic)")

    committer = threading.Thread(target=periodic_commit, daemon=True)
    committer.start()

    try:
        subprocess.run(["python", "/root/finetune_siglip2_v3_modal_body.py"], check=True)
    finally:
        stop_event.set()
        volume.commit()
        print("[modal_app_v3] final volume commit done")


@app.local_entrypoint()
def main():
    train.remote()
