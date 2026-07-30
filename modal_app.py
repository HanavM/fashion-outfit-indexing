"""Modal app for the SigLIP2 fine-tune -- replaces the Colab+SSH setup,
which kept dying to Colab's idle-disconnect (4 interruptions in one
session, even after a kernel-side keep-alive loop and a JS-level
browser-activity simulator both failed to prevent it). A Modal function
runs to completion in a container with no notebook/idle concept at all.

Usage:
    modal run modal_app.py::upload_dataset   # one-time: push local data to the Volume
    modal run modal_app.py                   # launch training
"""

import subprocess
import threading
import time
from pathlib import Path

import modal

app = modal.App("fashion-siglip2-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", "tqdm", "pandas", "numpy",
        "accelerate", "safetensors", "sentencepiece",
    )
    .add_local_file(
        str(Path(__file__).parent / "finetune_siglip2_modal_body.py"),
        "/root/finetune_siglip2_modal_body.py",
    )
)

volume = modal.Volume.from_name("fashion-dataset", create_if_missing=True)


@app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=6 * 60 * 60)
def train():
    stop_event = threading.Event()

    def periodic_commit():
        while not stop_event.wait(180):
            volume.commit()
            print("[modal_app] volume committed (periodic)")

    committer = threading.Thread(target=periodic_commit, daemon=True)
    committer.start()

    try:
        subprocess.run(["python", "/root/finetune_siglip2_modal_body.py"], check=True)
    finally:
        stop_event.set()
        volume.commit()
        print("[modal_app] final volume commit done")


@app.function(image=modal.Image.debian_slim(), volumes={"/data": volume}, timeout=3600)
def upload_dataset():
    """Not used directly -- data is pushed via `modal volume put` from the
    local CLI instead (simpler for a one-time bulk upload of ~7500 files)."""
    import subprocess as sp
    sp.run(["find", "/data", "-maxdepth", "2"], check=False)


@app.local_entrypoint()
def main():
    train.remote()
