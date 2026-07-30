"""Modal app for the SigLIP2 v4 fine-tune (finetune_siglip2_v4_modal_body.py).

Separate from modal_app_v3.py for the same reason v3 was separate from v2:
a Modal run's exact executed source should stay pinned per version. Reuses
the same `fashion-dataset` Volume, already populated. Same A10G GPU choice
as v3 (batch/unfreezing/text-multiplicity are all unchanged from v3 -- v4's
only change is LABEL_KIND_WEIGHTS/build_training_labels' facet tagging, see
finetune_siglip2_v4.py's own docstring).

Launch via deploy + spawn (modal_trigger_v4.py), not `modal run --detach`:
the first v3 launch attempt used --detach and got cancelled by an
unexplained client-side signal ~91 minutes in, with no billing/quota/error
anywhere in Modal's own logs -- coincided with the local background shell
process getting reaped in this environment, despite --detach being
documented to survive exactly that. deploy+spawn has no dependency on any
local process staying alive at all, so there's nothing left to reap.

Usage:
    modal deploy modal_app_v4.py
    python3 modal_trigger_v4.py
"""

import threading
from pathlib import Path

import modal

app = modal.App("fashion-siglip2-v4-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", "tqdm", "pandas", "numpy",
        "accelerate", "safetensors", "sentencepiece",
    )
    .add_local_file(
        str(Path(__file__).parent / "finetune_siglip2_v4_modal_body.py"),
        "/root/finetune_siglip2_v4_modal_body.py",
    )
)

volume = modal.Volume.from_name("fashion-dataset", create_if_missing=False)


@app.function(image=image, gpu="A10G", volumes={"/data": volume}, timeout=10 * 60 * 60)
def train():
    import subprocess

    stop_event = threading.Event()

    def periodic_commit():
        while not stop_event.wait(180):
            volume.commit()
            print("[modal_app_v4] volume committed (periodic)")

    committer = threading.Thread(target=periodic_commit, daemon=True)
    committer.start()

    try:
        subprocess.run(["python", "/root/finetune_siglip2_v4_modal_body.py"], check=True)
    finally:
        stop_event.set()
        volume.commit()
        print("[modal_app_v4] final volume commit done")


@app.local_entrypoint()
def main():
    train.remote()
