"""Modal app for evaluate_siglip2_by_label_kind_modal_body.py -- a short
eval-only job (image encoding once + 4 cheap text-candidate passes), so
unlike the training apps this doesn't strictly need the deploy+spawn
pattern for reliability. Using it anyway for consistency with
modal_app_v3.py and because it costs nothing extra.

Usage:
    modal deploy modal_app_eval_by_kind.py
    python3 modal_trigger_eval_by_kind.py
"""

from pathlib import Path

import modal

app = modal.App("fashion-siglip2-eval-by-kind")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", "tqdm", "numpy",
        "accelerate", "safetensors", "sentencepiece",
    )
    .add_local_file(
        str(Path(__file__).parent / "evaluate_siglip2_by_label_kind_modal_body.py"),
        "/root/evaluate_siglip2_by_label_kind_modal_body.py",
    )
)

volume = modal.Volume.from_name("fashion-dataset", create_if_missing=False)


@app.function(image=image, gpu="A10G", volumes={"/data": volume}, timeout=60 * 60)
def evaluate():
    import subprocess
    subprocess.run(["python", "/root/evaluate_siglip2_by_label_kind_modal_body.py"], check=True)
    volume.commit()


@app.local_entrypoint()
def main():
    evaluate.remote()
