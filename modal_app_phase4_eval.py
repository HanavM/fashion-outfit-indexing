"""Modal app for hierarchical_retrieval_pipeline_modal_body.py -- the real
Phase 4 end-to-end eval, now that both encoders have fine-tuned checkpoints
(SigLIP2 v3: 18.83% category-scoped R@1; DINOv3 identity: 56.55% final
test R@1). Runs --evaluate (both category-gate on/off) against the
fashion-dataset Volume.

Real blocker as of this launch: DINOv3's checkpoint only exists on Colab
Drive (that's where dino_identity_finetune.py ran), not on this Modal
Volume -- SigLIP2 v3 trained on Modal directly so it's already here. If
the DINOv3 checkpoint isn't present under
/data/apparel_dataset/finetuned_dinov3_identity_v1_supcon when this runs,
hierarchical_retrieval_pipeline.py's own checkpoint auto-detection falls
back to the frozen base DINOv3 model with a clear printed warning -- the
eval will still run, but the identity-rerank stage's number won't reflect
the real fine-tuned result until the checkpoint is pushed over, e.g.:
    modal volume put fashion-dataset \\
        /content/drive/MyDrive/apparel_dataset/finetuned_dinov3_identity_v1_supcon \\
        apparel_dataset/finetuned_dinov3_identity_v1_supcon -r
(run from wherever the Colab Drive folder is mounted/accessible, with
modal CLI installed and authenticated there.)

Usage:
    modal deploy modal_app_phase4_eval.py
    python3 modal_trigger_phase4_eval.py
"""

from pathlib import Path

import modal

app = modal.App("fashion-phase4-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", "tqdm", "numpy",
        "accelerate", "safetensors", "sentencepiece",
    )
    .add_local_file(
        str(Path(__file__).parent / "hierarchical_retrieval_pipeline_modal_body.py"),
        "/root/hierarchical_retrieval_pipeline_modal_body.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "docs" / "hierarchy.json"),
        "/root/docs/hierarchy.json",
    )
)

volume = modal.Volume.from_name("fashion-dataset", create_if_missing=False)
hf_secret = modal.Secret.from_name("hf-token")


@app.function(image=image, gpu="A10G", volumes={"/data": volume}, secrets=[hf_secret], timeout=60 * 60)
def evaluate():
    import subprocess
    subprocess.run(
        ["python", "/root/hierarchical_retrieval_pipeline_modal_body.py", "--evaluate"],
        check=True,
    )
    volume.commit()


@app.local_entrypoint()
def main():
    evaluate.remote()
