# pip install torch torchvision pillow matplotlib transformers accelerate
# pip install git+https://github.com/facebookresearch/sam2.git

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

IMAGE_PATH = "/Users/hanavmodasiya/Downloads/2c2b22425837ede3a3b0c61833f5a947.jpg"

# -------------------------
# 1. Load FashionCLIP
# -------------------------
processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
clip_model = AutoModelForZeroShotImageClassification.from_pretrained(
    "patrickjohncyh/fashion-clip"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# General fashion label set
candidate_labels = [
    "t-shirt", "shirt", "button-up shirt", "hoodie", "sweatshirt",
    "jacket", "coat", "jersey", "tank top",
    "blue jeans", "pants", "trousers", "shorts", "skirt", "dress",
    "sneakers", "running shoes", "basketball shoes", "boots", "sandals",
    "hat", "cap", "beanie", "socks", "bag", "backpack", "sambas", "nike jersey", "watch", "sling bag"
]

# -------------------------
# 2. Load SAM2
# -------------------------
# Download checkpoint from Meta SAM2 repo first.
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)

masks = mask_generator.generate(image_np)

print(f"SAM2 found {len(masks)} masks")

# -------------------------
# 3. Crop each mask and classify with FashionCLIP
# -------------------------
def crop_masked_region(image_pil, mask):
    mask = mask.astype(bool)
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    crop = image_pil.crop((x1, y1, x2, y2))
    return crop

results = []

for i, mask_data in enumerate(masks):
    mask = mask_data["segmentation"]
    area = mask_data["area"]

    # Skip tiny masks
    if area < 1000:
        continue

    crop = crop_masked_region(image, mask)
    if crop is None:
        continue

    inputs = processor(
        images=crop,
        text=candidate_labels,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=-1)[0]

    top_idx = probs.argmax().item()
    top_label = candidate_labels[top_idx]
    top_score = probs[top_idx].item()

    # Filter weak predictions
    if top_score < 0.25:
        continue

    results.append({
        "mask_id": i,
        "area": int(area),
        "label": top_label,
        "score": round(top_score, 3),
        "crop": crop
    })

# -------------------------
# 4. Print results
# -------------------------
for r in results:
    print(f"Mask {r['mask_id']} | {r['label']} | {r['score']} | area={r['area']}")

# -------------------------
# 5. Visualize masks + labels
# -------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.imshow(image_np)

colors = plt.colormaps["tab20"].resampled(max(len(results), 1))

for idx, r in enumerate(results):
    mask = masks[r["mask_id"]]["segmentation"].astype(bool)
    color = colors(idx)[:3]

    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask] = [*color, 0.45]
    ax.imshow(overlay)

    ys, xs = np.where(mask)
    cx, cy = int(xs.mean()), int(ys.mean())
    ax.text(
        cx, cy,
        f"{r['label']}\n{r['score']:.2f}",
        color="white",
        fontsize=8,
        fontweight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7, edgecolor="none"),
    )

ax.axis("off")
plt.tight_layout()
plt.savefig("output.png", dpi=150, bbox_inches="tight")
plt.show()