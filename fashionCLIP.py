from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image
import torch

processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model = AutoModelForZeroShotImageClassification.from_pretrained(
    "patrickjohncyh/fashion-clip"
)

image = Image.open('/Users/hanavmodasiya/Downloads/images (19).jpeg').convert("RGB")

candidate_labels = [
    "shoe",
    "shorts",
    "jeans",
    "pants",
    "white sneaker",
    "New Balance 530"
]

inputs = processor(
    images=image,
    text=candidate_labels,
    return_tensors="pt",
    padding=True,
)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits_per_image
probs = logits.softmax(dim=-1)

for label, score in zip(candidate_labels, probs[0]):
    print(f"{label}: {score:.3f}")