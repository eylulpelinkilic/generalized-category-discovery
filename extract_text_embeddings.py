
import json
from pathlib import Path

import numpy as np
import torch
import open_clip

# ----------------------------------------
# 1. Configuration
# ----------------------------------------

OUTPUT_DIR = Path("embeddings")
OUTPUT_DIR.mkdir(exist_ok=True)

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

# ----------------------------------------
# 2. Load CLIP ViT-B/32
# ----------------------------------------

model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer("ViT-B-32")

# ----------------------------------------
# 3. Load the 100 ImageNet concepts
# ----------------------------------------

with open("imagenet100_classes.json") as f:
    classes = json.load(f)

# Use the first name for classes with synonyms
class_names = [
    c["name"].split(",")[0].strip()
    for c in classes
]

class_ids = [
    c["imagenet_id"]
    for c in classes
]

# CLIP zero-shot prompt
prompts = [
    f"a photo of a {name}."
    for name in class_names
]

print("Number of concepts:", len(prompts))
print("Example prompts:", prompts[:5])

# ----------------------------------------
# 4. Extract text embeddings
# ----------------------------------------

tokens = tokenizer(prompts).to(device)

with torch.inference_mode():
    features = model.encode_text(tokens)

    features = torch.nn.functional.normalize(
        features.float(),
        dim=-1
    )

text_embeddings = features.cpu().numpy()

# ----------------------------------------
# 5. Save embeddings and metadata
# ----------------------------------------

np.save(
    OUTPUT_DIR / "text_embeddings.npy",
    text_embeddings
)

np.save(
    OUTPUT_DIR / "text_class_ids.npy",
    np.array(class_ids)
)

with open(
    OUTPUT_DIR / "text_prompts.json", "w"
) as f:
    json.dump(prompts, f, indent=2)

# ----------------------------------------
# 6. Verify results
# ----------------------------------------

norms = np.linalg.norm(text_embeddings, axis=1)

print("\nText extraction complete!")
print("Embedding shape:", text_embeddings.shape)
print("Number of classes:", len(class_ids))
print("Minimum norm:", norms.min())
print("Maximum norm:", norms.max())
