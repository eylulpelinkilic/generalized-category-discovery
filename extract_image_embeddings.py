
import json
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

# ----------------------------------------
# 1. Configuration
# ----------------------------------------

IMAGE_DIR = Path("imagenet100_images")
OUTPUT_DIR = Path("embeddings")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32

# Use Apple GPU if available
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

# ----------------------------------------
# 2. Load pretrained CLIP
# ----------------------------------------

model, _, preprocess = (
    open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )
)

model = model.to(device)
model.eval()

# ----------------------------------------
# 3. Collect image paths
# ----------------------------------------

with open("imagenet100_classes.json") as f:
    classes = json.load(f)

image_paths = []
labels = []

for c in classes:
    class_id = c["imagenet_id"]
    class_dir = IMAGE_DIR / str(class_id)

    paths = sorted(class_dir.glob("*.jpg"))

    for path in paths:
        image_paths.append(path)
        labels.append(class_id)

print("Total images:", len(image_paths))
assert len(image_paths) == 2000

# ----------------------------------------
# 4. Extract embeddings
# ----------------------------------------

all_embeddings = []

with torch.inference_mode():
    for start in tqdm(
        range(0, len(image_paths), BATCH_SIZE),
        desc="Extracting CLIP embeddings"
    ):
        batch_paths = image_paths[
            start:start + BATCH_SIZE
        ]

        images = []

        for path in batch_paths:
            with Image.open(path) as img:
                images.append(
                    preprocess(img.convert("RGB"))
                )

        batch = torch.stack(images).to(device)

        # CLIP image encoder
        features = model.encode_image(batch)

        # Normalize to unit length
        features = torch.nn.functional.normalize(
            features.float(),
            dim=-1
        )

        all_embeddings.append(
            features.cpu().numpy()
        )

# ----------------------------------------
# 5. Save embeddings and metadata
# ----------------------------------------

embeddings = np.concatenate(
    all_embeddings, axis=0
)

np.save(
    OUTPUT_DIR / "image_embeddings.npy",
    embeddings
)

np.save(
    OUTPUT_DIR / "image_labels.npy",
    np.array(labels)
)

with open(
    OUTPUT_DIR / "image_paths.json", "w"
) as f:
    json.dump(
        [str(p) for p in image_paths],
        f,
        indent=2
    )

# ----------------------------------------
# 6. Verify results
# ----------------------------------------

norms = np.linalg.norm(embeddings, axis=1)

print("\nExtraction complete!")
print("Embedding shape:", embeddings.shape)
print("Labels:", len(labels))
print("Minimum norm:", norms.min())
print("Maximum norm:", norms.max())
print("Files saved in:", OUTPUT_DIR)
