
import numpy as np
import json
from datasets import load_dataset

# Reproduce the GCD ImageNet-100 class selection
np.random.seed(0)
selected_classes = np.random.choice(
    range(1000), size=100, replace=False
)
selected_classes = np.sort(selected_classes)

# Load only dataset metadata
dataset = load_dataset(
    "ILSVRC/imagenet-1k",
    split="validation",
    streaming=True
)

class_names = dataset.features["label"].names

# Construct the class list
classes = []
for i, class_id in enumerate(selected_classes):
    classes.append({
        "gcd_id": i,
        "imagenet_id": int(class_id),
        "name": class_names[int(class_id)]
    })

# Save the selection
with open("imagenet100_classes.json", "w") as f:
    json.dump(classes, f, indent=2)

print("Selected", len(classes), "classes")

for c in classes[:10]:
    print(c)
