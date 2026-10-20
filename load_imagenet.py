from datasets import load_dataset

dataset = load_dataset(
    "ILSVRC/imagenet-1k",
    split="validation",
    streaming=True
)

print("Dataset features:")
print(dataset.features)

example = next(iter(dataset))
print("\nFirst image label:", example["label"])