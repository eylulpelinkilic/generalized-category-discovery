"""Image datasets keyed by manifest sample ids."""
from __future__ import annotations

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from e1.manifests import Manifest


def load_rgb(path) -> Image.Image:
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


class ImageList(Dataset):
    """Images in a fixed sample-id order; yields (image, position)."""

    def __init__(self, manifest: Manifest, sample_ids: list[str], transform):
        self.sample_ids = list(sample_ids)
        self.paths = manifest.paths(self.sample_ids)
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, i):
        return self.transform(load_rgb(self.paths[i])), i


class GCDTrainSet(Dataset):
    """Labelled anchors first, then unlabelled images, as in upstream `MergedDataset`.

    Only the labelled table is read; unlabelled images carry label -1, so their true classes
    cannot reach the losses. Yields (views, label, position, mask_lab) like the upstream loader.
    """

    def __init__(self, manifest: Manifest, transform):
        labelled = manifest.labels("labelled")
        self.ids_labelled = manifest.ids("labelled")
        self.ids_unlabelled = manifest.ids("unlabelled")
        self.sample_ids = self.ids_labelled + self.ids_unlabelled
        self.labels = np.array([int(labelled[s]) for s in self.ids_labelled] + [-1] * len(self.ids_unlabelled))
        self.mask_lab = np.array([1] * len(self.ids_labelled) + [0] * len(self.ids_unlabelled))
        self.paths = manifest.paths(self.sample_ids)
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def sample_weights(self) -> list[float]:
        """Upstream sampler weights: 1 for labelled, n_labelled / n_unlabelled for unlabelled."""
        n_l, n_u = len(self.ids_labelled), len(self.ids_unlabelled)
        return [1.0] * n_l + [n_l / n_u] * n_u

    def __getitem__(self, i):
        return self.transform(load_rgb(self.paths[i])), int(self.labels[i]), i, np.array([self.mask_lab[i]])
