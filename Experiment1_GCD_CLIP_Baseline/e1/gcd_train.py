"""GCD contrastive training (upstream objective) with resumable full-state checkpoints.

The training step is the upstream `train()` loop (methods/contrastive_training) with its losses
loaded verbatim. Differences, all recorded in `run.json`:
* unlabelled images carry label -1 (their labels are never loaded);
* no per-epoch k-means on labelled test data and no "best" checkpoint: the final epoch is used;
* the sampler and loader generators are seeded from (training seed, epoch), so a resumed run
  sees the same epoch inputs;
* optional mixed precision (`gcd_train.amp`), off by default.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from e1 import upstream
from e1.config import section_hash, stable_hash
from e1.datasets import GCDTrainSet
from e1.manifests import Manifest
from e1.provenance import environment, git_state, sha256_file, write_json


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return torch.device(name)


def epoch_seed(train_seed: int, epoch: int, stream: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{train_seed}:{epoch}:{stream}".encode()).digest()[:4], "little")


def build_backbone(gcd: dict, dino_path: str | None, require_weights: bool = True):
    vits = upstream.vision_transformer()
    if gcd["arch"] != "vit_base" and not gcd.get("smoke", False):
        raise RuntimeError("research configs must use the upstream ViT-B/16 (arch: vit_base)")
    model = vits.__dict__[gcd["arch"]]()
    if dino_path:
        model.load_state_dict(torch.load(dino_path, map_location="cpu"))
    elif require_weights and not gcd.get("smoke", False):
        raise RuntimeError("no DINO checkpoint given (gcd_train.dino_checkpoint)")
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "block" in name and int(name.split(".")[1]) >= gcd["grad_from_block"]:
            p.requires_grad = True
    feat_dim = model.embed_dim
    head = vits.__dict__["DINOHead"](in_dim=feat_dim, out_dim=gcd["mlp_out_dim"], nlayers=gcd["num_mlp_layers"])
    return model, head


class Trainer:
    def __init__(self, cfg: dict, layout, train_seed: int):
        self.cfg, self.gcd, self.seed = cfg, cfg["gcd_train"], train_seed
        self.run_dir = layout.gcd_run(train_seed)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest(layout.data)
        self.device = select_device(cfg.get("device", "auto"))
        self.config_hash = section_hash(cfg, "dataset", "gcd_train")
        train_tf, _ = upstream.get_transform(self.gcd["transform"], image_size=self.gcd["image_size"],
                                             interpolation=self.gcd["interpolation"], crop_pct=self.gcd["crop_pct"])
        view_tf = upstream.ContrastiveLearningViewGenerator(base_transform=train_tf, n_views=self.gcd["n_views"])
        self.dataset = GCDTrainSet(self.manifest, view_tf)
        self.ids_hash = stable_hash(self.dataset.sample_ids)
        # The projection head (and any non-DINO weight) is randomly initialised: seed it from the
        # training seed (upstream defines --seed but never uses it).
        init_seed = epoch_seed(train_seed, -1, "init")
        random.seed(init_seed)
        np.random.seed(init_seed % (2 ** 32))
        torch.manual_seed(init_seed)
        self.model, self.head = build_backbone(self.gcd, self.gcd.get("dino_checkpoint"))
        self.model.to(self.device)
        self.head.to(self.device)
        params = list(self.head.parameters()) + list(self.model.parameters())
        self.optimizer = torch.optim.SGD(params, lr=self.gcd["lr"], momentum=self.gcd["momentum"],
                                         weight_decay=self.gcd["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.gcd["epochs"],
                                                                    eta_min=self.gcd["lr"] * 1e-3)
        self.use_amp = bool(self.gcd.get("amp", False)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.sup_con = upstream.SupConLoss()
        self.epoch = 0

    # ------------------------------------------------------------ checkpoint state
    def state(self) -> dict:
        return {
            "epoch_completed": self.epoch, "train_seed": self.seed, "config_hash": self.config_hash,
            "manifest_sha256": self.manifest.sha256, "train_ids_hash": self.ids_hash,
            "model": self.model.state_dict(), "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(), "amp": self.use_amp,
            "trainable": [n for n, p in self.model.named_parameters() if p.requires_grad],
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }

    def save(self, name: str = "last.pt") -> Path:
        path = self.ckpt_dir / name
        tmp = path.with_suffix(".tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, path)
        return path

    def resume(self) -> bool:
        path = self.ckpt_dir / "last.pt"
        if not path.exists():
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        for key, ours in (("config_hash", self.config_hash), ("manifest_sha256", self.manifest.sha256),
                          ("train_ids_hash", self.ids_hash), ("train_seed", self.seed), ("amp", self.use_amp)):
            if state[key] != ours:
                raise RuntimeError(f"refusing to resume {path}: {key} differs ({state[key]!r} vs {ours!r})")
        self.model.load_state_dict(state["model"])
        self.head.load_state_dict(state["head"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        self.epoch = state["epoch_completed"]
        return True

    # ------------------------------------------------------------ training
    def loader(self, epoch: int) -> DataLoader:
        sampler = WeightedRandomSampler(torch.DoubleTensor(self.dataset.sample_weights()),
                                        num_samples=len(self.dataset), replacement=True,
                                        generator=torch.Generator().manual_seed(epoch_seed(self.seed, epoch, "sampler")))
        return DataLoader(self.dataset, batch_size=self.gcd["batch_size"], sampler=sampler, drop_last=True,
                          num_workers=self.gcd["num_workers"], pin_memory=self.device.type == "cuda",
                          generator=torch.Generator().manual_seed(epoch_seed(self.seed, epoch, "loader")),
                          persistent_workers=False)

    def step(self, batch) -> dict:
        images, class_labels, _, mask_lab = batch
        mask_lab = mask_lab[:, 0].to(self.device).bool()
        class_labels = class_labels.to(self.device)
        images = torch.cat(images, dim=0).to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            features = self.head(self.model(images))
        features = torch.nn.functional.normalize(features.float(), dim=-1)
        # ---- upstream objective (contrast_unlabel_only=False) ----
        logits, targets = upstream.info_nce_logits(features, n_views=self.gcd["n_views"],
                                                   temperature=self.gcd["unsup_temperature"])
        contrastive_loss = torch.nn.CrossEntropyLoss()(logits, targets)
        f1, f2 = [f[mask_lab] for f in features.chunk(2)]
        if mask_lab.any():
            sup_con_loss = self.sup_con(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1),
                                        labels=class_labels[mask_lab])
        else:   # no labelled image in the batch: upstream would divide by zero; counted in the log
            sup_con_loss = features.sum() * 0.0
        w = self.gcd["sup_con_weight"]
        loss = (1 - w) * contrastive_loss + w * sup_con_loss
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        acc = (logits.argmax(1) == targets).float().mean().item()
        return {"loss": loss.item(), "contrastive_acc": acc, "n_labelled": int(mask_lab.sum())}

    def train(self, time_budget_s: float | None = None, max_epochs: int | None = None) -> dict:
        resumed = self.resume()
        log_path = self.run_dir / "train_log.jsonl"
        started, done_now = time.time(), 0
        epoch_times = []
        while self.epoch < self.gcd["epochs"]:
            if max_epochs is not None and done_now >= max_epochs:
                break
            if time_budget_s is not None and epoch_times and \
                    time.time() - started + 1.15 * max(epoch_times) > time_budget_s:
                break
            torch.manual_seed(epoch_seed(self.seed, self.epoch, "torch"))
            self.model.train()
            self.head.train()
            t0, losses, accs, n_lab = time.time(), [], [], []
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for batch in self.loader(self.epoch):
                out = self.step(batch)
                losses.append(out["loss"])
                accs.append(out["contrastive_acc"])
                n_lab.append(out["n_labelled"])
            self.scheduler.step()
            self.epoch += 1
            done_now += 1
            epoch_times.append(time.time() - t0)
            record = {"epoch": self.epoch, "loss": float(np.mean(losses)), "contrastive_acc": float(np.mean(accs)),
                      "mean_labelled_per_batch": float(np.mean(n_lab)), "lr_next": self.scheduler.get_last_lr()[0],
                      "seconds": round(epoch_times[-1], 1), "steps": len(losses),
                      "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.device.type == "cuda" else None}
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            self.save()
        finished = self.epoch >= self.gcd["epochs"]
        summary = {"train_seed": self.seed, "epochs_completed": self.epoch, "epochs_target": self.gcd["epochs"],
                   "finished": finished, "resumed_from_checkpoint": resumed, "epochs_this_call": done_now}
        if finished and not (self.ckpt_dir / "final.pt").exists():
            final = self.ckpt_dir / "final.pt"
            try:
                os.link(self.ckpt_dir / "last.pt", final)      # same bytes as last.pt, no extra space
            except OSError:
                final = self.save("final.pt")
            backbone = self.ckpt_dir / "final_backbone.pt"
            torch.save(self.model.state_dict(), backbone)
            summary.update(final_checkpoint_sha256=sha256_file(final), final_backbone_sha256=sha256_file(backbone))
            write_json(self.run_dir / "run.json", self.run_record(summary))
        return summary

    def run_record(self, summary: dict) -> dict:
        n_l, n_u = len(self.dataset.ids_labelled), len(self.dataset.ids_unlabelled)
        return {
            "stage": "gcd_train", "config_hash": self.config_hash, "gcd_train": self.gcd, "summary": summary,
            "manifest_sha256": self.manifest.sha256, "n_labelled": n_l, "n_unlabelled": n_u,
            "batch": {"images_per_step": self.gcd["batch_size"], "views_per_step": self.gcd["batch_size"] * self.gcd["n_views"],
                      "contrastive_negatives": "all other views in the same step (no gradient accumulation)",
                      "sampler": "WeightedRandomSampler with replacement; labelled weight 1, unlabelled n_l/n_u"},
            "checkpoint_rule": "final epoch (prespecified); no label-based checkpoint selection",
            "deviations_from_upstream": [
                "unlabelled labels never loaded (upstream loads them and masks them out)",
                "no per-epoch k-means on labelled test data; no _best checkpoint",
                "sampler/loader generators seeded per (training seed, epoch) for exact resumption",
                f"num_workers={self.gcd['num_workers']} (upstream 16; no effect on the objective)",
                f"mixed precision={'on' if self.use_amp else 'off'} (upstream: off)",
            ],
            "environment": environment(), "git": git_state(),
        }


def benchmark(cfg: dict, layout, train_seed: int, iters: int, amp: bool) -> dict:
    """Throughput and memory of the real training step (no checkpoint written)."""
    cfg = json.loads(json.dumps(cfg))
    cfg["gcd_train"]["amp"] = amp
    trainer = Trainer(cfg, layout, train_seed)
    trainer.model.train()
    trainer.head.train()
    loader = iter(trainer.loader(10_000))
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times, waits = [], []
    for i in range(iters + 2):
        t0 = time.time()
        batch = next(loader)
        t1 = time.time()
        trainer.step(batch)
        if trainer.device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 2:                      # skip warm-up steps
            times.append(time.time() - t0)
            waits.append(t1 - t0)
    steps_per_epoch = len(trainer.dataset) // cfg["gcd_train"]["batch_size"]
    sec_per_step = float(np.mean(times))
    return {"amp": amp and trainer.device.type == "cuda", "device": str(trainer.device), "iters": iters,
            "batch_images": cfg["gcd_train"]["batch_size"], "sec_per_step": round(sec_per_step, 4),
            "data_wait_fraction": round(float(np.sum(waits) / np.sum(times)), 3),
            "images_per_sec": round(cfg["gcd_train"]["batch_size"] / sec_per_step, 1),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if trainer.device.type == "cuda" else None,
            "steps_per_epoch": steps_per_epoch,
            "estimated_hours_per_epoch": round(steps_per_epoch * sec_per_step / 3600, 3),
            "estimated_hours_full_run": round(steps_per_epoch * sec_per_step * cfg["gcd_train"]["epochs"] / 3600, 2),
            "environment": environment()}
