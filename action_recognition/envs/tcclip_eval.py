"""Run TC-CLIP's official evaluation and keep its per-video scores.

Runs inside the official TC-CLIP environment (Python 3.8, PyTorch 1.13), from any
directory, with the same Hydra arguments as the official evaluation, e.g.

    torchrun --nproc_per_node=1 envs/tcclip_eval.py --repo third_party/tc-clip -- \
        -cn zero_shot data=xov_ucf101 trainer=tc_clip eval=test resume=<ckpt> \
        output=<out> gather_filename=true

`main.main()` and `engine.validate` run unchanged. Two read-only observers are
added: the test loader records each batch's `file_id` (row of the annotation
file; the official sampler shuffles), and `accuracy_top1_top5` records the
score matrix it is given (the sum over views of the per-view softmax). After each
test dataset, `<output>/scores/<dataset_name>.npz` holds `file_id`, `label`,
`scores` and the official Acc@1/Acc@5. `ovar import-tcclip` turns it into a run.
"""
import argparse
import json
import os
import sys

import numpy as np


def _as_array(x):
    import torch

    return np.asarray(torch.as_tensor(x).reshape(-1).cpu().numpy())


class _RecordingLoader:
    """Iterates like the wrapped DataLoader and records the file ids it yields."""

    def __init__(self, loader, record):
        self.loader, self.record = loader, record
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            self.record["file_id"].append(_as_array(batch["file_id"]))
            yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", required=True, help="TC-CLIP repository (pinned clone)")
    args, hydra_args = parser.parse_known_args()
    if hydra_args[:1] == ["--"]:
        hydra_args = hydra_args[1:]
    if "gather_filename=true" not in hydra_args:
        raise SystemExit("pass gather_filename=true: per-video scores need the file ids")

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    os.chdir(repo)
    import engine
    import main as tc_main  # the official main.py

    record = {"file_id": [], "label": [], "scores": []}
    official_accuracy = engine.accuracy_top1_top5

    def recording_accuracy(pred, target):
        record["scores"].append(pred.detach().float().cpu().numpy())
        record["label"].append(_as_array(target))
        return official_accuracy(pred, target)

    def validate_and_save(val_loader, model, logger, config):
        for values in record.values():
            values.clear()
        stats = engine.validate(_RecordingLoader(val_loader, record), model, logger, config)
        name = val_loader.dataset.dataset_name
        file_id, label = np.concatenate(record["file_id"]), np.concatenate(record["label"])
        scores = np.concatenate(record["scores"])
        if not (len(file_id) == len(label) == len(scores) == len(val_loader.dataset)):
            raise RuntimeError(f"{name}: recorded {len(file_id)} ids, {len(label)} labels, {len(scores)} "
                               f"score rows for {len(val_loader.dataset)} videos")
        out = os.path.join(config.output, "scores")
        os.makedirs(out, exist_ok=True)
        meta = {"dataset_name": name, "ann_file": val_loader.dataset.ann_file,
                "num_frames": config.num_frames, "num_clip": config.num_clip, "num_crop": config.num_crop,
                "multi_view_inference": bool(config.multi_view_inference), "seed": config.seed,
                "wise_ft": config.wise_ft, "opt_level": config.opt_level, "resume": config.resume,
                "official_acc1": float(stats["acc1"]), "official_acc5": float(stats["acc5"])}
        np.savez(os.path.join(out, f"{name}.npz"), file_id=file_id, label=label, scores=scores,
                 meta=json.dumps(meta))
        logger.info(f"saved per-video scores to {out}/{name}.npz")
        return stats

    engine.accuracy_top1_top5 = recording_accuracy
    tc_main.validate = validate_and_save
    sys.argv = [os.path.join(repo, "main.py")] + hydra_args
    tc_main.main()


if __name__ == "__main__":
    main()
