from __future__ import annotations

from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


DIR_DATASET = Path("topk_dataset")


def make_topk_dataset(target: str) -> None:
    topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
    with open(topk_local, "rb") as f:
        topk_vals = pickle.load(f)

    if target not in topk_vals:
        raise ValueError(f"No top-k entry found for ROI: {target}")

    top_k = int(topk_vals[target]["k"])
    raster_4d = nu.significant_trial_raster(roi_uid=target, alpha=0.05, bin_size_ms=20)
    X = np.nanmean(raster_4d, axis=3)  # (units, time, images)
    ranked = np.asarray(tut.rank_images_by_response(X), dtype=int)
    idx_topk = ranked[:top_k]
    idx_random = np.random.default_rng(0).choice(ranked[top_k:], size=top_k, replace=False)

    dir_roi = DIR_DATASET / target
    dir_topk = dir_roi / "topk"
    dir_random = dir_roi / "random"
    dir_topk.mkdir(parents=True, exist_ok=True)
    dir_random.mkdir(parents=True, exist_ok=True)

    rows = []
    for split, indices, dir_split in [("topk", idx_topk, dir_topk), ("random", idx_random, dir_random)]:
        for rank, image_idx in enumerate(indices, start=1):
            image_idx = int(image_idx)
            image_name = f"{image_idx + 1:04d}.bmp" if image_idx < 1000 else f"MFOB{image_idx - 999:03d}.bmp"
            filename = f"{rank:03d}_{image_name.lower()}"
            filepath_src = vst.fetch(pth._join_path(pth.IMAGEDIR, image_name))
            shutil.copy2(filepath_src, dir_split / filename)
            rows.append(
                {
                    "roi": target,
                    "split": split,
                    "rank": rank,
                    "image_idx": image_idx,
                    "filename": filename,
                }
            )

    pd.DataFrame(rows).sort_values(["split", "rank"]).to_csv(dir_roi / "labels.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: uv run python cindy_audit.py <ROI>")
    make_topk_dataset(sys.argv[1])
