from __future__ import annotations

from pathlib import Path
import pickle

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import manifold_dynamics.paths as pth
import visionlab_utils.storage as vst


TARGET = "08.MF1.F"
SAVE = False
VERBOSE = True


def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)


def grassmann_fn(X: np.ndarray) -> np.ndarray:
    # out = np.nanmax(X, axis=-1)
    # out = np.nanmean(X, axis=-1)
    out = np.linalg.norm(np.sin(np.deg2rad(X)), axis=-1)
    return out


inpath = f"{pth.SAVEDIR}/dynamic_modes/shifting_subspace/{TARGET}.pkl"
local_inpath = vst.fetch(inpath)
with open(local_inpath, "rb") as f:
    principal_angles = pickle.load(f)

top_angles = np.asarray(principal_angles["top"], dtype=float)
all_angles = np.asarray(principal_angles["all"], dtype=float)
random_angles = np.asarray(principal_angles["random"], dtype=float)

random_angles = np.nanmean(random_angles, axis=0)

vprint(f"Loaded {TARGET}")
vprint(f"top angles shape: {top_angles.shape}")
vprint(f"all angles shape: {all_angles.shape}")
vprint(f"random mean angles shape: {random_angles.shape}")

top = grassmann_fn(top_angles)
all_pa = grassmann_fn(all_angles)
rand = grassmann_fn(random_angles)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
sns.heatmap(top, square=True, cmap="viridis", cbar=True, ax=axes[0])
sns.heatmap(all_pa, square=True, cmap="viridis", cbar=True, ax=axes[1])
sns.heatmap(rand, square=True, cmap="viridis", cbar=True, ax=axes[2])
axes[0].set_title("Top")
axes[1].set_title("All")
axes[2].set_title("Random mean")
plt.show()

if SAVE:
    out = {
        "target": TARGET,
        "top": top,
        "all": all_pa,
        "random": rand,
    }

    outpath = f"{pth.SAVEDIR}/dynamic_modes/grassmann/{TARGET}.pkl"
    with fsspec.open(outpath, "wb") as f:
        pickle.dump(out, f)
    vprint(f"Saved payload to {outpath}")

    png = f"{pth.SAVEDIR}/dynamic_modes/grassmann/{TARGET}.png"
    with fsspec.open(png, "wb") as f:
        fig.savefig(f, format="png", dpi=300, bbox_inches="tight")
    vprint(f"Saved figure to {png}")

    local_png = Path.home() / "Downloads" / f"grassmann_{TARGET}.png"
    fig.savefig(local_png, dpi=300, bbox_inches="tight")
