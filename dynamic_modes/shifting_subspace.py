from __future__ import annotations

import pickle
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


# Ad-hoc configuration
TARGET = "19.Unknown.F" # "19.Unknown.F"
ALPHA = 0.05
BIN_SIZE_MS = 20
N_RANDOM = 100
N_COMPONENTS = 2
WINDOW_SIZE = 100
STEP = 10
RANDOM_STATE = 0
SAVE = False
VERBOSE = True


def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)


target_parts = TARGET.split(".")
roi_label = f"{int(target_parts[0]):02d}.{target_parts[1]}.{target_parts[2]}"

topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
with open(topk_local, "rb") as f:
    topk_vals = pickle.load(f)
top_k = int(topk_vals[roi_label]["k"])

raster_4d = nu.significant_trial_raster(TARGET, alpha=ALPHA, bin_size_ms=BIN_SIZE_MS)
raster_3d = np.nanmean(raster_4d, axis=3)
image_order = tut.rank_images_by_response(raster_3d)
idx_topk = np.asarray(image_order[:top_k], dtype=int)
candidate_idxs = np.asarray(image_order[top_k:], dtype=int)

vprint(f"Resolved ROI target: {TARGET}")
vprint(f"Using top-k = {top_k}")
vprint(f"Responsive raster shape: {raster_4d.shape}")
vprint(f"Trial-averaged PSTH shape: {raster_3d.shape}")

time_starts = np.arange(0, raster_3d.shape[1] - WINDOW_SIZE, STEP)
n_time = len(time_starts)
n_components = min(N_COMPONENTS, top_k, raster_3d.shape[0])
if n_components < 1:
    raise ValueError(f"Invalid number of subspace dimensions: {n_components}")
if n_components != N_COMPONENTS:
    vprint(f"Adjusted n_components from {N_COMPONENTS} to {n_components}")

rng = np.random.default_rng(RANDOM_STATE)
random_idxs = np.stack(
    [rng.choice(candidate_idxs, size=top_k, replace=False) for _ in range(N_RANDOM)],
    axis=0,
)


def subspace_matrix(indices: np.ndarray) -> np.ndarray:
    subspaces = []
    for t in time_starts:
        R_t = np.nanmean(raster_3d[:, t : t + WINDOW_SIZE, :], axis=1).T
        A_t = PCA(n_components=n_components).fit(R_t[indices]).components_.T
        subspaces.append(A_t)

    angles_tt = np.full((n_time, n_time), np.nan, dtype=float)
    for i in range(n_time):
        for j in range(n_time):
            angles_tt[i, j] = float(np.degrees(subspace_angles(subspaces[i], subspaces[j])).mean())
    return angles_tt


angles_top = subspace_matrix(idx_topk)
angles_all = subspace_matrix(np.arange(raster_3d.shape[2], dtype=int))

angles_random = np.full((N_RANDOM, n_time, n_time), np.nan, dtype=float)
for i, idx_rand in enumerate(random_idxs):
    angles_random[i] = subspace_matrix(idx_rand)

# Averaging angle matrices is the cleanest random reference here because
# subspaces themselves do not have a meaningful elementwise average basis.
angles_random_mean = np.nanmean(angles_random, axis=0)

vprint(f"Top mean angle: {np.nanmean(angles_top):.6f}")
vprint(f"All mean angle: {np.nanmean(angles_all):.6f}")
vprint(f"Random mean angle: {np.nanmean(angles_random_mean):.6f}")

vmax = float(
    np.nanmax(
        [
            np.nanmax(angles_top),
            np.nanmax(angles_all),
            np.nanmax(angles_random_mean),
        ]
    )
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

sns.heatmap(angles_top, square=True, ax=axes[0])# , vmin=0, vmax=vmax, cbar=False)
axes[0].set_title("Top-k")
axes[0].set_xlabel("time window")
axes[0].set_ylabel("time window")

sns.heatmap(angles_all, square=True, ax=axes[1])# , vmin=0, vmax=vmax, cbar=False)
axes[1].set_title("All")
axes[1].set_xlabel("time window")
axes[1].set_ylabel("")

sns.heatmap(angles_random_mean, square=True, ax=axes[2])# , vmin=0, vmax=vmax, cbar=True)
axes[2].set_title("Random mean")
axes[2].set_xlabel("time window")
axes[2].set_ylabel("")

fig.suptitle(f"{roi_label} | window={WINDOW_SIZE} step={STEP} dims={n_components}")

if SAVE:
    s3_base = f"{pth.SAVEDIR}/dynamic_modes/shifting_subspace/{TARGET}"
    payload = {
        "roi": roi_label,
        "target": TARGET,
        "top_k": int(top_k),
        "n_components": int(n_components),
        "window_size": int(WINDOW_SIZE),
        "step": int(STEP),
        "time_starts": time_starts,
        "angles_top": angles_top,
        "angles_all": angles_all,
        "angles_random": angles_random,
        "angles_random_mean": angles_random_mean,
    }
    with fsspec.open(f"{s3_base}.pkl", "wb") as f:
        pickle.dump(payload, f)
    with fsspec.open(f"{s3_base}.png", "wb") as f:
        fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}.svg", "w") as f:
        fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")

download_png = Path.home() / "Downloads" / f"shifting_subspace_{TARGET}.png"
fig.savefig(download_png, dpi=300, transparent=False, bbox_inches="tight")
