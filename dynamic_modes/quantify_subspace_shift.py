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


TARGET = "08.MF1.F"
ALPHA = 0.05
BIN_SIZE_MS = 20
WINDOW_SIZE = 100
STEP = 10
N_COMPONENTS = 2
N_RANDOM = 500
MIN_BLOCK = 5
RANDOM_STATE = 0
SAVE = False
VERBOSE = True


def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)


# asks whether a split produces two within-block regions
# that are more similar to themselves than to the between-block region.
def change_point_score(M: np.ndarray, min_block: int = 5) -> tuple[int, np.ndarray]:
    n = M.shape[0]
    scores = np.full(n, np.nan, dtype=float)

    for s in range(min_block, n - min_block):
        ee = M[:s, :s]
        ll = M[s:, s:]
        el = M[:s, s:]

        ee_mask = ~np.eye(ee.shape[0], dtype=bool)
        ll_mask = ~np.eye(ll.shape[0], dtype=bool)

        ee_mean = np.nanmean(ee[ee_mask]) if ee.shape[0] > 1 else np.nan
        ll_mean = np.nanmean(ll[ll_mask]) if ll.shape[0] > 1 else np.nan
        el_mean = np.nanmean(el)
        scores[s] = el_mean - 0.5 * (ee_mean + ll_mean)

    best_split = int(np.nanargmax(scores))
    return best_split, scores


target_parts = TARGET.split(".")
if len(target_parts) not in (3, 4):
    raise ValueError(
        "TARGET must use 4-part UID (SesIdx.RoiIndex.AREALABEL.Categoty) "
        "or 3-part ROI key (RoiIndex.AREALABEL.Categoty)."
    )
if len(target_parts) == 4:
    roi_label = f"{int(target_parts[1]):02d}.{target_parts[2]}.{target_parts[3]}"
else:
    roi_label = f"{int(target_parts[0]):02d}.{target_parts[1]}.{target_parts[2]}"

topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
with open(topk_local, "rb") as f:
    topk_vals = pickle.load(f)

if roi_label not in topk_vals:
    raise ValueError(f"No top-k entry found for ROI: {roi_label}")
top_k = int(topk_vals[roi_label]["k"])

raster_4d = nu.significant_trial_raster(roi_uid=TARGET, alpha=ALPHA, bin_size_ms=BIN_SIZE_MS)
raster_3d = np.nanmean(raster_4d, axis=3)
image_order = np.asarray(tut.rank_images_by_response(raster_3d), dtype=int)
idx_topk = image_order[:top_k]
candidate_idxs = image_order[top_k:]
time_starts = np.arange(0, raster_3d.shape[1] - WINDOW_SIZE, STEP)
rng = np.random.default_rng(RANDOM_STATE)

vprint(f"Resolved ROI target: {TARGET}")
vprint(f"Using top-k = {top_k}")
vprint(f"Responsive raster shape: {raster_4d.shape}")
vprint(f"Trial-averaged PSTH shape: {raster_3d.shape}")

n_components = min(N_COMPONENTS, top_k, raster_3d.shape[0])
if n_components < 1:
    raise ValueError(f"Invalid number of subspace dimensions: {n_components}")
if n_components != N_COMPONENTS:
    vprint(f"Adjusted n_components from {N_COMPONENTS} to {n_components}")

R_windows = [
    np.nanmean(raster_3d[:, t : t + WINDOW_SIZE, :], axis=1).T
    for t in time_starts
]
n_time = len(R_windows)

subspaces_top = []
for R_t in R_windows:
    subspaces_top.append(PCA(n_components=n_components).fit(R_t[idx_topk]).components_.T)

angles_top = np.full((n_time, n_time), np.nan, dtype=float)
for i in range(n_time):
    for j in range(n_time):
        ang = subspace_angles(subspaces_top[i], subspaces_top[j])
        angles_top[i, j] = float(np.degrees(ang).mean())

top_best_split, top_scores = change_point_score(angles_top, min_block=MIN_BLOCK)
top_score = float(np.nanmax(top_scores))

subspaces_all = []
for R_t in R_windows:
    subspaces_all.append(PCA(n_components=n_components).fit(R_t).components_.T)

angles_all = np.full((n_time, n_time), np.nan, dtype=float)
for i in range(n_time):
    for j in range(n_time):
        ang = subspace_angles(subspaces_all[i], subspaces_all[j])
        angles_all[i, j] = float(np.degrees(ang).mean())

all_best_split, all_scores = change_point_score(angles_all, min_block=MIN_BLOCK)
all_score = float(np.nanmax(all_scores))

random_best_splits = np.full(N_RANDOM, np.nan, dtype=float)
random_best_scores = np.full(N_RANDOM, np.nan, dtype=float)
for i in range(N_RANDOM):
    idx_random = rng.choice(candidate_idxs, size=top_k, replace=False)

    subspaces_random = []
    for R_t in R_windows:
        subspaces_random.append(PCA(n_components=n_components).fit(R_t[idx_random]).components_.T)

    angles_random = np.full((n_time, n_time), np.nan, dtype=float)
    for j in range(n_time):
        for k in range(n_time):
            ang = subspace_angles(subspaces_random[j], subspaces_random[k])
            angles_random[j, k] = float(np.degrees(ang).mean())

    best_split, scores_cp = change_point_score(angles_random, min_block=MIN_BLOCK)
    random_best_splits[i] = best_split
    random_best_scores[i] = float(np.nanmax(scores_cp))

print(f"roi: {roi_label}")
print(f"top_k: {top_k}")
print(f"n_components: {n_components}")
print(f"window_size: {WINDOW_SIZE}")
print(f"step: {STEP}")

fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
sns.histplot(random_best_splits, stat="density", alpha=0.5, color="black", ax=ax)
ax.axvline(top_best_split, color="blue", linewidth=1, alpha=0.5, label="Top-k")
ax.axvline(all_best_split, color="red", linewidth=1, alpha=0.5, label="All")
ax.axvline(np.nanmean(random_best_splits), color="black", linewidth=1, alpha=0.5, label="Random (mean)")
ax.set_xlabel("Split point")
ax.set_ylabel("Density")
ax.set_title(f"{roi_label}")
ax.legend()

if SAVE:
    s3_base = f"{pth.SAVEDIR}/timextime/shifting_subspace/{TARGET}"
    payload = {
        "roi": roi_label,
        "target": TARGET,
        "top_k": int(top_k),
        "n_components": int(n_components),
        "window_size": int(WINDOW_SIZE),
        "step": int(STEP),
        "min_block": int(MIN_BLOCK),
        "time_starts": time_starts,
        "top_best_split": int(top_best_split),
        "top_score": top_score,
        "top_scores": top_scores,
        "all_best_split": int(all_best_split),
        "all_score": all_score,
        "all_scores": all_scores,
        "random_best_splits": random_best_splits,
        "random_best_scores": random_best_scores,
    }
    with fsspec.open(f"{s3_base}.pkl", "wb") as f:
        pickle.dump(payload, f)
    with fsspec.open(f"{s3_base}.png", "wb") as f:
        fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}.svg", "w") as f:
        fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")
    vprint(f"Saved figure to {s3_base}.png")
    vprint(f"Saved figure to {s3_base}.svg")

local_png = Path.home() / "Downloads" / f"shifting_subspace_{TARGET.replace('.', '_')}.png"
fig.savefig(local_png, dpi=300, transparent=False, bbox_inches="tight")
plt.close(fig)
vprint(f"Saved figure to {local_png}")
