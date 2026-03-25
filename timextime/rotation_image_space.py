from __future__ import annotations

import pickle
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut


# Ad-hoc configuration
ROI_TARGET = "07.MF1.F" # "19.Unknown.F", "07.MF1.F"
BIN_SIZE_MS = 20
ALPHA = 0.05
D_PCS = 3
BASE_SL = slice(0, 50)
POST_SL = slice(160, 200)
RANDOM_STATE = 0
VERBOSE = True

MINS_PATH = Path("./../datasets/NNN/face_mins.pkl")
OUTPUT_PATH = Path.home() / "Downloads" / "rotation_image_space_19_Unknown_F.png"


def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)


def pca_subspace_basis(X_u_by_img: np.ndarray, d: int, center: bool = True, eps: float = 1e-12) -> np.ndarray:
    """
    Return an orthonormal PCA basis in image space for one timepoint.
    """
    X = X_u_by_img.astype(np.float64, copy=False)
    if center:
        X = X - np.nanmean(X, axis=0, keepdims=True)

    X = np.nan_to_num(X, nan=0.0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)

    rank_eff = int(np.sum(S > eps))
    d_eff = int(min(d, rank_eff, Vt.shape[0]))
    if d_eff == 0:
        return np.zeros((Vt.shape[1], 0), dtype=np.float64)
    return Vt[:d_eff].T


def principal_angles(Q1: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    """
    Return principal angles between two orthonormal subspaces.
    """
    if Q1.size == 0 or Q2.size == 0:
        return np.array([], dtype=np.float64)

    singular_vals = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    singular_vals = np.clip(singular_vals, 0.0, 1.0)
    return np.arccos(singular_vals)


roi_key = ROI_TARGET.split(".")
if len(roi_key) == 4:
    roi_label = f"{roi_key[2]}_{int(roi_key[1])}_{roi_key[3]}"
else:
    roi_label = f"{roi_key[1]}_{int(roi_key[0])}_{roi_key[2]}"

with MINS_PATH.open("rb") as f:
    mins = pickle.load(f)

if roi_label not in mins:
    raise ValueError(f"{roi_label} not found in {MINS_PATH}")

local_k = int(mins[roi_label][0])
vprint(f"ROI label: {roi_label}")
vprint(f"Local scale k: {local_k}")


# -------------------------
# Collect data and matrices
# -------------------------
raster_4d = nu.significant_trial_raster(roi_uid=ROI_TARGET, alpha=ALPHA, bin_size_ms=BIN_SIZE_MS)
vprint(f"Responsive trial raster shape: {raster_4d.shape}")

split_a = raster_4d[:, :, :, 0::2]
split_b = raster_4d[:, :, :, 1::2]
psth_A = np.nanmean(split_a, axis=3)
psth_B = np.nanmean(split_b, axis=3)

if psth_A.shape != psth_B.shape:
    raise ValueError(f"A/B shape mismatch: A={psth_A.shape}, B={psth_B.shape}")

vprint(f"PSTH A shape: {psth_A.shape}")
vprint(f"PSTH B shape: {psth_B.shape}")

trial_avg = np.nanmean(raster_4d, axis=3)
image_order = tut.rank_images_by_response(trial_avg)
rng = np.random.default_rng(RANDOM_STATE)

image_sets = {
    f"Local ({local_k})": image_order[:local_k],
    f"Global ({trial_avg.shape[2]})": np.arange(trial_avg.shape[2]),
    f"Random ({local_k})": rng.choice(trial_avg.shape[2], size=local_k, replace=False),
}

T = psth_A.shape[1]
set_rotation = {}
set_summary = {}
for label, image_idx in image_sets.items():
    QA = []
    QB = []
    for t in range(T):
        QA.append(pca_subspace_basis(psth_A[:, t, image_idx], d=D_PCS))
        QB.append(pca_subspace_basis(psth_B[:, t, image_idx], d=D_PCS))

    sim_cv = np.full((T, T), np.nan, dtype=np.float64)
    for t1 in range(T):
        for t2 in range(T):
            ang1 = principal_angles(QA[t1], QB[t2])
            ang2 = principal_angles(QB[t1], QA[t2])
            s1 = float(np.mean(np.cos(ang1) ** 2))
            s2 = float(np.mean(np.cos(ang2) ** 2))
            sim_cv[t1, t2] = np.nanmean([s1, s2])

    baseline_mean = float(np.nanmean(sim_cv[BASE_SL, BASE_SL]))
    post_mean = float(np.nanmean(sim_cv[POST_SL, POST_SL]))
    set_rotation[label] = sim_cv
    set_summary[label] = {
        "baseline": baseline_mean,
        "post": post_mean,
        "percent_change": 100.0 * ((post_mean - baseline_mean) / baseline_mean),
    }


# -------------------------
# Plot collected results
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for col, (label, sim_cv) in enumerate(set_rotation.items()):
    ax_hm = axes[0, col]
    sns.heatmap(
        sim_cv,
        square=True,
        cmap=sns.color_palette("Greys", as_cmap=True),
        ax=ax_hm,
        cbar=False,
    )
    ax_hm.set_xlabel("time")
    ax_hm.set_ylabel("time")
    ax_hm.set_title(f"{label} | {set_summary[label]['percent_change']:.02f}%")

    ax_bar = axes[1, col]
    baseline_mean = set_summary[label]["baseline"]
    post_mean = set_summary[label]["post"]
    ax_bar.bar(["baseline", "post"], [baseline_mean, post_mean], color=["0.6", "0.2"])
    ax_bar.set_title(f"Mean similarity: {label}")
    ax_bar.set_ylabel("similarity")
    ax_bar.text(0, baseline_mean, f"{baseline_mean:.3f}", ha="center", va="bottom")
    ax_bar.text(1, post_mean, f"{post_mean:.3f}", ha="center", va="bottom")

s3_base = f"{pth.SAVEDIR}/timextime/rotation_image_space/{ROI_TARGET}"
with fsspec.open(f"{s3_base}.png", "wb") as f:
    fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
with fsspec.open(f"{s3_base}.svg", "w") as f:
    fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")
fig.savefig(OUTPUT_PATH, dpi=300, transparent=False, bbox_inches="tight")
print(f"Saved figure to: {OUTPUT_PATH}")
