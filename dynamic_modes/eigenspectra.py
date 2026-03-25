from __future__ import annotations

import pickle
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.spike_response_stats as srs
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

TARGET = "19.Unknown.F"
ALPHA = 0.05
BIN_SIZE_MS = 20
N_PCS = 10
N_RANDOM = 100
RANDOM_STATE = 0
SAVE = True
VERBOSE = True


def vprint(msg: str) -> None:
    """Print only when VERBOSE is enabled."""
    if VERBOSE:
        print(msg)


target_parts = TARGET.split(".")
if len(target_parts) not in (3, 4):
    raise ValueError(
        "TARGET must use 4-part UID (SesIdx.RoiIndex.AREALABEL.Categoty) "
        "or 3-part ROI key (RoiIndex.AREALABEL.Categoty)."
    )
if len(target_parts) == 4:
    roi_label = f"{int(target_parts[1]):02d}.{target_parts[2]}.{target_parts[3]}"
else:
    roi_label = TARGET

topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
with open(topk_local, "rb") as f:
    topk_vals = pickle.load(f)

if roi_label not in topk_vals:
    raise ValueError(f"No top-k entry found for ROI: {roi_label}")
top_k = int(topk_vals[roi_label]["k"])

vprint(f"ROI target: {TARGET}")
vprint(f"Using top-k = {top_k}")

raster_4d = nu.significant_trial_raster(
    roi_uid=TARGET,
    alpha=ALPHA,
    bin_size_ms=BIN_SIZE_MS,
)
X = np.nanmean(raster_4d, axis=3)
image_order = tut.rank_images_by_response(X)
rng = np.random.default_rng(RANDOM_STATE)

vprint(f"Responsive raster shape: {raster_4d.shape}")
vprint(f"Trial-averaged PSTH shape: {X.shape}")

idx_topk = np.asarray(image_order[:top_k], dtype=int)
random_sets = np.stack(
    [rng.choice(X.shape[2], size=top_k, replace=False) for _ in range(N_RANDOM)],
    axis=0,
)

top_pc1 = np.full(X.shape[1], np.nan, dtype=np.float64)
top_ed = np.full(X.shape[1], np.nan, dtype=np.float64)
rand_pc1 = np.full((N_RANDOM, X.shape[1]), np.nan, dtype=np.float64)
rand_ed = np.full((N_RANDOM, X.shape[1]), np.nan, dtype=np.float64)

for t in range(X.shape[1]):
    M_top = X[:, t, idx_topk]
    M_top = M_top - np.nanmean(M_top, axis=1, keepdims=True)
    M_top = M_top / (np.nanstd(M_top, axis=1, keepdims=True) + 1e-8)
    M_top = np.nan_to_num(M_top, nan=0.0, posinf=0.0, neginf=0.0)

    s_top = np.linalg.svd(M_top, full_matrices=False, compute_uv=False)
    lam_top = (s_top ** 2) / max(M_top.shape[1] - 1, 1)
    total_top = float(np.sum(lam_top))
    if total_top > 0:
        frac_top = lam_top / total_top
        top_pc1[t] = float(frac_top[0])
        top_ed[t] = float((np.sum(lam_top) ** 2) / np.sum(lam_top ** 2))

    for i, idx in enumerate(random_sets):
        M_rand = X[:, t, idx]
        M_rand = M_rand - np.nanmean(M_rand, axis=1, keepdims=True)
        M_rand = M_rand / (np.nanstd(M_rand, axis=1, keepdims=True) + 1e-8)
        M_rand = np.nan_to_num(M_rand, nan=0.0, posinf=0.0, neginf=0.0)

        s_rand = np.linalg.svd(M_rand, full_matrices=False, compute_uv=False)
        lam_rand = (s_rand ** 2) / max(M_rand.shape[1] - 1, 1)
        total_rand = float(np.sum(lam_rand))
        if total_rand <= 0:
            continue

        frac_rand = lam_rand / total_rand
        rand_pc1[i, t] = float(frac_rand[0])
        rand_ed[i, t] = float((np.sum(lam_rand) ** 2) / np.sum(lam_rand ** 2))

base_slice = srs._ms_to_slice(srs.ONSET_TIME, srs.BASE_WIN_MS, bin_ms=1)
post_slice = srs._ms_to_slice(srs.ONSET_TIME, srs.RESP_WIN_MS, bin_ms=1)

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)

axes[0].fill_between(
    np.arange(X.shape[1]),
    np.nanpercentile(rand_pc1, 2.5, axis=0),
    np.nanpercentile(rand_pc1, 97.5, axis=0),
    color="0.75",
    alpha=0.4,
)
axes[0].plot(np.nanmean(rand_pc1, axis=0), color="0.35", lw=2, label="Random-k mean")
axes[0].plot(top_pc1, color="#1f77b4", lw=2.5, label="Top-k")
axes[0].axvspan(base_slice.start, base_slice.stop, color="0.9", alpha=0.5)
axes[0].axvspan(post_slice.start, post_slice.stop, color="#ffddcc", alpha=0.25)
axes[0].axvline(50, color="k", lw=1, ls="--")
axes[0].set_ylabel("PC1 variance fraction")
axes[0].set_title(f"{roi_label}: instantaneous geometry control")
axes[0].legend(frameon=False, loc="upper left")

axes[1].fill_between(
    np.arange(X.shape[1]),
    np.nanpercentile(rand_ed, 2.5, axis=0),
    np.nanpercentile(rand_ed, 97.5, axis=0),
    color="0.75",
    alpha=0.4,
)
axes[1].plot(np.nanmean(rand_ed, axis=0), color="0.35", lw=2, label="Random-k mean")
axes[1].plot(top_ed, color="#1f77b4", lw=2.5, label="Top-k")
axes[1].axvspan(base_slice.start, base_slice.stop, color="0.9", alpha=0.5)
axes[1].axvspan(post_slice.start, post_slice.stop, color="#ffddcc", alpha=0.25)
axes[1].axvline(50, color="k", lw=1, ls="--")
axes[1].set_xlabel("time (ms)")
axes[1].set_ylabel("effective dimensionality")

print(f"PC1 post mean, top-k:   {np.nanmean(top_pc1[post_slice]):.6f}")
print(f"PC1 post mean, random:  {np.nanmean(np.nanmean(rand_pc1, axis=0)[post_slice]):.6f}")
print(f"ED post mean, top-k:    {np.nanmean(top_ed[post_slice]):.6f}")
print(f"ED post mean, random:   {np.nanmean(np.nanmean(rand_ed, axis=0)[post_slice]):.6f}")

if SAVE:
    payload = {
        "target": TARGET,
        "roi_label": roi_label,
        "top_k": int(top_k),
        "alpha": float(ALPHA),
        "bin_size_ms": int(BIN_SIZE_MS),
        "n_random": int(N_RANDOM),
        "random_state": int(RANDOM_STATE),
        "pc1_topk": top_pc1,
        "pc1_random_mean": np.nanmean(rand_pc1, axis=0),
        "pc1_random_low": np.nanpercentile(rand_pc1, 2.5, axis=0),
        "pc1_random_high": np.nanpercentile(rand_pc1, 97.5, axis=0),
        "ed_topk": top_ed,
        "ed_random_mean": np.nanmean(rand_ed, axis=0),
        "ed_random_low": np.nanpercentile(rand_ed, 2.5, axis=0),
        "ed_random_high": np.nanpercentile(rand_ed, 97.5, axis=0),
    }
    s3_base = f"{pth.SAVEDIR}/eigenspectra/{TARGET}"
    with fsspec.open(f"{s3_base}.pkl", "wb") as f:
        pickle.dump(payload, f)
    with fsspec.open(f"{s3_base}.png", "wb") as f:
        fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}.svg", "w") as f:
        fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")
    download_dir = Path.home() / "Downloads"
    fig.savefig(download_dir / f"eigenspectra_{roi_label}.png", dpi=300, transparent=False, bbox_inches="tight")
