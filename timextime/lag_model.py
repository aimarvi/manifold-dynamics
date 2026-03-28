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
import visionlab_utils.storage as vst


TARGET = "08.MF1.F"
ALPHA = 0.05
BIN_SIZE_MS = 20
TSTART = 100
TEND = 350
N_BOOTSTRAP = 100
RANDOM_STATE = 0
SAVE = False
VERBOSE = True


def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)


# This model keeps only lag structure by averaging each diagonal and reconstructing
# a matrix whose values depend only on time lag from the main diagonal.
def lag_model(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    M_hat = np.zeros_like(M)

    for k in range(n):
        diag = np.diag(M, k=k)
        mean = float(np.nanmean(diag))
        M_hat += np.diag(np.full(len(diag), mean), k=k)
        if k > 0:
            M_hat += np.diag(np.full(len(diag), mean), k=-k)

    return M_hat


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

raster_4d = nu.significant_trial_raster(
    roi_uid=TARGET,
    alpha=ALPHA,
    bin_size_ms=BIN_SIZE_MS,
)
raster_3d = np.nanmean(raster_4d, axis=3)
image_order = np.asarray(tut.rank_images_by_response(raster_3d), dtype=int)
idx_topk = image_order[:top_k]
idx_all = np.arange(raster_3d.shape[2], dtype=int)
rng = np.random.default_rng(RANDOM_STATE)

vprint(f"Resolved ROI target: {TARGET}")
vprint(f"Using top-k = {top_k}")
vprint(f"Responsive raster shape: {raster_4d.shape}")
vprint(f"Trial-averaged PSTH shape: {raster_3d.shape}")

R_all, _ = tut.tuning_rdm(
    X=raster_3d,
    indices=idx_all,
    tstart=TSTART,
    tend=TEND,
    metric="correlation",
)
R_topk, _ = tut.tuning_rdm(
    X=raster_3d,
    indices=idx_topk,
    tstart=TSTART,
    tend=TEND,
    metric="correlation",
)

R_all_hat = lag_model(R_all)
R_topk_hat = lag_model(R_topk)

x = R_all.reshape(-1)
y = R_all_hat.reshape(-1)
m = np.isfinite(x) & np.isfinite(y)
r_all = float(np.corrcoef(x[m], y[m])[0, 1])
r2_all = float(r_all ** 2)

x = R_topk.reshape(-1)
y = R_topk_hat.reshape(-1)
m = np.isfinite(x) & np.isfinite(y)
r_topk = float(np.corrcoef(x[m], y[m])[0, 1])
r2_topk = float(r_topk ** 2)

random_rows = []
for i in range(N_BOOTSTRAP):
    idx_random = rng.choice(raster_3d.shape[2], size=top_k, replace=False)
    R_random, _ = tut.tuning_rdm(
        X=raster_3d,
        indices=idx_random,
        tstart=TSTART,
        tend=TEND,
        metric="correlation",
    )
    R_random_hat = lag_model(R_random)
    x = R_random.reshape(-1)
    y = R_random_hat.reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    r_random = float(np.corrcoef(x[m], y[m])[0, 1])
    random_rows.append((r_random, float(r_random ** 2)))

random_rows = np.asarray(random_rows, dtype=float)

print(f"roi: {roi_label}")
print(f"top_k: {top_k}")
print(f"all     r={r_all:.6f}  r2={r2_all:.6f}")
print(f"top-k   r={r_topk:.6f}  r2={r2_topk:.6f}")
print(f"random  r_mean={np.nanmean(random_rows[:, 0]):.6f}  r_sd={np.nanstd(random_rows[:, 0], ddof=1):.6f}")
print(f"random  r2_mean={np.nanmean(random_rows[:, 1]):.6f}  r2_sd={np.nanstd(random_rows[:, 1], ddof=1):.6f}")

fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
cmap = sns.color_palette("rocket", as_cmap=True)
vmin = float(np.nanmin([np.nanmin(R_all), np.nanmin(R_topk)]))
vmax = float(np.nanmax([np.nanmax(R_all), np.nanmax(R_topk)]))

sns.heatmap(R_all, ax=axes[0, 0], square=True, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False)
axes[0, 0].set_title(f"All original | r={r_all:.3f}")
axes[0, 0].set_xlabel("time")
axes[0, 0].set_ylabel("time")

sns.heatmap(R_all_hat, ax=axes[0, 1], square=True, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False)
axes[0, 1].set_title(f"All lag model | r2={r2_all:.3f}")
axes[0, 1].set_xlabel("time")
axes[0, 1].set_ylabel("time")

sns.heatmap(R_topk, ax=axes[1, 0], square=True, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False)
axes[1, 0].set_title(f"Top-k original | r={r_topk:.3f}")
axes[1, 0].set_xlabel("time")
axes[1, 0].set_ylabel("time")

sns.heatmap(R_topk_hat, ax=axes[1, 1], square=True, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False)
axes[1, 1].set_title(f"Top-k lag model | r2={r2_topk:.3f}")
axes[1, 1].set_xlabel("time")
axes[1, 1].set_ylabel("time")

if SAVE:
    s3_base = f"{pth.SAVEDIR}/timextime/lag_model/{TARGET}"
    with fsspec.open(f"{s3_base}.png", "wb") as f:
        fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}.svg", "w") as f:
        fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")
    vprint(f"Saved figure to {s3_base}.png")
    vprint(f"Saved figure to {s3_base}.svg")

local_png = Path.home() / "Downloads" / f"lag_model_{TARGET.replace('.', '_')}.png"
fig.savefig(local_png, dpi=300, transparent=False, bbox_inches="tight")
plt.close(fig)
vprint(f"Saved figure to {local_png}")
