from __future__ import annotations

import pickle
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut


# Ad-hoc configuration
ROI_TARGET = "07.MF1.F" # "19.Unknown.F", "07.MF1.F"
BIN_SIZE_MS = 20
ALPHA = 0.05
L2_STEP = 5
K_MAX = 200
RANDOM_STATE = 0
VERBOSE = True
SAVE = True

def vprint(msg: str) -> None:
    if VERBOSE:
        print(msg)

vprint(f"Loading raster data for {ROI_TARGET}...")
raster_4d = nu.significant_trial_raster(roi_uid=ROI_TARGET, alpha=ALPHA, bin_size_ms=BIN_SIZE_MS)
X = np.nanmean(raster_4d, axis=3)  # (units, time, images)
order = np.asarray(tut.rank_images_by_response(X), dtype=int)
sizes_top = [k for k in range(2, min(K_MAX, X.shape[2]) + 1)]
top_rdms = []
for k in sizes_top:
    R, _ = tut.tuning_rdm(X=X, indices=order[:k], tstart=100, tend=350, metric="correlation")
    top_rdms.append(R)
roi_cache = {
    "sizes_top": sizes_top,
    "top_rdms": top_rdms,
}

cols = ["ROI", "Scale", "Derivative", "Mode"]
diffs = pd.DataFrame(columns=cols)
sizes = roi_cache["sizes_top"]
rdms = roi_cache["top_rdms"]
triu = np.triu_indices_from(rdms[0], k=1)

R0 = np.mean(np.array([rdm[triu] for rdm in rdms[0:L2_STEP]]), axis=0)
for t in np.arange(1 * L2_STEP, len(rdms), L2_STEP):
    prev = R0
    _ = prev
    R0 = np.mean(np.array([rdm[triu] for rdm in rdms[t:t + L2_STEP]]), axis=0)
    diff = np.sqrt(np.sum((R0) ** 2))
    diffs.loc[len(diffs)] = {"ROI": ROI_TARGET, "Scale": sizes[t - 1], "Derivative": diff, "Mode": "top"}

diffs["diff_smooth"] = diffs["Derivative"].groupby(diffs["Mode"]).transform(
    lambda v: gaussian_filter1d(v, sigma=1)
)

mins = {}
fig, ax = plt.subplots(1, 1, figsize=(3,2))
d = diffs[diffs["ROI"] == ROI_TARGET]

sns.lineplot(data=d, x="Scale", y="Derivative", hue="Mode", palette=sns.color_palette("husl"), alpha=0.5, ax=ax)
sns.lineplot(data=d, x="Scale", y="diff_smooth", hue="Mode", palette=sns.color_palette("husl"), ax=ax)

labels = list(ax.get_legend_handles_labels()[1])
dm = d[d["Mode"] == "top"]
idx_min = dm["diff_smooth"].idxmin()
if not np.isnan(idx_min):
    x_min = dm.loc[idx_min, "Scale"]
    y_min = dm.loc[idx_min, "diff_smooth"]
    mins[ROI_TARGET] = (x_min, y_min)
    ax.scatter(x_min, y_min, color="red", marker="|", zorder=5)
    labels[0] = f"top min @ {int(x_min)}"
vprint(mins)

ax.legend(ax.get_legend_handles_labels()[0], labels, frameon=False).remove()
ax.set_title("")
ax.set_ylabel("")
sns.despine(fig=fig, trim=True, offset=5)
plt.tight_layout()

if SAVE:
    s3_base = f"{pth.SAVEDIR}/timextime/l2-norm/{ROI_TARGET}"
    with fsspec.open(f"{s3_base}.png", "wb") as f:
        fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}.svg", "w") as f:
        fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")
    download_png = Path.home() / "Downloads" / f"l2_norm_{ROI_TARGET}.png"
    fig.savefig(download_png, dpi=300, transparent=False, bbox_inches="tight")

plt.close(fig)

if SAVE:
    payload = {
        "config": {
            "roi": ROI_TARGET,
            "alpha": ALPHA,
            "bin_size_ms": BIN_SIZE_MS,
            "l2_step": L2_STEP,
            "k_max": K_MAX,
            "random_state": RANDOM_STATE,
        },
        "roi_cache": roi_cache,
        "diffs": diffs,
        "mins": mins,
    }
    s3_pkl = f"{pth.SAVEDIR}/timextime/l2_norm/{ROI_TARGET}.pkl"
    with fsspec.open(s3_pkl, "wb") as f:
        pickle.dump(payload, f)
    download_pkl = Path.home() / "Downloads" / f"{ROI_TARGET}.pkl"
    with open(download_pkl, "wb") as f:
        pickle.dump(payload, f)
