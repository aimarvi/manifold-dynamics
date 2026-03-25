from __future__ import annotations

import pickle
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


# Ad-hoc configuration
TARGET = "07.MF1.F"
ALPHA = 0.05
BIN_SIZE_MS = 20
N_RANDOM = 100
N_COMPONENTS = 3
T_START = 100
T_STOP = 270
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

resp = np.nanmean(raster_3d[:, T_START:T_STOP, :], axis=1)
R = resp.T

n_components = min(N_COMPONENTS, top_k, R.shape[1])
if n_components < 1:
    raise ValueError(f"Invalid number of subspace dimensions: {n_components}")
if n_components != N_COMPONENTS:
    vprint(f"Adjusted n_components from {N_COMPONENTS} to {n_components}")

rng = np.random.default_rng(RANDOM_STATE)

A_top = PCA(n_components=n_components).fit(R[idx_topk]).components_.T
A_all = PCA(n_components=n_components).fit(R).components_.T
angles_top_all_deg = np.degrees(subspace_angles(A_top, A_all))

random_idxs = np.stack(
    [rng.choice(candidate_idxs, size=top_k, replace=False) for _ in range(N_RANDOM)],
    axis=0,
)

angles_top_rand_deg = np.full((N_RANDOM, n_components), np.nan, dtype=float)
angles_rand_rand_deg = np.full((N_RANDOM, n_components), np.nan, dtype=float)
angles_all_rand_deg = np.full((N_RANDOM, n_components), np.nan, dtype=float)

for i, idx_rand in enumerate(random_idxs):
    A_rand = PCA(n_components=n_components).fit(R[idx_rand]).components_.T
    angles_top_rand_deg[i] = np.degrees(subspace_angles(A_top, A_rand))
    angles_all_rand_deg[i] = np.degrees(subspace_angles(A_all, A_rand))

    idx_rand_b = rng.choice(candidate_idxs, size=top_k, replace=False)
    A_rand_b = PCA(n_components=n_components).fit(R[idx_rand_b]).components_.T
    angles_rand_rand_deg[i] = np.degrees(subspace_angles(A_rand, A_rand_b))

vprint(f"Top-k vs all-images angles: {angles_top_all_deg}")
vprint(f"Top-vs-random mean angles: {angles_top_rand_deg.mean(axis=0)}")
vprint(f"Random-vs-random mean angles: {angles_rand_rand_deg.mean(axis=0)}")
vprint(f"All-images-vs-random mean angles: {angles_all_rand_deg.mean(axis=0)}")

top_rand_mean = angles_top_rand_deg.mean(axis=1)
rand_rand_mean = angles_rand_rand_deg.mean(axis=1)
all_rand_mean = angles_all_rand_deg.mean(axis=1)
top_all_mean = float(angles_top_all_deg.mean())

df_out = pd.DataFrame(
    {
        "mean_angle_deg": np.concatenate(
            [
                top_rand_mean,
                rand_rand_mean,
                all_rand_mean,
                np.repeat(top_all_mean, len(top_rand_mean)),
            ]
        ),
        "comparison": (
            ["top-rand"] * len(top_rand_mean)
            + ["rand-rand"] * len(rand_rand_mean)
            + ["all-rand"] * len(all_rand_mean)
            + ["top-all"] * len(top_rand_mean)
        ),
    }
)

summary_rows = [
    {
        "roi": roi_label,
        "comparison": "top-all",
        "mean_angle_deg": top_all_mean,
        "component_angles_deg": angles_top_all_deg.tolist(),
        "n_components": int(n_components),
        "top_k": int(top_k),
    },
    {
        "roi": roi_label,
        "comparison": "top-rand",
        "mean_angle_deg": float(top_rand_mean.mean()),
        "component_angles_deg": angles_top_rand_deg.mean(axis=0).tolist(),
        "n_components": int(n_components),
        "top_k": int(top_k),
    },
    {
        "roi": roi_label,
        "comparison": "rand-rand",
        "mean_angle_deg": float(rand_rand_mean.mean()),
        "component_angles_deg": angles_rand_rand_deg.mean(axis=0).tolist(),
        "n_components": int(n_components),
        "top_k": int(top_k),
    },
    {
        "roi": roi_label,
        "comparison": "all-rand",
        "mean_angle_deg": float(all_rand_mean.mean()),
        "component_angles_deg": angles_all_rand_deg.mean(axis=0).tolist(),
        "n_components": int(n_components),
        "top_k": int(top_k),
    },
]
df_summary = pd.DataFrame(summary_rows)

print(df_summary.to_string(index=False))

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
sns.boxplot(
    data=df_out,
    x="comparison",
    y="mean_angle_deg",
    order=["top-rand", "rand-rand", "all-rand", "top-all"],
    ax=ax,
)
ax.set_ylabel("Mean principal angle (deg)")
ax.set_xlabel("")
ax.set_title(roi_label)
fig.tight_layout()

if SAVE:
    s3_base = f"{pth.SAVEDIR}/dynamic_modes/static_subspace/{TARGET}"
    payload = {
        "roi": roi_label,
        "target": TARGET,
        "top_k": int(top_k),
        "n_components": int(n_components),
        "t_start": int(T_START),
        "t_stop": int(T_STOP),
        "angles_top_all_deg": angles_top_all_deg,
        "angles_top_rand_deg": angles_top_rand_deg,
        "angles_rand_rand_deg": angles_rand_rand_deg,
        "angles_all_rand_deg": angles_all_rand_deg,
        "df_out": df_out,
        "df_summary": df_summary,
    }
    with fsspec.open(f"{s3_base}.pkl", "wb") as f:
        pickle.dump(payload, f)
    with fsspec.open(f"{s3_base}.png", "wb") as f:
        fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}.svg", "w") as f:
        fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")

download_png = Path.home() / "Downloads" / f"static_subspace_{TARGET}.png"
fig.savefig(download_png, dpi=300, transparent=False, bbox_inches="tight")
