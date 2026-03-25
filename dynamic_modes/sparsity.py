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


TARGET = "07.MF1.F"
ALPHA = 0.05
BIN_SIZE_MS = 10
N_BOOTSTRAP = 100
RANDOM_STATE = 0
SAVE = True
VERBOSE = True


def vprint(msg: str) -> None:
    """Print only when VERBOSE is enabled."""
    if VERBOSE:
        print(msg)


topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
with open(topk_local, "rb") as f:
    topk_vals = pickle.load(f)

if TARGET not in topk_vals:
    raise ValueError(f"No top-k entry found for ROI: {TARGET}")
top_k = int(topk_vals[TARGET]["k"])

raster_4d = nu.significant_trial_raster(
    roi_uid=TARGET,
    alpha=ALPHA,
    bin_size_ms=BIN_SIZE_MS,
)
split_a = raster_4d[:, :, :, 0::2]
split_b = raster_4d[:, :, :, 1::2]

rng = np.random.default_rng(RANDOM_STATE)
split_curves = {"A": {}, "B": {}}
for split_name, split_raster in [("A", split_a), ("B", split_b)]:
    pvals = srs.is_responsive(
        X=split_raster,
        roi_uid=TARGET,
        test_type="paired",
        bin_ms=1,
    ).squeeze()
    responsive = np.isfinite(pvals) & (pvals < ALPHA)
    if int(np.sum(responsive)) == 0:
        raise ValueError(f"{TARGET} split {split_name}: no responsive units.")

    X = np.nanmean(split_raster, axis=3)[responsive]
    image_order = tut.rank_images_by_response(X)
    top_indices = np.asarray(image_order[:top_k], dtype=int)
    random_sets = np.stack(
        [rng.choice(X.shape[2], size=top_k, replace=False) for _ in range(N_BOOTSTRAP)],
        axis=0,
    )

    top_image_sparsity = np.full(X.shape[1], np.nan, dtype=float)
    top_population_sparsity = np.full(X.shape[1], np.nan, dtype=float)
    top_psth = np.full(X.shape[1], np.nan, dtype=float)

    random_image_sparsity = np.full((N_BOOTSTRAP, X.shape[1]), np.nan, dtype=float)
    random_population_sparsity = np.full((N_BOOTSTRAP, X.shape[1]), np.nan, dtype=float)
    random_psth = np.full((N_BOOTSTRAP, X.shape[1]), np.nan, dtype=float)

    for t in range(X.shape[1]):
        Xt = X[:, t, :]
        Xt_use = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
        Xt_top = Xt_use[:, top_indices]
#         lower = np.nanpercentile(Xt, 2.5, axis=1, keepdims=True)
#         upper = np.nanpercentile(Xt, 97.5, axis=1, keepdims=True)
#         Xt_norm = (Xt - lower) / (upper - lower + 1e-8)
#         Xt_norm = np.nan_to_num(Xt_norm, nan=0.0, posinf=0.0, neginf=0.0)

#         Xt_top = Xt_norm[:, top_indices]
        top_image_vals = tut.treves_rolls_sparsity(Xt_top, axis=1)
        top_population_vals = tut.treves_rolls_sparsity(Xt_top, axis=0)
        top_image_sparsity[t] = float(np.nanmean(top_image_vals))
        top_population_sparsity[t] = float(np.nanmean(top_population_vals))
        top_psth[t] = float(np.nanmean(X[:, t, top_indices]))

        for i, idx in enumerate(random_sets):
            Xt_rand = Xt_use[:, idx]
#             Xt_rand = Xt_norm[:, idx]
            rand_image_vals = tut.treves_rolls_sparsity(Xt_rand, axis=1)
            rand_population_vals = tut.treves_rolls_sparsity(Xt_rand, axis=0)
            random_image_sparsity[i, t] = float(np.nanmean(rand_image_vals))
            random_population_sparsity[i, t] = float(np.nanmean(rand_population_vals))
            random_psth[i, t] = float(np.nanmean(X[:, t, idx]))

    split_curves[split_name]["top_image"] = top_image_sparsity
    split_curves[split_name]["top_population"] = top_population_sparsity
    split_curves[split_name]["top_psth"] = top_psth
    split_curves[split_name]["random_image"] = random_image_sparsity
    split_curves[split_name]["random_population"] = random_population_sparsity
    split_curves[split_name]["random_psth"] = random_psth
    split_curves[split_name]["n_units"] = int(np.sum(responsive))


def minmax_timecourse(x: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D timecourse to [0, 1]."""
    x = np.asarray(x, dtype=float)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return np.full_like(x, np.nan, dtype=float)
    return (x - xmin) / (xmax - xmin)


top_image_mean = 0.5 * (split_curves["A"]["top_image"] + split_curves["B"]["top_image"])
top_population_mean = 0.5 * (
    split_curves["A"]["top_population"] + split_curves["B"]["top_population"]
)
top_psth_mean = 0.5 * (split_curves["A"]["top_psth"] + split_curves["B"]["top_psth"])

random_image_mean = 0.5 * (
    split_curves["A"]["random_image"] + split_curves["B"]["random_image"]
)
random_population_mean = 0.5 * (
    split_curves["A"]["random_population"] + split_curves["B"]["random_population"]
)
random_psth_mean = 0.5 * (split_curves["A"]["random_psth"] + split_curves["B"]["random_psth"])

top_image_plot = minmax_timecourse(top_image_mean)
top_population_plot = minmax_timecourse(top_population_mean)
top_psth_plot = minmax_timecourse(top_psth_mean)

random_image_plot = np.array([minmax_timecourse(x) for x in random_image_mean])
random_population_plot = np.array([minmax_timecourse(x) for x in random_population_mean])
random_psth_plot = np.array([minmax_timecourse(x) for x in random_psth_mean])

random_image_plot_mean = np.nanmean(random_image_plot, axis=0)
random_image_plot_sem = np.nanstd(random_image_plot, axis=0, ddof=1) / np.sqrt(N_BOOTSTRAP)
random_population_plot_mean = np.nanmean(random_population_plot, axis=0)
random_population_plot_sem = np.nanstd(random_population_plot, axis=0, ddof=1) / np.sqrt(N_BOOTSTRAP)
random_psth_plot_mean = np.nanmean(random_psth_plot, axis=0)
random_psth_plot_sem = np.nanstd(random_psth_plot, axis=0, ddof=1) / np.sqrt(N_BOOTSTRAP)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True)
time = np.arange(top_image_mean.shape[0])

axes[0, 0].plot(time, top_psth_plot, color="0.35", lw=2, label="PSTH")
axes[0, 0].plot(time, top_image_plot, color="#1f77b4", lw=2.5, label="image sparsity")
axes[0, 0].set_title("Top-k image sparsity")

axes[0, 1].fill_between(
    time,
    random_psth_plot_mean - random_psth_plot_sem,
    random_psth_plot_mean + random_psth_plot_sem,
    color="0.75",
    alpha=0.35,
)
axes[0, 1].plot(time, random_psth_plot_mean, color="0.35", lw=2, label="PSTH")
axes[0, 1].fill_between(
    time,
    random_image_plot_mean - random_image_plot_sem,
    random_image_plot_mean + random_image_plot_sem,
    color="#1f77b4",
    alpha=0.18,
)
axes[0, 1].plot(time, random_image_plot_mean, color="#1f77b4", lw=2.5, label="image sparsity")
axes[0, 1].set_title("Random-k image sparsity")

axes[1, 0].plot(time, top_psth_plot, color="0.35", lw=2, label="PSTH")
axes[1, 0].plot(time, top_population_plot, color="#ff7f0e", lw=2.5, label="population sparsity")
axes[1, 0].set_title("Top-k population sparsity")

axes[1, 1].fill_between(
    time,
    random_psth_plot_mean - random_psth_plot_sem,
    random_psth_plot_mean + random_psth_plot_sem,
    color="0.75",
    alpha=0.35,
)
axes[1, 1].plot(time, random_psth_plot_mean, color="0.35", lw=2, label="PSTH")
axes[1, 1].fill_between(
    time,
    random_population_plot_mean - random_population_plot_sem,
    random_population_plot_mean + random_population_plot_sem,
    color="#ff7f0e",
    alpha=0.18,
)
axes[1, 1].plot(time, random_population_plot_mean, color="#ff7f0e", lw=2.5, label="population sparsity")
axes[1, 1].set_title("Random-k population sparsity")

for ax in axes.flat:
    ax.axvline(srs.ONSET_TIME, color="k", lw=1, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("normalized value")
    ax.legend(frameon=False, fontsize=8)

axes[1, 0].set_xlabel("time (ms)")
axes[1, 1].set_xlabel("time (ms)")
fig.suptitle(f"{TARGET}: normalized sparsity and PSTH", fontsize=14)

base_sl = srs._ms_to_slice(srs.ONSET_TIME, srs.BASE_WIN_MS, bin_ms=1)
post_sl = srs._ms_to_slice(srs.ONSET_TIME, srs.RESP_WIN_MS, bin_ms=1)

print(f"{TARGET}: top-k = {top_k}")
print(
    "top image sparsity "
    f"baseline={np.nanmean(top_image_mean[base_sl]):.6f} "
    f"post={np.nanmean(top_image_mean[post_sl]):.6f}"
)
print(
    "random image sparsity "
    f"baseline={np.nanmean(np.nanmean(random_image_mean[:, base_sl], axis=0)):.6f} "
    f"post={np.nanmean(np.nanmean(random_image_mean[:, post_sl], axis=0)):.6f}"
)
print(
    "top population sparsity "
    f"baseline={np.nanmean(top_population_mean[base_sl]):.6f} "
    f"post={np.nanmean(top_population_mean[post_sl]):.6f}"
)
print(
    "random population sparsity "
    f"baseline={np.nanmean(np.nanmean(random_population_mean[:, base_sl], axis=0)):.6f} "
    f"post={np.nanmean(np.nanmean(random_population_mean[:, post_sl], axis=0)):.6f}"
)

if SAVE:
    s3_base = f"{pth.SAVEDIR}/dynamic_modes/sparsity_{TARGET}"
    payload = {
        "target": TARGET,
        "top_k": top_k,
        "n_bootstrap": N_BOOTSTRAP,
        "top_image_mean": top_image_mean,
        "top_population_mean": top_population_mean,
        "top_psth_mean": top_psth_mean,
        "random_image_mean": np.nanmean(random_image_mean, axis=0),
        "random_image_sem": np.nanstd(random_image_mean, axis=0, ddof=1) / np.sqrt(N_BOOTSTRAP),
        "random_population_mean": np.nanmean(random_population_mean, axis=0),
        "random_population_sem": np.nanstd(random_population_mean, axis=0, ddof=1) / np.sqrt(N_BOOTSTRAP),
        "random_psth_mean": np.nanmean(random_psth_mean, axis=0),
        "random_psth_sem": np.nanstd(random_psth_mean, axis=0, ddof=1) / np.sqrt(N_BOOTSTRAP),
        "n_units_A": split_curves["A"]["n_units"],
        "n_units_B": split_curves["B"]["n_units"],
    }
    with fsspec.open(f"{s3_base}.pkl", "wb") as f:
        pickle.dump(payload, f)
    with fsspec.open(f"{s3_base}.png", "wb") as f:
        fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}.svg", "w") as f:
        fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")

    download_dir = Path.home() / "Downloads"
    fig.savefig(download_dir / f"sparsity_{TARGET}.png", dpi=300, transparent=False, bbox_inches="tight")
