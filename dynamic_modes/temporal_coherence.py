from __future__ import annotations

import pickle
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


# -----------------------------------------------------------------------------
# Ad Hoc Configuration
# -----------------------------------------------------------------------------

ROI_TARGETS = ["19.Unknown.F", "07.MF1.F", "08.MF1.F", "09.MF1.F"]
EXAMPLE_TARGET = "19.Unknown.F"
ALPHA = 0.05
BIN_SIZE_MS = 20
TSTART = 100
TEND = 350
N_RANDOM = 100
RANDOM_STATE = 0
MAX_LAG = None
SAVE = True
VERBOSE = True


def vprint(msg: str) -> None:
    """Print only when VERBOSE is enabled."""
    if VERBOSE:
        print(msg)


topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
with open(topk_local, "rb") as f:
    topk_vals = pickle.load(f)

rng = np.random.default_rng(RANDOM_STATE)
payloads = []

for target in ROI_TARGETS:
    top_k = int(topk_vals[target]["k"])
    raster_4d = nu.significant_trial_raster(
        roi_uid=target,
        alpha=ALPHA,
        bin_size_ms=BIN_SIZE_MS,
    )
    X = np.nanmean(raster_4d, axis=3)
    image_order = tut.rank_images_by_response(X)
    _, n_time_total, n_images = X.shape
    if TEND > n_time_total:
        raise ValueError(f"{target}: TEND {TEND} exceeds available timepoints {n_time_total}")
    if top_k > n_images:
        raise ValueError(f"{target}: top-k {top_k} exceeds n_images {n_images}")

    idx_topk = np.asarray(image_order[:top_k], dtype=int)
    random_sets = np.stack(
        [rng.choice(n_images, size=top_k, replace=False) for _ in range(N_RANDOM)],
        axis=0,
    )

    time = np.arange(TSTART, TEND)
    n_time = len(time)
    n_pairs = int(top_k * (top_k - 1) / 2)
    max_components = min(n_pairs, n_time)
    max_lag = MAX_LAG if MAX_LAG is not None else max(1, n_time // 3)
    max_lag = min(max_lag, n_time - 1)

    vprint(f"{target}: X shape={X.shape}, top-k={top_k}, time=[{TSTART},{TEND})")

    top_varfrac = None
    top_cumvar = None
    top_tpcs = None
    top_l_total = np.nan
    top_l_norm = np.nan
    top_auc = np.nan
    top_half = np.nan
    top_mean_turn = np.nan
    top_median_turn = np.nan
    top_step_tc = None
    top_angle_tc = None
    top_autocorr = None

    rand_varfrac = np.full((N_RANDOM, max_components), np.nan, dtype=np.float64)
    rand_cumvar = np.full((N_RANDOM, max_components), np.nan, dtype=np.float64)
    rand_pc1 = np.full(N_RANDOM, np.nan, dtype=np.float64)
    rand_pc3 = np.full(N_RANDOM, np.nan, dtype=np.float64)
    rand_n80 = np.full(N_RANDOM, np.nan, dtype=np.float64)
    rand_n90 = np.full(N_RANDOM, np.nan, dtype=np.float64)
    rand_autocorr = np.full((N_RANDOM, max_lag), np.nan, dtype=np.float64)
    rand_l_total = np.full(N_RANDOM, np.nan, dtype=np.float64)
    rand_l_norm = np.full(N_RANDOM, np.nan, dtype=np.float64)
    rand_auc = np.full(N_RANDOM, np.nan, dtype=np.float64)
    rand_half = np.full(N_RANDOM, np.nan, dtype=np.float64)
    rand_mean_turn = np.full(N_RANDOM, np.nan, dtype=np.float64)
    rand_median_turn = np.full(N_RANDOM, np.nan, dtype=np.float64)
    rand_step_tc = np.full((N_RANDOM, n_time - 1), np.nan, dtype=np.float64)
    rand_angle_tc = np.full((N_RANDOM, n_time - 2), np.nan, dtype=np.float64)

    for set_i, idx in enumerate([idx_topk, *random_sets]):
        rdv_time = np.full((n_time, n_pairs), np.nan, dtype=np.float64)
        for j, t in enumerate(time):
            M = X[:, t, idx]
            M = M - np.nanmean(M, axis=1, keepdims=True)
            M = M / (np.nanstd(M, axis=1, keepdims=True) + 1e-8)
            M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
            rdv_time[j] = pdist(M.T, metric="correlation")

        R = rdv_time.T
        R = R - np.mean(R, axis=1, keepdims=True)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        _, s, vt = np.linalg.svd(R, full_matrices=False)
        s2 = s ** 2
        total = float(np.sum(s2))
        frac = s2 / total if total > 0 else np.full_like(s2, np.nan)
        cum = np.cumsum(frac) if total > 0 else np.full_like(s2, np.nan)

        steps = np.diff(rdv_time, axis=0)
        step_norms = np.linalg.norm(steps, axis=1)
        l_total = float(np.sum(step_norms))
        l_net = float(np.linalg.norm(rdv_time[-1] - rdv_time[0]))
        l_norm = float(l_total / l_net) if l_net > 0 else np.nan

        autocorr = np.full(max_lag, np.nan, dtype=np.float64)
        for lag in range(1, max_lag + 1):
            vals = []
            for t0 in range(n_time - lag):
                a = rdv_time[t0] - np.mean(rdv_time[t0])
                b = rdv_time[t0 + lag] - np.mean(rdv_time[t0 + lag])
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                if denom > 0:
                    vals.append(float(np.dot(a, b) / denom))
            if vals:
                autocorr[lag - 1] = float(np.mean(vals))

        auc = float(np.trapezoid(np.nan_to_num(autocorr, nan=0.0), dx=1.0))
        below = np.where(autocorr < 0.5)[0]
        half = float(below[0] + 1) if below.size > 0 else float(max_lag)

        angles = np.full(n_time - 2, np.nan, dtype=np.float64)
        for t0 in range(steps.shape[0] - 1):
            v1 = steps[t0]
            v2 = steps[t0 + 1]
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom > 0:
                c = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
                angles[t0] = float(np.arccos(c))

        if set_i == 0:
            top_varfrac = frac
            top_cumvar = cum
            top_tpcs = vt[: min(5, len(vt))]
            top_l_total = l_total
            top_l_norm = l_norm
            top_auc = auc
            top_half = half
            top_mean_turn = float(np.nanmean(angles))
            top_median_turn = float(np.nanmedian(angles))
            top_step_tc = step_norms
            top_angle_tc = angles
            top_autocorr = autocorr
        else:
            i = set_i - 1
            rand_varfrac[i, : frac.size] = frac
            rand_cumvar[i, : cum.size] = cum
            rand_pc1[i] = float(frac[0])
            rand_pc3[i] = float(np.sum(frac[: min(3, frac.size)]))
            rand_n80[i] = float(np.searchsorted(cum, 0.80) + 1) if np.isfinite(cum).any() else np.nan
            rand_n90[i] = float(np.searchsorted(cum, 0.90) + 1) if np.isfinite(cum).any() else np.nan
            rand_autocorr[i] = autocorr
            rand_l_total[i] = l_total
            rand_l_norm[i] = l_norm
            rand_auc[i] = auc
            rand_half[i] = half
            rand_mean_turn[i] = float(np.nanmean(angles))
            rand_median_turn[i] = float(np.nanmedian(angles))
            rand_step_tc[i] = step_norms
            rand_angle_tc[i] = angles

    payloads.append(
        {
            "target": target,
            "top_k": top_k,
            "time": time,
            "topk": {
                "variance_fraction": top_varfrac,
                "cumulative_variance": top_cumvar,
                "pc1_variance": float(top_varfrac[0]),
                "pc3_variance": float(np.sum(top_varfrac[:3])),
                "n_components_80": float(np.searchsorted(top_cumvar, 0.80) + 1),
                "n_components_90": float(np.searchsorted(top_cumvar, 0.90) + 1),
                "temporal_components": top_tpcs,
                "L_total": top_l_total,
                "L_norm": top_l_norm,
                "autocorr_curve": top_autocorr,
                "AUC_autocorr": top_auc,
                "half_decay_lag": top_half,
                "step_size_timecourse": top_step_tc,
                "turn_angle_timecourse": top_angle_tc,
                "mean_turn_angle": top_mean_turn,
                "median_turn_angle": top_median_turn,
            },
            "random": {
                "variance_fraction_mean": np.nanmean(rand_varfrac, axis=0),
                "variance_fraction_low": np.nanpercentile(rand_varfrac, 2.5, axis=0),
                "variance_fraction_high": np.nanpercentile(rand_varfrac, 97.5, axis=0),
                "cumulative_variance_mean": np.nanmean(rand_cumvar, axis=0),
                "cumulative_variance_low": np.nanpercentile(rand_cumvar, 2.5, axis=0),
                "cumulative_variance_high": np.nanpercentile(rand_cumvar, 97.5, axis=0),
                "pc1_variance_mean": float(np.nanmean(rand_pc1)),
                "pc3_variance_mean": float(np.nanmean(rand_pc3)),
                "n_components_80_mean": float(np.nanmean(rand_n80)),
                "n_components_90_mean": float(np.nanmean(rand_n90)),
                "autocorr_curve_mean": np.nanmean(rand_autocorr, axis=0),
                "autocorr_curve_low": np.nanpercentile(rand_autocorr, 2.5, axis=0),
                "autocorr_curve_high": np.nanpercentile(rand_autocorr, 97.5, axis=0),
                "L_total_mean": float(np.nanmean(rand_l_total)),
                "L_norm_mean": float(np.nanmean(rand_l_norm)),
                "AUC_autocorr_mean": float(np.nanmean(rand_auc)),
                "half_decay_lag_mean": float(np.nanmean(rand_half)),
                "mean_turn_angle_mean": float(np.nanmean(rand_mean_turn)),
                "median_turn_angle_mean": float(np.nanmean(rand_median_turn)),
                "step_size_timecourse_mean": np.nanmean(rand_step_tc, axis=0),
                "step_size_timecourse_low": np.nanpercentile(rand_step_tc, 2.5, axis=0),
                "step_size_timecourse_high": np.nanpercentile(rand_step_tc, 97.5, axis=0),
                "turn_angle_timecourse_mean": np.nanmean(rand_angle_tc, axis=0),
                "turn_angle_timecourse_low": np.nanpercentile(rand_angle_tc, 2.5, axis=0),
                "turn_angle_timecourse_high": np.nanpercentile(rand_angle_tc, 97.5, axis=0),
            },
        }
    )

n_components = min(len(p["topk"]["variance_fraction"]) for p in payloads)
example_payload = next(p for p in payloads if p["target"] == EXAMPLE_TARGET)
time = example_payload["time"]
lags = np.arange(1, len(example_payload["topk"]["autocorr_curve"]) + 1)
step_time = np.arange(TSTART, TEND - 1)
angle_time = np.arange(TSTART + 1, TEND - 1)

top_cum = np.stack([p["topk"]["cumulative_variance"][:n_components] for p in payloads], axis=0)
rand_cum = np.stack([p["random"]["cumulative_variance_mean"][:n_components] for p in payloads], axis=0)
top_var = np.stack([p["topk"]["variance_fraction"][:n_components] for p in payloads], axis=0)
rand_var = np.stack([p["random"]["variance_fraction_mean"][:n_components] for p in payloads], axis=0)

fig_cum, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
for ax, payload in zip(axes.flat, payloads):
    pcs = np.arange(1, n_components + 1)
    ax.fill_between(
        pcs,
        payload["random"]["cumulative_variance_low"][:n_components],
        payload["random"]["cumulative_variance_high"][:n_components],
        color="0.75",
        alpha=0.4,
    )
    ax.plot(
        pcs,
        payload["random"]["cumulative_variance_mean"][:n_components],
        color="0.35",
        lw=2,
        label="Random-k mean",
    )
    ax.plot(
        pcs,
        payload["topk"]["cumulative_variance"][:n_components],
        color="#1f77b4",
        lw=2.5,
        label="Top-k",
    )
    ax.axhline(0.80, color="#ff7f0e", lw=1, ls="--")
    ax.axhline(0.90, color="#d62728", lw=1, ls="--")
    ax.set_title(payload["target"])
    ax.set_xlabel("temporal PC")
    ax.set_ylabel("cumulative variance explained")
    ax.set_ylim(0, 1.02)
axes.flat[0].legend(frameon=False, loc="lower right")
fig_cum.suptitle("Temporal PCA cumulative variance by ROI", fontsize=14)

fig_pair, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
for ax, (metric_name, ylabel) in zip(
    axes,
    [
        ("pc1_variance", "PC1 variance"),
        ("pc3_variance", "Top-3 variance"),
        ("n_components_80", "PCs for 80%"),
        ("n_components_90", "PCs for 90%"),
    ],
):
    for payload in payloads:
        top_val = payload["topk"][metric_name]
        rand_val = payload["random"][f"{metric_name}_mean"]
        ax.plot([0, 1], [top_val, rand_val], color="0.75", lw=1)
        ax.scatter([0], [top_val], color="#1f77b4", s=35)
        ax.scatter([1], [rand_val], color="0.35", s=35)
        ax.text(1.03, rand_val, payload["target"], fontsize=7, va="center")
    ax.set_xticks([0, 1], ["top-k", "random"])
    ax.set_title(ylabel)
    ax.set_ylabel(ylabel)
fig_pair.suptitle("Temporal PCA summary across ROIs", fontsize=14)

fig_avg, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
pcs = np.arange(1, n_components + 1)
axes[0].fill_between(
    pcs,
    np.nanpercentile(rand_cum, 2.5, axis=0),
    np.nanpercentile(rand_cum, 97.5, axis=0),
    color="0.75",
    alpha=0.4,
)
axes[0].plot(pcs, np.nanmean(rand_cum, axis=0), color="0.35", lw=2, label="Random-k mean")
axes[0].plot(pcs, np.nanmean(top_cum, axis=0), color="#1f77b4", lw=2.5, label="Top-k")
axes[0].axhline(0.80, color="#ff7f0e", lw=1, ls="--")
axes[0].axhline(0.90, color="#d62728", lw=1, ls="--")
axes[0].set_xlabel("temporal PC")
axes[0].set_ylabel("cumulative variance explained")
axes[0].set_title("Across-ROI cumulative variance")
axes[0].legend(frameon=False, loc="lower right")
axes[1].fill_between(
    pcs,
    np.nanpercentile(rand_var, 2.5, axis=0),
    np.nanpercentile(rand_var, 97.5, axis=0),
    color="0.75",
    alpha=0.4,
)
axes[1].plot(pcs, np.nanmean(rand_var, axis=0), color="0.35", lw=2, label="Random-k mean")
axes[1].plot(pcs, np.nanmean(top_var, axis=0), color="#1f77b4", lw=2.5, label="Top-k")
axes[1].set_xlabel("temporal PC")
axes[1].set_ylabel("variance explained")
axes[1].set_title("Across-ROI variance spectrum")
fig_avg.suptitle("Across-ROI temporal PCA curves", fontsize=14)

fig_example, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
axes[0].fill_between(
    step_time,
    example_payload["random"]["step_size_timecourse_low"],
    example_payload["random"]["step_size_timecourse_high"],
    color="0.75",
    alpha=0.4,
)
axes[0].plot(
    step_time,
    example_payload["random"]["step_size_timecourse_mean"],
    color="0.35",
    lw=2,
    label="Random-k mean",
)
axes[0].plot(
    step_time,
    example_payload["topk"]["step_size_timecourse"],
    color="#1f77b4",
    lw=2.5,
    label="Top-k",
)
axes[0].set_title(f"{example_payload['target']} step size over time")
axes[0].set_xlabel("time (ms)")
axes[0].set_ylabel("step norm")
axes[0].legend(frameon=False, loc="upper right")
axes[1].fill_between(
    angle_time,
    example_payload["random"]["turn_angle_timecourse_low"],
    example_payload["random"]["turn_angle_timecourse_high"],
    color="0.75",
    alpha=0.4,
)
axes[1].plot(
    angle_time,
    example_payload["random"]["turn_angle_timecourse_mean"],
    color="0.35",
    lw=2,
    label="Random-k mean",
)
axes[1].plot(
    angle_time,
    example_payload["topk"]["turn_angle_timecourse"],
    color="#1f77b4",
    lw=2.5,
    label="Top-k",
)
axes[1].set_title(f"{example_payload['target']} turning angle over time")
axes[1].set_xlabel("time (ms)")
axes[1].set_ylabel("turn angle (rad)")
fig_example.suptitle("Example temporal coherence dynamics", fontsize=14)

fig_auto, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True)
for ax, payload in zip(axes.flat, payloads):
    ax.fill_between(
        lags,
        payload["random"]["autocorr_curve_low"],
        payload["random"]["autocorr_curve_high"],
        color="0.75",
        alpha=0.4,
    )
    ax.plot(lags, payload["random"]["autocorr_curve_mean"], color="0.35", lw=2, label="Random-k mean")
    ax.plot(lags, payload["topk"]["autocorr_curve"], color="#1f77b4", lw=2.5, label="Top-k")
    ax.set_title(payload["target"])
    ax.set_xlabel("lag")
    ax.set_ylabel("trajectory autocorrelation")
axes.flat[0].legend(frameon=False, loc="upper right")
fig_auto.suptitle("Trajectory autocorrelation by ROI", fontsize=14)

fig_constraints, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
for ax, (metric_name, ylabel) in zip(
    axes,
    [
        ("L_norm", "Normalized path length"),
        ("AUC_autocorr", "Autocorr AUC"),
        ("half_decay_lag", "Half-decay lag"),
        ("mean_turn_angle", "Mean turn angle"),
    ],
):
    for payload in payloads:
        top_val = payload["topk"][metric_name]
        rand_val = payload["random"][f"{metric_name}_mean"]
        ax.plot([0, 1], [top_val, rand_val], color="0.75", lw=1)
        ax.scatter([0], [top_val], color="#1f77b4", s=35)
        ax.scatter([1], [rand_val], color="0.35", s=35)
        ax.text(1.03, rand_val, payload["target"], fontsize=7, va="center")
    ax.set_xticks([0, 1], ["top-k", "random"])
    ax.set_title(ylabel)
    ax.set_ylabel(ylabel)
fig_constraints.suptitle("Trajectory constraint summaries across ROIs", fontsize=14)

df_summary = pd.DataFrame(
    [
        {
            "roi": payload["target"],
            "pc1_topk": payload["topk"]["pc1_variance"],
            "pc1_random": payload["random"]["pc1_variance_mean"],
            "pc3_topk": payload["topk"]["pc3_variance"],
            "pc3_random": payload["random"]["pc3_variance_mean"],
            "n80_topk": payload["topk"]["n_components_80"],
            "n80_random": payload["random"]["n_components_80_mean"],
            "n90_topk": payload["topk"]["n_components_90"],
            "n90_random": payload["random"]["n_components_90_mean"],
            "L_norm_topk": payload["topk"]["L_norm"],
            "L_norm_random": payload["random"]["L_norm_mean"],
            "AUC_topk": payload["topk"]["AUC_autocorr"],
            "AUC_random": payload["random"]["AUC_autocorr_mean"],
            "turn_topk": payload["topk"]["mean_turn_angle"],
            "turn_random": payload["random"]["mean_turn_angle_mean"],
        }
        for payload in payloads
    ]
)
print(df_summary.to_string(index=False))

if SAVE:
    s3_base = f"{pth.SAVEDIR}/temporal_coherence/four_roi"
    payload = {
        "rois": ROI_TARGETS,
        "n_random": int(N_RANDOM),
        "payloads": payloads,
    }
    with fsspec.open(f"{s3_base}.pkl", "wb") as f:
        pickle.dump(payload, f)
    with fsspec.open(f"{s3_base}_cumvar.png", "wb") as f:
        fig_cum.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}_cumvar.svg", "w") as f:
        fig_cum.savefig(f, format="svg", transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}_pca_pair.png", "wb") as f:
        fig_pair.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}_pca_pair.svg", "w") as f:
        fig_pair.savefig(f, format="svg", transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}_pca_average.png", "wb") as f:
        fig_avg.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}_pca_average.svg", "w") as f:
        fig_avg.savefig(f, format="svg", transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}_example.png", "wb") as f:
        fig_example.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}_example.svg", "w") as f:
        fig_example.savefig(f, format="svg", transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}_autocorr.png", "wb") as f:
        fig_auto.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}_autocorr.svg", "w") as f:
        fig_auto.savefig(f, format="svg", transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}_constraints.png", "wb") as f:
        fig_constraints.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}_constraints.svg", "w") as f:
        fig_constraints.savefig(f, format="svg", transparent=True, bbox_inches="tight")
    vprint(f"Saved outputs to {s3_base}*")

download_dir = Path.home() / "Downloads"
fig_cum.savefig(download_dir / "temporal_coherence_cumvar.png", dpi=300, transparent=False, bbox_inches="tight")
fig_pair.savefig(download_dir / "temporal_coherence_pca_pair.png", dpi=300, transparent=False, bbox_inches="tight")
fig_avg.savefig(download_dir / "temporal_coherence_pca_average.png", dpi=300, transparent=False, bbox_inches="tight")
fig_example.savefig(download_dir / "temporal_coherence_example.png", dpi=300, transparent=False, bbox_inches="tight")
fig_auto.savefig(download_dir / "temporal_coherence_autocorr.png", dpi=300, transparent=False, bbox_inches="tight")
fig_constraints.savefig(download_dir / "temporal_coherence_constraints.png", dpi=300, transparent=False, bbox_inches="tight")
