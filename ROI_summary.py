from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


def grassmann_fn(X: np.ndarray) -> np.ndarray:
    return np.nanmean(X, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the composite single-ROI analysis figure."
    )
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bin-size-ms", type=int, default=20)
    parser.add_argument("--tstart", type=int, default=100)
    parser.add_argument("--tend", type=int, default=350)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    def vprint(msg: str) -> None:
        if args.verbose:
            print(msg)

    target_parts = args.target.split(".")
    if len(target_parts) != 3:
        raise ValueError("TARGET must be a 3-part ROI key: RoiIndex.AREALABEL.Categoty")

    roi_label = f"{int(target_parts[0]):02d}.{target_parts[1]}.{target_parts[2]}"

    topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
    with open(topk_local, "rb") as f:
        topk_vals = pickle.load(f)

    if roi_label not in topk_vals:
        raise ValueError(f"No top-k entry found for ROI: {roi_label}")

    top_k = int(topk_vals[roi_label]["k"])
    raster_4d = nu.significant_trial_raster(
        roi_uid=args.target,
        alpha=args.alpha,
        bin_size_ms=args.bin_size_ms,
    )
    X = np.nanmean(raster_4d, axis=3)
    image_order = np.asarray(tut.rank_images_by_response(X), dtype=int)
    idx_topk = image_order[:top_k]
    idx_global = np.arange(X.shape[2], dtype=int)
    candidate_idxs = image_order[top_k:]

    rng = np.random.default_rng(args.random_state)
    if candidate_idxs.size >= top_k:
        idx_random = np.asarray(rng.choice(candidate_idxs, size=top_k, replace=False), dtype=int)
    else:
        idx_random = np.asarray(rng.choice(idx_global, size=top_k, replace=False), dtype=int)

    vprint(f"Loaded {args.target}")
    vprint(f"Responsive raster shape: {raster_4d.shape}")
    vprint(f"Trial-averaged PSTH shape: {X.shape}")
    vprint(f"Using top-k = {top_k}")

    ### calculate time-time RDMs
    R_topk, _ = tut.tuning_rdm(X=X, indices=idx_topk,
        tstart=args.tstart, tend=args.tend, metric="correlation")
    R_global, _ = tut.tuning_rdm(X=X, indices=idx_global,
        tstart=args.tstart, tend=args.tend, metric="correlation")
    R_random, _ = tut.tuning_rdm(X=X, indices=idx_random,
        tstart=args.tstart, tend=args.tend, metric="correlation")

    pa_local = vst.fetch(f"{pth.SAVEDIR}/dynamic_modes/shifting_subspace/{args.target}.pkl")
    with open(pa_local, "rb") as f:
        pa_payload = pickle.load(f)

    ### calculate principal angles
    PA_top = grassmann_fn(np.asarray(pa_payload["top"], dtype=float))
    PA_all = grassmann_fn(np.asarray(pa_payload["all"], dtype=float))
    pa_vmin = float(np.nanmin([np.nanmin(PA_top), np.nanmin(PA_all)]))
    pa_vmax = float(np.nanmax([np.nanmax(PA_top), np.nanmax(PA_all)]))

    ### start of plotting
    # create the global figure layout
    fig = plt.figure(figsize=(9, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])

    ax_00 = fig.add_subplot(gs[0, 0])
    ax_10 = fig.add_subplot(gs[1, 0])
    ax_11 = fig.add_subplot(gs[1, 1])
    ax_02 = fig.add_subplot(gs[0, 2])
    ax_12 = fig.add_subplot(gs[1, 2])
    ax_20 = fig.add_subplot(gs[2, 0], box_aspect=1)
    ax_21 = fig.add_subplot(gs[2, 1:])

    # subgrid for top images
    gs_images = gs[0, 1].subgridspec(
        5,
        5,
        wspace=0.01,
        hspace=0.01,
        height_ratios=[0.125, 0.25, 0.25, 0.25, 0.125],
        width_ratios=[0.125, 0.25, 0.25, 0.25, 0.125],
    )
    image_axes = [fig.add_subplot(gs_images[i, j]) for i in range(1, 4) for j in range(1, 4)]

    # shrink whole 3x3 block inside the cell
    image_block_scale = 0.5

    ### time-time RDMs
    rdm_cmap = sns.color_palette("rocket", as_cmap=True)
    pa_cmap = sns.color_palette("viridis", as_cmap=True)

    for ax, R, title in [
        (ax_00, R_topk, "Top-k"),
        (ax_10, R_global, "Global"),
        (ax_11, R_random, "Random-k"),
    ]:
        sns.heatmap(R, vmin=0, vmax=1,
            cmap=rdm_cmap, square=True, cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")

    ### principal angles
    for ax, R, title in [
        (ax_02, PA_top, "Top-k principal angles"),
        (ax_12, PA_all, "Global principal angles"),
    ]:
        sns.heatmap(R, vmin=pa_vmin, vmax=pa_vmax,
            cmap=pa_cmap, square=True, cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")

    ed_local = vst.fetch(f"{pth.SAVEDIR}/timextime/ed_main/{args.target}.pkl")
    with open(ed_local, "rb") as f:
        df_ed = pickle.load(f)

    df_ed["condition"] = df_ed["condition"].replace({"local": "top-k", "global": "global", "random": "random"})

    ### ED boxplot
    sns.boxplot(data=df_ed, x="condition", y="ED", hue="condition", order=["global", "top-k", "random"],
        palette=["white"], ax=ax_20)
    sns.stripplot(data=df_ed, x="condition", y="ED", hue="condition", order=["global", "top-k", "random"],
        palette=["black"], alpha=0.05, dodge=False, legend=False, ax=ax_20)
    ax_20.set_title("ED")
    ax_20.set_xlabel("")
    ax_20.set_ylabel("ED")
    ax_20.set_xticklabels(["Global", "Top-k", "Random"])
    sns.despine(ax=ax_20, trim=True, offset=5)
    if ax_20.legend_ is not None:
        ax_20.legend_.remove()

    resp = np.nanmean(X[:, tut.RESP, :], axis=(0, 1))
    base_mean = np.nanmean(X[:, tut.BASE, :], axis=(0, 1))
    base_std = np.nanstd(X[:, tut.BASE, :], axis=(0, 1))
    base_std = np.where(base_std == 0, np.nan, base_std)
    zscores = (resp - base_mean) / base_std

    ### response z-score
    colors = sns.color_palette("coolwarm_r", len(image_order))
    ax_21.vlines(np.arange(len(image_order)), 0, zscores[image_order],
        colors=colors, linewidth=0.6 )
    ax_21.axhline(0, color="black", linewidth=0.8)
    ax_21.set_title("")
    ax_21.set_xlabel("Ranked image")
    ax_21.set_ylabel("Response z-score")
    sns.despine(ax=ax_21, trim=True, offset=5)

    ### top images
    for i, ax in enumerate(image_axes):
        nu.plot_stimulus_image(int(image_order[i]), ax=ax)

    panel_box = fig.add_subplot(gs[0, 1])
    panel_box.set_title("")
    panel_box.set_xticks([])
    panel_box.set_yticks([])
    panel_box.patch.set_alpha(0)
    for spine in panel_box.spines.values():
        spine.set_visible(False)

    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.05, hspace=0.08, wspace=0.06)

    ### colorbars
    rdm_pos = ax_00.get_position()
    pa_pos = ax_02.get_position()
    cbar_y = 1.02
    cbar_h = 0.012

    scale = 0.8  # fraction of column width used by colorbar
    cax_rdm = fig.add_axes([
        rdm_pos.x0 + rdm_pos.width * (1 - scale) / 2 + 0.01,
        cbar_y,
        rdm_pos.width * scale, cbar_h,])
    cb_rdm = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=rdm_cmap),
        cax=cax_rdm,
        orientation="horizontal")
    cb_rdm.ax.set_title("Pearson's r", fontsize=10, pad=2)

    cax_pa = fig.add_axes([
        pa_pos.x0 + pa_pos.width * (1 - scale) / 2 + 0.02,
        cbar_y,
        pa_pos.width * scale, cbar_h])
    cb_pa = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=pa_vmin, vmax=pa_vmax), cmap=pa_cmap),
        cax=cax_pa,
        orientation="horizontal")
    cb_pa.ax.set_title("Degrees", fontsize=10, pad=2)

    # only show min/max values on colorbar
    cb_rdm.set_ticks([0, 1])
    cb_rdm.ax.xaxis.set_ticks_position("bottom")
    cb_pa.set_ticks([pa_vmin, pa_vmax])
    cb_pa.ax.xaxis.set_ticks_position("bottom")

    fig.suptitle(args.target, y=1.015)

    if args.save:
        s3_png = f"{pth.SAVEDIR}/analysis/{args.target}.png"
        with fsspec.open(s3_png, "wb") as f:
            fig.savefig(f, format="png", dpi=300, bbox_inches="tight")
        vprint(f"Saved figure to {s3_png}")

    download_png = Path.home() / "Downloads" / f"analysis_{args.target}.png"
    fig.savefig(download_png, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
