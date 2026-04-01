from __future__ import annotations

import argparse
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


def main() -> None:
    """Plot the time-time tuning RDM for one ROI target."""
    parser = argparse.ArgumentParser(
        description="Plot the time-time tuning RDM for one ROI target."
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help=(
            "ROI UID (4-part: SesIdx.RoiIndex.AREALABEL.Categoty) "
            "or ROI key (3-part: RoiIndex.AREALABEL.Categoty)."
        ),
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bin-size-ms", type=int, default=20)
    parser.add_argument("--tstart", type=int, default=100)
    parser.add_argument("--tend", type=int, default=350)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--savedir", type=str, default="local")
    parser.add_argument("--colorbar", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    def vprint(msg: str) -> None:
        if args.verbose:
            print(msg)

    target_parts = args.target.split(".")
    if len(target_parts) not in (3, 4):
        raise ValueError(
            "Invalid --target format. Use 4-part UID (SesIdx.RoiIndex.AREALABEL.Categoty) "
            "or 3-part ROI key (RoiIndex.AREALABEL.Categoty)."
        )
    if len(target_parts) == 4:
        roi_label = f"{int(target_parts[1]):02d}.{target_parts[2]}.{target_parts[3]}"
    else:
        roi_label = f"{int(target_parts[0]):02d}.{target_parts[1]}.{target_parts[2]}"

    topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
    with open(topk_local, "rb") as f:
        topk_vals = pickle.load(f)

    top_k = args.top_k
    if top_k is None:
        if roi_label not in topk_vals:
            raise ValueError(f"No top-k entry found for ROI: {roi_label}")
        top_k = int(topk_vals[roi_label]["k"])

    raster_4d = nu.significant_trial_raster(
        roi_uid=args.target,
        alpha=args.alpha,
        bin_size_ms=args.bin_size_ms,
    )
    raster_3d = np.nanmean(raster_4d, axis=3)
    image_order = tut.rank_images_by_response(raster_3d)
    idx_topk = np.asarray(image_order[:top_k], dtype=int)

    vprint(f"Resolved ROI target: {args.target}")
    vprint(f"Using top-k = {top_k}")
    vprint(f"Raster shape after binning: {raster_4d.shape}")
    vprint(f"Trial-averaged PSTH shape: {raster_3d.shape}")

    R, rdv = tut.tuning_rdm(
        X=raster_3d,
        indices=idx_topk,
        tstart=args.tstart,
        tend=args.tend,
        metric="correlation",
    )

    use_cbar = args.colorbar
    fig = plt.figure(figsize=(2 + 0.5 * use_cbar, 2))
    if use_cbar:
        gs = fig.add_gridspec(1, 2, width_ratios=[20, 1])
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
    else:
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        cax = None

    cmap = sns.color_palette("rocket", as_cmap=True)
    im = sns.heatmap(
        R,
        vmin=0,
        vmax=1,
        cmap=cmap,
        square=True,
        ax=ax,
        cbar=False,
    )
    if use_cbar:
        fig.colorbar(im.collections[0], cax=cax, ticks=[0, 0.5, 1])

    tick_positions = np.arange(0, R.shape[0]+50, 50)
    tick_labels = tick_positions + args.tstart
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("")
    ax.set_ylabel("")
    # ax.set_xlabel("Time (msec)")
    # ax.set_ylabel("Time")
    sns.despine(ax=ax, trim=True, offset=5)
    fig.tight_layout()

    if args.save:
        s3_png = f"{pth.SAVEDIR}/timextime/{args.savedir}/{args.target}.png"
        s3_svg = f"{pth.SAVEDIR}/timextime/{args.savedir}/{args.target}.svg"
        with fsspec.open(s3_png, "wb") as f:
            fig.savefig(f, format="png", dpi=args.dpi, transparent=True, bbox_inches="tight")
        with fsspec.open(s3_svg, "w") as f:
            fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")
        vprint(f"Saved figure to {s3_png}")
        vprint(f"Saved figure to {s3_svg}")
        vprint(f"Saved figure to {local_png}")

    local_png = Path.home() / "Downloads" / f"{args.savedir}_{args.target.replace('.', '_')}.png"
    fig.savefig(local_png, dpi=args.dpi, transparent=False, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
