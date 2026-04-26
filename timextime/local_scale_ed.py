from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


def main() -> None:
    """
    Compare one ROI's local-scale ED against image rankings from other ROIs.

    For a target ROI:
      - compute its own top-k ED
      - compute a random-k ED distribution from its non-top image pool
      - compute an alternative ED distribution using the top-k rankings from all
        other ROIs whose selectivity label differs from the target's
    """
    parser = argparse.ArgumentParser(
        description="Compare local-scale ED against rankings from other ROI patches."
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
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=0)
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

    target_selectivity = roi_label.split(".")[-1]

    topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
    with open(topk_local, "rb") as f:
        topk_vals = pickle.load(f)

    top_k = args.top_k
    if top_k is None:
        if roi_label not in topk_vals:
            raise ValueError(f"No top-k entry found for ROI: {roi_label}")
        top_k = int(topk_vals[roi_label]["k"])

    other_rois = sorted(
        roi
        for roi in topk_vals
        if roi != roi_label and str(roi).split(".")[-1] != target_selectivity
    )
    if len(other_rois) == 0:
        raise ValueError(f"No alternative ROIs found for target {roi_label}")

    raster_4d_target = nu.significant_trial_raster(
        roi_uid=args.target,
        alpha=args.alpha,
        bin_size_ms=args.bin_size_ms,
    )
    X_target = np.nanmean(raster_4d_target, axis=3)  # (units, time, images)
    order_target = np.asarray(tut.rank_images_by_response(X_target), dtype=int)
    idx_topk = order_target[:top_k]
    idx_pool = order_target[top_k:]

    if idx_pool.size < top_k:
        raise ValueError(
            f"Target {roi_label} has only {idx_pool.size} non-top images, need at least {top_k}"
        )

    vprint(f"Target ROI: {roi_label}")
    vprint(f"Target raster shape: {raster_4d_target.shape}")
    vprint(f"Using top-k = {top_k}")
    vprint(f"Alternative ROI count: {len(other_rois)}")

    rows = []

    R_top, _ = tut.tuning_rdm(
        X=X_target,
        indices=idx_topk,
        tstart=args.tstart,
        tend=args.tend,
        metric="correlation",
    )
    rows.append(
        {
            "target": roi_label,
            "source_roi": roi_label,
            "condition": "top-k",
            "bootstrap": np.nan,
            "top_k": int(top_k),
            "ED": float(tut.ED2(R_top)),
        }
    )

    rng = np.random.default_rng(args.random_state)
    for i in range(args.n_random):
        idx_random = np.asarray(rng.choice(idx_pool, size=top_k, replace=False), dtype=int)
        R_random, _ = tut.tuning_rdm(
            X=X_target,
            indices=idx_random,
            tstart=args.tstart,
            tend=args.tend,
            metric="correlation",
        )
        rows.append(
            {
                "target": roi_label,
                "source_roi": np.nan,
                "condition": "random",
                "bootstrap": i,
                "top_k": int(top_k),
                "ED": float(tut.ED2(R_random)),
            }
        )

    for roi_alt in other_rois:
        raster_4d_alt = nu.significant_trial_raster(
            roi_uid=roi_alt,
            alpha=args.alpha,
            bin_size_ms=args.bin_size_ms,
        )
        X_alt = np.nanmean(raster_4d_alt, axis=3)
        idx_alt = np.asarray(tut.rank_images_by_response(X_alt), dtype=int)[:top_k]
        R_alt, _ = tut.tuning_rdm(
            X=X_target,
            indices=idx_alt,
            tstart=args.tstart,
            tend=args.tend,
            metric="correlation",
        )
        rows.append(
            {
                "target": roi_label,
                "source_roi": roi_alt,
                "condition": "alternative",
                "bootstrap": np.nan,
                "top_k": int(top_k),
                "ED": float(tut.ED2(R_alt)),
            }
        )
        vprint(f"Computed alternative ED from {roi_alt}")

    df_out = pd.DataFrame(rows)

    summary = (
        df_out.groupby("condition")["ED"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    order = ["top-k", "random", "alternative"]
    sns.boxplot(
        data=df_out,
        x="condition",
        y="ED",
        hue="condition",
        order=order,
        palette=["white"],
        ax=ax,
    )
    sns.stripplot(
        data=df_out,
        x="condition",
        y="ED",
        hue="condition",
        order=order,
        palette=["black"],
        alpha=0.15,
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title(roi_label)
    ax.set_xlabel("")
    ax.set_ylabel("Effective Dimensionality (ED)")
    ax.set_xticklabels(["Top-k", "Random", "Other ROI"])
    ax.tick_params(axis="x", labelrotation=35)
    sns.despine(ax=ax, trim=True, offset=5)
    if ax.legend_ is not None:
        ax.legend_.remove()
    fig.tight_layout()

    if args.save:
        s3_pkl = f"{pth.SAVEDIR}/timextime/local_scale_ed/{roi_label}.pkl"
        s3_png = f"{pth.SAVEDIR}/timextime/local_scale_ed/{roi_label}.png"
        s3_svg = f"{pth.SAVEDIR}/timextime/local_scale_ed/{roi_label}.svg"
        with fsspec.open(s3_pkl, "wb") as f:
            pickle.dump({"df_out": df_out, "summary": summary}, f)
        with fsspec.open(s3_png, "wb") as f:
            fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
        with fsspec.open(s3_svg, "w") as f:
            fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")
        vprint(f"Saved payload to {s3_pkl}")
        vprint(f"Saved figure to {s3_png}")
        vprint(f"Saved figure to {s3_svg}")

    local_png = Path.home() / "Downloads" / f"local_scale_ed_{roi_label}.png"
    # fig.savefig(local_png, dpi=300, transparent=False, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
