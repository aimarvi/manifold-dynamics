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


def lag_model(M: np.ndarray) -> np.ndarray:
    """
    Keep only lag structure by averaging each diagonal and reconstructing a
    matrix whose values depend only on distance from the main diagonal.
    """
    n = M.shape[0]
    M_hat = np.zeros_like(M)

    for k in range(n):
        diag = np.diag(M, k=k)
        mean = float(np.nanmean(diag))
        M_hat += np.diag(np.full(len(diag), mean), k=k)
        if k > 0:
            M_hat += np.diag(np.full(len(diag), mean), k=-k)

    return M_hat


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a diagonal-lag surrogate model to one ROI time-time RDM."
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
    parser.add_argument("--n-bootstrap", type=int, default=100)
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
        roi_label = args.target

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
    image_order = np.asarray(tut.rank_images_by_response(raster_3d), dtype=int)
    idx_topk = image_order[:top_k]
    idx_all = np.arange(raster_3d.shape[2], dtype=int)
    rng = np.random.default_rng(args.random_state)

    vprint(f"Resolved ROI target: {args.target}")
    vprint(f"Using top-k = {top_k}")
    vprint(f"Responsive raster shape: {raster_4d.shape}")
    vprint(f"Trial-averaged PSTH shape: {raster_3d.shape}")

    R_all, _ = tut.tuning_rdm(
        X=raster_3d,
        indices=idx_all,
        tstart=args.tstart,
        tend=args.tend,
        metric="correlation",
    )
    R_topk, _ = tut.tuning_rdm(
        X=raster_3d,
        indices=idx_topk,
        tstart=args.tstart,
        tend=args.tend,
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
    for i in range(args.n_bootstrap):
        idx_random = rng.choice(raster_3d.shape[2], size=top_k, replace=False)
        R_random, _ = tut.tuning_rdm(
            X=raster_3d,
            indices=idx_random,
            tstart=args.tstart,
            tend=args.tend,
            metric="correlation",
        )
        R_random_hat = lag_model(R_random)
        x = R_random.reshape(-1)
        y = R_random_hat.reshape(-1)
        m = np.isfinite(x) & np.isfinite(y)
        r_random = float(np.corrcoef(x[m], y[m])[0, 1])
        random_rows.append(
            {
                "roi": roi_label,
                "target": args.target,
                "condition": "random",
                "bootstrap": i,
                "r": r_random,
                "r2": float(r_random ** 2),
            }
        )

    df_random = pd.DataFrame(random_rows)
    df_summary = pd.DataFrame(
        [
            {
                "roi": roi_label,
                "target": args.target,
                "condition": "all",
                "top_k": int(top_k),
                "r": r_all,
                "r2": r2_all,
            },
            {
                "roi": roi_label,
                "target": args.target,
                "condition": "top-k",
                "top_k": int(top_k),
                "r": r_topk,
                "r2": r2_topk,
            },
            {
                "roi": roi_label,
                "target": args.target,
                "condition": "random_mean",
                "top_k": int(top_k),
                "r": float(df_random["r"].mean()),
                "r2": float(df_random["r2"].mean()),
            },
        ]
    )

    print(df_summary.to_string(index=False))
    print(f"random r_sd:  {df_random['r'].std(ddof=1):.6f}")
    print(f"random r2_sd: {df_random['r2'].std(ddof=1):.6f}")

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

    payload = {
        "roi": roi_label,
        "target": args.target,
        "top_k": int(top_k),
        "tstart": int(args.tstart),
        "tend": int(args.tend),
        "R_all": R_all,
        "R_all_hat": R_all_hat,
        "R_topk": R_topk,
        "R_topk_hat": R_topk_hat,
        "r_all": r_all,
        "r2_all": r2_all,
        "r_topk": r_topk,
        "r2_topk": r2_topk,
        "df_random": df_random,
        "df_summary": df_summary,
    }

    if args.save:
        s3_base = f"{pth.SAVEDIR}/timextime/lag_model/{args.target}"
        with fsspec.open(f"{s3_base}.pkl", "wb") as f:
            pickle.dump(payload, f)
        with fsspec.open(f"{s3_base}.png", "wb") as f:
            fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
        with fsspec.open(f"{s3_base}.svg", "w") as f:
            fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")
        vprint(f"Saved payload to {s3_base}.pkl")
        vprint(f"Saved figure to {s3_base}.png")
        vprint(f"Saved figure to {s3_base}.svg")

    local_png = Path.home() / "Downloads" / f"lag_model_{args.target.replace('.', '_')}.png"
    fig.savefig(local_png, dpi=300, transparent=False, bbox_inches="tight")
    plt.close(fig)
    vprint(f"Saved figure to {local_png}")


if __name__ == "__main__":
    main()
