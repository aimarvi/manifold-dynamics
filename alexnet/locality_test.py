from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import manifold_dynamics.model_utils as mut
import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


def main() -> None:
    """
    Compare ED across preferred images, preferred-image neighborhoods, general
    AlexNet neighborhoods, and random image sets for one ROI target.

    Target format options:
      - ROI UID (4 parts): ``SesIdx.RoiIndex.AREALABEL.Categoty``
      - ROI key (3 parts): ``RoiIndex.AREALABEL.Categoty``
    """
    parser = argparse.ArgumentParser(
        description="Locality test for one ROI target."
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
    parser.add_argument("--feature-uri", type=str, default=f"{pth.SAVEDIR}/alexnet/alexnet_acts.pkl")
    parser.add_argument(
        "--feature-layers",
        nargs="+",
        default=["classifier.5"],
        help="AlexNet layers used to define neighbor sets.",
    )
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
    X = np.nanmean(raster_4d, axis=3)
    vprint(f"Responsive trial raster shape (units, time, images, trials): {raster_4d.shape}")
    vprint(f"Trial-averaged PSTH shape (units, time, images): {X.shape}")

    image_order = tut.rank_images_by_response(X)
    top_indices = np.asarray(image_order[:top_k], dtype=int)
    vprint(f"Using top-k = {top_k}")
    vprint(f"Top-{top_k} image indices: {top_indices.tolist()}")

    acts_local = vst.fetch(args.feature_uri)
    with open(acts_local, "rb") as f:
        acts = pickle.load(f)

    feature_blocks = []
    for layer in args.feature_layers:
        if layer not in acts:
            raise ValueError(f"Missing AlexNet feature layer: {layer}")
        arr = acts[layer]
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D activations for {layer}, got shape {arr.shape}")
        feature_blocks.append(arr)
        vprint(f"{layer}: feature shape={arr.shape}")

    feature_matrix = np.concatenate(feature_blocks, axis=1)
    if feature_matrix.shape[0] != X.shape[2]:
        raise ValueError(
            f"Feature/image mismatch: feature_matrix has {feature_matrix.shape[0]} images, "
            f"but PSTH has {X.shape[2]} images."
        )

    rng = np.random.default_rng(args.random_state)
    non_top_pool = np.setdiff1d(np.arange(X.shape[2], dtype=int), top_indices, assume_unique=False)
    if non_top_pool.size < top_k:
        raise ValueError(
            f"Need at least {top_k} non-top images for general-local seeds, got {non_top_pool.size}."
        )

    preferred_local_sets = mut.neighbor_sets(
        feature_matrix=feature_matrix,
        seed_indices=top_indices,
        k=top_k,
    )
    general_seed_indices = rng.choice(non_top_pool, size=top_k, replace=False)
    general_local_sets = mut.neighbor_sets(
        feature_matrix=feature_matrix,
        seed_indices=general_seed_indices,
        k=top_k,
    )

    R_topk, _ = tut.tuning_rdm(
        X=X,
        indices=top_indices,
        tstart=args.tstart,
        tend=args.tend,
        metric="correlation",
    )
    ed_topk = float(tut.ED2(R_topk))

    rows = []
    for set_idx in range(top_k):
        preferred_seed = int(top_indices[set_idx])
        preferred_local_indices = preferred_local_sets[set_idx]
        general_seed = int(general_seed_indices[set_idx])
        general_local_indices = general_local_sets[set_idx]
        random_indices = rng.choice(non_top_pool, size=top_k, replace=False) # rng.choice(X.shape[2], size=top_k, replace=False)

        R_pref_local, _ = tut.tuning_rdm(
            X=X,
            indices=preferred_local_indices,
            tstart=args.tstart,
            tend=args.tend,
            metric="correlation",
        )
        R_general_local, _ = tut.tuning_rdm(
            X=X,
            indices=general_local_indices,
            tstart=args.tstart,
            tend=args.tend,
            metric="correlation",
        )
        R_random, _ = tut.tuning_rdm(
            X=X,
            indices=random_indices,
            tstart=args.tstart,
            tend=args.tend,
            metric="correlation",
        )

        ed_pref_local = float(tut.ED2(R_pref_local))
        ed_general_local = float(tut.ED2(R_general_local))
        ed_random = float(tut.ED2(R_random))

        rows.append(
            {
                "roi": args.target,
                "set_idx": int(set_idx),
                "preferred_seed_idx": preferred_seed,
                "preferred_local_indices": preferred_local_indices.tolist(),
                "general_seed_idx": general_seed,
                "general_local_indices": general_local_indices.tolist(),
                "random_indices": random_indices.tolist(),
                "ED_topk": ed_topk,
                "ED_preferred_local": ed_pref_local,
                "ED_general_local": ed_general_local,
                "ED_random": ed_random,
                "delta_preferred_local_minus_random": ed_pref_local - ed_random,
                "delta_general_local_minus_random": ed_general_local - ed_random,
                "delta_preferred_local_minus_general_local": ed_pref_local - ed_general_local,
                "delta_preferred_local_minus_topk": ed_pref_local - ed_topk,
                "delta_general_local_minus_topk": ed_general_local - ed_topk,
                "delta_random_minus_topk": ed_random - ed_topk,
            }
        )

    df_out = pd.DataFrame(rows)

    print(
        df_out[
            [
                "set_idx",
                "preferred_seed_idx",
                "general_seed_idx",
                "ED_preferred_local",
                "ED_general_local",
                "ED_random",
            ]
        ].to_string(index=False)
    )
    print(f"ED_topk:                 {ed_topk:.6f}")
    print(f"mean ED_preferred_local: {df_out['ED_preferred_local'].mean():.6f}")
    print(f"mean ED_general_local:   {df_out['ED_general_local'].mean():.6f}")
    print(f"mean ED_random:          {df_out['ED_random'].mean():.6f}")
    print(
        f"mean delta pref-local minus general-local: "
        f"{df_out['delta_preferred_local_minus_general_local'].mean():.6f}"
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.boxplot(
        [
            np.array([ed_topk]),
            df_out["ED_preferred_local"].to_numpy(),
            df_out["ED_general_local"].to_numpy(),
            df_out["ED_random"].to_numpy(),
        ],
        positions=[1, 2, 3, 4],
        widths=0.5,
        tick_labels=["top-k", "preferred local", "general local", "random"],
    )
    ax.set_ylabel("effective dimensionality")
    ax.set_title(args.target)
    ax.scatter(1, ed_topk, color="black", zorder=3)
    ax.text(1, ed_topk, f"{ed_topk:.2f}", ha="center", va="bottom")

    if args.save:
        s3_base = f"{pth.SAVEDIR}/neighbors/locality_test/{'_'.join(args.feature_layers)}/{args.target}"
        with fsspec.open(f"{s3_base}.pkl", "wb") as f:
            df_out.to_pickle(f)
        with fsspec.open(f"{s3_base}.png", "wb") as f:
            fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
        with fsspec.open(f"{s3_base}.svg", "w") as f:
            fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")
        local_png = Path.home() / "Downloads" / f"locality_test_{args.target.replace('.', '_')}.png"
        fig.savefig(local_png, dpi=300, transparent=False, bbox_inches="tight")


if __name__ == "__main__":
    main()
