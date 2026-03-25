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


LAYER_FILE = Path(__file__).with_name("layers.txt")


def parse_layer_map(path: Path) -> list[tuple[str, str]]:
    """
    Parse the AlexNet layer label -> key mapping from ``alexnet/layers.txt``.

    Returns:
        List of ``(label, key)`` pairs in file order.
    """
    layers = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if "->" not in line:
            continue
        left, right = line.split("->", 1)
        label = left.strip()
        key = right.strip()
        if label and key:
            layers.append((label, key))
    if not layers:
        raise ValueError(f"No layer mappings found in {path}")
    return layers


def sem(x: np.ndarray) -> float:
    """Return the standard error of the mean for a 1D array."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(np.nanstd(x, ddof=1) / np.sqrt(x.size))


def main() -> None:
    """
    Sweep AlexNet layers and compare locality effects in ED.

    For each layer defined in ``alexnet/layers.txt``, the script computes:
      - preferred local neighborhoods: neighbors around top-k preferred seeds
      - general local neighborhoods: neighbors around random non-top seeds

    The plot shows the mean ED for those two local conditions with uncertainty
    bands, plus horizontal reference lines for the fixed top-k and random sets.
    """
    parser = argparse.ArgumentParser(
        description="Layer sweep for AlexNet locality effects."
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
    parser.add_argument(
        "--feature-uri",
        type=str,
        default=f"{pth.SAVEDIR}/alexnet/alexnet_acts.pkl",
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
    image_order = tut.rank_images_by_response(X)
    top_indices = np.asarray(image_order[:top_k], dtype=int)
    non_top_pool = np.setdiff1d(np.arange(X.shape[2], dtype=int), top_indices, assume_unique=False)
    if non_top_pool.size < top_k:
        raise ValueError(
            f"Need at least {top_k} non-top images for general-local seeds, got {non_top_pool.size}."
        )

    vprint(f"Responsive raster shape: {raster_4d.shape}")
    vprint(f"Trial-averaged PSTH shape: {X.shape}")
    vprint(f"Using top-k = {top_k}")

    rng_ref = np.random.default_rng(args.random_state)
    random_reference_sets = np.stack(
        [rng_ref.choice(non_top_pool, size=top_k, replace=False) for _ in range(top_k)],
        axis=0,
    )

    R_topk, _ = tut.tuning_rdm(
        X=X,
        indices=top_indices,
        tstart=args.tstart,
        tend=args.tend,
        metric="correlation",
    )
    ed_topk = float(tut.ED2(R_topk))

    ed_random_reference = []
    for random_indices in random_reference_sets:
        R_random, _ = tut.tuning_rdm(
            X=X,
            indices=random_indices,
            tstart=args.tstart,
            tend=args.tend,
            metric="correlation",
        )
        ed_random_reference.append(float(tut.ED2(R_random)))
    ed_random_reference = np.asarray(ed_random_reference, dtype=float)

    acts_local = vst.fetch(args.feature_uri)
    with open(acts_local, "rb") as f:
        acts = pickle.load(f)

    layer_map = parse_layer_map(LAYER_FILE)
    available_layer_map = [(label, key) for label, key in layer_map if key in acts]
    missing_layer_map = [(label, key) for label, key in layer_map if key not in acts]
    if not available_layer_map:
        raise ValueError("No layer keys from alexnet/layers.txt were found in the activation file.")
    if missing_layer_map:
        vprint(
            "Skipping unavailable layers: "
            + ", ".join(f"{label} ({key})" for label, key in missing_layer_map)
        )

    rows = []
    for layer_idx, (layer_label, layer_key) in enumerate(available_layer_map):
        arr = acts[layer_key]
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        feature_matrix = np.asarray(arr)
        if feature_matrix.ndim < 2:
            raise ValueError(f"Expected activation array with image axis for {layer_key}, got shape {feature_matrix.shape}")
        if feature_matrix.ndim > 2:
            feature_matrix = feature_matrix.reshape(feature_matrix.shape[0], -1)
        if feature_matrix.shape[0] != X.shape[2]:
            raise ValueError(
                f"Feature/image mismatch for {layer_key}: {feature_matrix.shape[0]} != {X.shape[2]}"
            )

        rng_layer = np.random.default_rng(args.random_state)
        general_seed_indices = rng_layer.choice(non_top_pool, size=top_k, replace=False)

        preferred_local_sets = mut.neighbor_sets(
            feature_matrix=feature_matrix,
            seed_indices=top_indices,
            k=top_k,
        )
        general_local_sets = mut.neighbor_sets(
            feature_matrix=feature_matrix,
            seed_indices=general_seed_indices,
            k=top_k,
        )

        ed_preferred_local = []
        ed_general_local = []
        for preferred_local_indices, general_local_indices in zip(preferred_local_sets, general_local_sets):
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
            ed_preferred_local.append(float(tut.ED2(R_pref_local)))
            ed_general_local.append(float(tut.ED2(R_general_local)))

        ed_preferred_local = np.asarray(ed_preferred_local, dtype=float)
        ed_general_local = np.asarray(ed_general_local, dtype=float)
        rows.append(
            {
                "layer_idx": int(layer_idx),
                "layer_label": layer_label,
                "layer_key": layer_key,
                "ED_topk": ed_topk,
                "ED_random_mean": float(np.nanmean(ed_random_reference)),
                "ED_random_sem": sem(ed_random_reference),
                "ED_preferred_local_mean": float(np.nanmean(ed_preferred_local)),
                "ED_preferred_local_sem": sem(ed_preferred_local),
                "ED_general_local_mean": float(np.nanmean(ed_general_local)),
                "ED_general_local_sem": sem(ed_general_local),
            }
        )
        vprint(
            f"{layer_label} ({layer_key}): "
            f"preferred-local={np.nanmean(ed_preferred_local):.6f}, "
            f"general-local={np.nanmean(ed_general_local):.6f}"
        )

    df_out = pd.DataFrame(rows)
    print(
        df_out[
            [
                "layer_label",
                "ED_preferred_local_mean",
                "ED_general_local_mean",
                "ED_topk",
                "ED_random_mean",
            ]
        ].to_string(index=False)
    )

    x = np.arange(len(df_out), dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    ax.fill_between(
        x,
        df_out["ED_preferred_local_mean"].to_numpy() - df_out["ED_preferred_local_sem"].to_numpy(),
        df_out["ED_preferred_local_mean"].to_numpy() + df_out["ED_preferred_local_sem"].to_numpy(),
        color="#1f77b4",
        alpha=0.18,
    )
    ax.fill_between(
        x,
        df_out["ED_general_local_mean"].to_numpy() - df_out["ED_general_local_sem"].to_numpy(),
        df_out["ED_general_local_mean"].to_numpy() + df_out["ED_general_local_sem"].to_numpy(),
        color="#ff7f0e",
        alpha=0.18,
    )
    ax.plot(
        x,
        df_out["ED_preferred_local_mean"].to_numpy(),
        color="#1f77b4",
        lw=2.5,
        label="preferred local",
    )
    ax.plot(
        x,
        df_out["ED_general_local_mean"].to_numpy(),
        color="#ff7f0e",
        lw=2.5,
        label="general local",
    )
    ax.axhline(ed_topk, color="black", lw=1.5, ls="-", label="top-k")
    ax.axhline(float(np.nanmean(ed_random_reference)), color="0.35", lw=1.5, ls="--", label="random")
    ax.set_xticks(x)
    ax.set_xticklabels(df_out["layer_label"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("effective dimensionality")
    ax.set_title(args.target)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    if args.save:
        s3_base = f"{pth.SAVEDIR}/neighbors/locality_layers/{args.target}"
        with fsspec.open(f"{s3_base}.pkl", "wb") as f:
            df_out.to_pickle(f)
        with fsspec.open(f"{s3_base}.png", "wb") as f:
            fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
        with fsspec.open(f"{s3_base}.svg", "w") as f:
            fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")
        local_png = Path.home() / "Downloads" / f"locality_layers_{args.target.replace('.', '_')}.png"
        fig.savefig(local_png, dpi=300, transparent=False, bbox_inches="tight")


if __name__ == "__main__":
    main()
