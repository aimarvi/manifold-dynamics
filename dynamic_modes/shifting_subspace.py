from __future__ import annotations

import argparse
import pickle

import fsspec
import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

import manifold_dynamics.neural_utils as nu
import manifold_dynamics.paths as pth
import manifold_dynamics.tuning_utils as tut
import visionlab_utils.storage as vst


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute principal-angle spectra for one ROI target across sliding "
            "time-window subspaces and save a lightweight payload."
        )
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
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output path. Defaults to "
            "s3://.../manifold-dynamics/dynamic_modes/shifting_subspace/<target>.pkl"
        ),
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    def vprint(msg: str) -> None:
        if args.verbose:
            print(msg)

    target_parts = args.target.split(".")
    if len(target_parts) not in (3, 4):
        raise ValueError(
            "Invalid --target format. Use 4-part UID "
            "(SesIdx.RoiIndex.AREALABEL.Categoty) or 3-part ROI key "
            "(RoiIndex.AREALABEL.Categoty)."
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
    elif top_k < 1:
        raise ValueError(f"--top-k must be >= 1, got {top_k}")

    raster_4d = nu.significant_trial_raster(
        roi_uid=args.target,
        alpha=args.alpha,
        bin_size_ms=args.bin_size_ms,
    )
    raster_3d = np.nanmean(raster_4d, axis=3)
    image_order = np.asarray(tut.rank_images_by_response(raster_3d), dtype=int)

    if top_k > image_order.size:
        raise ValueError(
            f"top_k={top_k} exceeds the number of available images ({image_order.size})"
        )

    idx_topk = image_order[:top_k]
    candidate_idxs = image_order[top_k:]

    if args.window_size < 1:
        raise ValueError(f"--window-size must be >= 1, got {args.window_size}")
    if args.step < 1:
        raise ValueError(f"--step must be >= 1, got {args.step}")
    if raster_3d.shape[1] < args.window_size:
        raise ValueError(
            f"window_size={args.window_size} exceeds time axis length ({raster_3d.shape[1]})"
        )

    time_starts = np.arange(0, raster_3d.shape[1] - args.window_size, args.step, dtype=int)
    if time_starts.size == 0:
        raise ValueError("No valid sliding windows were generated.")

    n_components = min(args.n_components, top_k, raster_3d.shape[0])
    if n_components < 1:
        raise ValueError(f"Invalid number of subspace dimensions: {n_components}")

    rng = np.random.default_rng(args.random_state)
    if candidate_idxs.size >= top_k and args.n_random > 0:
        random_idxs = np.stack(
            [rng.choice(candidate_idxs, size=top_k, replace=False) for _ in range(args.n_random)],
            axis=0,
        )
    else:
        random_idxs = np.empty((0, top_k), dtype=int)
        if args.n_random > 0:
            vprint(
                "Skipping random index generation because there are not enough "
                "non-top-k candidate images."
            )

    vprint(f"Resolved ROI target: {args.target}")
    vprint(f"Using top-k = {top_k}")
    vprint(f"Responsive raster shape: {raster_4d.shape}")
    vprint(f"Trial-averaged PSTH shape: {raster_3d.shape}")
    vprint(f"Using n_components = {n_components}")

    subspaces = []
    for t in time_starts:
        R_t = np.nanmean(raster_3d[:, t : t + args.window_size, :], axis=1).T
        subspaces.append(
            {
                "top": PCA(n_components=n_components).fit(R_t[idx_topk]).components_.T,
                "all": PCA(n_components=n_components).fit(R_t).components_.T,
            }
        )

    n_time = time_starts.size
    principal_angles_top = np.full((n_time, n_time, n_components), np.nan, dtype=float)
    principal_angles_all = np.full((n_time, n_time, n_components), np.nan, dtype=float)
    principal_angles_random = np.full(
        (random_idxs.shape[0], n_time, n_time, n_components),
        np.nan,
        dtype=float,
    )

    for i in range(n_time):
        for j in range(n_time):
            principal_angles_top[i, j] = np.degrees(
                subspace_angles(subspaces[i]["top"], subspaces[j]["top"])
            )
            principal_angles_all[i, j] = np.degrees(
                subspace_angles(subspaces[i]["all"], subspaces[j]["all"])
            )

    for r, idx_rand in enumerate(random_idxs):
        random_subspaces = []
        for t in time_starts:
            R_t = np.nanmean(raster_3d[:, t : t + args.window_size, :], axis=1).T
            random_subspaces.append(
                PCA(n_components=n_components).fit(R_t[idx_rand]).components_.T
            )
        for i in range(n_time):
            for j in range(n_time):
                principal_angles_random[r, i, j] = np.degrees(
                    subspace_angles(random_subspaces[i], random_subspaces[j])
                )

    payload = {
        "target": args.target,
        "top_k": int(top_k),
        "n_components": int(n_components),
        "window_size": int(args.window_size),
        "step": int(args.step),
        "time_starts": time_starts,
        "principal_angles_top": principal_angles_top,
        "principal_angles_all": principal_angles_all,
        "principal_angles_random": principal_angles_random,
    }

    vprint(f"roi: {roi_label}")
    vprint(f"target: {args.target}")
    vprint(f"top_k: {top_k}")
    vprint(f"n_components: {n_components}")
    vprint(f"n_windows: {time_starts.size}")
    vprint(f"n_random_sets: {random_idxs.shape[0]}")
    vprint(f"principal angle sizes: {principal_angles_top.shape}")

    if args.save:
        output = args.output
        if output is None:
            output = f"{pth.SAVEDIR}/dynamic_modes/shifting_subspace/{args.target}.pkl"
        with fsspec.open(output, "wb") as f:
            pickle.dump(payload, f)
        vprint(f"Saved payload to {output}")


if __name__ == "__main__":
    main()
