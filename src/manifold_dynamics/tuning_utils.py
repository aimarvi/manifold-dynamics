from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Global Analysis Defaults
# -----------------------------------------------------------------------------
RAND = 0
ONSET = 50
RESP = slice(ONSET + 50, ONSET + 220)
BASE = slice(ONSET - 50, ONSET + 0)


# -----------------------------------------------------------------------------
# Data Selection and Preparation
# -----------------------------------------------------------------------------

def select_significant_roi_rows(dat: pd.DataFrame, roi: str, pval_threshold: float = 0.05) -> pd.DataFrame:
    """
    Return rows for one ROI after p-value filtering.

    Args:
        dat: Input unit-level table.
        roi: ROI label (e.g., ``MF1_7_F``).
        pval_threshold: Significance threshold for ``p_value``.

    Returns:
        Filtered DataFrame containing only significant units for the ROI.
    """
    sig = dat[dat["p_value"] < pval_threshold]
    df = sig[sig["roi"] == roi]
    if len(df) == 0:
        raise ValueError(f"No data for ROI {roi}")
    return df


def trial_averaged_psth(dat: pd.DataFrame, roi: str, pval_threshold: float = 0.05) -> np.ndarray:
    """
    Return trial-averaged PSTH tensor for one ROI.

    Returns:
        Array with shape ``(units, time, images)``.
    """
    df = select_significant_roi_rows(dat=dat, roi=roi, pval_threshold=pval_threshold)
    return np.stack(df["img_psth"].to_numpy())


# NOTE: Original name kept for compatibility.
def response_array(dat, roi):
    """Backward-compatible wrapper for ``trial_averaged_psth``."""
    return trial_averaged_psth(dat=dat, roi=roi, pval_threshold=0.05)


# -----------------------------------------------------------------------------
# Image Ordering and Selection
# -----------------------------------------------------------------------------

def rank_images_by_response(
    X: np.ndarray,
    response_window=RESP,
    baseline_window=BASE,
) -> np.ndarray:
    """
    Rank images by baseline-subtracted response magnitude (descending).

    accepts:
        (units, time, images)
        (units, time, images, trials)  # averages over trials

    returns:
        indices of images sorted by descending score
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 4:
        # average over trials
        X = np.nanmean(X, axis=3)
    elif X.ndim != 3:
        raise ValueError(f'expected 3d or 4d array, got shape {X.shape}')

    # compute mean response across units and time window
    resp = np.nanmean(X[:, response_window, :], axis=(0, 1))
    base = np.nanmean(X[:, baseline_window, :], axis=(0, 1))

    scores = resp - base

    return np.argsort(scores)[::-1]


def resolve_image_indices(X: np.ndarray, images="all", random_state: int = RAND):
    """
    Resolve image selection spec into concrete indices/slices.

    Supported `images` values:
      - ``'all'``
      - ``'nsd'``
      - ``'localizer'``
      - ``'shuff_nsd'``
      - slice
      - tuple(start, end) -> slice(start, end)
      - explicit 1D bool/int array
    """
    rng = np.random.default_rng(random_state)

    if isinstance(images, str):
        if images == "all":
            return slice(None)
        if images == "nsd":
            return slice(0, min(1000, X.shape[2]))
        if images == "localizer":
            return slice(1000, X.shape[2])
        if images == "shuff_nsd":
            n = min(1000, X.shape[2])
            return rng.permutation(np.arange(n))
        raise ValueError(
            f"unknown images='{images}' (use 'all', 'nsd', 'localizer', or indices)"
        )

    if isinstance(images, slice):
        return images

    if (
        isinstance(images, tuple)
        and len(images) == 2
        and all(isinstance(x, (int, np.integer, type(None))) for x in images)
    ):
        return slice(images[0], images[1])

    idx = np.asarray(images)
    if idx.ndim != 1:
        raise ValueError("images indices must be 1d")

    if idx.dtype == bool:
        if idx.size != X.shape[2]:
            raise ValueError(f"boolean mask length {idx.size} != n_images {X.shape[2]}")
        return idx

    idx = idx.astype(int)
    if (idx < 0).any() or (idx >= X.shape[2]).any():
        raise ValueError("image indices out of bounds")
    return idx


# -----------------------------------------------------------------------------
# Core Time-Time Tuning RDM
# -----------------------------------------------------------------------------

def tuning_rdm(X, indices, tstart=100, tend=350, metric="correlation"):
    """
    Build a time-time tuning RDM from trial-averaged unit responses.

    Canonical pipeline:
      1. Select image subset and time window.
      2. Compute image RDV at each timepoint.
      3. Rank-transform each timepoint RDV.
      4. Compute pairwise distance between timepoint RDVs.

    Args:
        X: Response tensor ``(units, time, images)``.
        indices: Image indices/slice/mask.
        tstart: Time start index (inclusive).
        tend: Time end index (exclusive).
        metric: Distance metric for time-time RDM.

    Returns:
        R: Time-time RDM (square).
        Xrdv: Per-time image-pair RDV matrix ``(time, n_pairs)``.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (units,time,images), got {X.shape}")

    Ximg = X[:, tstart:tend, indices]
    if Ximg.shape[2] < 2:
        raise ValueError("Need at least 2 images to form an image RDM.")

    Xrdv = np.array([pdist(Ximg[:, t, :].T, metric="correlation") for t in range(Ximg.shape[1])])
    Xrank = np.apply_along_axis(rankdata, 1, Xrdv)
    R = squareform(pdist(Xrank, metric=metric))
    return R, Xrdv


# -----------------------------------------------------------------------------
# Time-Time Analysis Wrappers (Built on tuning_rdm)
# -----------------------------------------------------------------------------

def geo_rdm(
    dat,
    roi,
    mode="top",
    step=5,
    k_max=200,
    metric="correlation",
    random_state=RAND,
    tstart=100,
    tend=350,
):
    """
    Compute a sequence of time-time RDMs across manifold scales.

    Returns:
        sizes: list of manifold scales.
        rdms: list of time-time RDMs.
    """
    rng = np.random.default_rng(random_state)
    X = trial_averaged_psth(dat, roi)

    order = rank_images_by_response(X) if mode == "top" else rng.permutation(X.shape[2])
    sizes = [k for k in range(max(step,2), min(k_max, X.shape[2]) + 1, step)]

    rdms = []
    for k in tqdm(sizes):
        idx = order[:k]
        R, _ = tuning_rdm(X=X, indices=idx, tstart=tstart, tend=tend, metric=metric)
        rdms.append(R)

    return sizes, rdms


def static_rdm(
    dat,
    roi,
    mode="top",
    scale=30,
    tstart=100,
    tend=350,
    metric="correlation",
    random_state=RAND,
):
    """
    Compute one time-time RDM at a single manifold scale.

    Special case:
      - ``scale == -1`` uses localizer image set ``[1000:]``.
    """
    rng = np.random.default_rng(random_state)
    X = trial_averaged_psth(dat, roi)

    order = rank_images_by_response(X) if mode == "top" else rng.permutation(X.shape[2])
    idx = order[:scale]
    if scale == -1:
        idx = slice(1000, X.shape[2])

    return tuning_rdm(X=X, indices=idx, tstart=tstart, tend=tend, metric=metric)


def specific_static_rdm(
    dat,
    roi,
    indices,
    tstart=100,
    tend=350,
    metric="correlation",
    random_state=RAND,
):
    """
    Compute one time-time RDM for an explicitly provided image index set.
    """
    _ = random_state  # kept for API compatibility
    X = trial_averaged_psth(dat, roi)
    return tuning_rdm(X=X, indices=indices, tstart=tstart, tend=tend, metric=metric)


# -----------------------------------------------------------------------------
# Time-Averaged and Unit-Response Views
# -----------------------------------------------------------------------------

def time_avg_rdm(dat, roi, window=RESP, images="all", metric="correlation", random_state=RAND):
    """
    Compute image-image RDM after averaging unit responses across a time window.

    Note:
        This keeps existing indexing behavior (including tuple window behavior)
        for compatibility with current analyses.
    """
    X = trial_averaged_psth(dat, roi)
    idx = resolve_image_indices(X=X, images=images, random_state=random_state)

    Xw = np.nanmean(X[:, window, idx], axis=1)  # (units, images_sel)
    Xrdv = pdist(Xw.T, metric=metric)
    R = squareform(Xrdv)
    return R, Xrdv


def unit_responses(dat, roi, window=RESP, images="all", random_state=RAND):
    """
    Return unit responses during a given time window for a selected image set.

    Returns:
        Array with shape ``(units, images_selected)``.
    """
    X = trial_averaged_psth(dat, roi)
    idx = resolve_image_indices(X=X, images=images, random_state=random_state)

    # Keep original behavior: average over the selected time window only.
    Xw = np.nanmean(X[:, window, idx], axis=1)
    return Xw


def landscape(dat, roi, response_window=RESP):
    """
    Compute baseline-subtracted response score for each image.

    Returns:
        1D score vector of length ``n_images``.
    """
    X = trial_averaged_psth(dat, roi)
    return np.nanmean(X[:, response_window, :], axis=(0, 1)) - np.nanmean(X[:, BASE, :], axis=(0, 1))

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def rdv(X):
    """Return upper-triangle entries (k=1) from a square matrix."""
    ind = np.triu_indices_from(X, k=1)
    return X[ind]


def l2(X):
    """Return L2 norm of a vector-like input."""
    return np.sqrt(np.sum(X ** 2))


def treves_rolls_sparsity(X, axis=1):
    """
    Compute Treves-Rolls sparsity along one axis of an array.

    The sparsity measure is:

        S = 1 - ((mean(r)) ** 2 / mean(r ** 2))

    where the mean is taken along the requested axis. The output contains one
    sparsity value for each slice orthogonal to that axis.

    Args:
        X: Input array.
        axis: Axis along which to compute sparsity.
            ``axis=1`` on a ``(units, images)`` matrix returns one value per unit.
            ``axis=0`` returns one value per image.

    Returns:
        1D array of Treves-Rolls sparsity values.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim < 2:
        raise ValueError(f"treves_rolls_sparsity expects an array with ndim >= 2, got {X.shape}")
    if axis < 0 or axis >= X.ndim:
        raise ValueError(f"axis {axis} is out of bounds for shape {X.shape}")

    mean_r = np.nanmean(X, axis=axis)
    mean_r2 = np.nanmean(X ** 2, axis=axis)

    out = np.full_like(mean_r, np.nan, dtype=float)
    valid = np.isfinite(mean_r) & np.isfinite(mean_r2) & (mean_r2 > 0)
    out[valid] = 1.0 - ((mean_r[valid] ** 2) / mean_r2[valid])
    return out

# -----------------------------------------------------------------------------
# Effective Dimensionality & Entropy Metrics
# -----------------------------------------------------------------------------

def ED1(R):
    """
    Standard effective dimensionality from similarity-like matrices.
    """
    S = -0.5 * R ** 2
    lam = np.linalg.eigvalsh(S)
    lam = np.clip(lam, 0, None)
    return (lam.sum() ** 2) / (lam ** 2).sum()

def ED2(R):
    """
    Effective dimensionality from distance matrices.

    Includes finite-entry filtering before eigendecomposition.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"ED2 expects a square 2D matrix, got shape {R.shape}")

    finite_mask = np.isfinite(R).all(axis=0) & np.isfinite(R).all(axis=1)
    R = R[np.ix_(finite_mask, finite_mask)]
    if R.shape[0] < 2:
        return np.nan

    n = R.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (R ** 2) @ J
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)

    lam = np.linalg.eigvalsh(B)
    lam = np.clip(lam, 0, None)
    denom = (lam ** 2).sum()
    if denom <= 0:
        return np.nan
    return (lam.sum() ** 2) / denom

def entropy(V):
    """Normalize absolute values of a vector to sum to 1."""
    v = np.abs(V)
    return v / v.sum()
