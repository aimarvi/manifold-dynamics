import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tqdm import tqdm

# =========================
# CONFIG
# =========================
ONSET_MS = 0          # stimulus onset (ms)
WINDOW_MS = 350        # analyze 0-300 ms post onset
BIN_MS = 1             # PSTH bin width

# start and end times calculated from CONFIG
start = int(ONSET_MS / BIN_MS)
end   = start + int(WINDOW_MS / BIN_MS)

dat = pd.read_pickle('../datasets/NNN/unit_data_full.pkl')
cluster_ids = pd.read_pickle('../datasets/NNN/cluster_ids.pkl')
assert len(dat) == len(cluster_ids)
dat['cluster'] = cluster_ids['labels']

# =========================
# RSA per cluster (time × time similarity)
# =========================
# For RSA, we use img_psth (time x images) for units in each cluster.
# At each time bin within the window, build population-by-image matrix for that cluster:
#   M_t = (n_units_in_cluster x n_images_sub)
# Compute image×image correlation matrix at that time => similarity
# Convert to RDM vector = 1 - corr (upper triangle).
# Compare RDM vectors across all time pairs via correlation => time×time RSA.

def rdm_vector_from_matrix(M, metric='correlation'):
    """
    Args:
        M: (ndarray)
        metric: (str)
    Given a matrix M with shape (n_units, n_images), compute the image×image
    RDM and return the vectorized upper triangle (excluding diagonal).
    """
    M = np.asarray(M, dtype=float)
    # center columns (images) across units, matching the corrcoef approach
    M -= M.mean(axis=0, keepdims=True)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    # pdist expects observations in rows, so pass images as rows -> M.T
    rdm_vec = pdist(M.T, metric=metric)  # already 1 - corr
    return rdm_vec

def rsa_time_by_time_for_cluster(cluster_id, metric='correlation', vectorized=True):
    idx_units = np.where(dat['cluster'] == cluster_id)[0]

    # Select only the rows you want, based on cluster id
    subset = dat.iloc[idx_units]["img_psth"]

    # Convert to list of arrays
    arrays = [np.asarray(x) for x in subset]
    # shape: (units, time_points, images)
    img_psth_array = np.stack(arrays, axis=0)
    Z = img_psth_array[:, start:end, :]
    U, T, I = Z.shape

    # 1) z-score across units for each (t, i)
    mu = Z.mean(axis=0, keepdims=True)                  # (1, T, I)
    sd = Z.std(axis=0, keepdims=True) + 1e-8           # (1, T, I)
    Zu = (Z - mu) / sd                                  # (U, T, I)

    # 2) per-time image×image correlation matrices (population-based)
    #    For each t: C_t = (Zu_t^T @ Zu_t) / (U-1)


    vectorized = True
    if vectorized:
        C_time = np.einsum('uti,utj->tij', Zu, Zu) / (U - 1)   # (T, I, I)
        # C_time = np.clip(C_time, -1.0, 1.0)
    else:
        C_time = Zu.map(lambda A: pdist(np.asarray(A).T, metric=metric))

    RDM_time = 1.0 - C_time                                # (T, I, I)
    # 3) vectorize upper triangles to get one RDM vector per time
    iu = np.triu_indices(I, k=1)
    RDM_vecs = RDM_time[:, iu[0], iu[1]]                   # (T, P), P = I*(I-1)/2

    X = RDM_vecs - RDM_vecs.mean(axis=1, keepdims=True)     # demean per time
    den = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / den
    rsa = Xn @ Xn.T

    return rsa

rsa_results = {}
best_k = 3 # empirically defined, see other scripts
for c in tqdm(range(best_k)):
    print(f"[RSA] Computing time×time similarity for cluster {c} ...")
    out = rsa_time_by_time_for_cluster(c)
    if out is None:
        continue
    rsa_mat = out
    rsa_results[c] = rsa_mat

    plt.figure(figsize=(6, 5))
    plt.imshow(rsa_mat, origin='lower', aspect='auto', vmin=0, vmax=1,
              cmap='Spectral')
    plt.colorbar(label="RSA (RDM–RDM correlation)")

    # add contour lines at correlation levels 0.3 and 0.5
    levels = [0.3, 0.5]
    contours = plt.contour(
        np.linspace(start, end-1, rsa_mat.shape[0]),  # x-axis
        np.linspace(start, end-1, rsa_mat.shape[1]),  # y-axis
        rsa_mat,
        levels=levels,
        colors='gray',
        linewidths=0.8,
    )
    plt.clabel(contours, fmt="%.1f", colors='gray', fontsize=8)

    plt.title(f"Time×Time RSA — Cluster {c})")
    plt.xlabel("Time (ms)")
    plt.ylabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(f'./cluster_{c}.png', dpi=300, transparent=False)
