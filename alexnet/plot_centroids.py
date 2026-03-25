from __future__ import annotations

import pickle
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import manifold_dynamics.paths as pth
import visionlab_utils.storage as vst


# -----------------------------------------------------------------------------
# Ad Hoc Configuration
# -----------------------------------------------------------------------------

SELECTIVITIES_TO_PLOT = {"B", "F", "O"}
LAYER_KEY = "classifier.2"
SAVE = True
VERBOSE = True


def vprint(msg: str) -> None:
    """Print only when VERBOSE is enabled."""
    if VERBOSE:
        print(msg)


topk_local = vst.fetch(f"{pth.OTHERS}/topk_vals.pkl")
with open(topk_local, "rb") as f:
    topk_vals = pickle.load(f)

roi_targets = sorted(
    target for target in topk_vals.keys() if target.split(".")[-1] in SELECTIVITIES_TO_PLOT
)

feature_uri = f"{pth.SAVEDIR}/alexnet/alexnet_acts.pkl"
acts_local = vst.fetch(feature_uri)
with open(acts_local, "rb") as f:
    acts = pickle.load(f)

if LAYER_KEY not in acts:
    raise ValueError(f"Missing AlexNet feature layer: {LAYER_KEY}")

feature_matrix = acts[LAYER_KEY]
if hasattr(feature_matrix, "detach"):
    feature_matrix = feature_matrix.detach().cpu().numpy()
feature_matrix = np.asarray(feature_matrix)
if feature_matrix.ndim != 2:
    raise ValueError(f"Expected 2D feature matrix, got shape {feature_matrix.shape}")

pca = PCA(n_components=2)
Z = pca.fit_transform(feature_matrix)

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
ax.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.25, marker="o", c="black", label=None)

selectivities = sorted({target.split(".")[-1] for target in roi_targets})
palette = plt.cm.tab10(np.linspace(0, 1, len(selectivities)))
color_map = {sel: palette[i] for i, sel in enumerate(selectivities)}
seen_labels = set()
plot_labels = {"F": "Face", "O": "Object", "B": "Body"}
rows = []

for target in roi_targets:
    selectivity = target.split(".")[-1]
    centroid_uri = (
        f"{pth.SAVEDIR}/alexnet/roi_centroids/{LAYER_KEY.replace('.', '_')}/{target}.pkl"
    )
    try:
        centroid_local = vst.fetch(centroid_uri)
    except Exception as e:
        print(f"Could not load centroid for {target}: {type(e).__name__}: {e}")
        continue

    with open(centroid_local, "rb") as f:
        payload = pickle.load(f)

    centroid = np.array([payload["centroid_pc1"], payload["centroid_pc2"]], dtype=float)
    color = color_map[selectivity]

    ax.scatter(
        centroid[0],
        centroid[1],
        s=120,
        marker="X",
        edgecolor="white", 
        color=color,
        linewidth=0.8,
        label=plot_labels[selectivity] if selectivity not in seen_labels else None,
    )
    seen_labels.add(selectivity)
    # add text label next to each ROI
    # ax.text(centroid[0], centroid[1], f" {target}", color=color, va="center", fontsize=7)

    rows.append(payload)
    vprint(
        f"{target}: layer={payload['layer_key']}, "
        f"centroid=({payload['centroid_pc1']:.4f}, {payload['centroid_pc2']:.4f})"
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"ROI centroids in {LAYER_KEY} space")
ax.legend(frameon=False, title="selectivity", loc="best")

x_low, x_high = np.percentile(Z[:, 0], [1, 99])
y_low, y_high = np.percentile(Z[:, 1], [1, 99])
ax.set_xlim(x_low, x_high)
ax.set_ylim(y_low, y_high)

fig.tight_layout()

if SAVE:
    s3_base = f"{pth.SAVEDIR}/alexnet/roi_centroids_{LAYER_KEY.replace('.', '_')}"
    with fsspec.open(f"{s3_base}.pkl", "wb") as f:
        pickle.dump(rows, f)
    with fsspec.open(f"{s3_base}.png", "wb") as f:
        fig.savefig(f, format="png", dpi=300, transparent=True, bbox_inches="tight")
    with fsspec.open(f"{s3_base}.svg", "w") as f:
        fig.savefig(f, format="svg", transparent=True, bbox_inches="tight")

    download_dir = Path.home() / "Downloads"
    fig.savefig(
        download_dir / f"roi_centroids_{LAYER_KEY.replace('.', '_')}.png",
        dpi=300,
        transparent=False,
        bbox_inches="tight",
    )
