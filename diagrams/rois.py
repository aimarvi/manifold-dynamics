#!/usr/bin/env python

import numpy as np
import pandas as pd
import nibabel as nib
import re
import os
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

# =========================
# paths (EDIT THESE)
# =========================
DIR = '~/Desktop/HVRD/workspace/datasets/NMT_v2.0_sym'
SURF_GII = os.path.join(DIR, 'NMT_v2.0_sym_surfaces', 'lh.mid_surface.inf_300.rsl.gii')
CHARM_NII = os.path.join(DIR, 'NMT_v2.0_sym', 'CHARM_in_NMT_v2.0_sym.nii.gz')
CHARM_TABLE = os.path.join(DIR, 'tables_CHARM', 'CHARM_key_table.csv')

# =========================
# ROI definitions
# =========================
ap_targets = {
    'MiddleBody': 6.3, 'AnteriorBody': 11.3,
    'MiddleFace': 8.0, 'AnteriorFace': 13.6,
    'MiddleObject': 9.0, 'AnteriorObject': 12.8,
    'MiddleColor': 8.3, 'AnteriorColor': 14.5,
    'Scene': 5.3, 'Unknown': 7.3,
}

roi_family = {
    'MiddleBody': 'body', 'AnteriorBody': 'body',
    'MiddleFace': 'face', 'AnteriorFace': 'face',
    'MiddleObject': 'object', 'AnteriorObject': 'object',
    'MiddleColor': 'color', 'AnteriorColor': 'color',
    'Scene': 'scene', 'Unknown': 'unknown',
}

family_label = {
    'background': 0,
    'face': 1,
    'body': 2,
    'object': 3,
    'color': 4,
    'scene': 5,
    'unknown': 6,
}

cmap = [
    'lightgrey',   # background
    'dodgerblue',  # face
    'limegreen',   # body
    'orange',      # object
    'gold',        # color
    'purple',      # scene
    'dimgray',     # unknown
]

cmap = [
    'lightgrey',  # background
    'red',      # face
    'red',      # body
    'red',      # object
    'red',      # color
    'red',      # scene
    'red',      # unknown
]

# =========================
# knobs
# =========================
ap_axis = 1        # usually y in NMT
ap_tol = 0.5
roi_radius = 1.5 
charm_level = 'Level 4'

# =========================
# load surface
# =========================
gii = nib.load(SURF_GII)
coords = gii.agg_data('NIFTI_INTENT_POINTSET')
faces = gii.agg_data('NIFTI_INTENT_TRIANGLE')

faces_pv = np.hstack(
    [np.full((faces.shape[0], 1), 3), faces]
).astype(np.int64).ravel()

mesh = pv.PolyData(coords, faces_pv)

# =========================
# load CHARM atlas + sample
# =========================
atlas_img = nib.load(CHARM_NII)
atlas = atlas_img.get_fdata().astype(np.int32)
charm_lvl = 4 # manually specify which level of granulaity (1-6)
atlas3 = atlas[:, :, :, 0, charm_lvl-1].astype(np.int32)  # shape (256, 312, 200)

inv_aff = np.linalg.inv(atlas_img.affine)

good = np.isfinite(coords).all(axis=1)
ijk = np.full((coords.shape[0], 3), -1, dtype=np.int64)  # sentinel

ijk_good = nib.affines.apply_affine(inv_aff, coords[good])
ijk[good] = np.rint(ijk_good).astype(np.int64)

shape3 = np.array(atlas3.shape, dtype=np.int64)
ok = good & np.all((ijk >= 0) & (ijk < shape3[None, :]), axis=1)

labels_at_verts = np.zeros(coords.shape[0], dtype=np.int32)
labels_at_verts[ok] = atlas3[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]

# =========================
# parse CHARM table
# =========================
charm = pd.read_csv(CHARM_TABLE)

def split_id_name(s):
    m = re.match(r'^\s*(\d+)\s*:\s*(.*)\s*$', str(s))
    return (int(m.group(1)), m.group(2)) if m else (None, str(s))

lvl = charm[charm_level].dropna().drop_duplicates()
pairs = lvl.apply(split_id_name).tolist()
id2name = {i: n for i, n in pairs if i is not None}

def ids_matching(pattern):
    rx = re.compile(pattern, flags=re.IGNORECASE)
    return {i for i, n in id2name.items() if rx.search(n)}

family_allowed = {
    'face':   ids_matching(r'superior temporal|sts|inferior temporal|temporal'),
    'object': ids_matching(r'superior temporal|sts|inferior temporal|temporal'),
    'color':  ids_matching(r'inferior temporal|temporal|fusiform|occipito'),
    'body':   ids_matching(r'superior temporal|temporal'),
    'scene':  ids_matching(r'para.?hipp|hipp|medial temporal|cingulate|retrosplenial'),
    'unknown': set(id2name.keys()),
}

# =========================
# geodesic distance
# =========================
def geodesic_ball(mesh, seed, radius_mm):
    pts = mesh.points
    f = mesh.faces.reshape(-1, 4)[:, 1:]

    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    w = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
    n = pts.shape[0]
    A = sp.csr_matrix((w, (edges[:, 0], edges[:, 1])), shape=(n, n))
    A = A + A.T

    dist = csgraph.dijkstra(A, directed=False, indices=int(seed))
    return dist <= radius_mm

# =========================
# seed selection
# =========================
def pick_seed(coords, labels_at_verts, allowed, ap_mm):
    ap = coords[:, ap_axis]
    cand = np.where(
        (np.abs(ap - ap_mm) <= ap_tol) &
        np.isin(labels_at_verts, list(allowed))
    )[0]

    if cand.size == 0:
        raise RuntimeError(f'no candidates for ap={ap_mm}')

    # left hemi heuristics
    x = coords[cand, 0]
    cand = cand[x <= np.quantile(x, 0.2)]
    z = coords[cand, 2]

    return int(cand[np.argmin(z)])

# =========================
# paint ROIs
# =========================
labels = np.zeros(mesh.n_points, dtype=np.uint8)

for roi, ap_mm in ap_targets.items():
    fam = roi_family[roi]
    if 'unknown' in fam:
        continue
    seed = pick_seed(coords, labels_at_verts, family_allowed[fam], ap_mm)
    # mask = geodesic_ball(mesh, seed, roi_radius * np.random.uniform(0.8, 1.2)) & np.isin(labels_at_verts, list(family_allowed[fam]))
    mask = geodesic_ball(mesh, seed, roi_radius) & np.isin(labels_at_verts, list(family_allowed[fam]))
    # mask = geodesic_ball(mesh, seed, roi_radius)
    labels[mask] = family_label[fam]

mesh.point_data['roi_family'] = labels

# =========================
# launch server
# =========================
pv.global_theme.smooth_shading = False

pl = pv.Plotter(off_screen=True)
pl.add_mesh(
    mesh,
    scalars='roi_family',
    cmap=cmap,
    clim=[0, len(cmap) - 1],
    categories=True,
    interpolate_before_map=False,
    show_scalar_bar=False, # disable colorbar
)
pl.set_background('white')
pl.reset_camera()

print('\nStarting PyVista server...')
# pl.view_vector((-0.6, 0, -1), viewup=(0, 1, 0)) # look from below and slightly from the right
pl.view_vector((-1, 0, 0), viewup=(0, 0, 1)) # lateral view
pl.show(auto_close=False)
pl.screenshot('~/Downloads/macaque_brain.png', transparent_background=True)
pl.close()
