import numpy as np
import pandas as pd
import nibabel as nib
import re
import os
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import matplotlib
import matplotlib.cm as cm

# =========================
# paths (EDIT THESE)
# =========================
DIR = '/Users/aim/Desktop/HVRD/workspace/dynamics/datasets/NMT_v2.0_sym'
D99_TABLE = os.path.join(DIR, 'tables_D99', 'D99_labeltable.txt')
D99_VOL = os.path.join(DIR, 'NMT_v2.0_sym', 'D99_atlas_in_NMT_v2.0_sym.nii.gz')

SURF_GII = os.path.join(DIR, 'NMT_v2.0_sym_surfaces', 'lh.mid_surface.inf_300.rsl.gii')
CHARM_NII = os.path.join(DIR, 'NMT_v2.0_sym', 'CHARM_in_NMT_v2.0_sym.nii.gz')
CHARM_TABLE = os.path.join(DIR, 'tables_CHARM', 'CHARM_key_table.csv')

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
# load in D99 labels and vol
# =========================
rows = []
with open(D99_TABLE, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)\s+(.*)$', line)
        if m:
            rows.append((int(m.group(1)), m.group(2).strip()))

d99 = pd.DataFrame(rows, columns=['id', 'name'])
id2name = dict(zip(d99['id'], d99['name']))
name2id = dict(zip(d99['name'], d99['id']))

atlas_img = nib.load(D99_VOL)
atlas = np.asanyarray(atlas_img.dataobj)
atlas = np.squeeze(atlas).astype(np.int32)
if atlas.ndim != 3:
    raise ValueError(f'expected 3D D99 atlas, got shape={atlas.shape}')

inv_aff = np.linalg.inv(atlas_img.affine)

# robust handling of any bad vertices
good = np.isfinite(coords).all(axis=1)

ijk = np.full((coords.shape[0], 3), -1, dtype=np.int64)
ijk_good = nib.affines.apply_affine(inv_aff, coords[good])
ijk[good] = np.rint(ijk_good).astype(np.int64)

shape3 = np.array(atlas.shape, dtype=np.int64)
ok = good & np.all((ijk >= 0) & (ijk < shape3[None, :]), axis=1)

labels_at_verts = np.zeros(coords.shape[0], dtype=np.int32)
labels_at_verts[ok] = atlas[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]

# ===========================
# regex pattern matching
# =========================
def ids_matching(pattern):
    rx = re.compile(pattern, flags=re.IGNORECASE)
    return set(d99.loc[d99['name'].str.contains(rx, na=False), 'id'].tolist())

# example: tweak these once you inspect what matches
family_allowed = {
    # sts / it / te* neighborhoods for face/object/color
    'face':   ids_matching(r'\bTE|\bTG|TGsts|STG|FST|MT|MST|TEO|TPO|Tpt'),
    'object': ids_matching(r'\bTE|\bTG|TGsts|STG|FST|MT|MST|TEO|TPO|Tpt'),
    'color':  ids_matching(r'\bTE|\bTG|TEO|TFO|TF|TH'),

    # body often more dorsal temporal / sts / motion-adjacent
    'body':   ids_matching(r'STG|FST|MT|MST|TGsts|Tpt'),

    # scenes: ventromedial temporal / parahippocampal-ish proxies in D99
    'scene':  ids_matching(r'\bTF\b|\bTH\b|EC|ER|paraS|preS|proS'),

    'unknown': set(d99['id'].tolist()),
}

for fam, ids in family_allowed.items():
    ex = sorted(list(ids))[:15]
    print(fam, 'n=', len(ids))
    print(' ', [(i, id2name[i]) for i in ex if i in id2name])

# ===========================
# visualize D99 labels on the surface
# ===========================
mesh.point_data['d99'] = labels_at_verts.astype(np.int32)

# only labels that actually appear on this hemi surface
u = np.unique(mesh.point_data['d99'])
u = u[u != 0]  # 0 usually background/outside cortex
print('n unique d99 labels on surface:', len(u))
print('first 20 labels:', u[:20])
print('first 20 names:', [id2name.get(int(i), 'unknown') for i in u[:20]])

# make a stable random colormap for all possible ids up to max label
# this avoids needing a huge hand-made palette
max_id = int(mesh.point_data['d99'].max())
rng = np.random.default_rng(0)  # fixed seed for consistent colors
lut = np.ones((max_id + 1, 4), dtype=float)
lut[:, :3] = rng.random((max_id + 1, 3))
lut[0, :3] = (0.85, 0.85, 0.85)  # background as light grey

pv.set_jupyter_backend('trame')
pl = pv.Plotter()

actor = pl.add_mesh(
    mesh,
    scalars='d99',
    categories=True,
    smooth_shading=False,
    interpolate_before_map=False,
)

pl.add_axes()
pl.reset_camera()
pl.show()
