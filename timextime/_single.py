import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import imageio.v2 as iio

from _grid import *

font = ImageFont.load_default(size=32)

def build_grid_gif(dat, ROI_LIST, step=10, k_max=200, metric='correlation',
                   out_path='rdm_grid.gif', random_state=0, cmap='rocket',
                   duration=2, loop=0, cbar=True, reverse=False):
    """
    build a gif showing, for each time step t, a row of top-k rdms across rois.
    - only the 'top' rdms are rendered (no shuffled controls)
    - all frames share the same vmin/vmax for comparable color scaling
    """

    # ---- compute rdm sequences (top only) ----
    seqs_top = {}
    sizes = None  # same schedule for every roi; we'll grab it once
    for roi in tqdm(ROI_LIST):
        sizes_roi, rdms_top = rdv_redo(
            dat, roi, mode='top', step=step, k_max=k_max,
            metric=metric, random_state=random_state
        )
        if sizes is None:
            sizes = sizes_roi
        seqs_top[roi] = rdms_top

    # ---- global color limits for consistency across rois + frames ----
    # vmin is 0 for correlation distance; vmax from the data (nan-safe)
    vmin = 0.0
    vmax = np.nanmax([np.nanmax(R) for v in seqs_top.values() for R in v])

    # number of frames (assumes same schedule for all rois)
    T = len(next(iter(seqs_top.values())))
    frames = []

    # ---- precompute a single colorbar image (shared across frames) ----
    first_tile = rdm_to_image(next(iter(seqs_top.values()))[0], vmin, vmax, cmap=cmap)
    tile_w, tile_h = first_tile.size
    cb_img = make_colorbar_image(vmin, vmax, cmap=cmap, height=tile_h + 40)

    # layout constants (pixels)
    title_h = 10   # space for roi labels at top
    footer_h = 20  # space for "images used" at bottom

    # ---- render frames ----
    for t in range(T):
        # convert each roi's rdm at time t into an image tile
        tiles = [rdm_to_image(seqs_top[roi][t], vmin, vmax, cmap=cmap) for roi in ROI_LIST]

        W, H = tiles[0].size
        row = Image.new('RGB', (W * len(ROI_LIST), H))
        for j, img in enumerate(tiles):
            row.paste(img, (j * W, 0))

        # grid canvas: title + row + footer
        grid = Image.new('RGB', (row.width, H + title_h + footer_h), color=(255, 255, 255))
        grid.paste(row, (0, title_h))
        draw = ImageDraw.Draw(grid)

        # roi names centered over each tile
        for j, roi in enumerate(ROI_LIST):
            xmid = j * W + W // 2
            draw.text((xmid, title_h // 2), str(roi), fill=(0, 0, 0), anchor='ma', font=font)

        # frame footer: how many images were used at this step
        used_k = int(sizes[t]) if sizes is not None else (t * step)
        draw.text((grid.width // 2, H - footer_h),
                  f'images used: top {used_k}', fill=(0, 0, 0), anchor='ma', font=font)

        # add shared colorbar on the right
        cb_w, cb_h = cb_img.size
        canvas = Image.new('RGB', (grid.width + cb_w + 20, grid.height), color=(255, 255, 255))
        canvas.paste(grid, (0, 0))
        y0 = (canvas.height - cb_h) // 2
        canvas.paste(cb_img, (grid.width + 10, y0))

        # canvas contains the colorbar, if wanted
        if cbar:
            frames.append(canvas)
        else:
            frames.append(grid)

    # play the gif in reverse
    if reverse:
        frames = frames[::-1]

    # ---- write gif ----
    iio.mimsave(out_path, frames, duration=duration, loop=loop)
    return out_path


if __name__ == '__main__':
    
    # DATA_DIR = ../../datasets/NNN/object_roi_data.pkl
    DATA_DIR = './../../datasets/NNN/face_roi_data.pkl'
    dat = pd.read_pickle(DATA_DIR)
    # ROI_LIST = ['AB3_18_B', 'MB3_12_B', 'AB3_12_B', 'AB3_17_B']
    # ROI_LIST = ['Unknown_19_F']# , 'MF1_7_F', 'MF1_8_F', 'MF1_9_F']
    # ROI_LIST = ['PITP4_10_O', 'Unknown_6_O', 'MO1s2_5_O', 'Unknown_16_O',
    #  'Unknown_26_O', 'AO5_25_O']
    # ROI_LIST = ['Unknown_19_F', 'MF1_8_F', 'MF1_9_F', 'MF1_7_F']
    ROI_LIST = ["MF1_7_F"]

    SAVE_DIR = '~/Downloads/'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for roi in ROI_LIST:
        metric = 'correlation'
        rflag = 1
        rev = 'reverse' if rflag else 'forward'
        out_path = os.path.join(SAVE_DIR, f'{roi}_{metric}_{rev}.gif')
        out = build_grid_gif(dat, [roi], step=5, k_max=200, duration=1, cbar=False, metric=metric, out_path=out_path, reverse=rflag)
        print('Saved:', roi)
