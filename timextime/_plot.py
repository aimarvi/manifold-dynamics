import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
import manifold_dynamics.tuning_utils as tut

'''
makes a GIF for a single ROI in dat
the time x time RDM can shows representational geometry across all time points (-50 -> 400 msec)
calculate time x time RDMs using different sets of images (inc. random selection)
'''

def rdm_gif(dat, ROI, mode='top', step=10, k_max=200, metric='correlation',
            onset=50, resp=(50,220), base=(-50,0), session=None, tstart=100, tend=350):
    """
    mode: 'top' or 'shuffle'
    step/k_max: grow subset sizes [step, 2*step, ..., k_max]
    session: None=all sessions pooled; or pass a specific session value
    """
    rng = np.random.default_rng(0)
    ONSET = 50
    RESP = slice(ONSET + resp[0], ONSET + resp[1])
    BASE = slice(ONSET + base[0], ONSET + base[1])

    # --- subset data ---
    sig = dat[dat['p_value'] < 0.05]
    roi_df = sig[sig['roi'] == ROI] if session is None else sig[(sig['roi']==ROI) & (sig['session']==session)]
    if len(roi_df) == 0: raise ValueError("No rows for given ROI/session.")
    X = np.stack(roi_df['img_psth'].to_numpy())  # (units, time, images)

    # --- image scores and ordering ---
    resp_mean = np.nanmean(X[:, RESP, :], axis=(0,1))
    base_mean = np.nanmean(X[:, BASE, :], axis=(0,1))
    scores = resp_mean - base_mean
    order = np.argsort(scores)[::-1] if mode == 'top' else rng.permutation(scores.size)

    # --- precompute vmax for consistent color scale ---
    sizes = [k for k in range(step, min(k_max, X.shape[2]) + 1, step)]
    rdms = []
    for k in sizes:
        idx = order[:k]
        R, _ = tut.tuning_rdm(X=X, indices=idx, tstart=tstart, tend=tend, metric=metric)
        rdms.append(R)
    vmin, vmax = 0.0, np.nanmax([R.max() for R in rdms])  # lock scale across frames

    # --- animate ---
    fig, ax = plt.subplots(figsize=(5,5))
    hm = sns.heatmap(rdms[0], cmap=sns.color_palette('magma'),
                     vmin=vmin, vmax=vmax, square=True, cbar=False, ax=ax)
    title = ax.set_title(f"{ROI} | {mode} | {metric} | k={sizes[0]}")

    def update(i):
        ax.clear()
        sns.heatmap(rdms[i], cmap=sns.color_palette('magma', as_cmap=True),
                    vmin=vmin, vmax=vmax, square=True, cbar=False, ax=ax)
        ax.set_xticks([]); ax.set_yticks([])
        title = ax.set_title(f"{ROI} | {mode} | {metric} | k={sizes[i]}")
        return [title]

    anim = FuncAnimation(fig, update, frames=len(rdms), interval=400, blit=False)
    
    return anim
    anim.save(outpath, writer=PillowWriter(fps=10))
    plt.close(fig)
    return outpath

# ============== USAGE ==================
DATA_DIR = './../../datasets/NNN/object_roi_data.pkl'
dat = pd.read_pickle(DATA_DIR)
rois = list(dat['roi'].unique())
# choose ROI and save path
for ROI in tqdm(rois):
    OUT_DIR=f'~/Downloads/{ROI}_timextime_slow.gif'
    out = rdm_gif(dat, ROI, mode='top', step=10, k_max=500, metric='correlation')
    
    # change fps
    out.save(OUT_DIR, writer=PillowWriter(fps=6))
    print("Saved:", ROI)
    
#     # shuffled version
#     OUT_DIR=f'../../gifs/{ROI}_shuff_timextime_slow.gif'
#     out = rdm_gif(dat, ROI, mode='shuffle', step=10, k_max=500, metric='correlation')
#     out.save(OUT_DIR, writer=PillowWriter(fps=6))
#     print("Saved:", ROI)
