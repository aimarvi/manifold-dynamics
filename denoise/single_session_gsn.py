import os, fsspec, time, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import manifold_dynamics.session_raster_extraction as sre
import manifold_dynamics.io_matlab_s3 as ims
from manifold_dynamics.session_gsn import session_gsn
import manifold_dynamics.paths as pth

fs = fsspec.filesystem("s3")

outdir = os.path.join(pth.SAVEDIR, 'denoising')
uid_sheet = pd.read_csv(os.path.join(pth.OTHERS, 'roi-uid.csv'))
unique_rois = uid_sheet['uid'].unique()

# run via sbatch array
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
win = int(sys.argv[1])
out_dir  = os.path.join(pth.PROCESSED, sys.argv[2])
plot = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else True

roi_uid = unique_rois[task_id]
inpath = os.path.join(pth.PROCESSED, 'single-session-raster', f'{roi_uid}.npy')

### time how long it takes to load via fsspec
size_bytes = fs.size(inpath)
print('starting to load data...')
t0 = time.perf_counter()
with fs.open(inpath, 'rb') as f:
    out = np.load(f, allow_pickle=False)
dt = time.perf_counter() - t0
print(f'size: {size_bytes/1e9:.2f} GB')
print(f'time: {dt:.2f} sec')
print(f'throughput: {(size_bytes/1e6)/dt:.2f} MB/s')
### end time

savepath = os.path.join(out_dir, f'{roi_uid}_w{win:02d}.parquet')
if not ims.exists(savepath):
    # run GSN over time
    cov_df, ncsnr_df = session_gsn(out, win=win, overlap=False, verbose=False)

    # save
    combined = pd.concat(
        {'cov_df': cov_df, 'ncsnr_df': ncsnr_df}, 
        names=['table']
    )
    combined.to_parquet(savepath)
    print(f'Saved data for {roi_uid}!')

    if plot:
        #### PLOT ###
        customp = ['red', 'black']# sns.color_palette('Dark2')
        fig,ax = plt.subplots(1,1, figsize=(5,3))
        sns.scatterplot(cov_df, x='time', y='mean_abs_covariance', hue='type', 
                        marker='o', palette=customp, alpha=.75, ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('Cov.')
        ax.legend(title='')
        sns.despine(fig=fig, trim=True, offset=5)
        
        # save
        s3_base = os.path.join(outdir, f'{roi_uid}_w{win:02d}_covariance')
        with fs.open(f'{s3_base}.png', 'wb') as f:
            plt.savefig(f, dpi=300, transparent=True, bbox_inches='tight')
        with fs.open(f'{s3_base}.svg', 'w') as f:
            plt.savefig(f, format='svg', transparent=True, bbox_inches='tight')
        plt.savefig(Path.home() / 'Downloads' / f'{roi_uid}_w{win:02d}_covariance.png', dpi=300, transparent=False, bbox_inches='tight')
        
        fig,ax = plt.subplots(1,1, figsize=(5,3))
        sns.lineplot(ncsnr_df, x='time', y='mean_abs_ncsnr', color='gray', ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('NCSNR')
        sns.despine(fig=fig, trim=True, offset=5)
        
        # save
        s3_base = os.path.join(outdir, f'{roi_uid}_w{win:02d}_ncsnr')
        with fs.open(f'{s3_base}.png', 'wb') as f:
            plt.savefig(f, dpi=300, transparent=True, bbox_inches='tight')
        with fs.open(f'{s3_base}.svg', 'w') as f:
            plt.savefig(f, format='svg', transparent=True, bbox_inches='tight')
        plt.savefig(Path.home() / 'Downloads' / f'{roi_uid}_w{win:02d}_ncsnr.png', dpi=300, transparent=False, bbox_inches='tight')
        
        print(f'Saved figures for {roi_uid}!')
else:
    print(f'Skipping {roi_uid} ({win}, already processed...')
