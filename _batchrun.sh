#!/bin/bash
#SBATCH -p shared
#SBATCH -c 1
#SBATCH --mem=80G
#SBATCH -t 01:00:00
#SBATCH -o errlog/lag.%j.out

set -euo pipefail

cd /n/holylabs/LABS/konkle_lab/Users/amarvi/workspace/manifold-dynamics

# ========== cross-validation of 51 unique ROIs
idx=${SLURM_ARRAY_TASK_ID}

target="$(
uv run python -c "
import sys
import pandas as pd
import manifold_dynamics.paths as pth
import visionlab_utils.storage as vst

idx = int(sys.argv[1])

uid_csv_uri = f'{pth.OTHERS}/roi-uid.csv'
uid_csv_local = vst.fetch(uid_csv_uri)
df = pd.read_csv(uid_csv_local)

col = 'uid' if 'uid' in df.columns else df.columns[0]

vals = (
    df[col].astype(str)
    .str.split('.', n=1).str[1]   # drop first portion before first '.'
    .dropna()
    .unique()
)

if idx < 0 or idx >= len(vals):
    raise SystemExit(f'idx {idx} out of range (0..{len(vals)-1})')

print(vals[idx])
" "$idx"
)"

if [[ -z "${target}" ]]; then
  echo "no target for idx=${idx}" >&2
  exit 1
fi

# uv run python crossval.py --target "$target" --save --verbose
# uv run python neighbor_scales.py --target "$target" --feature-layers classifier.2 classifier.5 --save --verbose
# uv run python eigenspectra.py --target "$target" --save --verbose --log-scale
# uv run python alexnet/compute_centroids.py --target "$target" --layer-key classifier.4 --save --verbose
# uv run python timextime/ed_main.py --target "$target" --verbose --save
# uv run python dynamic_modes/shifting_subpsace.py --target "$target" --verbose --save
uv run python timextime/lag_model.py --target "$target" --verbose --save
