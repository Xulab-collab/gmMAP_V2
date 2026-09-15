#!/usr/bin/env bash
set -euo pipefail

# Replace paths with local/private files.
gmmap-score \
  --h5ad data/input.h5ad \
  --magma-z data/MAGMA_zstat.csv \
  --celltype-key celltype \
  --out results/gmmap_scores \
  --top-n-genes 1000 \
  --score-method waucell \
  --stage S3

gmmap-association \
  --up-scores results/gmmap_scores/gmmap_scores_UP_S3.csv \
  --down-scores results/gmmap_scores/gmmap_scores_DOWN_S3.csv \
  --metadata results/gmmap_scores/obs_metadata.csv \
  --celltype-key celltype \
  --out results/gmmap_association
