#!/usr/bin/env bash
set -euo pipefail

# Example commands for the full gmMAP workflows.
# Replace paths with your own data paths before running.

gmmap-main-full   --h5ad data/input.h5ad   --magma-z data/MAGMA_zstat.csv   --celltype-key celltype   --prefix Meta   --outdir results/Meta_SchemeC_AUROC_UpDownNet_outputs   --n-jobs 16   --top-n-genes 1000   --min-valid-genes 200   --methods-to-run wAUCell   --stages-to-run S1_raw,S3_geneSetZ_cellZ

gmmap-pseudotime-pyucell   --h5ad data/trajectory.h5ad   --magma-z data/MAGMA_zstat.csv   --celltype-col celltype   --pseudotime-col pseudotime   --outdir results/pseudotime

gmmap-flux-full   --annotation src/gmmap/resources/flux/module_ratio_template.csv   --score-matrix results/Meta_wAUCell_up_S3_geneSetZ_cellZ.csv.gz   --gene-matrix data/cell_by_gene_expression_matrix.csv.gz   --enzyme-annotation src/gmmap/resources/flux/Module_round7.xlsx   --metadata data/cell_metadata.csv   --id-col cell_id   --celltype-col celltype   --agg mean   --outdir results/gmmap_flux
