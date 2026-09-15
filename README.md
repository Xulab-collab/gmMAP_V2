> **v1.1 counts input:** `gmmap-bulk-smrs analyze --input-scale counts` now computes per-sample CPM followed by natural log1p and freezes preprocessing settings for prediction. Existing log1p inputs remain supported.

> **Bulk exposure validation:** `gmmap-bulk-smrs analyze` / `predict` now support donor-aware SMRS validation, frozen matched backgrounds and sample-label permutation tests. See [bulk SMRS methods and usage](BULK_SMRS_说明.md).

> Optional matched gene-set background calibration is available through `gmmap-score --background matched`. See [Chinese methods and usage](MATCHED_BACKGROUND_说明.md). This is scDRS-inspired, not the original scDRS algorithm. Full workflows are unchanged.

# gmMAP

**gmMAP** (*Genetically informed Metabolite trait Mapping across single-cell and spatial tissues*) is a Python toolkit for projecting GWAS-informed metabolite trait programs onto single-cell and spatial transcriptomic datasets. The package is designed to identify cell-type-specific, tissue-level, spatially organized, trajectory-associated, perturbation-associated and drug-reversible metabolite programs.


---

## Overview

gmMAP links metabolite trait genetics with transcriptomic cell states. Given a metabolite-gene association matrix, such as MAGMA gene-level Z statistics derived from metabolite GWAS summary statistics, gmMAP constructs directional metabolite-associated gene programs and scores these programs in single-cell or spatial transcriptomic observations. Downstream modules then test whether metabolite programs are enriched in specific cell types, spatial regions, developmental trajectories, inflammatory states, cancer states or drug-response signatures.

The package currently provides the following command-line modules:

| Command | Purpose |
|---|---|
| `gmmap-score` | Build UP/DOWN metabolite-gene programs and calculate gmMAP scores. |
| `gmmap-association` | Perform one-vs-rest metabolite-cell-type association analysis. |
| `gmmap-spatial` | Project metabolite programs onto spatial transcriptomic tissues. |
| `gmmap-pseudotime` | Associate metabolite programs with pseudotime or disease trajectories. |
| `gmmap-flux` | Estimate pathway-level metabolic-flow potential from metabolite programs. |
| `gmmap-perturb` | Predict effects of metabolite perturbation on cell-state transitions. |
| `gmmap-drug` | Rank drugs that may reverse disease-associated metabolic programs. |

---

## Repository structure

```text
gmmap/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── configs/
│   ├── default.yaml
│   └── flux_modules.yaml
├── examples/
│   ├── README.md
│   └── run_minimal.sh
├── src/gmmap/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── io.py
│   ├── logging_utils.py
│   ├── core/
│   │   ├── signatures.py
│   │   ├── scoring.py
│   │   ├── normalization.py
│   │   └── association.py
│   └── apps/
│       ├── spatial.py
│       ├── pseudotime.py
│       ├── flux.py
│       ├── perturb.py
│       └── drug.py
└── tests/
    ├── test_import.py
    └── test_core_toy.py
```

The core workflow is organized into reusable modules for signature construction, scoring, normalization and association testing, with application modules for spatial mapping, pseudotime analysis, flux-potential inference, perturbation analysis and drug-reversal analysis.

---

## Installation

### Option 1: conda environment

```bash
conda env create -f environment.yml
conda activate gmmap
python -m pip install -e ".[dev,spatial]"
```

### Option 2: editable installation for development

```bash
git clone https://github.com/Xulab-collab/gmMAP_V2.git
cd gmMAP

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

python -m pip install -U pip
python -m pip install -e ".[dev,spatial]"
```

### Option 3: install from a GitHub repository

```bash
python -m pip install "gmmap @ git+https://github.com/Xulab-collab/gmMAP.git"
```

For a private repository using SSH:

```bash
python -m pip install "gmmap @ git+ssh://git@github.com/YOUR_USERNAME/gmmap.git"
```



---

## Input files

### 1. Metabolite-gene Z-score matrix

gmMAP expects a gene-by-trait matrix. Rows are genes and columns are metabolite traits, such as GCST accessions, metabolite names or curated trait identifiers.

Example:

| gene | GCST000001 | GCST000002 | Lactate_levels |
|---|---:|---:|---:|
| LDHA | 2.84 | -0.41 | 3.21 |
| ACO2 | -1.32 | 2.16 | -0.86 |
| CPT1A | 0.74 | 3.02 | -1.11 |

Requirements:

- The first column should contain gene symbols or stable gene identifiers.
- The remaining columns should contain numeric Z statistics.
- Gene names should match `adata.var_names` in the transcriptomic object.
- Missing or non-finite values should be removed or imputed before running the main workflow.

### 2. Single-cell or spatial transcriptomic data

gmMAP uses an AnnData `.h5ad` object as the main transcriptomic input.

Required fields:

```text
adata.X                  expression matrix, preferably raw counts or normalized expression
adata.var_names           gene names
adata.obs[celltype_key]   cell type, state, region or cluster annotation
```

Recommended optional fields:

```text
adata.obs["Sample"]          sample identifier
adata.obs["organ_tissue"]    organ or tissue label
adata.obs["Subregion"]       spatial subregion label
adata.obs["x_image"]         spatial x coordinate
adata.obs["y_image"]         spatial y coordinate
adata.obs["pseudotime"]      pseudotime value
adata.obs["branch"]          trajectory branch label
```

---

## Quick start

### Step 1: score metabolite programs

```bash
gmmap-score \
  --h5ad data/input.h5ad \
  --magma-z data/MAGMA_zstat.csv \
  --celltype-key celltype \
  --out results/gmmap_scores \
  --top-n-genes 1000 \
  --score-method waucell \
  --stage S3
```

Main outputs:

```text
results/gmmap_scores/
├── gmmap_scores_UP_S3.csv
├── gmmap_scores_DOWN_S3.csv
├── gmmap_scores_NET_S3.csv
├── obs_metadata.csv
└── signature_summary.csv
```

### Step 2: identify cell-type-associated metabolite programs

```bash
gmmap-association \
  --up-scores results/gmmap_scores/gmmap_scores_UP_S3.csv \
  --down-scores results/gmmap_scores/gmmap_scores_DOWN_S3.csv \
  --metadata results/gmmap_scores/obs_metadata.csv \
  --celltype-key celltype \
  --out results/gmmap_association \
  --min-cells-in-type 20
```

Main outputs:

```text
results/gmmap_association/
├── association_statistics.csv
├── final_signed_associations.csv
└── top_metabolite_celltype_pairs.csv
```

### Step 3: run an application module

For spatial projection:

```bash
gmmap-spatial \
  --h5ad data/spatial.h5ad \
  --scores results/gmmap_scores/gmmap_scores_NET_S3.csv \
  --trait "Lactate_levels" \
  --sample-key Sample \
  --x-key x_image \
  --y-key y_image \
  --out results/spatial_lactate
```

For pseudotime association:

```bash
gmmap-pseudotime \
  --scores results/gmmap_scores/gmmap_scores_NET_S3.csv \
  --metadata results/gmmap_scores/obs_metadata.csv \
  --pseudotime-key pseudotime \
  --traits "Lactate_levels" "N_lactoyl_valine_levels" \
  --out results/pseudotime
```

---

## Application modules

### gmMAP-spatial

`gmmap-spatial` projects metabolite program scores onto spatial transcriptomic coordinates and summarizes spatial enrichment across samples, organs, tissue regions or manually annotated domains.

Typical use cases:

- spatial metabolite program visualization;
- organ-level or region-level metabolite enrichment;
- inflammatory versus homeostatic spatial comparison;
- spatial validation of genetically informed metabolite programs.

### gmMAP-pseudotime

`gmmap-pseudotime` links metabolite-associated scores with pseudotime, trajectory branches or disease-state progression.

Typical use cases:

- developmental metabolic remodeling;
- inflammatory fibroblast differentiation;
- cell-state transition analysis;
- branch-specific metabolite program dynamics.

### gmMAP-flux

`gmmap-flux` estimates relative metabolic-flow potential by integrating gmMAP scores with module definitions, directionality information and transcriptome-derived feasibility constraints.

Important note: gmMAP-flux estimates **relative metabolic-flow potential**. It is not a stoichiometric flux-balance analysis model and should not be interpreted as absolute biochemical flux.

### gmMAP-perturb

`gmmap-perturb` compares activity and inhibition signatures to predict how metabolite perturbation may shift cells along developmental or disease trajectories.

Typical outputs include:

- activity score;
- inhibition score;
- activity-minus-inhibition score;
- branch-specific perturbation trend;
- candidate metabolite prioritization.

### gmMAP-drug

`gmmap-drug` ranks candidate drugs according to their potential to reverse disease-associated metabolite programs.

Typical inputs include:

- disease-associated metabolite signature;
- drug-associated metabolite or transcriptional signature;
- optional metabolite annotation table;
- optional drug class and clinical annotation table.

---

## Recommended default parameters

The following defaults are suitable for initial analyses and can be modified in `configs/default.yaml`:

```yaml
top_n_genes: 1000
min_valid_genes: 200
min_cells_in_type: 20
score_method: waucell
top_frac_waucell: 0.05
normalization_stage: S3
multiple_testing: bh
q_cutoff: 0.05
empirical_null_background_q: 0.20
empirical_null_quantile: 0.95
specificity_gap_min: 0.03
```

Suggested interpretation:

- `UP` program: genes positively associated with a metabolite trait.
- `DOWN` program: genes negatively associated with a metabolite trait.
- `NET` score: signed or contrastive score derived from UP and DOWN programs.
- `S3` score: recommended normalized score for cross-cell-type or cross-state comparison.
- `final_effect`: signed metabolite-cell-type effect determined from the dominant UP/DOWN direction.

---

## Output interpretation

gmMAP identifies metabolite-associated cellular programs inferred from genetically informed metabolite-gene relationships and transcriptomic states. A positive metabolite-cell-type association should generally be interpreted as evidence that the cell type is transcriptionally aligned with a metabolite-associated genetic program.

Depending on the metabolite trait and validation context, this may suggest metabolite exposure, accumulation, utilization, signaling, epigenetic regulation, redox activity or pathway remodeling. However, gmMAP scores should not be interpreted as direct absolute metabolite concentrations unless supported by orthogonal metabolomics, imaging mass spectrometry, isotope tracing or targeted biochemical validation.

---

## Data privacy and repository hygiene

Do not commit private or unpublished files to the repository.

The following files and directories should remain local or be stored in controlled-access storage:

```text
data/
results/
outputs/
*.h5ad
*.h5seurat
*.rds
*.loom
*.csv.gz
*.tsv.gz
*.pkl
*.npz
*.npy
private_annotations/
raw_gwas/
intermediate_results/
```

Before pushing to GitHub, check staged files carefully:

```bash
git status
git diff --cached --stat
```

For private projects, initialize the repository as private:

```bash
gh repo create gmmap --private --source=. --remote=origin --push
```

---


## Full manuscript workflow commands

In addition to the lightweight package API, this repository includes full gmMAP workflow modules under `src/gmmap/workflows/`. These commands provide command-line entry points for the manuscript analysis workflows.

### Main Scheme C pipeline

```bash
gmmap-main-full   --h5ad data/input.h5ad   --magma-z data/MAGMA_zstat.csv   --celltype-key celltype   --prefix Meta   --outdir results/Meta_SchemeC_AUROC_UpDownNet_outputs   --n-jobs 16   --top-n-genes 1000   --min-valid-genes 200   --methods-to-run wAUCell   --stages-to-run S1_raw,S3_geneSetZ_cellZ
```

### Spatial workflow

```bash
gmmap-spatial-auroc-full   --h5ad data/spatial_input.h5ad   --magma-z data/MAGMA_zstat.csv   --celltype-key Subregion   --prefix Meta   --outdir results/spatial_auroc
```

Additional spatial plotting commands:

```bash
gmmap-spatial-identify --help
gmmap-spatial-violin-batch --help
gmmap-spatial-violin-similarity --help
gmmap-spatial-groupdiff --help
```

### Pseudotime and perturbation workflows

```bash
gmmap-pseudotime-pyucell   --h5ad data/trajectory.h5ad   --magma-z data/MAGMA_zstat.csv   --celltype-col celltype   --pseudotime-col pseudotime   --outdir results/pseudotime
```

```bash
gmmap-perturb-full   --h5ad data/trajectory.h5ad   --magma-z data/MAGMA_zstat.csv   --celltype-col celltype   --pseudotime-col pseudotime   --outdir results/perturb
```

For dataset-specific lineages or trajectory definitions, pass a Python literal dictionary using `--lineages` or `--trajectory-config`, or use repeated `--set KEY=VALUE` overrides.

### gmMAP-flux

```bash
gmmap-flux-full   --annotation src/gmmap/resources/flux/module_ratio_template.csv   --score-matrix results/Meta_wAUCell_up_S3_geneSetZ_cellZ.csv.gz   --gene-matrix data/cell_by_gene_expression_matrix.csv.gz   --enzyme-annotation src/gmmap/resources/flux/Module_round7.xlsx   --metadata data/cell_metadata.csv   --id-col cell_id   --celltype-col celltype   --agg mean   --outdir results/gmmap_flux
```

```bash
gmmap-flux-plot --input-dir results/gmmap_flux --outdir results/gmmap_flux_figures
```

### gmMAP-drug

```bash
gmmap-drug-rank-full   --diff-dir results/TraitScore_GroupDiff_allCelltypes   --drug-table src/gmmap/resources/drug/custom_cmap_metabolic_drugs_level.csv.gz   --outdir results/gmmap_drug_reversal
```

```bash
gmmap-drug-plot --help
```

### Bundled reference resources

The repository includes selected non-private reference files:

```bash
gmmap-resources
```

These resources include the gmMAP-flux module-ratio reference table, enzyme annotation workbook and the custom CMap metabolic drug-level table used by the analysis workflow. Private raw datasets, unpublished intermediate results and patient/animal metadata should not be committed to the repository.

## Development

Install development dependencies:

```bash
python -m pip install -e ".[dev,spatial]"
```

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check src tests
```

Build a local package:

```bash
python -m build
```

The generated distribution files will be placed under:

```text
dist/
├── gmmap-*.tar.gz
└── gmmap-*.whl
```

---

## Citation

If you use gmMAP in a manuscript, please cite the associated paper or preprint after publication.

Suggested citation placeholder:

```text
Xu H. et al. gmMAP: Genetically informed Metabolite trait Mapping across single-cell and spatial tissues. Manuscript in preparation.
```

A `CITATION.cff` file is included so that GitHub can display citation metadata after the repository is released.

---

## License

This repository is currently distributed as proprietary research software unless a different license is added by the authors.

Recommended private-stage license statement:

```text
Copyright (c) Heng Xu.
All rights reserved. This software is provided for internal research use only.
Redistribution, sublicensing, public release or commercial use is not permitted without written permission.
```

Before public release, replace this section with an appropriate open-source license, such as MIT, BSD-3-Clause, Apache-2.0 or GPL-3.0, depending on institutional and collaborator requirements.

---

## Contact

For questions, bug reports or collaboration requests, please open an issue in the GitHub repository after the project is released, or contact the repository maintainer directly for private-stage development.
