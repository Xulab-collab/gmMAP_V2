#代谢药物
python batch_rank_drug_reversal_all_celltypes.py \
  --trait_diff_dir /path/to/input \
  --pattern "trait_score_diff_*.csv" \
  --cmap_drug /path/to/input \
  --outdir Drug_Reversal_ALL_Celltypes_GEOdataset \
  --effect_col z_wilcoxon \
  --q_col q_wilcoxon \
  --q_max 0.05 \
  --score_col median_score \
  --mode reverse \
  --min_traits_per_drug 3 \
  --standardize_drug_by_trait \
  --skip_allcelltypes_merged

#cMAP药物
python batch_rank_drug_reversal_all_celltypes.py \
  --trait_diff_dir /path/to/input \
  --pattern "trait_score_diff_*.csv" \
  --cmap_drug /path/to/input \
  --outdir Drug_Reversal_ALL_Celltypes_CMAP_dataset \
  --effect_col z_wilcoxon \
  --q_col q_wilcoxon \
  --q_max 0.05 \
  --score_col median_score \
  --mode reverse \
  --min_traits_per_drug 3 \
  --standardize_drug_by_trait \
  --skip_allcelltypes_merged
  
  
  
######带统计  rank_percentile版本
python batch_rank_drug_reversal_all_celltypes_with_rank_percentile.py \
  --trait_diff_dir TraitScore_GroupDiff_allCelltypes \
  --pattern "trait_score_diff_*.csv" \
  --cmap_drug custom_CMap_trait_all_merged/custom_CMap_ALL_drug_level.csv.gz \
  --outdir Drug_Reversal_ALL_Celltypes \
  --effect_col z_wilcoxon \
  --q_col q_wilcoxon \
  --q_max 0.05 \
  --score_col median_score \
  --mode reverse \
  --min_traits_per_drug 3 \
  --standardize_drug_by_trait \
  --skip_allcelltypes_merged
  
  
  
  
  
  
#######可视化结果  
python plot_panCancer_epithelial_metabolism_drugs.py \
  --trait_diff TraitScore_GroupDiff_allCelltypes/trait_score_diff_ALLCELLTYPES_Tumor_vs_Adjacent.csv.gz \
  --drug_rank Drug_Reversal_ALL_Celltypes/ALL_celltype_drug_reversal_rankings.csv.gz \
  --outdir Fig_panCancer_epithelial_metabolism_drugs \
  --effect_col mean_diff \
  --q_col q_wilcoxon \
  --drug_score_col final_rank_score \
  --q_cutoff 0.05 \
  --top_n_drugs 18 \
  --n_label_traits 10 \
  --epithelial_pattern "epi|epithelial" \
  --max_celltypes 30 
  


python plot_panCancer_epithelial_metabolism_drugs_v2.py \
  --trait_diff TraitScore_GroupDiff_allCelltypes/trait_score_diff_ALLCELLTYPES_Tumor_vs_Adjacent.csv.gz \
  --drug_rank Drug_Reversal_ALL_Celltypes/ALL_celltype_drug_reversal_rankings.csv.gz \
  --outdir Fig_panCancer_epithelial_metabolism_drugs_v2 \
  --effect_col mean_diff \
  --q_col q_wilcoxon_global \
  --drug_score_col final_rank_score \
  --q_cutoff 0.01 \
  --effect_cutoff auto \
  --null_q_min 0.20 \
  --null_effect_quantile 0.95 \
  --effect_fallback_quantile 0.90 \
  --max_neglog10q 60 \
  --x_quantile 0.995 \
  --top_n_drugs 8 \
  --n_label_traits 10 \
  --y_limit 330 \
  --celltypes_per_page 6 \
  --max_celltypes 20 \
  --epithelial_pattern "epi|epithelial" \
  --only_positive_drug_score
  
  
  


python plot_panCancer_epithelial_metabolism_drugs_v2.py \
  --trait_diff TraitScore_GroupDiff_allCelltypes/trait_score_diff_ALLCELLTYPES_Tumor_vs_Adjacent.csv.gz \
  --drug_rank Drug_Reversal_ALL_Celltypes_CMAP_dataset/ALL_celltype_drug_reversal_rankings.csv.gz \
  --outdir Fig_panCancer_epithelial_metabolism_drugs_CMAPdataset \
  --effect_col mean_diff \
  --q_col q_wilcoxon_global \
  --drug_score_col final_rank_score \
  --q_cutoff 0.01 \
  --effect_cutoff auto \
  --null_q_min 0.20 \
  --null_effect_quantile 0.95 \
  --effect_fallback_quantile 0.90 \
  --max_neglog10q 60 \
  --x_quantile 0.995 \
  --top_n_drugs 8 \
  --n_label_traits 10 \
  --y_limit 330 \
  --celltypes_per_page 6 \
  --max_celltypes 20 \
  --epithelial_pattern "epi|epithelial" \
  --only_positive_drug_score