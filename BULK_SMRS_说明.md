# bulk-SMRS：外源代谢物暴露识别

本版本增加 `gmmap-bulk-smrs`，也可用 `gmmap bulk-smrs`。原有 gmmap-score、wAUCell、full workflows 与 flux 保留。SMRS 指核心代码的遗传关联权重加权平均表达，不是另一个名为 scMRS 的算法。本扩展是研究分析工具，尚未用用户真实样本验证。

## 适用实验与解释

适用：一类原代细胞、预先指定的一个外源代谢物、处理和对照两组。主要输出该代谢物 GWAS 候选程序的处理效应与供体外识别能力。不根据本批差异基因重新训练目标基因集，不寻找最佳代谢物，不推断浓度、通量或作用方向。

标准 MAGMA ZSTAT 是关联强度而非效应方向。因此使用排名靠前的正关联统计量作为权重，不把负 ZSTAT 当作抑制。处理可能提高或降低 score，效应检验为双侧；分类方向只从训练供体学习。原始 SMRS 和校正 SMRS 并行输出，建议事先指定主要终点为校正 SMRS，不要事后挑表现更好的结果。

## 已完成的流程

1. 输入检查与预处理：支持 --input-scale counts（整数原始计数，自动归一化并取 log1p）或 --input-scale log1p（已经转换的表达）；样本必须一一对应、ID 唯一、数值非负且有限、恰好两组。
2. 以供体×条件为单位；重复样本默认报错，可明确选择先平均。
3. SMRS 原始评分保留。默认权重不附加方差惩罚，大小、表达均值/方差分箱匹配的随机集使用同样的遗传权重分布。
4. 全队列拟合用于描述效应；供体内处理减对照的差、配对 t 检验与近似 95% CI，或独立供体的 Welch 检验与 CI。
5. 处理效应的主要 P 值使用样本标签置换，独立于随机基因集 P 值。
6. 留一供体交叉验证：该供体所有样本共同留出；基因资格筛选、分箱、随机集、可选方差惩罚、基因集中心、分类方向、阈值、尺度均从训练样本确定。
7. 报告 OOF ROC-AUC、敏感度、特异度、平衡准确率；配对实验另报供体内排序正确率（并列记 0.5）。没有事后把 AUC<0.5 翻转为 1-AUC。
8. AUC 显著性置换时重新学习每个训练折的方向和阈值。固定折的背景不依赖标签，缓存结果与每次重算等价，避免无意义重复拟合。
9. 保存最终全训练队列拟合模型；新样本单独预测，不重新估计背景，不用其分组标签。

## 输入准备

- counts.csv 或 expression_log1p.csv：行=基因，列=样本，首列=基因 ID。counts 文件只含基因 ID 和数值样本列，不包含 Length/Chr/Start 等注释列或汇总行。
- metadata.csv：行=样本，首列=样本 ID，另有 condition 与 donor。
- magma_z.csv：行=基因，列=代谢物性状，首列=基因 ID。
- 三者 ID 应事先统一。以送检样本为列，不能把多个供体先合并成一个处理组均值。

当前版本支持原始 counts 自动预处理，方法见下方新增章节。已有转换矩阵用 `--input-scale log1p`；旧参数 `--input-is-log1p` 作为兼容别名保留，两者不能同时使用。程序不会对 log1p 模式重复转换，也不自行做差异表达分析。不要在全队列先进行利用处理标签的特征选择/批次校正；若上游预处理需要跨样本学习参考，也必须在训练折拟合。独立预测数据须使用同一平台/基因注释与预处理定义。

配对元数据示例：

```csv
sample_id,condition,donor
D01_control,control,D01
D01_treated,treated,D01
D02_control,control,D02
D02_treated,treated,D02
```

以上仅展示格式。程序要求配对验证至少 4 个完整供体，独立设计每组至少 3 个不同供体；这些是计算运行下限，不是推荐样本量或统计效能保证。没有独立供体重复时，不能验证跨供体推广。

同一供体多个培养孔/技术重复默认拒绝自动当作独立样本。若实验设计适合先汇总，可增加 `--aggregate-replicates`：程序对已变换表达的供体×条件样本取均值，再计算分数。它不是原始计数 pseudobulk 求和，不适用于需要保留不同时间/剂量结构的数据。

## 安装与运行

在解压项目目录：

```powershell
python -m pip install -e .
```

配对设计（同一供体分成处理和对照）：

```powershell
gmmap-bulk-smrs analyze --expression expression_log1p.csv --metadata metadata.csv --magma-z magma_z.csv --trait YOUR_METABOLITE --group-key condition --treated treated --control control --design paired --donor-key donor --input-is-log1p --top-n-genes 1000 --min-valid-genes 200 --n-ctrl 1000 --mean-bins 10 --var-bins 10 --n-perm 999 --seed 0 --out bulk_smrs_result
```

独立设计使用 `--design independent`，仍建议明确 donor 列；供体不能跨组。只有每个样本确实来自不同独立供体时，才可省略 donor-key，以样本 ID 充当供体 ID。若每个供体包含两种条件，应使用 paired，不要用 independent。

输出目录必须为空或不存在，避免混合不同运行结果。

新样本独立识别：

```powershell
gmmap-bulk-smrs predict --expression new_samples_log1p.csv --model bulk_smrs_result/fitted_model.npz --input-is-log1p --out new_sample_predictions.csv
```

预测不需要 metadata 或 GWAS，不重新选择基因或背景。log1p 模型要求所有目标和对照基因都存在，允许额外基因；counts 模型要求完整输入基因集合与训练时一致，以保持总 counts 分母定义。缺失基因明确报错，不用 0 冒充缺失表达。基因顺序不影响结果。输出 decision 是训练尺度的分类分数，不是暴露概率。大于 0 判为训练时的处理标签，否则为对照；零信息时方向为 0。

## 校准定义与训练/测试隔离

SMRS 为正关联权重的平均表达：R(s,j)=Σ w(g,j)X(s,g)/Σw(g,j)，j=0 为目标、其余为匹配集。默认 alpha=0，与核心 SMRS 一致。

每个训练折记录各基因集训练样本均值 μ(j)。对于任意训练或测试样本，先算 A(s,j)=R(s,j)−μ(j)，再对该样本的目标与全部匹配集分数做中心化/标准差缩放。全程不估计测试队列的均值、方差或分箱；样本内部的缩放仅依赖它自己的表达和固定基因集，故同一测试样本单独预测与批量预测一致。

注意：这是上次 scDRS-inspired 校准的训练参考扩展，不是原版 scDRS。校正可能提升也可能降低分类能力，不能预先保证它优于原始 SMRS。全队列 sample_scores 是描述性结果，预测能力必须读 out_of_fold_predictions，不能拿最终训练模型在同一队列的预测当作独立验证。

## 检验、共表达与小样本

处理效应：配对设计平均供体内差；独立设计为两组供体平均差。样本标签置换以绝对差为统计量，保留每个样本整条表达谱。配对设计只在供体内部交换处理/对照；独立设计在独立单位间置换。因此不会打散基因的共表达。这处理的是样本层面零假设，不等同于声称匹配随机集已经控制所有基因相关性。

如果配对供体不超过 12 个且全部 2^D 种交换不超过 --n-perm，则自动枚举全部配对交换，用 count/N 计算精确 P；否则 Monte Carlo 采样并采用 (count+1)/(B+1)。run.json 记录实际方式与次数。增加随机次数不能突破独立供体少导致的真实统计分辨率：例如 4 个完整供体的双侧平均差检验，最小精确 P 通常为 2/16=0.125。

AUC 使用相同设计的标签置换，并在每次置换中重新学习训练方向、阈值。OOF 预测之间共享训练样本，不能把它们当独立观测直接计算普通独立样本 AUC P。本版不提供未经校正的简单 AUC 置信区间；独立外部供体验证仍是重要补充。

参数 t/Welch P 与 CI 作为辅助结果；CI 条件于已经拟合的评分，不包含基因集选择/背景估计的全部不确定性。优先解读样本置换 P。置换有效性要求处理标签在相应设计中可交换；处理与批次完全重合、存在时间序列等情况不满足默认设计。

本版未加入年龄/批次等多协变量回归或一般混合模型。配对比较控制稳定的供体差异，但不能修复处理与测序批次混杂。均值/方差匹配也不能替代此类研究设计。

## 输出

- validation_summary.csv：每种评分的处理效应、近似 CI、参数 P、置换 P、OOF AUC、AUC 置换 P、敏感度/特异度/平衡准确率。effect_q_two_scores 和 auc_q_two_scores 分别只校正本次两种分数，不是跨所有代谢物/剂量/时间的全局 FDR。
- sample_scores.csv：全队列描述性原始/校正分数、随机背景 P、供体、标签、原始样本列表。gene_set_p_upper 仅检验向上富集，不是处理效应双侧 P，不用于筛选哪些样本进入分类。
- paired_differences.csv：各供体处理减对照（配对设计）。
- out_of_fold_predictions.csv：真正留出供体的分类分数、类别、该折训练方向。
- fold_diagnostics.csv：各留出折训练/测试数量、实际目标基因数、目标/随机集均值方差、重叠比例。
- gene_matching_stats.csv.gz、target_genes.csv：最终全训练队列的基因匹配和候选集。
- fitted_model.npz：固定基因集/权重/方差/训练中心/分类参数；使用 allow_pickle=False 加载。
- permutation_null.npz：效应与 AUC 的置换零分布。
- run.json：参数、设计与解释边界。

高重叠或匹配分布差异大时，应查看诊断并做预先规定的敏感性分析。bulk 默认最多 10×10 个箱，比上一版单细胞默认分箱更粗；这只是起点，小样本不应机械增加箱数。单个性状包含过多目标基因时，对照多样性可能不足。

## 示例与验证

```powershell
python examples/bulk_smrs/demo.py --out synthetic_demo
python -m pytest tests -q
```

示例是人为加入较强处理信号的合成数据，不是真实细胞实验或灵敏度基准。本次完整测试 30 项通过，覆盖原流程、SMRS 一致、配对/独立设计、重复拒绝与汇总、测试表达/标签隔离、冻结模型单样本预测、置换结构、精确配对枚举、无组间信息时 AUC=0.5/P=1，以及 analyze/predict 命令入口。

对实际研究的适当结论是“识别 X 处理相关的成纤维细胞转录状态”。只测试一种 X 对照组不能证明它区别于其他代谢物或一般应激；本版未将暴露识别等同于代谢物特异性。


## counts 自动预处理（v1.1 新增）

对每个样本 s，先用输入文件中全部基因求总 counts L(s)，再计算：

```
CPM(g,s) = counts(g,s) / L(s) * 1,000,000
X(g,s)   = ln(1 + CPM(g,s))
```

`log1p(x)` 就是自然对数 ln(1+x)。不是直接对原始 counts 做 ln(counts+1)。总 counts 在 GWAS 取交集、方差过滤、候选基因筛选之前计算；不要只提交差异基因或目标代谢物基因。

此方法不使用其他样本的统计量或分组标签，所以可以先对每个样本独立执行，再做留出供体的背景拟合，不产生跨样本归一化泄漏。它是简单总量缩放，不是 TMM、DESeq2 size factor 或 TPM（没有基因长度校正）。是否适合真实数据仍需 QC；该实现不宣称纠正了所有 RNA 组成偏倚或批次效应。

原始 counts 运行：

```powershell
gmmap-bulk-smrs analyze --expression counts.csv --input-scale counts --metadata metadata.csv --magma-z magma_z.csv --trait YOUR_METABOLITE --group-key condition --treated treated --control control --design paired --donor-key donor --out bulk_smrs_counts_result
```

默认总量为 1,000,000，因此称 CPM。高级选项 `--normalization-target-sum` 可以改变总量；若改变，就不能再把结果称为默认 log1p(CPM)。所有折使用同一固定总量，模型保存该值。不要根据留出测试表现选择该参数。

新样本也提交原始 counts：

```powershell
gmmap-bulk-smrs predict --expression new_counts.csv --input-scale counts --model bulk_smrs_counts_result/fitted_model.npz --out new_predictions.csv
```

预测从模型读取缩放总量，禁止把 log1p 输入交给 counts 模型，或把 counts 输入交给 log1p 模型。counts 预测需提供与训练相同的完整基因集合；顺序可以不同，真实测得的 0 可以保留，未测得的基因不能自行补零。最终模型格式已升级为 v2；新代码仍可读取上一版本的 v1 log1p 模型。

支持非负整数 counts（浮点存储的 10.0 可以）。负数、NaN/Inf、全零样本和非整数计数会报错。Salmon/RSEM 等产生的非整数 estimated counts 不在当前严格 counts 模式范围内，不要通过四舍五入偷偷改变真实数据；需在适当的上游流程处理后提供明确的 log1p 输入。数值检查不能证明数据来源，整数化的 TPM 也不能冒充 counts。

新增输出：

- preprocessing_qc.csv：每个原始样本的总 counts、非零基因数、缩放系数和目标总量。
- normalized_expression_log1p.csv.gz：转换后的完整基因×原始样本矩阵，保存于供体内重复汇总之前；log1p 输入则原样导出。
- run.json 和 fitted_model.npz：记录输入尺度、归一化参数和预测要求的基因集合。

测试新增：公式核对、测序深度倍增不改变归一化结果、单样本/批量一致、自动和手工转换的完整分析与预测一致、无效 counts 拒绝、输入尺度/基因集合保护，以及 counts analyze/predict 命令端到端运行。
