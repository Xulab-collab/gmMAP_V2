# gmMAP-score 匹配随机基因集背景校准

## 交付范围

基于用户提供的 gmMAP-main (1).zip 修改。原始 ZIP 未改动。
新增 src/gmmap/core/background.py，修改 src/gmmap/cli.py，新增 tests/test_background.py。
功能通过 gmmap-score --background matched 启用；默认 --background none 保留旧流程。
**没有修改 gmmap-main-full、空间 full 工作流和 gmMAP-flux。** 若实际生产使用 full 脚本，需要另外对接，不能认为这些脚本已经自动应用本次校准。

本实现是 scDRS-inspired matched empirical calibration for gmMAP，不是原版 scDRS。

## 1. 原流程的问题与插入位置

原核心 gmMAP-score 先按 MAGMA 基因统计量选集，计算 wAUCell 或 SMRS，再作 S1–S4 标准化。跨细胞/跨性状 z-score 不等于与大小、表达量、方差匹配的随机基因集比较。原 empirical_null_threshold 也不是此类基因背景。

新流程：

MAGMA 关联强度 → 与表达基因取交集 → 选目标基因集 → 按均值/方差联合分箱抽取随机集 → 目标与对照采用同一原始评分 → 背景校准 → 单细胞/细胞群经验 P 与 BH 校正。

**在原始评分之后、S2/S3/S4 之前接入；新模式不再套 S1–S4。** 重复标准化会改变校准后的解释，不能直接沿用旧流程的显著性阈值。

## 2. 如何构建匹配基因集

输入必须是 library-size normalized、非负的 log1p 表达矩阵，不是原始 counts，也不是中心化/缩放后的 residual。程序检查有限值和非负性，无法仅凭数值确定你是否已完成正确预处理，所以需要 --input-is-log1p 明确声明。使用全体质量合格表达基因，建议不要只保留 HVG；基因命名必须和 MAGMA 一致。

对每个基因计算所有分析细胞上的平均表达 μ 和方差 σ²。排除全零或近零方差基因。先按 μ 分位数分箱，再在各均值箱内按 σ² 分箱，默认最多 20×20 个联合箱；相同值不强制拆分。

例如目标集实际有 K=1000 个基因，在各箱分别占 n₁,…,nⱼ 个。每个随机集也从相同箱中分别抽 n₁,…,nⱼ 个，因此：

- 集合大小严格为 K，每个集合内部没有重复基因。
- 每个联合箱的基因数与目标集严格相同。
- 表达均值/方差属于同一分箱，**不是每个数值完全相等**。
- 不同随机集允许重用基因；默认生成 B=1000 个。
- 默认目标基因也可进入随机池，遵循 scDRS 的抽样思路。可用 --exclude-target-controls 排除，但池不够时会报错，不会暗中放宽匹配。

若目标集与对照重叠很高、箱内替代基因太少，应检查匹配诊断，并考虑减少箱数。箱数越多不一定越好：匹配更细，但对照多样性下降。改变参数后应完整重跑。

目标集是在表达基因交集内选取最多 --top-n-genes 个，实际 K 可能小于 1000；随机集匹配实际 K。K 小于 --min-valid-genes 时跳过该性状。所有排除/跳过需在下游报告中说明。

## 3. 权重也必须匹配

目标基因权重 wᵢ 复制到对应箱内抽出的随机基因槽位。**不使用随机基因自己的 GWAS 统计量。** 否则随机对照的遗传权重通常更弱，会人为放大目标分数。

目标集与所有对照使用同一 wAUCell/SMRS 函数、同一 top-frac 和同一权重归一化。

--variance-alpha 0 是默认，保持原核心评分：权重直接按总和归一化。
若使用 --variance-alpha 0.5，则每个选中基因先按 w/(SD+1e-8)^0.5 调整，再归一化；目标与对照均如此。它是可选的表达方差惩罚，不是 scDRS 的技术噪声方差估计。full 脚本的实现与核心命令不能自动视为等价。

## 4. 背景校准和检验的精确定义

令 R(c,j) 为原始分数，j=0 为目标集，j=1,…,B 为随机集。

1. 每个基因集跨细胞中心化：A(c,j)=R(c,j)−mean_c R(c,j)。
2. 每个细胞内，使用目标与对照全部 B+1 个值计算均值 m(c)、标准差 s(c)。
3. T(c,j)=[A(c,j)−m(c)]/s(c)，零方差时记为 0。
4. P(c)=[1+Σ_j I(T(c,j)≥T(c,0))]/(B+1)，j 仅包含随机集；数值并列保守计入。

目标与对照对称参与背景估计，避免因单独处理目标破坏列置换对称性。不除以各基因集自身跨细胞 SD，因为那会把真实细胞群异质性吸收进分母。这里的 T 是标准化背景分数，不应套正态分布计算 P。

该经验检验依赖“在匹配条件下，目标与随机集近似可交换”的零假设；均值、方差匹配本身不能保证它在真实生物数据中成立，尤其不能匹配基因间所有共表达结构。因此需要实际数据的负对照与敏感性分析。

单细胞同时输出 raw_score、background_raw_mean、background_excess（原始目标减原始对照均值）、calibrated_score、p_mc、q_bh_within_trait。q 的检验范围仅为当前性状内所有细胞，不是所有代谢物×细胞的全局 FDR。

细胞群分析对目标和每个随机集均计算相同的 one-vs-rest 均值差，再构建经验 P。若提供 --sample-key，则先算每个供体内部的差，再对供体等权平均；缺少某一侧细胞的供体不参与该群比较。group_associations.csv 中 BH 范围为本次输出的所有性状×细胞群组合。

这种组检验是随机基因集背景下的关联检验，不是供体层面的处理效应检验，不提供生物学重复置信区间。批次、供体构成、疾病分组等混杂仍需研究设计与额外模型处理。BH 校正也不能弥补一个失准的零分布。

## 5. 为什么不是“完整复刻 scDRS”

沿用的思想：同尺寸、均值/方差分箱匹配、匹配遗传权重、同一评分规则、经验背景比较。

不同之处：保留 gmMAP 的 rank-based wAUCell（也支持 SMRS）；未使用 scDRS 的完整预处理/协变量校正、技术噪声模型、加权表达解析方差校正及 pooled-cell 零分布。本实现使用上面明确给出的对称校准与同细胞 Monte Carlo P。即使选择 SMRS，也不是原版 scDRS。

因此方法学写作应称“借鉴 scDRS 的匹配随机基因集背景校准”，不能称“与 scDRS 完全相同”。如果目标是报告原版 scDRS 结果，应另外运行官方 scDRS，作为独立对照方法。

参考：
- scDRS 论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC9891382/
- 官方实现：https://github.com/martinjzhang/scDRS/blob/master/scdrs/method.py

## 6. 使用方法

解压后在项目目录安装（建议独立环境）：

```powershell
python -m pip install -e .
```

实际数据运行；下面 paths、cell_type 和 donor 是占位符，需替换为你的文件/元数据列名：

```powershell
gmmap-score --h5ad cells.h5ad --magma-z magma_z.csv --background matched --input-is-log1p --top-n-genes 1000 --min-valid-genes 200 --n-ctrl 1000 --n-mean-bins 20 --n-var-bins 20 --score-method waucell --top-frac 0.05 --celltype-key cell_type --sample-key donor --random-seed 0 --out gmmap_calibrated
```

如果 log1p 数据在 layer 中，增加 --layer log1p；没有供体信息时删除 --sample-key donor。
建议先加 --traits 目标性状名 跑少数性状检查资源与匹配质量，然后扩大规模。

--n-ctrl 1000 时 P 下限为 1/1001≈0.001。在海量检验中，BH 可能因分辨率不足而不显著；这不等于无关联。更多对照可提高分辨率，但本实现不假借跨细胞 pooled P 获得额外精度。应按检验数量、算力及预先规定的分析层级规划。

默认工作数组预算 2 GiB，超限提前报错。它是估计，不是进程内存硬上限，也不包括所有原始输入/依赖开销。可用 --background-memory-gb 调整；按细胞分批排名，按性状依次计算，避免整个 cells×traits×controls 三维矩阵。

## 7. 输出与解释

结果在 out/matched_background/：

- trait_manifest.csv：性状与哈希子目录对应关系、实际基因数、是否跳过、P 分辨率、目标重叠率。
- gene_matching_stats.csv.gz：基因均值、方差、可用标记、联合箱。
- 每个性状目录：target_genes.csv、control_gene_indices.npz、matching_diagnostics.csv、cell_scores.csv.gz。
- 索引文件中的整数对应 gene_matching_stats.csv.gz 的行序（从 0 开始）。
- 可加 --save-control-scores 保存全部对照分数以进一步审计。
- group_associations.csv：细胞群结果（提供 --celltype-key 时）。
- background_run.json：关键分析参数。

校正后高分的适当解释：该细胞中，代谢物性状遗传关联候选基因程序的相对表达富集，超过表达特征相近的随机基因程序背景。

不表示该细胞内代谢物浓度高、不表示代谢物输入/输出方向，也不证明代谢物直接导致该细胞状态。跨细胞中心化意味着全体细胞一致升高的信号可能被消除；分数依赖分析细胞总体，不能直接把不同队列各自标准化后的分数视作绝对量。

另一个必要修正：标准 MAGMA ZSTAT 是由基因关联 P 转换而来，负值不是降低代谢物浓度或抑制细胞的效应方向。本模式使用正的关联强度候选并取排名靠前者，不生成 DOWN/NET。前 1000 个是候选多基因集合，不等于 1000 个已证实关联基因。若输入是真实带方向的效应估计，应另行设计方向一致的聚合，而不是直接把这个模式当作有符号效应分析。

校准结果不要传入旧 gmmap-association 的 UP/DOWN fallback 或沿用其原阈值；本模式已给出匹配背景的组检验。若用于 flux，应重新定义下游统计量，并保留“关联而非方向/因果”的解释。

## 8. 验证与仍需补充的研究证据

本次 14 项自动测试通过：包含原仓库测试以及精确分箱计数、集合大小、重复/种子、稀疏/稠密一致、wAUCell 与 SMRS 原始分数一致、方差权重、并列/退化情况、列置换对称、理想可交换零假设模拟、注入状态信号模拟、供体内群比较、实际 H5AD/CSV 新旧 CLI 流程、输出完整性和输入检查。

这些验证说明程序按定义运行；模拟仅为理想化合成数据，不是你真实数据上灵敏度/FDR 已获验证。此次未获得配套真实 h5ad 与 MAGMA 矩阵，未计算真实生物学结果。

发表前建议：检查每个性状的目标/对照分布与重叠；比较不同箱数和基因数；在真实表达数据上重复构造符合零假设的目标集；独立队列/供体重复；报告细胞群层面的稳定性；与官方 scDRS 和原 gmMAP 并行比较。匹配均值/方差只针对这类背景偏倚，不等于“去掉所有背景干扰”。
