# AAAI-27 Evaluation and Aggregation Contract

## 1. Raw input

正式 evaluator 输入为 `raw_metrics.jsonl`，而不是历史汇总表。每个 seed 必须具有：

- 1 条 metadata：seed、asset fingerprint、target-forward boundary、efficiency；
- 20 条 episode：五套 outfit × 四个固定 view；
- 80 条 cross-outfit swap；
- permutation、single-reference、two-reference dropout 和 zero/base replacement records；
- O07 只允许作为独立 held-out record。

任何重复或缺失 episode、swap 不足 80、non-finite、reference-target overlap、asset mismatch 或 target-forward leakage 都会拒绝整个 evaluation。

## 2. Metrics

共 27 项：

- Coefficient 5：standardized/restored RMSE、nearest-teacher accuracy、correct outfit rank、pairwise margin；
- Residual 4：normalized RMSE、cosine、top-10/top-20 support overlap；
- Render 6：garment RGB/alpha MAE、edit reduction、target-closer、protected/background RGB MAE；
- Reference 6：swap wins、permutation max diff、single/dropout accuracy、zero/base replacement sensitivity；
- Efficiency 6：trainable parameters、basis storage、peak VRAM、training/inference/render time。

Coefficient、residual 和 ranking 从 raw vectors 重新计算；render scalar 必须是 per-episode 原始测量，禁止只交付跨 outfit 聚合值。

## 3. Aggregation

固定顺序：

1. 每个 episode 计算；
2. 每个 outfit 的四 views 等权平均；
3. 五个 outfit 宏平均得到 per-seed；
4. seeds `0/1/2` 计算 mean ± sample standard deviation。

禁止 pixel-weighted outfit micro average、缺 seed 后继续、缺 episode 后继续、只报告最佳 seed，或把 O07 混入 seen aggregate。

保留 `raw_metrics.jsonl`、`per_episode_metrics.csv`、`per_outfit_metrics.csv`、`per_seed_metrics.csv`、`aggregate_metrics.json`、`aggregate_metrics.csv`。

## 4. Tables and figures

Table 1 固定 B0–B5 + Ours；B1 标记 Optimization Upper Bound，B2 标记 Seen-only Lookup。Table 2 只含 A1–A7，Table 3 报 efficiency，Table 4 标题固定为 `Held-out Diagnostic — FAIL`。缺失值写 `N/A`，不能写 0。

Figure 1–6 使用 `figure_source_path_manifest.json` 记录源路径、固定顺序、crop 和允许操作。Figure 2 的 outfit 顺序固定为 O01/O02/O03/O04/O08，view 顺序固定为 front/back/left/right 对应四个 condition；禁止 cherry-pick、retouch、artifact removal 或 color change。

Synthetic fixture 标记 `SYNTHETIC_TEST_ONLY`，只验证 schema、数学聚合和导出布局，禁止写入 registry 或正式 paper-ready 目录。
