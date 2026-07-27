# AAAI-27 Paper Experiment Command Templates

本文件只冻结命令接口，不授权运行训练。`PAPER_RUNNER`、`PAPER_EVALUATOR`、`PAPER_AGGREGATOR` 和 `PAPER_FIGURE_EXPORTER` 必须在下一任务中实现并通过协议测试后才能执行。所有命令从 repo root 运行，不硬编码用户名或 checkout 路径。

## 1. 环境变量

Windows PowerShell：

```powershell
$env:CANONDRESSGS_REPO_ROOT = (Resolve-Path .).Path
$env:CANONDRESSGS_OUTPUT_ROOT = '<repository-external output root>'
$env:CANONDRESSGS_PYTHON = '<python executable>'
$env:PAPER_RUNNER = '<validated paper runner path>'
$env:PAPER_EVALUATOR = '<validated evaluator path>'
$env:PAPER_AGGREGATOR = '<validated aggregator path>'
$env:PAPER_FIGURE_EXPORTER = '<validated figure exporter path>'
```

Cloud Linux：

```bash
export CANONDRESSGS_REPO_ROOT="$(pwd)"
export CANONDRESSGS_OUTPUT_ROOT='<repository-external output root>'
export CANONDRESSGS_PYTHON='<python executable>'
export PAPER_RUNNER='<validated paper runner path>'
export PAPER_EVALUATOR='<validated evaluator path>'
export PAPER_AGGREGATOR='<validated aggregator path>'
export PAPER_FIGURE_EXPORTER='<validated figure exporter path>'
```

## 2. Asset verification

Windows：

```powershell
& $env:CANONDRESSGS_PYTHON tools/paper/verify_seen_outfit_paper_assets.py `
  --manifest paper_protocol/frozen_asset_manifest.json `
  --repo-root $env:CANONDRESSGS_REPO_ROOT `
  --output-root $env:CANONDRESSGS_OUTPUT_ROOT
if ($LASTEXITCODE -ne 0) { throw 'PAPER_ASSET_MISMATCH' }
```

Linux：

```bash
"$CANONDRESSGS_PYTHON" tools/paper/verify_seen_outfit_paper_assets.py \
  --manifest paper_protocol/frozen_asset_manifest.json \
  --repo-root "$CANONDRESSGS_REPO_ROOT" \
  --output-root "$CANONDRESSGS_OUTPUT_ROOT" || { echo PAPER_ASSET_MISMATCH; exit 2; }
```

## 3. Baseline run template

```powershell
& $env:CANONDRESSGS_PYTHON $env:PAPER_RUNNER `
  --registry paper_protocol/experiment_registry.yaml `
  --experiment-id PAPER-B3-S0
```

```bash
"$CANONDRESSGS_PYTHON" "$PAPER_RUNNER" \
  --registry paper_protocol/experiment_registry.yaml \
  --experiment-id PAPER-B3-S0
```

B0/B1/B2 只执行 deterministic evaluation；B3/B4/B5 按 registry 分别执行 seeds 0/1/2。B1 输出必须标记 optimization upper bound，B2 禁止 O07。

## 4. Ablation run template

```powershell
& $env:CANONDRESSGS_PYTHON $env:PAPER_RUNNER `
  --registry paper_protocol/experiment_registry.yaml `
  --experiment-id PAPER-A1-K1-S0
```

```bash
"$CANONDRESSGS_PYTHON" "$PAPER_RUNNER" \
  --registry paper_protocol/experiment_registry.yaml \
  --experiment-id PAPER-A1-K1-S0
```

每个 experiment ID 只能产生唯一新输出目录；存在 COMPLETE/PASS 结果时幂等跳过，存在失败目录时创建新 attempt，不覆盖。

## 5. Three-seed run

Windows：

```powershell
0,1,2 | ForEach-Object {
  & $env:CANONDRESSGS_PYTHON $env:PAPER_RUNNER `
    --registry paper_protocol/experiment_registry.yaml `
    --experiment-id ("PAPER-OURS-S{0}" -f $_)
  if ($LASTEXITCODE -ne 0) { throw "seed $_ failed" }
}
```

Linux：

```bash
for seed in 0 1 2; do
  "$CANONDRESSGS_PYTHON" "$PAPER_RUNNER" \
    --registry paper_protocol/experiment_registry.yaml \
    --experiment-id "PAPER-OURS-S${seed}" || exit 1
done
```

禁止根据中间结果跳过 seed 或只保留最佳 seed。

## 6. Evaluator

```powershell
& $env:CANONDRESSGS_PYTHON $env:PAPER_EVALUATOR `
  --registry paper_protocol/experiment_registry.yaml `
  --config configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml `
  --aggregation outfit_macro `
  --output-root $env:CANONDRESSGS_OUTPUT_ROOT
```

```bash
"$CANONDRESSGS_PYTHON" "$PAPER_EVALUATOR" \
  --registry paper_protocol/experiment_registry.yaml \
  --config configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml \
  --aggregation outfit_macro \
  --output-root "$CANONDRESSGS_OUTPUT_ROOT"
```

Evaluator 必须先输出 per-episode/per-view/per-outfit/per-seed 数据，再做 outfit macro 和 seed mean/std。

## 7. Table aggregation

```powershell
& $env:CANONDRESSGS_PYTHON $env:PAPER_AGGREGATOR `
  --registry paper_protocol/experiment_registry.yaml `
  --source-data artifacts/paper_ready/source_data `
  --tables 1,2,3,4
```

```bash
"$CANONDRESSGS_PYTHON" "$PAPER_AGGREGATOR" \
  --registry paper_protocol/experiment_registry.yaml \
  --source-data artifacts/paper_ready/source_data \
  --tables 1,2,3,4
```

Table 4 必须保留 `Held-out diagnostic — FAIL`，不得进入 Table 1 aggregate。

## 8. Paper figure export

```powershell
& $env:CANONDRESSGS_PYTHON $env:PAPER_FIGURE_EXPORTER `
  --plan docs/PAPER/AAAI27_PAPER_TABLE_AND_FIGURE_PLAN_20260720.md `
  --output artifacts/paper_ready/figures
```

```bash
"$CANONDRESSGS_PYTHON" "$PAPER_FIGURE_EXPORTER" \
  --plan docs/PAPER/AAAI27_PAPER_TABLE_AND_FIGURE_PLAN_20260720.md \
  --output artifacts/paper_ready/figures
```

Exporter 必须拒绝未固定视角、缺少 source SHA、非等比例裁剪或缺少 O07 failure panel 的请求。

## 9. Reproducibility archive

```powershell
git rev-parse HEAD
git status --short
& $env:CANONDRESSGS_PYTHON tools/paper/verify_seen_outfit_paper_assets.py `
  --repo-root $env:CANONDRESSGS_REPO_ROOT --output-root $env:CANONDRESSGS_OUTPUT_ROOT
```

```bash
git rev-parse HEAD
git status --short
"$CANONDRESSGS_PYTHON" tools/paper/verify_seen_outfit_paper_assets.py \
  --repo-root "$CANONDRESSGS_REPO_ROOT" --output-root "$CANONDRESSGS_OUTPUT_ROOT"
```

Archive 至少包含 commit、dirty state、resolved config、asset verification、registry entry、command、environment、checkpoint SHA、raw metrics、aggregation、figures 和 final adjudication。禁止把 checkpoint、dataset 或历史 outputs 加入 Git。
