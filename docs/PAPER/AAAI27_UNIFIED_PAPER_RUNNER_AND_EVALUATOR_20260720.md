# AAAI-27 Unified Paper Runner and Evaluator

任务：`AAAI27-UNIFIED-PAPER-RUNNER-EVALUATOR-001`。

本实现把已冻结的 **Reference-Conditioned Explicit Gaussian Residual Basis** 协议转换为一个统一、append-only、可审计的 paper experiment 基础设施。本任务只建设与 CPU synthetic 验收，不运行任何正式实验。

## 1. 冻结边界

- 五套 seen outfits：`O01/O02/O03/O04/O08`。
- Held-out diagnostic：`O07`，固定为 `Held-out Diagnostic — FAIL`。
- Reserve：`O06`，不使用。
- Views：`cond_000000/cond_000318/cond_000017/cond_000347`。
- Seeds：`0/1/2`；trainable experiment 每 seed 精确 300 steps。
- Basis、base Gaussian、image backbone、MMLP-Human、renderer 全部冻结。
- Prediction forward 禁止 target RGB/mask、outfit ID、teacher residual/coefficient。

## 2. Registry 一致性修正

Registry 保留 55 条，不增删。51 条正式可执行实验使用：

```yaml
executable: true
status: NOT_RUN
```

4 条 A8 历史诊断使用：

```yaml
executable: false
status: HISTORICAL_EVIDENCE
evidence_class: HISTORICAL_EVIDENCE
```

A8 没有 adapter、没有 run command、不能迁移到 `RUNNING` 或 `PAPER_FINAL`。Stage 0 机器审计位于 `paper_protocol/audits/protocol_registry_audit.json`。

## 3. 统一 CLI

入口为 `tools/paper/aaai27_paper_pipeline.py`，提供：`validate`、`plan`、`run`、`resume`、`evaluate`、`aggregate`、`export-tables`、`export-figures`、`archive`、`status`。

所有子命令统一支持 `--dry-run`、`--registry`、`--config`、`--experiment-id`、`--seed`、`--output-root`、`--asset-root`、`--repo-root`。正式 `run/resume` 还要求显式 `--executor-command`；没有用户授权和 executor 时不会创建 attempt 或 optimizer。

## 4. Run output contract

正式根为：

```text
<OUTPUT_ROOT>/AAAI27-SEEN-OUTFIT-PAPER/<experiment_id>/seed_<seed>/attempt_<NNN>/
```

每个 attempt 固定包含 `contract/preflight/logs/checkpoints/raw_metrics/evaluated_metrics/visuals/tables/provenance/final_adjudication`。Attempt 使用首次不存在的三位编号创建；0-step 工具失败也保留，禁止覆盖。

每次真实 run 在 attempt 创建前完成 19 项 asset preflight。Mismatch 直接抛出 `PAPER_ASSET_MISMATCH`，不会创建 optimizer、checkpoint 或 attempt。通过后快照 registry、config、manifest、Git/environment/command、adapter contract、parameter freeze 和 training plan。

## 5. Adapters

8 个 adapter 类型覆盖 B0–B5、Ours 和 A1–A7。B0/B1/B2 是 fixed evaluation；B1 标记 `optimization_upper_bound`，B2 标记 `seen_only`。B3/B4/B5/Ours/A1–A7 的 training plan 固定 seed、300 steps、five-outfit balanced batch、round-robin view、milestones `0/20/50/100/200/300`、exact resume fields 和禁止 best-seed selection。

## 6. Evaluator 与 exporter

Evaluator 只读 raw per-episode JSONL；强制 20 个唯一 seen episode、80 个 cross-outfit swap、完整 permutation/single/dropout/replacement、target-forward boundary 和 asset fingerprint。共输出 27 个冻结指标，其中 zero/base replacement sensitivity 分开保存。

Aggregator 严格按 episode → outfit macro → seed → three-seed mean ± sample std。O07 永远单独导出。Table 1–4 同时输出 CSV/Markdown/LaTeX/JSON；Figure 1–6 使用固定 outfit/view/column 和统一 crop，不进行图像增强、伪影删除或成功样本选择。

## 7. 状态机

`NOT_RUN → PREFLIGHT_PASS → RUNNING → TRAINED → EVALUATED → MANUAL_REVIEW_REQUIRED → PAPER_FINAL`，任一执行态可按合同进入 `FAILED`。`FAILED` 与 `HISTORICAL_EVIDENCE` 不可回退。`PAPER_FINAL` 需要 evaluator PASS、产物完整、人工裁决存在和显式确认；代码不会根据指标自动升级。

下一唯一任务：`RUN_UNIFIED_PAPER_SMOKE_AND_EVALUATOR_ACCEPTANCE`。
