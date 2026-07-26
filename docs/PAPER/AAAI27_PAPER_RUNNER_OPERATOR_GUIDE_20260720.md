# AAAI-27 Paper Runner Operator Guide

## 1. 环境

Windows PowerShell：

```powershell
$env:CANONDRESSGS_REPO_ROOT = '<clean-repository>'
$env:CANONDRESSGS_ASSET_ROOT = '<frozen-asset-root>'
$env:CANONDRESSGS_OUTPUT_ROOT = '<paper-output-root>'
$env:CUDA_VISIBLE_DEVICES = '-1'
Set-Location -LiteralPath $env:CANONDRESSGS_REPO_ROOT
```

Linux：

```bash
export CANONDRESSGS_REPO_ROOT='<clean-repository>'
export CANONDRESSGS_ASSET_ROOT='<frozen-asset-root>'
export CANONDRESSGS_OUTPUT_ROOT='<paper-output-root>'
export CUDA_VISIBLE_DEVICES=-1
cd "$CANONDRESSGS_REPO_ROOT"
```

路径优先级为 CLI override → environment → protocol-relative。示例见 `paper_protocol/path_mapping.example.yaml`。

## 2. 安全命令

```bash
python tools/paper/aaai27_paper_pipeline.py validate --repo-root "$CANONDRESSGS_REPO_ROOT" --asset-root "$CANONDRESSGS_ASSET_ROOT" --output-root "$CANONDRESSGS_OUTPUT_ROOT"
python tools/paper/aaai27_paper_pipeline.py plan --dry-run --repo-root "$CANONDRESSGS_REPO_ROOT"
python tools/paper/aaai27_paper_pipeline.py status --repo-root "$CANONDRESSGS_REPO_ROOT"
```

`plan` 生成 JSON/CSV/Linux/Windows 共 51 条 executable commands，不执行命令、不创建正式 attempt、不改 registry。A8/O07/O06 不出现在 seen run commands。

## 3. Run 与失败处理

正式 run 必须由后续授权任务提供 `CANONDRESSGS_PAPER_EXECUTOR`。生成命令显式传入 executor；统一 runner 先完成全资产 preflight，再创建 append-only attempt 和快照，最后调用 executor。Preflight failure 发生在 attempt/optimizer 创建前。

Executor 非零退出时，attempt 保留，registry 原子进入 `FAILED`，不得改回 `NOT_RUN`。禁止删除 0-step attempt 或复用编号。

## 4. Resume

`resume` 要求 checkpoint 位于指定 attempt 内，并存在同名 `.resume.json` sidecar。Sidecar 必须含 model、optimizer、RNG、scheduler、global step、condition position、fixed-output parity。Resume 前重新验证 19 项 assets；缺字段或 fingerprint drift 立即停止。

## 5. Evaluate / Aggregate

`evaluate` 输入 raw JSONL，不接受只含聚合数值的文件。`aggregate` 必须显式提供 seed 0/1/2 三份 evaluator JSON；缺 seed、outfit、view 或 episode 均失败，不回退到 available-seed 或 micro average。

## 6. Table / Figure

`export-tables` 读取 source-data JSON，缺失指标输出 `N/A`。Synthetic fixture 只能写临时或 debug 目录，不能写 `artifacts/paper_ready`。`export-figures` 固定 Figure 1–6 layout；Figure 2 使用全部五 outfits、全部四 views，strongest baseline column 预注册为 B5，不按当前结果挑选。

## 7. PAPER_FINAL

Evaluator 通过后只允许进入 `EVALUATED`，不会自动成为 `PAPER_FINAL`。操作员必须补充人工视觉裁决、完整性检查和显式确认，再执行状态迁移。历史 A8 永远不参与该流程。
