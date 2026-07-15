# Project Ledger

本文件只允许在末尾追加记录。既有条目不得改写或删除；纠错使用新条目引用旧条目。

## LEDGER-20260715-001 — 治理基线启动

- 时间：2026-07-15 Asia/Shanghai
- 类型：GOVERNANCE
- 状态：IN PROGRESS
- 本地权威仓库：`E:\model_train\mmlphuman_code`
- 起始分支/HEAD：`dev-dressable` / `07d4c34b7fe80a9336ed6ee29ec0c7c95b7b4c59`
- 新分支：`pipeline/imagecond-mvp-20260715`
- 云端 legacy：冻结，不修改
- 变更：创建项目治理文件、只读文件清单和部署/检查/拉取脚本
- 限制：尚未 stage、commit 或部署；等待用户确认精确 stage 清单

## LEDGER-20260715-002 — Gate 3-C2 证据登记

- 类型：EXPERIMENT-AUDIT
- 状态：PARTIAL
- Run ID：`GATE-REFONLY-THRESHOLD-ABLATION-001`
- 执行代码 SHA256：`37cf977fec43c29aaa56fc406f4eba5f98567f9f6411166f810cc94d7069d074`
- 当前 backbone SHA256：`abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`
- 结论：threshold 0.40 相对 0.50 小幅改善；不是历史 backbone reproduction，不关闭 Gate 3-C
- 输出：`/root/autodl-tmp/canondressgs_work/outputs/gates/gate_3b/GATE-REFONLY-THRESHOLD-ABLATION-001`

## LEDGER-20260715-003 — 本地治理与 MVP 静态基线检查

- 时间：2026-07-15 Asia/Shanghai
- 类型：TEST
- 状态：PASS-WITH-SCOPE-LIMIT
- 分支/HEAD：`pipeline/imagecond-mvp-20260715` / `07d4c34b7fe80a9336ed6ee29ec0c7c95b7b4c59`
- PowerShell parse：3/3 PASS
- Python compile：Gate 4 核心闭包 PASS
- 接口测试：21 PASS；dataset：20 PASS；clothing loss：10 PASS；Dressable model：9 PASS
- synthetic/mock image-conditioned training、gradient、checkpoint resume：PASS
- 范围限制：未运行真实 MMLPHuman 端到端 rendering，不构成 Gate 4 MVP acceptance
- Git：未 stage、未 commit、未部署

## LEDGER-20260715-004 — 云端干净执行基线部署

- 时间：2026-07-15T22:02:26+08:00
- 类型：DEPLOYMENT
- 状态：PASS
- 本地分支：`pipeline/imagecond-mvp-20260715`
- 本地 commit：`aa9bdaa918554fae18bd51132be023f5d9759dbe`
- bare remote：`/root/autodl-tmp/canondressgs_work/git/canondressgs.git`
- cloud remote URL：`canondress-cloud:/root/autodl-tmp/canondressgs_work/git/canondressgs.git`
- 云端执行目录：`/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_pipeline_mvp`
- 云端分支：`pipeline/imagecond-mvp-20260715`
- 云端 commit：`aa9bdaa918554fae18bd51132be023f5d9759dbe`
- 云端 Git 状态：clean，modified count 0
- Python：3.10.20
- PyTorch：2.4.1+cu121
- CUDA runtime：12.1
- CUDA available：True
- GPU：NVIDIA GeForce RTX 4090
- py_compile：PASS
- image-conditioned interfaces：21 PASS
- dataset：20 PASS
- clothing losses：10 PASS
- Dressable model：9 PASS
- synthetic/mock training、gradient、checkpoint roundtrip：PASS
- 范围限制：未运行真实 MMLPHuman rendering，未启动 Gate 4 训练
- legacy 仓库：`/root/autodl-tmp/canondressgs_work/mmlphuman_code` 继续冻结，未触碰
