# Project Ledger

## LEDGER-20260716-004 — Full Dressable Module 1 Gaussian Attribute Contract

- 类型/状态：IMPLEMENTATION + REAL ACCEPTANCE / PASS
- branch：`pipeline/full-dressable-20260715`
- 基线 tag：`gate4c-geometry-baseline-20260715` -> `1d46d018eef08453cef7151bb39c9fb5b5fc07d4`
- 核心 commits：`7a82dbb`、`d00b812`、`9f4ee2c`、`1cd44ba`
- base checkpoint SHA256：`abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`
- 实际 shapes：xyz/scaling/rotation/opacity/sh0/shN = `[200000,3] / [200000,3] / [200000,4] / [200000] / [200000,1,3] / [200000,3,3]`
- quaternion：`wxyz`；local `normalize(q_base * q_delta)`
- 六通道真实 render diff 与梯度：PASS；base gradient count 0
- zero override 与 base render：RGB/alpha max diff 0
- state restore：pose/Rh/Th/cache/SH degree 和六个 raw base tensors restored
- legacy step-100 checkpoint：restored step 100；缺失新 metadata 时仅 `delta_xyz` enabled，其余 disabled/zero
- 输出：`/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/GATE5-FULL-ATTRIBUTE-CONTRACT-001`
- 训练：未启动

## LEDGER-20260716-002 — Gate 4-C 100-step Overfit and Independent Inference

- 时间：2026-07-16 Asia/Shanghai
- 类型/状态：EXPERIMENT / PIPELINE PASS / EFFECT STRONG
- Run ID：`GATE4-REAL-OVERFIT100-001`
- branch/commit：`pipeline/imagecond-mvp-20260715` / `e8baa03efb46c5244693f5aa798810dd5dbaacde`
- config：`configs/canon_dress_gs_gate4c_overfit100.yaml`；resolved SHA256 `3ee295ea0fa09e6eecb3f77d386dcf8dd0b546daa7d30b489eeb1df39db42503`
- 命令：`/root/autodl-tmp/conda_envs/mmlphuman/bin/python -m tools.run_gate4c_overfit100`
- 输出：`/root/autodl-tmp/canondressgs_work/outputs/pipeline_mvp/GATE4-REAL-OVERFIT100-001`
- resume：Gate 4-B step 10 -> step 100；90 个新增 step 全部 finite
- step-100 checkpoint SHA256：`784b5af0b55de6ce40946c64d7774ee7f745ed86240491d8ada1525f41b172e3`
- loss：step 11 `0.050672099`，step 100 `0.024047337`；early/late mean 相对下降 `48.6181%`
- 冻结/禁用：base grad count 始终 0；scaling/opacity 均严格为 0；末十步显存 range `4513280` bytes
- unified evaluation：step-100 RGB-all/foreground/alpha L1 `0.00124963 / 0.0147991 / 0.000519729`，foreground PSNR `27.7492`，mask IoU `0.993598`
- anchor：step-100 xyz MAE `0.00391920`，L2 mean error `0.00823399`，active MAE `0.00771490`，cosine `0.376112`
- sensitivity：step-100 raw/RGB MAE `7.30441e-06 / 5.04538e-06`，分别为 step-10 的约 `11.11x / 7.81x`
- roundtrip：raw/gated anchor、RGB、alpha max/mean diff 全部为 0
- independent inference：step=100；target/teacher reads 均 false；两次推理 max diff 全部为 0；anchor/Gaussian shapes `[10000,3] / [200000,3]`
- 保守限制：独立 inference 事后图像指标与统一评估有漂移，后续需复核；未启动 300-step

## LEDGER-20260716-001 — Gate 4-B Real Image-conditioned 10-step Smoke

- 时间：2026-07-16T00:49:03+08:00
- 类型/状态：EXPERIMENT / PASS
- Run ID：`GATE4-REAL-SMOKE10-001`
- branch/commit：`pipeline/imagecond-mvp-20260715` / `6f394339727d1b9e351ff1c98d3ff484397923c7`
- config：`configs/canon_dress_gs_gate4b_smoke10.yaml`，SHA256 `935d188956a4c44d4dbf8a3081fd335f8d68d299a192f993508d848f676fbbdd`
- 命令：`/root/autodl-tmp/conda_envs/mmlphuman/bin/python -m tools.run_gate4b_smoke10`
- 输出：`/root/autodl-tmp/canondressgs_work/outputs/pipeline_mvp/GATE4-REAL-SMOKE10-001`
- 协议：A=`f000_c018 + f1000_c000`，B=`f000_c018`，target=`f2000_c009`，`target_view_used=false`
- 当前 base SHA256：`abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`；不得与历史 `64ac7f2d...` 合并表述
- step-10 checkpoint SHA256：`6d001694e28e9b1065abc731691288ccd73afdd8483afd0351152a041bb5446c`
- total loss：`0.069504388 -> 0.052702554`
- step-10 gradients：encoder `0.00462416`，aggregator `1.78386e-05`，HyperNetwork `0.0242637`，Anchor MLP `0.999694`；base grad count `0`
- sensitivity：raw anchor `6.57478e-07`，gated anchor `4.95539e-07`，Gaussian `4.86841e-07`，RGB `6.46310e-07`，alpha `6.42821e-07`（MAE）
- roundtrip：raw/gated anchor、RGB、alpha max/mean diff 均为 `0`
- 稳定性：全部 step finite；禁用通道为 0；末三步 allocated-memory range `1122816` bytes
- Git：云端 worktree clean；legacy 仓库继续冻结；未启动 100/300-step 训练

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

## LEDGER-20260715-005 — Gate 4-A Real Image-conditioned One-batch

- 时间：2026-07-15 Asia/Shanghai
- 类型：EXPERIMENT
- 状态：PARTIAL
- Run ID：`GATE4-REAL-ONEBATCH-001`
- 最终执行 commit：`774e95d3e80488cf0968d4112dd641e81f9bf623`
- 修复 commits：`436c07ccb4a863e79370bd514e66c8fb1a0a6bc6`、`9f723c079b85c5f854ac0b3d725ac03e87e29333`、`774e95d3e80488cf0968d4112dd641e81f9bf623`
- 配置：`configs/canon_dress_gs_mvp_real.yaml`
- 当前 checkpoint SHA256：`abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`
- episode：`/root/autodl-tmp/canondressgs_work/data_dressable/gate3b_synthetic_hoodie_expand_018.json`
- reference-only region SHA256：`20ed6dbc95664b16b6fedac0acf046ea7368a35490dfb440b2d3bfa2eccdc126`
- gate：threshold 0.40，`target_view_used=false`
- 输出：`/root/autodl-tmp/canondressgs_work/outputs/pipeline_mvp/GATE4-REAL-ONEBATCH-001`
- 结果：真实 gated step-1 forward/backward 成功；total loss `0.05546812`
- 梯度门禁：最终运行已越过 HyperNetwork/Anchor MLP/encoder/aggregator 梯度检查
- 阻塞：checkpoint roundtrip 比较使用了 warmup 后内存输出与 reload 的 step-1 checkpoint 输出，比较状态不一致
- 结论：不得写 PASS；未启动 100/300-step 训练

## LEDGER-20260715-006 — Gate 4-A Same-state Roundtrip 重跑

- 时间：2026-07-15 Asia/Shanghai
- 类型：EXPERIMENT
- 状态：PARTIAL
- Run ID：`GATE4-REAL-ONEBATCH-001`
- 执行 commit：`a6639d375355a09d0d451630e46cb37e813e363f`
- 修复 commit：`a6639d375355a09d0d451630e46cb37e813e363f` (`fix(gate4): compare checkpoint roundtrip at identical state`)
- same-state roundtrip：global embedding、raw/gated anchor xyz、Gaussian xyz、RGB、alpha 的 max/mean diff 全为 0，全部 allclose
- anchor/Gaussian shape：`[10000,3]` / `[200000,3]`
- render shape：RGB `[1150,1330,3]`，alpha `[1150,1330,1]`，finite ratio 1.0
- 梯度：encoder `1.0067e-4`、aggregator `3.1657e-7`、HyperNetwork `4.3848e-3`、Anchor MLP `1.0781`；base grad count 0
- sensitivity：embedding L2 `0.0018960`，anchor feature MAE `0.00464356`；offset/render 差异为 0，整体非全零
- 协议偏差：实际 reference 为 `f2000_c009 + f1000_c000`、target 为 `f000_c018`，不匹配 gate 的预注册 reference views
- 持久化阻塞：renderer 输出是 HWC，直接传给 `torchvision.save_image` 导致 predicted PNG 与 `GATE_ACCEPTANCE.md` 未生成
- 结论：PARTIAL；未启动 100/300-step 训练

## LEDGER-20260718-007 — Module 4B Canonical Representation Oracle Micro-pilot

- 时间：2026-07-18 Asia/Shanghai
- 类型：EXPERIMENT / VISUAL ACCEPTANCE / SEAL
- Run ID：`SUBJECT02-MODULE4B-MICROPILOT-001/attempt_001`
- 正式候选运行 commit：`dce29e089cc422abd661c622800e2d4c435bce26`
- 封存工具 commit：`a5c5c02055dd5df7e91acd44e604e2c4a3b84b8b`
- 配置：`configs/oracle/module4b_canonical_capacity_v1.yaml`，SHA256 `82229cbc62c083b1373b6f6bed35be54a468036b49c06d50c3640a0a357a135f`
- 数据：V5.3 12-sample fixture，`dual_target_region_aware_v1`；post-run aggregate SHA256 `457b4f93c9c04c337e7222750ca0ad8f9b6cf4f388775de1f10a18c640dba6fc`，全部输入未变
- 基线 checkpoint：`chkpnt100000.pth`，SHA256 `abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`
- 协议：O00/O01/O05 × Gaussian/Anchor；每个 480 steps；front/back/left/right round-robin
- 结果：Gaussian 三套 `PARTIAL_FIT + VISUAL_FAIL`；Anchor O00 `NO_FIT + VISUAL_FAIL`，O01/O05 `PARTIAL_FIT + VISUAL_FAIL`
- Anchor retention：O00 edit/clothing `0.5686/0.6057`；O01 `0.5919/0.6121`；O05 `0.6450/0.6604`，全部低于 0.75
- 冻结：base fingerprint exact、base grad 0、image backbone 未实例化、SHN strict zero
- Resume：O00 Gaussian step-40 model/optimizer/global-step/sampler exact，PASS
- 视觉：实际打开六组四视角、O05 中间阶段、residual/gate/protected panels；六组均 FAIL，未形成目标服装结构
- 测试：Module 4B 12/12、full-attribute Oracle、full-training checkpoint、py_compile、`git diff --check` 全部 PASS
- 状态：**FAIL / decision case D**
- 输出：`/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-MODULE4B-MICROPILOT-001/attempt_001`
- 停止：未启动正式 image-conditioned training，未实现 garment Gaussian layer，未扩展数据规模
