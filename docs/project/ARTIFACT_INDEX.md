# Artifact Index

| Run ID | Module | 状态 | Git commit | Base | 输出目录 | 关键证据 |
|---|---|---|---|---|---|---|
| GATE5-FULL-ATTRIBUTE-CONTRACT-001 | Full Pipeline Module 1 | PASS | `1cd44ba61ad4ee0be08eb762bb24987918a6d616` | `abbf67b5...` | `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/GATE5-FULL-ATTRIBUTE-CONTRACT-001` | six-channel real render/gradient PASS; zero/state exact; old xyz checkpoint PASS |

| Run ID | Gate | 状态 | Git commit | Config | Checkpoint | 输出目录 | 关键证据 |
|---|---|---|---|---|---|---|---|
| GATE4-REAL-OVERFIT100-001 | 4-C | PASS / STRONG | `e8baa03efb46c5244693f5aa798810dd5dbaacde` | `configs/canon_dress_gs_gate4c_overfit100.yaml` | step-100 `784b5af0...`; base `abbf67b5...` | `/root/autodl-tmp/canondressgs_work/outputs/pipeline_mvp/GATE4-REAL-OVERFIT100-001` | 90 resumed steps finite; RGB-all L1 vs zero -63.23%; raw/RGB sensitivity 11.11x/7.81x step-10; independent inference deterministic |

| Run ID | Gate | 状态 | Git commit | Config | Checkpoint | 输出目录 | 关键证据 |
|---|---|---|---|---|---|---|---|
| GATE4-REAL-SMOKE10-001 | 4-B | PASS | `6f394339727d1b9e351ff1c98d3ff484397923c7` | `configs/canon_dress_gs_gate4b_smoke10.yaml` | step-10 `6d001694...`; base `abbf67b5...` | `/root/autodl-tmp/canondressgs_work/outputs/pipeline_mvp/GATE4-REAL-SMOKE10-001` | loss `0.069504388 -> 0.052702554`; raw sensitivity `6.57478e-07`; RGB sensitivity `6.46310e-07`; roundtrip zero diff |

正式实验必须先分配唯一 Run ID。路径使用绝对路径，二进制产物记录 SHA256；未知字段写 `PENDING`，禁止猜测。

| Run ID | Gate | 状态 | Git commit | Config | Checkpoint | 输出目录 | 关键证据 |
|---|---|---|---|---|---|---|---|
| GATE-MATCHED-EVAL-001 | 3-B | PARTIAL | legacy dirty worktree | manifest 内记录 | current backbone `abbf67b5...` | `/root/autodl-tmp/canondressgs_work/outputs/gates/gate_3b/GATE-MATCHED-EVAL-001` | precision 0.998516, recall 0.929191, IoU 0.927910 |
| GATE-REFONLY-FN-DIAG-001 | 3-C1 | PASS-DIAGNOSIS | legacy dirty worktree | CLI manifest | current backbone `abbf67b5...` | `/root/autodl-tmp/canondressgs_work/outputs/gates/gate_3b/GATE-REFONLY-FN-DIAG-001` | 359 FN；334 在两个 reference views 均不可见 |
| GATE-REFONLY-THRESHOLD-ABLATION-001 | 3-C2 | PARTIAL | legacy dirty worktree | input manifest | current backbone `abbf67b5...` | `/root/autodl-tmp/canondressgs_work/outputs/gates/gate_3b/GATE-REFONLY-THRESHOLD-ABLATION-001` | 0.40 相对 0.50：float RGB-all MAE 改善 0.537% |
| GATE4-MVP-BASELINE-001 | 4 | PLANNED | PENDING | PENDING | PENDING | PENDING | 14 项 Gate 4 验收 |
| GATE4-REAL-ONEBATCH-001 | 4-A | PARTIAL | `a6639d375355a09d0d451630e46cb37e813e363f` | `configs/canon_dress_gs_mvp_real.yaml` | step-1；base `abbf67b5...` | `/root/autodl-tmp/canondressgs_work/outputs/pipeline_mvp/GATE4-REAL-ONEBATCH-001` | same-state roundtrip 全零差异；实际 reference/target 采样与预注册协议不一致，最终图片/acceptance 不完整 |

## Checkpoint 漂移警告

- 历史 Gate 验收指纹：`64ac7f2dd1dd705307258620f76967fdc93c66869d3ff5c7f32e1f70055635c4`
- 当前路径文件指纹：`abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`
- 状态：历史验收指纹与当前文件不一致，需进一步追溯。不得写成同一 checkpoint，也不因此否定已保存的历史 Gate 验收结果。
