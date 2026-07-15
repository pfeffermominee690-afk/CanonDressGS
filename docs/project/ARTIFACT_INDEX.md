# Artifact Index

正式实验必须先分配唯一 Run ID。路径使用绝对路径，二进制产物记录 SHA256；未知字段写 `PENDING`，禁止猜测。

| Run ID | Gate | 状态 | Git commit | Config | Checkpoint | 输出目录 | 关键证据 |
|---|---|---|---|---|---|---|---|
| GATE-MATCHED-EVAL-001 | 3-B | PARTIAL | legacy dirty worktree | manifest 内记录 | current backbone `abbf67b5...` | `/root/autodl-tmp/canondressgs_work/outputs/gates/gate_3b/GATE-MATCHED-EVAL-001` | precision 0.998516, recall 0.929191, IoU 0.927910 |
| GATE-REFONLY-FN-DIAG-001 | 3-C1 | PASS-DIAGNOSIS | legacy dirty worktree | CLI manifest | current backbone `abbf67b5...` | `/root/autodl-tmp/canondressgs_work/outputs/gates/gate_3b/GATE-REFONLY-FN-DIAG-001` | 359 FN；334 在两个 reference views 均不可见 |
| GATE-REFONLY-THRESHOLD-ABLATION-001 | 3-C2 | PARTIAL | legacy dirty worktree | input manifest | current backbone `abbf67b5...` | `/root/autodl-tmp/canondressgs_work/outputs/gates/gate_3b/GATE-REFONLY-THRESHOLD-ABLATION-001` | 0.40 相对 0.50：float RGB-all MAE 改善 0.537% |
| GATE4-MVP-BASELINE-001 | 4 | PLANNED | PENDING | PENDING | PENDING | PENDING | 14 项 Gate 4 验收 |

## Checkpoint 漂移警告

- 历史 Gate 验收指纹：`64ac7f2dd1dd705307258620f76967fdc93c66869d3ff5c7f32e1f70055635c4`
- 当前路径文件指纹：`abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`
- 状态：历史验收指纹与当前文件不一致，需进一步追溯。不得写成同一 checkpoint，也不因此否定已保存的历史 Gate 验收结果。
