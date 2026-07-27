# AAAI-27 Seen-Outfit Claim–Evidence Matrix

统一配置：[aaai27_seen_outfit_explicit_basis_v1.yaml](../../configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml)
冻结资产：[frozen_asset_manifest.json](../../paper_protocol/frozen_asset_manifest.json)
历史正式根：`$CANONDRESSGS_OUTPUT_ROOT/pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003`

本矩阵中的历史数值只标记 `HISTORICAL_EVIDENCE`；论文最终数字必须依据 `experiment_registry.yaml` 重新运行。

## 允许 claims

| ID | 允许表述 | 配置/方法证据 | 指标证据 | 视觉证据 | 正式路径 |
|---|---|---|---|---|---|
| C1 | 固定 subject02 上的多套 seen garment reference control | config `data.seen_outfits`、`pipeline` | seen RMSE `0.0407073`、nearest `20/20` | `five_outfit_four_view_predictions.png` | `attempt_003/stage_c_seen/seen_metrics.json` |
| C2 | Prediction 是 reference-conditioned，forward 不输入 outfit ID 或 target image | config `permissions`；predictor forward signature | `target_forward_leakage=false` | method boundary 由 Figure 1 明示 | `attempt_003/final_adjudication/final_adjudication.json` |
| C3 | Rank-4 explicit canonical Gaussian residual basis 可重建五套 seen teachers | config `basis.rank=4` | explained variance `0.999999999998`；各 outfit RMSE `≤5.05e-7` | `basis_rank_4_reconstruction.png` | `attempt_003/stage_b_basis/rank_ladder_metrics.json` |
| C4 | Reference swap 可控制五套 seen garment endpoints | correct/swap 固定同 target pose/camera | correct-vs-swapped `80/80` | `five_outfit_correct_swapped_unified_contact.png` | `attempt_003/stage_c_seen/seen_metrics.json` |
| C5 | Multi-reference、single-reference、dropout 在 seen outfits 上保持稳定判别 | A7 与 reference metrics 合同 | permutation `3.57628e-7`；single/dropout `40/40` | `permutation_dropout_comparison.png` | `attempt_003/stage_c_seen/seen_metrics.json` |
| C6 | 在已验证的两服装 collapse 场景中，frozen F2 + linear coefficient control 比 legacy complex fusion 更稳定 | B5/Ours 与 A5 预注册；CS-PASS | CS-PASS sign `8/8`、separation `1.99957410` | Figure 5 必须同时展示旧 collapse 与 CS-PASS | `$CANONDRESSGS_OUTPUT_ROOT/pipeline_full/SUBJECT02-REFERENCE-COEFFICIENT-SUPERVISION-CALIBRATION-001/attempt_002` |

## 禁止 claims

| ID | 禁止表述 | 反证或证据缺口 | 必须如何呈现 |
|---|---|---|---|
| N1 | unseen outfit generalization | O07 projection RMSE `0.314913`、cosine `0.557144`，reference prediction FAIL | 明确写 `Held-out diagnostic — FAIL` |
| N2 | arbitrary garment generation | 只覆盖五套预注册 seen endpoints | 使用“seen garment control”，不得使用“arbitrary” |
| N3 | 跨身份泛化 | 唯一 identity 为 subject02 | Abstract、Method、Experiments 均限定 subject02 |
| N4 | novel-pose 或 novel-view generalization | 仅四个固定 conditions/views | 不使用 novel-view/novel-pose 表述 |
| N5 | 完整长期 canonical projection/completion pipeline 闭环 | 当前正式方法无 projection/completion | 只描述 explicit-basis coefficient pipeline |
| N6 | 从未见 reference 生成新的 residual field | Predictor 只能组合冻结 seen basis | 写作 coefficient prediction，不写 residual generation |
| N7 | O07 held-out PASS | O07 basis 和 reference 两部分均 FAIL | Table 4/Figure 6 保留失败，不进入主 PASS 表 |

## O07 负面证据链接

- Metrics：`attempt_003/stage_d_held_out/held_out_metrics.json`
- Adjudication：`attempt_003/stage_d_held_out/held_out_adjudication.json`
- Figure source：`attempt_003/visual_acceptance/o07_teacher_projection_prediction.png`
- Nearest seen endpoint：四个固定条件均为 O03。

任何超出 C1–C6 的论文句子必须先在本矩阵新增证据绑定并重新评审；不得通过改写措辞绕过 N1–N7。
