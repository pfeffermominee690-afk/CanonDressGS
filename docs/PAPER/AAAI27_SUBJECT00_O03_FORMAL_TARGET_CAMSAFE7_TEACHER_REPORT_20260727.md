# Subject00 O03 正式 Target Camera-safe7 Teacher 执行报告

任务编号：`AAAI27-SUBJECT00-O03-FORMAL-TARGET-CAMSAFE7-TEACHER-RERUN-001`

## 结论

本次执行在 optimizer 之前按冻结合同停止。正式 CPU loader 已通过：7/7 个 O03
camera-safe 记录按固定顺序加载，12 个科学字段共 84 个 tensor 均完成 shape、
dtype、范围、非零计数、finite 与 value SHA 审计。Base60747 的文件大小、SHA256、
CPU `torch.load` 和内部 step=60747 也全部通过。

阻断原因不是 target 数据失败，而是正式字段到冻结 loss 的实际绑定不完整：
`target_edit_core_mask`、`target_preserve_mask`、`target_revealed_skin_mask` 在
`CAPACITY_ORACLE_LOSS_V1` 中没有 consumer；另外五个字段只出现在
materialization 分支的 capacity adapter 中，冻结的执行源 HEAD 仍只有旧的
run-local snapshot loss 路径。合同中的示意语义不能代替代码绑定，也不能切换到
`SUPPORT_AWARE_REGION_TRUSTED_OBJECTIVE_V6_1`，因为那会改变冻结 loss 合同。

因此最终分类为：

`SUBJECT00_O03_FORMAL_TARGET_RERUN_BLOCKED_BY_LOSS_FIELD_BINDING_AMBIGUITY`

唯一后续任务：

`RESOLVE_SUBJECT00_FORMAL_TARGET_LOSS_FIELD_BINDING`

## 八字段实际 consumer 审计

| 正式字段 | capacity 参数 | 审计状态 |
|---|---|---|
| `target_base_rgb` | `base_rgb` | `BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION` |
| `target_edit_mask` | `edit_mask` | `BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION` |
| `target_edit_core_mask` | `None` | `UNBOUND_IN_CAPACITY_ORACLE_LOSS_V1` |
| `target_preserve_mask` | `None` | `UNBOUND_IN_CAPACITY_ORACLE_LOSS_V1` |
| `target_transition_mask` | `transition_mask` | `BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION` |
| `target_base_foreground_mask` | `base_foreground` | `BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION` |
| `target_old_clothing_mask` | `old_clothing_mask` | `BOUND_IN_MATERIALIZATION_ADAPTER_BUT_NOT_FROZEN_SOURCE_EXECUTION` |
| `target_revealed_skin_mask` | `None` | `UNBOUND_IN_CAPACITY_ORACLE_LOSS_V1` |

## 执行边界

- 本任务未创建新 output root；检测到后续外部任务占用同一路径。
- CUDA forward / backward：0 / 0。
- optimizer：未构造；optimizer step：0。
- 新 checkpoint：0。
- 自动重试、第二次运行、超参扫描：均未发生。
- 正式 target、raw、person/garment masks、camera、Base60747、历史 Teacher
  和 Formal Base：零修改。
- Formal Base 保持 `USER_AUTHORIZED_PAUSED`，durable step=60747，
  resume authorization=false。
- 论文正文：零修改；paper eligible=false；paper final=false。

首次完整门禁快照完成时固定 output root 不存在。随后另一个已授权任务
`AAAI27-SUBJECT00-BASE60747-ACCELERATED-METHOD-LAUNCH-001` 使用同一路径启动，
形成外部并发碰撞。该进程、其 optimizer steps 和 checkpoint 不属于本任务；本任务
在检测到路径存在后立即停止，没有覆盖、删除或创建 `attempt_002`。

冻结任务末尾同时列出了 `OPTIMIZER_STEPS_COMPLETED=1200`，但同一合同更早明确
规定：任一正式字段没有唯一 consumer 时不得启动 optimizer。这里记录实际值 0，
以服从安全门禁并避免伪造训练完成状态。

## 关键证据

- 正式 loader Windows 冻结 SHA256：
  `786c93355776093e610dfe1bc74efd61233da134550a220bc06601f4f2365508`
- 正式 loader 云端 LF 容器 SHA256：
  `ee1270ca18a4a85698a03f8fdab693ef78d4d7eb303c4fe4effc4efe912c7d00`
- Base60747 SHA256：`2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7`
- loss consumer map：
  `paper_protocol/reviewer_risk/subject00_O03_formal_target_loss_consumer_map_20260727.json`
- 84 项字段注册表：
  `paper_protocol/reviewer_risk/subject00_O03_formal_target_scientific_field_registry_20260727.json`
- 53 项条件化机器检查：
  `paper_protocol/reviewer_risk/subject00_O03_formal_target_execution_tests_20260727.json`

## 最终字段摘要

- `FORMAL_FIELD_LOAD_STATUS`: `PASS_12_FIELDS_X_7_RECORDS`
- `FORMAL_FIELD_VALUE_HASH_STATUS`: `PASS_84_OF_84_FINITE_VALUE_HASHES_RECORDED`
- `LOSS_CONSUMER_MAP_STATUS`: `FAIL_3_FIELDS_UNBOUND_AND_5_BINDINGS_NOT_IN_FROZEN_SOURCE_EXECUTION`
- `OPTIMIZER_STEPS_COMPLETED`: `0`
- `CHECKPOINT_COUNT`: `0`
- `PAPER_ELIGIBLE`: `false`
- `NEXT_TASK`: `RESOLVE_SUBJECT00_FORMAL_TARGET_LOSS_FIELD_BINDING`
