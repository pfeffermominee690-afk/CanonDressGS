# Subject00 O03 loss binding、并发来源与方法有效性报告

任务：`AAAI27-SUBJECT00-O03-LOSS-BINDING-CONCURRENT-RUN-PROVENANCE-001`

## 唯一裁决

并发完成的 O03 formal safe-7 Teacher 是单一 accelerated-launch 写入者产生的
有效正式 Teacher。候选的五个 checkpoint 均可解析，内部步数为
`0/300/600/900/1200`，Base60747、正式 materialized target、冻结
`CAPACITY_ORACLE_LOSS_V1`、7-view 分母与 slot04 排除均通过。没有发现 blocked
task 的 metadata、日志或 checkpoint 混写。

方法 rotation0/seed0 的启动封存与 step300 checkpoint 都精确绑定 O03
`054b9efe1086d89b18310b9831d3ed7aea59e286e31200509b58756545fc3920`，因此该 cell 科学有效；在另行授权后可继续
剩余 11 个预注册 run。本任务没有执行任何新 Teacher 或 method optimizer step。

## 实际 loss-field binding

- loss-active：`target_edit_rgb, target_base_rgb, target_foreground_mask, target_base_foreground_mask, target_edit_mask, target_clothing_mask, target_old_clothing_mask, target_protected_mask, target_transition_mask`
- evaluation-active：`target_edit_rgb, target_base_rgb, target_foreground_mask, target_edit_mask, target_clothing_mask, target_old_clothing_mask, target_protected_mask`
- runtime-derived：`protected, garment_union_unprotected, foreground, base_foreground, new_silhouette, boundary_unprotected`
- schema-required、当前 loss inactive：`target_edit_core_mask, target_preserve_mask, target_revealed_skin_mask`
- 严格 QA-only：无；上述三个 inactive 字段同时有其他模式消费者
- 完全 unused：无

`garment_rgb` 使用 edit/clothing/old-clothing 的并集并去掉 protected；
`new_silhouette_alpha` 使用 foreground 与 base-foreground；
`boundary_rgb` 使用正式 transition mask；protected RGB/alpha 分别使用正式
base RGB/base foreground。stability 仅使用当前 trainable field。

## 旧 safe-7 等价性重审

旧 snapshot 在 raw、person、garment、protected 四个 active 字段上保持 exact，
但 `target_base_rgb`、`target_edit_mask`、`target_transition_mask`、
`target_base_foreground_mask`、`target_old_clothing_mask` 五个实际 active 字段
存在科学来源或 tensor 差异。因此仍为
`D_ACTIVE_LOSS_INPUT_SCIENTIFIC_MISMATCH`。三个 schema-only 字段没有被用来维持
D 类。

## 并发写入边界

- 实际 writer task：`AAAI27-SUBJECT00-BASE60747-ACCELERATED-METHOD-LAUNCH-001`
- writer branch/head：`research/subject00-base60747-accelerated-method-launch-20260727` / `3d4307f581f74efc173bb27c982277d952e32e31`
- 候选 output：`/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-O03-TEACHER-BASE60747-FORMAL-CAMSAFE7-001/attempt_001`
- 单写入者：通过
- 跨任务文件混写：未发现
- blocked task：0 optimizer step、0 checkpoint、0 candidate-root write
- 首次失败：同一 accelerated task 的 preoptimizer runtime mismatch
- 恢复：同 attempt、0 既有 checkpoint、0 既有 optimizer step

PID/PPID/命令在冲突发现时由进程表捕获；进程退出后 tmux/session 标识无法恢复。
这一缺失不影响唯一归属，因为 run-root task markers、checkpoint metadata、
preflight Git snapshot、config snapshot、日志结果与全部时间戳一致。

## 候选与方法依赖

- O03 final：`054b9efe1086d89b18310b9831d3ed7aea59e286e31200509b58756545fc3920`
- method step300：`6f79082ea32bf4cccffc98ab56da17a0051ab1700715bb9bc9c46d737a0294bf`
- method O01/O03/O04 Teacher：
  `c7881862c4eddf5f58538a2278ab7765aa047784681fb02e7cd89cfe846c1892` /
  `054b9efe1086d89b18310b9831d3ed7aea59e286e31200509b58756545fc3920` /
  `2fa7764097d8577c1610bbf222b26d9ea287bd18074371de400a66cd2270f3a1`
- O01 validity：`VALID_FORMAL_TARGET_TEACHER`
- O03 validity：`VALID_FORMAL_TARGET_TEACHER`
- O04 validity：`VALID_FORMAL_TARGET_TEACHER`
- method rotation0/seed0：有效

## 不可变性与范围

Base、正式 target、O01/O03/O04 Teacher、旧 O03 safe-7、method step300、raw/mask
均为只读且任务前后 SHA 不变。论文正文修改为 0，`PAPER_FINAL=false`。

## 最终状态

- `FINAL_CLASSIFICATION`: `SUBJECT00_O03_CONCURRENT_FORMAL_TEACHER_VALID_METHOD_STEP300_VALID_MATRIX_READY_TO_CONTINUE`
- `NEXT_TASK`: `CONTINUE_SUBJECT00_BASE60747_REMAINING_11_METHOD_RUNS_AND_FAIR_BASELINES`
- `TEST_RESULT`: `PASS_45_OF_45; FORMAL_PYTEST_5_PASSED; AUDIT_PYTEST_10_PASSED; UPSTREAM_SCIENTIFIC_CONTRACT_12_OF_12_PASSED`
