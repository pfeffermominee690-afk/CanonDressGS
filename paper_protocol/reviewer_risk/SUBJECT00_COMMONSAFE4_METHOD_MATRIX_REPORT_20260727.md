# Subject00 CommonSafe4 Base60747 方法矩阵与公平基线报告

任务 `AAAI27-SUBJECT00-BASE60747-COMMONSAFE4-METHOD-MATRIX-001` 已达到技术通过，等待人工科学审查。原 slot04 因 O01/O03
永久 quarantine 后不具备三服装共同覆盖，被归类为科学 fold-coverage blocker，
不是 GPU 或训练器故障。旧 R0/S0 保留为真实诊断运行，但不进入新矩阵。

## CommonSafe4 合同

- common-safe slots：`['slot00', 'slot01', 'slot02', 'slot03', 'slot05', 'slot06', 'slot07']`
- replacement candidates：`['slot01', 'slot02', 'slot05', 'slot06']`
- 确定性选择：`slot06 / cam09 / back-right`
- anchors：`[slot00, slot07, slot03, slot06]`
- 方法合同：`PURE_ENDPOINT`；Dual-Support：`false`
- 每个 rotation 的 train/calibration/test 分母严格为 `6/3/3`

## 方法矩阵

12/12 个新运行均从头执行 300 步，总计 3600 步；每个运行均有
0/20/50/100/200/300 六个 checkpoint 和 3/3 正式测试记录。

| Run | Test top-1 | Checkpoints | Formal test |
|---|---:|---|---|
| COMMONSAFE4-METHOD-R0-S0 | 1.000000 | PASS_6_OF_6 | PASS_3_OF_3 |
| COMMONSAFE4-METHOD-R0-S1 | 1.000000 | PASS_6_OF_6 | PASS_3_OF_3 |
| COMMONSAFE4-METHOD-R0-S2 | 1.000000 | PASS_6_OF_6 | PASS_3_OF_3 |
| COMMONSAFE4-METHOD-R1-S0 | 0.666667 | PASS_6_OF_6 | PASS_3_OF_3 |
| COMMONSAFE4-METHOD-R1-S1 | 0.666667 | PASS_6_OF_6 | PASS_3_OF_3 |
| COMMONSAFE4-METHOD-R1-S2 | 0.666667 | PASS_6_OF_6 | PASS_3_OF_3 |
| COMMONSAFE4-METHOD-R2-S0 | 0.666667 | PASS_6_OF_6 | PASS_3_OF_3 |
| COMMONSAFE4-METHOD-R2-S1 | 1.000000 | PASS_6_OF_6 | PASS_3_OF_3 |
| COMMONSAFE4-METHOD-R2-S2 | 1.000000 | PASS_6_OF_6 | PASS_3_OF_3 |
| COMMONSAFE4-METHOD-R3-S0 | 0.666667 | PASS_6_OF_6 | PASS_3_OF_3 |
| COMMONSAFE4-METHOD-R3-S1 | 0.666667 | PASS_6_OF_6 | PASS_3_OF_3 |
| COMMONSAFE4-METHOD-R3-S2 | 1.000000 | PASS_6_OF_6 | PASS_3_OF_3 |

矩阵 top-1 为 `30/36`
(`0.833333`)，每运行均值/总体标准差为
`0.833333 / 0.166667`。
O01 与 O03 均为 12/12；O04 有 6 次预测为 O03、6 次预测为 O04。

## 公平内部基线

| Baseline | Endpoint top-1 | Optimizer steps |
|---|---:|---:|
| Reference Classifier Lookup | 0.861111 | 3600 |
| Nearest-Centroid Lookup | 0.972222 | 0 |
| Outfit-ID Oracle | 1.000000 | 0 |
| Teacher Endpoint | 1.000000 | 0 |

CanonDressGS-Endpoint 为 30/36；Reference Classifier 为 31/36；
Nearest-Centroid 为 35/36；两个非部署 oracle 均为 36/36。
因此当前结果是描述性的“方法低于两种 hard lookup”，不作优越性或统计显著性声称。

基线启动曾出现一次执行器 self-PID GPU 门禁错误：第一个分类器单元已完整结束，
第二个单元在模型/优化器初始化前停止。修复后在同一 `attempt_001` 续接，
首单元未重跑，科学运行 retry 数为 0，未创建 `attempt_002`。

## 安全性与边界

- quarantine optimizer/test usage：`0/0`
- Dual-Support calls：`0`
- NaN/Inf/OOM：`NONE/NONE`
- Base、三 Teacher、正式 targets、旧方法输出 mutation：`0`
- 人工视觉结论、科学通过、paper eligible：`null / null / false`
- 执行测试：`PASS_44_OF_44`

最终技术分类：
`SUBJECT00_BASE60747_COMMONSAFE4_METHOD_MATRIX_AND_BASELINES_TECHNICAL_PASS_PENDING_HUMAN_SCIENTIFIC_REVIEW`

唯一下一任务：
`PREPARE_SUBJECT00_COMMONSAFE4_METHOD_TEACHER_BASELINE_HUMAN_SCIENTIFIC_REVIEW_PACK`
