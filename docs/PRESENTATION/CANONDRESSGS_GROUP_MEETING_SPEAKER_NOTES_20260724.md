# CanonDressGS Group Meeting Speaker Notes — 2026-07-24

These are speaking prompts, not a verbatim script.

## S01 · CanonDressGS 当前进展
- 开场直接给出任务和边界。
- 说明这不是最终论文答辩，而是研究收敛汇报。
- 强调 LOO 尚无结果。
- 过渡到服装绑定问题。
- **Likely follow-up:** Figure 1 候选仍需人工选图
- **Recommended answer boundary:** 展示当前封闭服装库任务和方法边界
- **Transition:** 先从真实任务动机开始。

## S02 · 为什么可动画 Avatar 仍被采集服装绑定？
- 用输入输出三元组讲任务。
- 指出身份和姿态必须被冻结。
- 不要使用 arbitrary editing 表述。
- 引出三因素耦合。
- **Likely follow-up:** 避免把 closed-bank 说成 open-vocabulary
- **Recommended answer boundary:** 定义 reference-controlled garment state selection
- **Transition:** 任务看似是外观控制，实际同时涉及几何、可见性和外观。

## S03 · 核心困难：服装状态不是单一颜色变量
- 先对比 lookup 与 interpolation。
- 再指出三因素并非同等重要。
- 预告后续因果实验。
- 转入研究问题地图。
- **Likely follow-up:** 早期图仅作为历史设计证据
- **Recommended answer boundary:** 提出 geometry/visibility/appearance 三因素问题
- **Transition:** 因此需要把问题拆成可验证的研究问题。

## S04 · 研究问题地图
- 逐项标记 answered/pending。
- 强调 Q4 目前只有描述性等价。
- Q5 由 LOO 决定。
- 进入真实研究演化。
- **Likely follow-up:** 最终论文问题表述必须随 LOO 更新
- **Recommended answer boundary:** 明确已证与未证问题
- **Transition:** 下面按实验如何逐步收缩方法空间来讲。

## S05 · 方法路线如何被证据逐步收缩
- 按时间顺序讲失败链。
- 强调 support capacity PASS 排除了表达容量不足。
- 把 controller 与 headroom 放在后段。
- 引出冻结底座。
- **Likely follow-up:** 必须区分历史协议与当前协议
- **Recommended answer boundary:** 历史负结果推动方法收缩
- **Transition:** 收缩后的路线建立在冻结的 avatar 技术底座上。

## S06 · 冻结的技术底座
- 从 frozen/trainable 两列讲。
- 强调 renderer 和 deformation 不改。
- 区分 offline endpoint construction 与 inference。
- 过渡到 Teacher Endpoint。
- **Likely follow-up:** Figure 2 仍为 METHOD_FREEZE_PENDING_LOO
- **Recommended answer boundary:** 说明冻结边界
- **Transition:** 在这个底座上先建立每件服装的有效 Teacher Endpoint。

## S07 · Garment Teacher Endpoint
- 先讲优化目标，再讲 residual 组合。
- 强调 Teacher 不等于上界。
- 指出五个 seen endpoint 的范围。
- 引出端点间插值。
- **Likely follow-up:** 需避免 upper bound 旧称
- **Recommended answer boundary:** Teacher Endpoint 是多视图优化得到的稳定状态
- **Transition:** 但两个有效 endpoint 之间的线性几何并不一定有效。

## S08 · 为什么不能直接插值端点？
- 先解释 intervention 逻辑。
- 重点报 0.9959 与两个 10/10。
- 限定为当前协议。
- 转到 Dual-Support。
- **Likely follow-up:** 当前没有 Figure Bank 内的专用 geometry 图，建议用数值 callout
- **Recommended answer boundary:** geometry 是当前中间 artifact 主因
- **Transition:** 既然几何不能线性混合，下一步测试保留两套有效支持。

## S09 · Dual-Support：保留有效几何支持
- 先讲结构，再报三组指标。
- 指出 active Gaussian 与时间开销增加。
- 明确它不进入主线。
- 引出显式 basis。
- **Likely follow-up:** O01_O03 与 O02_O03 仍困难
- **Recommended answer boundary:** oracle Dual-Support 证明 geometry-safe mechanism 有效
- **Transition:** 主线仍需要一个可解释、最小化的 endpoint coordinate system。

## S10 · Explicit Global Basis
- 解释 n=5 时 centered rank≤4。
- 强调坐标系而非压缩。
- 区分 exact endpoint reconstruction 与 generalization。
- 转入 Pure Endpoint。
- **Likely follow-up:** basis 的论文定位由 LOO 决定
- **Recommended answer boundary:** rank-4 精确覆盖五个 centered endpoints
- **Transition:** 在这个坐标系中，用最小 reference predictor 选择 endpoint。

## S11 · Pure Endpoint Pipeline
- 按箭头顺序讲 pipeline。
- 在 raw coefficient 处停顿。
- 强调实际 render 使用 snapped endpoint。
- 转入协议。
- **Likely follow-up:** Figure 2 状态 METHOD_FREEZE_PENDING_LOO
- **Recommended answer boundary:** 最小控制器可靠选择 seen endpoints；raw 与 realized coefficient 明确分开
- **Transition:** 可靠性需要严格的 condition-fold cross-fit 验证。

## S12 · Pure Endpoint 实验设计
- 先解释 fold 划分单位。
- 报 24 runs。
- 说明所有比较共享 endpoint realization。
- 转入主结果。
- **Likely follow-up:** 需口头区分 condition-held-out 与 garment-held-out
- **Recommended answer boundary:** 协议支持固定 subject02、五个 seen garments 的闭集评估
- **Transition:** 在这个边界内，Pure Endpoint 的 clean 结果非常明确。

## S13 · Pure Endpoint 主结果
- 先报完成数，再报准确率。
- 补充安全性为零污染。
- 明确 closed-bank。
- 主动引出 hard lookup 等价。
- **Likely follow-up:** clean benchmark 已饱和
- **Recommended answer boundary:** closed five-garment bank endpoint control supported
- **Transition:** 但 clean benchmark 上的成功与 hard lookup 功能等价。

## S14 · Hard Lookup 等价性
- 先展示 agreement。
- 再解释 render equivalence。
- 主动说不能宣称 superiority。
- 转到 raw coefficient。
- **Likely follow-up:** Figure 5 仍需人工裁决
- **Recommended answer boundary:** clean closed-bank 下功能等价
- **Transition:** 区别在于 predictor 输出了连续坐标，但实际成功仍来自 endpoint realization。

## S15 · Raw Coefficient 与 Endpoint Snapping
- 先指 raw error。
- 再指 realized endpoint。
- 解释二者不是同一输出。
- 转入 perturbation。
- **Likely follow-up:** 图例必须固定 raw/snapped 术语
- **Recommended answer boundary:** snapping 有效修正 endpoint realization
- **Transition:** 接下来检查这种 endpoint control 对 reference 扰动是否稳定。

## S16 · Perturbation 与安全性
- 按 single/blur/dropout 顺序讲。
- 不要只展示安全项。
- 强调 blur flip 是真实限制。
- 引出 headroom。
- **Likely follow-up:** blur failure 必须保留在主汇报
- **Recommended answer boundary:** 报告具体 perturbation 边界与安全行为
- **Transition:** 既然 raw coefficient 不精确，下一步自然问题是 rendering loss 能否优化它。

## S17 · 为什么做 Coefficient Headroom？
- 从 raw coefficient 误差连接动机。
- 解释 Teacher/SVD parity 的必要性。
- 说明 full residual 是 comparator。
- 进入实验设计。
- **Likely follow-up:** 不得将 headroom 动机写成最终 pipeline
- **Recommended answer boundary:** 提出 seen-endpoint refinement 假设
- **Transition:** 实验只优化 4 个系数，并用 full residual 检查是否存在更大 headroom。

## S18 · Headroom 实验设计
- 先报 parity，再报 120/36000/960。
- 解释 λ selection。
- 强调本次规划未运行任何 optimizer。
- 转入结果。
- **Likely follow-up:** 数字应与 sealed summary 原样一致
- **Recommended answer boundary:** 说明 sealed attempt_002 的完整 protocol
- **Transition:** 完整执行后，结果否定了 seen-endpoint refinement 路线。

## S19 · Headroom 结果：Teacher Span 已在局部最优
- 先报 0/5。
- 再解释 macro gain 的符号。
- 最后展示 full-residual artifact。
- 用 classification 收束。
- **Likely follow-up:** caption 必须写明不支持 render refinement 进入主线
- **Recommended answer boundary:** Headroom 没有改善 Teacher Endpoint；full residual 退化并产生 artifacts
- **Transition:** 这个负结果直接改变当前 pipeline 的取舍。

## S20 · Headroom 如何改变方法设计
- 清楚划掉 refinement 分支。
- 区分 not supported 与 proven impossible。
- 说明 spatial field 没有启动。
- 引出 LOO。
- **Likely follow-up:** spatial field 是未启动 contingency，不是负实验
- **Recommended answer boundary:** 负结果支持冻结 refinement 与 local spatial basis
- **Transition:** 剩下真正能判断 basis 是否有独立价值的问题是 LOO adaptation。

## S21 · LOO：basis 剩余的核心价值检验
- 先解释 held-out 的信息边界。
- 强调 rank≤3。
- 说明历史 O07 不等于新 LOO。
- 进入当前状态。
- **Likely follow-up:** 不得读取 partial/active output
- **Recommended answer boundary:** LOO 是判断 basis adaptation value 的关键实验
- **Transition:** 协议已经修复完成，但当前没有 sealed execution result。

## S22 · LOO 当前状态
- 一句话说明没有结果。
- 报告 files_read=0。
- 不要填任何占位数字。
- 转入历史设计证据。
- **Likely follow-up:** 汇报当天需再次做 metadata-only 状态确认
- **Recommended answer boundary:** 只报告 protocol readiness 与 pending 状态
- **Transition:** 在等待 LOO 的同时，历史消融只用于解释设计演化。

## S23 · 历史 51 实验如何塑造最小模型
- 开头先打 Historical 标签。
- 报 51 与 14,400。
- 只挑三个最影响设计的消融。
- 转入 failure modes。
- **Likely follow-up:** 表格必须用 Historical 标签
- **Recommended answer boundary:** 历史消融解释设计选择
- **Transition:** 这些证据也暴露了当前仍必须正面呈现的失败模式。

## S24 · Failure Modes / Limitations
- 按 generalization/robustness/artifact 三类讲。
- 每类只给一个结论。
- 说明失败如何影响 claim。
- 转入 multi-identity。
- **Likely follow-up:** 避免堆叠太多小图，最终 PPT 需人工选 3 类
- **Recommended answer boundary:** 集中展示已知失败边界
- **Transition:** 其中第二身份目前只完成了 base-avatar foundation。

## S25 · Multi-Identity 基础进展
- 先讲 Subject00 的 base-avatar 范围。
- 再讲 formal 与 garment pending。
- AvatarReX 只讲 metadata。
- 转到方法冻结状态。
- **Likely follow-up:** Subject00 图不能被解释为 garment editing
- **Recommended answer boundary:** 仅称 base-avatar foundation 与 protocol readiness
- **Transition:** 因此论文冻结必须把已完成核心与待验证扩展分开。

## S26 · 当前论文方法冻结状态
- 按 core/supplementary/pending 三列讲。
- 点名 Figure 2 状态。
- 强调 Controller 不在主线。
- 转入风险。
- **Likely follow-up:** PAPER_FINAL 必须保持 0
- **Recommended answer boundary:** 展示当前冻结矩阵
- **Transition:** 冻结矩阵对应一组明确的 reviewer risks。

## S27 · 当前不足与 Reviewer 风险
- 先说最大风险。
- 再列支撑风险的三条事实。
- 说明负结果已阻止无效扩张。
- 转入 next steps。
- **Likely follow-up:** LOO 失败时需收缩论文主张
- **Recommended answer boundary:** 主动呈现 reviewer 风险和对应证据
- **Transition:** 下一步必须按风险优先级执行，而不是继续扩大方法。

## S28 · 下一步：先完成决定性实验，再冻结论文
- 按优先级读，不展开工程细节。
- 第一项必须是 LOO。
- 明确 PPTX 不在本任务生成。
- 收束并进入 Q&A。
- **Likely follow-up:** NEXT_TASK 仅为人工审阅 outline 与选图
- **Recommended answer boundary:** 按证据门槛安排 next steps
- **Transition:** 汇报结束，进入问题讨论。

## A01 · 数据集与服装定义
- 只在被问数据集时使用。
- 先身份再服装。
- 说明 O07 历史角色。
- **Likely follow-up:** 需最终核对展示名称
- **Recommended answer boundary:** 定义 closed bank
- **Transition:** 按提问跳转。

## A02 · Exact Rotation Manifests
- 说明 fold 单位。
- 说明重复处理。
- 指向正式 JSON。
- **Likely follow-up:** 只展示摘要，不暴露过密 manifest
- **Recommended answer boundary:** 协议可复核
- **Transition:** 按提问跳转。

## A03 · Pure Endpoint 七方法注册表
- 先列方法。
- 再列 realization。
- 强调公平性。
- **Likely follow-up:** 最终 PPT 可转为小表
- **Recommended answer boundary:** 方法注册完整
- **Transition:** 按提问跳转。

## A04 · 完整 Perturbation 表
- 按三种扰动讲。
- 先失败再安全。
- 不做平均美化。
- **Likely follow-up:** 保持原始分母
- **Recommended answer boundary:** 完整 robustness 边界
- **Transition:** 按提问跳转。

## A05 · 历史 Rank Ablation
- 先打 Historical 标签。
- 再讲 rank 上限。
- 不称压缩。
- **Likely follow-up:** 双重理由要分开
- **Recommended answer boundary:** 历史趋势
- **Transition:** 按提问跳转。

## A06 · Controller 诊断
- 先讲失败门槛。
- 再讲预算轨迹。
- 说明冻结决策。
- **Likely follow-up:** 避免只报 AUROC 而忽略 pair gate
- **Recommended answer boundary:** controller 未成功
- **Transition:** 按提问跳转。

## A07 · Dual-Support 全 10 Pair
- 先讲 aggregate。
- 再点困难 pair。
- 最后报 oracle 定位。
- **Likely follow-up:** 指出困难 pair
- **Recommended answer boundary:** oracle all-pair result
- **Transition:** 按提问跳转。

## A08 · Headroom λ Selection
- 先讲 split。
- 再讲 λ。
- 报完整运行数。
- **Likely follow-up:** 口头解释 split
- **Recommended answer boundary:** selection protocol closed
- **Transition:** 按提问跳转。

## A09 · Full-Residual Failure
- 先展示 artifact。
- 再联系指标。
- 最后限定结论。
- **Likely follow-up:** 机制解释保持假设语气
- **Recommended answer boundary:** full residual degrades
- **Transition:** 按提问跳转。

## A10 · Subject00 / AvatarReX 基础
- 先 Subject00。
- 再 pending。
- AvatarReX 不展示媒体。
- **Likely follow-up:** license audit
- **Recommended answer boundary:** foundation only
- **Transition:** 按提问跳转。

## A11 · Claim Boundary Matrix
- 先讲 supported。
- 再讲 pending。
- 最后讲 forbidden。
- **Likely follow-up:** 随 LOO 更新
- **Recommended answer boundary:** claim audit
- **Transition:** 按提问跳转。

## A12 · 执行治理与证据完整性
- 展示 provenance 字段。
- 报 mutation=0。
- 区分 cloud 状态与科学状态。
- **Likely follow-up:** Cloud sync DNS pending is infrastructure only
- **Recommended answer boundary:** audit trail complete
- **Transition:** 按提问跳转。
