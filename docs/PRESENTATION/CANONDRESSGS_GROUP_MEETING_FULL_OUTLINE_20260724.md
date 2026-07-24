# CanonDressGS Group Meeting Full Outline — 2026-07-24

**Plan:** 28 slides, 25–30 minutes, plus A1–A12 appendix.

## S01 · CanonDressGS 当前进展
- **Section / status**: Opening / `PAPER_CORE_METHOD_CANDIDATE`
- **Core question**: 当前项目真正解决到哪一步？
- **One-sentence conclusion**: 当前已支持固定身份、封闭五服装库中的 reference-controlled endpoint selection，但方法价值仍需 LOO 判定。
- **Main content**: Group Meeting Progress Report；subject02 / closed five-garment bank；PAPER_FINAL=0
- **Formulas**: none
- **Primary assets**: GM-A01-ENDPOINT-TEASER
- **Secondary assets**: GM-A35-FIGURE-MAP
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/candidates/fig01_teaser/pure_endpoint_endpoint_only_v1/contact_sheet_for_manual_adjudication.png; paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json
- **Allowed claims**: 展示当前封闭服装库任务和方法边界
- **Forbidden claims**: 宣称 unseen garment 或 multi-identity garment editing 已完成
- **Transition**: 先从真实任务动机开始。
- **Timing**: 40 seconds
- **Paper eligibility**: `CANDIDATE_AFTER_LOO_SCOPE_FREEZE`
- **LOO dependency**: `WAIT_FOR_LOO_FOR_FINAL_TITLE`
- **Risks / TODO**: Figure 1 候选仍需人工选图

## S02 · 为什么可动画 Avatar 仍被采集服装绑定？
- **Section / status**: Problem / `PAPER_CORE_METHOD_CANDIDATE`
- **Core question**: 如何在保持身份与动作的同时指定服装？
- **One-sentence conclusion**: 目标是用 reference garment 与 target pose/camera 生成同一身份的指定服装状态。
- **Main content**: 输入：reference RGB/mask；控制：target pose/camera；输出：same identity, specified garment
- **Formulas**: none
- **Primary assets**: GM-A01-ENDPOINT-TEASER
- **Secondary assets**: GM-A04-TEACHER-ENDPOINTS
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/candidates/fig01_teaser/pure_endpoint_endpoint_only_v1/contact_sheet_for_manual_adjudication.png; paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_clean_all_garments.png
- **Allowed claims**: 定义 reference-controlled garment state selection
- **Forbidden claims**: arbitrary garment synthesis；cross-identity completion
- **Transition**: 任务看似是外观控制，实际同时涉及几何、可见性和外观。
- **Timing**: 55 seconds
- **Paper eligibility**: `PAPER_MOTIVATION_CANDIDATE`
- **LOO dependency**: `NO`
- **Risks / TODO**: 避免把 closed-bank 说成 open-vocabulary

## S03 · 核心困难：服装状态不是单一颜色变量
- **Section / status**: Problem / `PAPER_CORE_EVIDENCE`
- **Core question**: 为什么连续 residual 插值容易失败？
- **One-sentence conclusion**: 服装同时改变 geometry、visibility 与 appearance，而 Gaussian 几何插值会制造无效中间支持。
- **Main content**: lookup：稳定但离散；continuous residual：灵活但产生 artifact；研究焦点：在可控性与几何有效性之间取舍
- **Formulas**: F06
- **Primary assets**: GM-A02-EARLY-FAILURES
- **Secondary assets**: GM-A05-GEOMETRY-CAUSAL-DATA
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/contact_sheets/limitations/early_failures.png; paper_protocol/reviewer_risk/continuous_control_causal_attribution_final_summary.json
- **Allowed claims**: 提出 geometry/visibility/appearance 三因素问题
- **Forbidden claims**: 将失败归因于单一网络容量而无因果证据
- **Transition**: 因此需要把问题拆成可验证的研究问题。
- **Timing**: 55 seconds
- **Paper eligibility**: `PAPER_PROBLEM_FORMULATION_CANDIDATE`
- **LOO dependency**: `NO`
- **Risks / TODO**: 早期图仅作为历史设计证据

## S04 · 研究问题地图
- **Section / status**: Problem / `PROJECT_GOVERNANCE_APPENDIX`
- **Core question**: 哪些问题已回答，哪些仍开放？
- **One-sentence conclusion**: 当前证据回答了 support、geometry 与 seen-endpoint control，LOO 和第二身份仍是开放问题。
- **Main content**: Q1 support capacity；Q2 geometry interpolation failure；Q3 seen endpoint control；Q4 basis vs hard lookup；Q5 out-of-bank adaptation；Q6 second identity
- **Formulas**: none
- **Primary assets**: GM-A35-FIGURE-MAP
- **Secondary assets**: GM-A33-LOO-PENDING
- **Backup assets**: none
- **Formal paths**: paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json; paper_draft/figures/pure_endpoint_refresh/pending/loo/repaired_protocol_pending.json
- **Allowed claims**: 明确已证与未证问题
- **Forbidden claims**: 提前判定 Q5/Q6 成功
- **Transition**: 下面按实验如何逐步收缩方法空间来讲。
- **Timing**: 55 seconds
- **Paper eligibility**: `PRESENTATION_ONLY`
- **LOO dependency**: `YES_FOR_Q4_Q5`
- **Risks / TODO**: 最终论文问题表述必须随 LOO 更新

## S05 · 方法路线如何被证据逐步收缩
- **Section / status**: Research Evolution / `HISTORICAL_DESIGN_EVIDENCE`
- **Core question**: 为什么最终路线与最初设想不同？
- **One-sentence conclusion**: 历史 reference-to-residual 路线连续失败，capacity 与 causal 诊断把方案收缩到 valid endpoints、显式 basis 和最小控制器。
- **Main content**: full reference-to-residual → oracle failure；support capacity PASS → representation/objective diagnosis；Teacher endpoints → explicit basis → Pure Endpoint；Headroom negative → LOO remains
- **Formulas**: none
- **Primary assets**: GM-A02-EARLY-FAILURES
- **Secondary assets**: GM-A27-HIST-REPRESENTATION
- **Backup assets**: GM-A29-CONTROLLER-VISUALS
- **Formal paths**: paper_draft/figures/contact_sheets/limitations/early_failures.png; paper_draft/figures/contact_sheets/ablations/representation_ablations.png; paper_draft/figures/contact_sheets/controller/controller_visuals.png
- **Allowed claims**: 历史负结果推动方法收缩
- **Forbidden claims**: 声称从一开始就设计了最终 pipeline
- **Transition**: 收缩后的路线建立在冻结的 avatar 技术底座上。
- **Timing**: 75 seconds
- **Paper eligibility**: `PRESENTATION_NARRATIVE_ONLY`
- **LOO dependency**: `NO`
- **Risks / TODO**: 必须区分历史协议与当前协议

## S06 · 冻结的技术底座
- **Section / status**: Method Foundation / `PAPER_CORE_METHOD_CANDIDATE`
- **Core question**: 哪些模块固定，哪些模块允许学习？
- **One-sentence conclusion**: MMLP-Human canonical Gaussians、deformation、LBS 与 renderer 均冻结，仅 garment residual 表示和 reference control 按实验契约变化。
- **Main content**: Frozen: base avatar, deformation, LBS, renderer；Trainable by stage: garment endpoint residual or small controller；No target-forward leakage in current protocol
- **Formulas**: none
- **Primary assets**: GM-A03-METHOD-WIREFRAME
- **Secondary assets**: GM-A35-FIGURE-MAP
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/candidates/fig02_method/layout_wireframe.svg; paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json
- **Allowed claims**: 说明冻结边界
- **Forbidden claims**: 暗示基础 avatar 在本任务中重新训练；将 target views 用于 inference
- **Transition**: 在这个底座上先建立每件服装的有效 Teacher Endpoint。
- **Timing**: 55 seconds
- **Paper eligibility**: `PAPER_METHOD_CANDIDATE`
- **LOO dependency**: `NO`
- **Risks / TODO**: Figure 2 仍为 METHOD_FREEZE_PENDING_LOO

## S07 · Garment Teacher Endpoint
- **Section / status**: Method Foundation / `PAPER_CORE_EVIDENCE`
- **Core question**: 如何得到稳定的服装状态？
- **One-sentence conclusion**: 每件 seen garment 通过多视图 rendering objective 优化 canonical residual，得到可稳定复用的 valid endpoint。
- **Main content**: 每件 garment 独立优化；canonical residual 与 base state 组合；Teacher 是训练目标与稳定 endpoint，不是 inference network
- **Formulas**: F01, F02
- **Primary assets**: GM-A04-TEACHER-ENDPOINTS
- **Secondary assets**: GM-A15-PARITY-SAFETY
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_clean_all_garments.png; paper_draft/figures/pure_endpoint_refresh/candidates/supplementary/pure_endpoint_parity_safety_v1/pure_endpoint_parity_safety_v1.png
- **Allowed claims**: Teacher Endpoint 是多视图优化得到的稳定状态
- **Forbidden claims**: 称 Teacher 为 Upper Bound；声称其覆盖 unseen garment
- **Transition**: 但两个有效 endpoint 之间的线性几何并不一定有效。
- **Timing**: 70 seconds
- **Paper eligibility**: `PAPER_CORE_EVIDENCE`
- **LOO dependency**: `NO`
- **Risks / TODO**: 需避免 upper bound 旧称

## S08 · 为什么不能直接插值端点？
- **Section / status**: Causal Diagnosis / `PAPER_CORE_EVIDENCE`
- **Core question**: 中间 artifact 的主因是什么？
- **One-sentence conclusion**: 因果注入显示 geometry main effect 约 0.9959，geometry sufficiency 与 necessity 均为 10/10。
- **Main content**: geometry / visibility / appearance factorial injection；main effect ≈ 0.9959；sufficiency 10/10; necessity 10/10；visibility/appearance 不是主因
- **Formulas**: F06
- **Primary assets**: GM-A05-GEOMETRY-CAUSAL-DATA
- **Secondary assets**: GM-A02-EARLY-FAILURES
- **Backup assets**: none
- **Formal paths**: paper_protocol/reviewer_risk/continuous_control_causal_attribution_final_summary.json; paper_draft/figures/contact_sheets/limitations/early_failures.png
- **Allowed claims**: geometry 是当前中间 artifact 主因
- **Forbidden claims**: 推广到所有 avatar/garment 场景；说 visibility/appearance 永远无影响
- **Transition**: 既然几何不能线性混合，下一步测试保留两套有效支持。
- **Timing**: 70 seconds
- **Paper eligibility**: `PAPER_CORE_EVIDENCE`
- **LOO dependency**: `NO`
- **Risks / TODO**: 当前没有 Figure Bank 内的专用 geometry 图，建议用数值 callout

## S09 · Dual-Support：保留有效几何支持
- **Section / status**: Causal Diagnosis / `SUPPLEMENTARY_EXTENSION`
- **Core question**: 不插值 geometry 能否改善 mixed state？
- **One-sentence conclusion**: Oracle Dual-Support 在全部 10 对服装上改善指标与严重 artifact，但它是 oracle mechanism extension，不是成功的自动 controller。
- **Main content**: 保留两套 endpoint geometry；只对 effective opacity 加权；LPIPS 0.085303→0.056985；IoU 0.710052→0.725290; Boundary F 0.300263→0.379103
- **Formulas**: F07
- **Primary assets**: GM-A06-DUAL-LPIPS
- **Secondary assets**: GM-A07-DUAL-SHEET
- **Backup assets**: GM-A29-CONTROLLER-VISUALS
- **Formal paths**: paper_draft/figures/plots/dual_support/dual_support_all_pair_garment_lpips.png; paper_draft/figures/contact_sheets/dual_support/dual_support.png; paper_draft/figures/contact_sheets/controller/controller_visuals.png
- **Allowed claims**: oracle Dual-Support 证明 geometry-safe mechanism 有效
- **Forbidden claims**: 自动 reference-conditioned mixed controller 已成功；Dual-Support 是当前主 pipeline
- **Transition**: 主线仍需要一个可解释、最小化的 endpoint coordinate system。
- **Timing**: 75 seconds
- **Paper eligibility**: `SUPPLEMENTARY_METHOD_EXTENSION`
- **LOO dependency**: `NO`
- **Risks / TODO**: O01_O03 与 O02_O03 仍困难；需持续强调 oracle

## S10 · Explicit Global Basis
- **Section / status**: Core Method / `PAPER_CORE_METHOD_CANDIDATE`
- **Core question**: 五个 Teacher residual 如何形成显式坐标？
- **One-sentence conclusion**: 对五个 centered Teacher residual 做 SVD 得到 rank-4 global basis，它首先是 endpoint 坐标系，adaptation 价值尚待 LOO。
- **Main content**: five residuals centered by mean；maximum centered rank is 4；endpoint coefficients are exact coordinates；not yet a universal semantic manifold
- **Formulas**: F03
- **Primary assets**: GM-A17-HEADROOM-PARITY
- **Secondary assets**: GM-A04-TEACHER-ENDPOINTS
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/headroom_refresh/plots/headroom/teacher_svd_parity.png; paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_clean_all_garments.png
- **Allowed claims**: rank-4 精确覆盖五个 centered endpoints
- **Forbidden claims**: 称 rank-4 为强压缩；称其已证明通用 semantic manifold
- **Transition**: 在这个坐标系中，用最小 reference predictor 选择 endpoint。
- **Timing**: 65 seconds
- **Paper eligibility**: `PAPER_CORE_METHOD_CANDIDATE`
- **LOO dependency**: `WAIT_FOR_LOO_FOR_BASIS_CLAIM`
- **Risks / TODO**: basis 的论文定位由 LOO 决定

## S11 · Pure Endpoint Pipeline
- **Section / status**: Core Method / `PAPER_CORE_METHOD_CANDIDATE`
- **Core question**: reference 如何控制最终 Gaussian 状态？
- **One-sentence conclusion**: Frozen F2 特征经 mean/max aggregation 和 LayerNorm+Linear(4) 预测系数，再以 endpoint snapping 实现稳定状态。
- **Main content**: reference RGB/mask → frozen F2；mean/max aggregation → LN + Linear(4)；raw coefficient → nearest endpoint coefficient；snapped endpoint → frozen renderer
- **Formulas**: F04, F05
- **Primary assets**: GM-A03-METHOD-WIREFRAME
- **Secondary assets**: GM-A12-RAW-SNAPPED-SHEET
- **Backup assets**: GM-A01-ENDPOINT-TEASER
- **Formal paths**: paper_draft/figures/candidates/fig02_method/layout_wireframe.svg; paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_raw_vs_snapped.png; paper_draft/figures/pure_endpoint_refresh/candidates/fig01_teaser/pure_endpoint_endpoint_only_v1/contact_sheet_for_manual_adjudication.png
- **Allowed claims**: 最小控制器可靠选择 seen endpoints；raw 与 realized coefficient 明确分开
- **Forbidden claims**: 把 snapping 描述为精确 continuous regression；把方法写成 unseen garment synthesis
- **Transition**: 可靠性需要严格的 condition-fold cross-fit 验证。
- **Timing**: 75 seconds
- **Paper eligibility**: `PAPER_CORE_METHOD_CANDIDATE`
- **LOO dependency**: `WAIT_FOR_LOO_FOR_FINAL_FIGURE2`
- **Risks / TODO**: Figure 2 状态 METHOD_FREEZE_PENDING_LOO

## S12 · Pure Endpoint 实验设计
- **Section / status**: Core Method / `PROJECT_GOVERNANCE_APPENDIX`
- **Core question**: 如何避免把服装识别成功误当成泛化？
- **One-sentence conclusion**: 4 rotations×3 seeds 的 condition-fold cross-fit 连同 7 methods、perturbation、parity 与 safety 检查，限定了 clean closed-bank 结论。
- **Main content**: 24 completed runs；4 rotations × 3 seeds；train/calibration/test condition folds；7 methods + hard lookup + perturbations；endpoint parity and identity safety
- **Formulas**: none
- **Primary assets**: GM-A38-PURE-ROTATIONS
- **Secondary assets**: GM-A15-PARITY-SAFETY
- **Backup assets**: none
- **Formal paths**: paper_protocol/reviewer_risk/pure_endpoint_rotation_manifests.json; paper_draft/figures/pure_endpoint_refresh/candidates/supplementary/pure_endpoint_parity_safety_v1/pure_endpoint_parity_safety_v1.png
- **Allowed claims**: 协议支持固定 subject02、五个 seen garments 的闭集评估
- **Forbidden claims**: 把 condition-fold 说成 garment-held-out；把 hard lookup 当 unseen baseline
- **Transition**: 在这个边界内，Pure Endpoint 的 clean 结果非常明确。
- **Timing**: 70 seconds
- **Paper eligibility**: `PAPER_EXPERIMENT_PROTOCOL_CANDIDATE`
- **LOO dependency**: `NO`
- **Risks / TODO**: 需口头区分 condition-held-out 与 garment-held-out

## S13 · Pure Endpoint 主结果
- **Section / status**: Core Results / `PAPER_CORE_EVIDENCE`
- **Core question**: reference 能否可靠选择五个 seen endpoints？
- **One-sentence conclusion**: 24/24 runs 完成，五件 garment 的 clean top-1 与 precision/recall 均为 1，parity 与 identity safety PASS。
- **Main content**: clean top-1 = 1.0；per-garment precision/recall = 1.0；endpoint parity PASS；identity contamination = 0; severe wrong outfit = 0
- **Formulas**: none
- **Primary assets**: GM-A08-PURE-TOP1
- **Secondary assets**: GM-A09-ENDPOINT-MATCH, GM-A15-PARITY-SAFETY
- **Backup assets**: GM-A04-TEACHER-ENDPOINTS
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/clean_method_top1.png; paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/endpoint_exact_match.png; paper_draft/figures/pure_endpoint_refresh/candidates/supplementary/pure_endpoint_parity_safety_v1/pure_endpoint_parity_safety_v1.png; paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_clean_all_garments.png
- **Allowed claims**: closed five-garment bank endpoint control supported
- **Forbidden claims**: unseen garment generalization supported；优于 hard lookup
- **Transition**: 但 clean benchmark 上的成功与 hard lookup 功能等价。
- **Timing**: 70 seconds
- **Paper eligibility**: `PAPER_CORE_EVIDENCE`
- **LOO dependency**: `NO`
- **Risks / TODO**: clean benchmark 已饱和

## S14 · Hard Lookup 等价性
- **Section / status**: Core Results / `PAPER_CORE_EVIDENCE`
- **Core question**: Pure Endpoint 是否优于 hard lookup？
- **One-sentence conclusion**: clean 条件下正确方法选择同一 endpoint 并产生等价 render，因此当前只能描述功能等价，不能宣称优于 lookup。
- **Main content**: clean agreement；error overlap under perturbation；endpoint-equivalent render；HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY
- **Formulas**: none
- **Primary assets**: GM-A10-HARD-LOOKUP
- **Secondary assets**: GM-A36-RENDER-EQUIVALENCE
- **Backup assets**: GM-A08-PURE-TOP1
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/candidates/fig05_hard_lookup/pure_endpoint_hard_lookup_v1/pure_endpoint_hard_lookup_v1.png; paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/render_equivalence_and_cache_summary.png; paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/clean_method_top1.png
- **Allowed claims**: clean closed-bank 下功能等价
- **Forbidden claims**: CanonDressGS 优于所有 hard lookup；将 descriptive relation 写成 superiority
- **Transition**: 区别在于 predictor 输出了连续坐标，但实际成功仍来自 endpoint realization。
- **Timing**: 70 seconds
- **Paper eligibility**: `PAPER_CORE_EVIDENCE_DESCRIPTIVE_ONLY`
- **LOO dependency**: `WAIT_FOR_LOO_FOR_VALUE_BEYOND_LOOKUP`
- **Risks / TODO**: Figure 5 仍需人工裁决

## S15 · Raw Coefficient 与 Endpoint Snapping
- **Section / status**: Core Results / `PAPER_CORE_EVIDENCE`
- **Core question**: 连续系数预测本身是否精确？
- **One-sentence conclusion**: raw continuous coefficient 误差仍高，而 snapping 将预测可靠地映射到正确 endpoint；当前成功来自正确 realization。
- **Main content**: raw predicted coefficient；nearest endpoint coefficient；realized snapped coefficient；render uses realized state
- **Formulas**: F04, F05
- **Primary assets**: GM-A11-RAW-SNAPPED-PLOT
- **Secondary assets**: GM-A12-RAW-SNAPPED-SHEET
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/raw_vs_realized_coefficient_error.png; paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_raw_vs_snapped.png
- **Allowed claims**: snapping 有效修正 endpoint realization
- **Forbidden claims**: raw regression 已精确；连续 manifold 已学得
- **Transition**: 接下来检查这种 endpoint control 对 reference 扰动是否稳定。
- **Timing**: 65 seconds
- **Paper eligibility**: `PAPER_CORE_EVIDENCE`
- **LOO dependency**: `NO`
- **Risks / TODO**: 图例必须固定 raw/snapped 术语

## S16 · Perturbation 与安全性
- **Section / status**: Core Results / `LIMITATION`
- **Core question**: reference 质量下降时系统如何失败？
- **One-sentence conclusion**: single reference 基本稳定、complete dropout 可安全 abstain，但 mild blur 产生明显 endpoint flip，限制了当前鲁棒性。
- **Main content**: single reference: largely stable；mild blur: elevated endpoint flips；complete dropout: safe abstention；identity contamination remains zero
- **Formulas**: none
- **Primary assets**: GM-A13-PERTURBATION
- **Secondary assets**: GM-A14-BLUR-FAILURES, GM-A15-PARITY-SAFETY
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/candidates/fig06_limitations/pure_endpoint_perturbation_v1/pure_endpoint_perturbation_v1.png; paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_mild_blur_failures.png; paper_draft/figures/pure_endpoint_refresh/candidates/supplementary/pure_endpoint_parity_safety_v1/pure_endpoint_parity_safety_v1.png
- **Allowed claims**: 报告具体 perturbation 边界与安全行为
- **Forbidden claims**: 宣称鲁棒 reference understanding 已解决
- **Transition**: 既然 raw coefficient 不精确，下一步自然问题是 rendering loss 能否优化它。
- **Timing**: 65 seconds
- **Paper eligibility**: `PAPER_LIMITATION_CANDIDATE`
- **LOO dependency**: `NO`
- **Risks / TODO**: blur failure 必须保留在主汇报

## S17 · 为什么做 Coefficient Headroom？
- **Section / status**: Headroom / `NEGATIVE_DIAGNOSTIC`
- **Core question**: basis 是否不仅能分类 endpoint，还能提供优化空间？
- **One-sentence conclusion**: Headroom 实验检验固定 rank-4 span 内 test-time coefficient refinement 是否能超过 Teacher Endpoint。
- **Main content**: Teacher vs exact SVD reconstruction；refined global coefficient；unconstrained full residual comparator；pre-registered LPIPS gates
- **Formulas**: F08
- **Primary assets**: GM-A24-REFINED-DECOMPOSITION
- **Secondary assets**: GM-A17-HEADROOM-PARITY
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/headroom_refresh/plots/headroom/refined_lookup_decomposition.png; paper_draft/figures/headroom_refresh/plots/headroom/teacher_svd_parity.png
- **Allowed claims**: 提出 seen-endpoint refinement 假设
- **Forbidden claims**: 预设 refinement 会改善结果
- **Transition**: 实验只优化 4 个系数，并用 full residual 检查是否存在更大 headroom。
- **Timing**: 55 seconds
- **Paper eligibility**: `SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC`
- **LOO dependency**: `NO`
- **Risks / TODO**: 不得将 headroom 动机写成最终 pipeline

## S18 · Headroom 实验设计
- **Section / status**: Headroom / `NEGATIVE_DIAGNOSTIC`
- **Core question**: 如何区分 basis 限制与优化失败？
- **One-sentence conclusion**: 120 个 optimizer runs 在 optimize/calibration/test 上比较 Teacher、SVD、refined coefficient 与 full residual，并进行 λ 选择与门槛审计。
- **Main content**: 120/120 optimizer runs；36,000 optimizer steps；960 checkpoints；Teacher/SVD parity 20/20 PASS；fixed rank-4 basis; four trainable coefficients
- **Formulas**: F08
- **Primary assets**: GM-A17-HEADROOM-PARITY
- **Secondary assets**: GM-A18-HEADROOM-FOUR-METHOD, GM-A21-HEADROOM-DISPLACEMENT
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/headroom_refresh/plots/headroom/teacher_svd_parity.png; paper_draft/figures/headroom_refresh/plots/headroom/four_method_test_metrics.png; paper_draft/figures/headroom_refresh/plots/headroom/coefficient_displacement_trajectory.png
- **Allowed claims**: 说明 sealed attempt_002 的完整 protocol
- **Forbidden claims**: 将 full residual 结果归因于未完成运行；暗示 renderer 重新运行于本规划任务
- **Transition**: 完整执行后，结果否定了 seen-endpoint refinement 路线。
- **Timing**: 65 seconds
- **Paper eligibility**: `SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC`
- **LOO dependency**: `NO`
- **Risks / TODO**: 数字应与 sealed summary 原样一致

## S19 · Headroom 结果：Teacher Span 已在局部最优
- **Section / status**: Headroom / `NEGATIVE_DIAGNOSTIC`
- **Core question**: refinement 是否改善 Teacher Endpoint？
- **One-sentence conclusion**: global rank-4 coefficient refinement 未改善 Teacher，0/5 garments 过门槛；full residual 反而产生 patch/cloud/mottle 与 edge scatter。
- **Main content**: macro Teacher-minus-refined LPIPS < 0；0/5 garments pass improvement gate；full residual severe degradation；classification: TEACHER_SPAN_AT_LOCAL_OPTIMUM
- **Formulas**: none
- **Primary assets**: GM-A16-HEADROOM-OVERVIEW
- **Secondary assets**: GM-A19-HEADROOM-GATES, GM-A20-HEADROOM-PER-GARMENT
- **Backup assets**: GM-A22-FULL-RESIDUAL-ARTIFACT, GM-A23-SPAN-NULL
- **Formal paths**: paper_draft/figures/headroom_refresh/candidates/supplementary/headroom_negative_diagnostic_v1/supplementary_negative_diagnostic_v1.png; paper_draft/figures/headroom_refresh/plots/headroom/lpips_gate_summary.png; paper_draft/figures/headroom_refresh/plots/headroom/per_garment_teacher_minus_refined_lpips.png; paper_draft/figures/headroom_refresh/contact_sheets/headroom/full_residual_artifact_examples.png; paper_draft/figures/headroom_refresh/plots/headroom/span_recovery_denominator_null.png
- **Allowed claims**: Headroom 没有改善 Teacher Endpoint；full residual 退化并产生 artifacts
- **Forbidden claims**: 将 Headroom 写成正结果；声称 basis span recoverable denominator 非零
- **Transition**: 这个负结果直接改变当前 pipeline 的取舍。
- **Timing**: 80 seconds
- **Paper eligibility**: `SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC`
- **LOO dependency**: `NO`
- **Risks / TODO**: caption 必须写明不支持 render refinement 进入主线

## S20 · Headroom 如何改变方法设计
- **Section / status**: Headroom / `LIMITATION`
- **Core question**: 哪些路线因此被冻结？
- **One-sentence conclusion**: reference→coefficient→test-time refinement 不进入主 pipeline，spatial coefficient field 也因缺乏 full-residual headroom 仅保留 contingency。
- **Main content**: remove test-time render refinement from main pipeline；Headroom: CONSUMED_FROM_SEALED_RESULT；Spatial field: CONTINGENCY_ONLY_NOT_STARTED；negative result narrows claims and implementation
- **Formulas**: none
- **Primary assets**: GM-A24-REFINED-DECOMPOSITION
- **Secondary assets**: GM-A22-FULL-RESIDUAL-ARTIFACT, GM-A35-FIGURE-MAP
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/headroom_refresh/plots/headroom/refined_lookup_decomposition.png; paper_draft/figures/headroom_refresh/contact_sheets/headroom/full_residual_artifact_examples.png; paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json
- **Allowed claims**: 负结果支持冻结 refinement 与 local spatial basis
- **Forbidden claims**: 宣称 spatial basis 已测试失败；宣称所有优化都无效
- **Transition**: 剩下真正能判断 basis 是否有独立价值的问题是 LOO adaptation。
- **Timing**: 60 seconds
- **Paper eligibility**: `PAPER_LIMITATION_OR_SUPPLEMENTARY`
- **LOO dependency**: `NO`
- **Risks / TODO**: spatial field 是未启动 contingency，不是负实验

## S21 · LOO：basis 剩余的核心价值检验
- **Section / status**: LOO Decision / `PENDING_LOO`
- **Core question**: explicit basis 能否超越 closed-bank lookup？
- **One-sentence conclusion**: LOO 用四件 garment 构建 rank≤3 basis，并完全排除第五件，直接检验 out-of-bank basis adaptation。
- **Main content**: build μ_-g and B_-g from four garments；exclude held-out garment completely；compare K=1/K=2, hard lookup, oracle projection, full residual；few-view coefficient fitting
- **Formulas**: F09, F10
- **Primary assets**: GM-A33-LOO-PENDING
- **Secondary assets**: GM-A28-O07-LIMITATION
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/pending/loo/repaired_protocol_pending.json; paper_draft/figures/contact_sheets/limitations/o07.png
- **Allowed claims**: LOO 是判断 basis adaptation value 的关键实验
- **Forbidden claims**: 预判 LOO 成功；把 historical O07 当正式 LOO 结果
- **Transition**: 协议已经修复完成，但当前没有 sealed execution result。
- **Timing**: 70 seconds
- **Paper eligibility**: `WAIT_FOR_LOO`
- **LOO dependency**: `REQUIRED`
- **Risks / TODO**: 不得读取 partial/active output

## S22 · LOO 当前状态
- **Section / status**: LOO Decision / `PENDING_LOO`
- **Core question**: 我们现在能报告什么？
- **One-sentence conclusion**: LOO protocol 已 READY，但 metadata-only 检查显示 LOO_ATTEMPT=ABSENT，因此结果页只保留占位。
- **Main content**: protocol: LOO_FEW_VIEW_FOLD_REPAIRED_AND_READY；execution: READY / PENDING EXECUTION；LOO_ATTEMPT=ABSENT；files_read=0; no fabricated metrics
- **Formulas**: none
- **Primary assets**: GM-A33-LOO-PENDING
- **Secondary assets**: GM-A35-FIGURE-MAP
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/pending/loo/repaired_protocol_pending.json; paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json
- **Allowed claims**: 只报告 protocol readiness 与 pending 状态
- **Forbidden claims**: 任何 LOO 指标、成功率或图片；使用正在运行或 partial output
- **Transition**: 在等待 LOO 的同时，历史消融只用于解释设计演化。
- **Timing**: 40 seconds
- **Paper eligibility**: `WAIT_FOR_LOO`
- **LOO dependency**: `REQUIRED`
- **Risks / TODO**: 汇报当天需再次做 metadata-only 状态确认

## S23 · 历史 51 实验如何塑造最小模型
- **Section / status**: Historical Evidence / `HISTORICAL_DESIGN_EVIDENCE`
- **Core question**: 哪些旧实验仍有设计解释价值？
- **One-sentence conclusion**: rank、mask、mean-only、standardization 与 reference-count 消融解释了为何收缩到显式 basis 与最小 predictor，但它们属于历史 view-transductive 协议。
- **Main content**: 51/51 experiments; 14,400 steps；rank 1–4；clothing mask / mean-only / standardization；reference count 1/2/3；historical protocol kept separate
- **Formulas**: none
- **Primary assets**: GM-A37-HIST-ALL
- **Secondary assets**: GM-A25-HIST-RANK, GM-A26-HIST-REFERENCE
- **Backup assets**: GM-A27-HIST-REPRESENTATION
- **Formal paths**: paper_draft/figures/plots/historical_ablations/historical_ablation_edit_reduction.png; paper_draft/figures/plots/historical_ablations/rank_1_4_edit_reduction.png; paper_draft/figures/plots/historical_ablations/reference_count_1_3_edit_reduction.png; paper_draft/figures/contact_sheets/ablations/representation_ablations.png
- **Allowed claims**: 历史消融解释设计选择
- **Forbidden claims**: 与 current Pure Endpoint 无标记合并成主表；把 historical view-transductive 写成 current generalization
- **Transition**: 这些证据也暴露了当前仍必须正面呈现的失败模式。
- **Timing**: 70 seconds
- **Paper eligibility**: `HISTORICAL_ONLY`
- **LOO dependency**: `NO`
- **Risks / TODO**: 表格必须用 Historical 标签

## S24 · Failure Modes / Limitations
- **Section / status**: Limitations / `LIMITATION`
- **Core question**: 当前系统最脆弱的地方是什么？
- **One-sentence conclusion**: O07 collapse、blur flip、neural/full-residual artifacts、困难 Dual-Support pair 与第二身份服装容量共同限定当前 claim。
- **Main content**: historical O07 → O03 collapse；mild-blur endpoint flip；neural/full-residual patch-cloud-mottle；O01_O03 and O02_O03 remain difficult；Subject00 loose-clothing capacity unproven
- **Formulas**: none
- **Primary assets**: GM-A28-O07-LIMITATION
- **Secondary assets**: GM-A14-BLUR-FAILURES, GM-A22-FULL-RESIDUAL-ARTIFACT
- **Backup assets**: GM-A07-DUAL-SHEET
- **Formal paths**: paper_draft/figures/contact_sheets/limitations/o07.png; paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_mild_blur_failures.png; paper_draft/figures/headroom_refresh/contact_sheets/headroom/full_residual_artifact_examples.png; paper_draft/figures/contact_sheets/dual_support/dual_support.png
- **Allowed claims**: 集中展示已知失败边界
- **Forbidden claims**: 把失败归为展示偶例而忽略定量证据
- **Transition**: 其中第二身份目前只完成了 base-avatar foundation。
- **Timing**: 70 seconds
- **Paper eligibility**: `PAPER_LIMITATION_CANDIDATE`
- **LOO dependency**: `NO`
- **Risks / TODO**: 避免堆叠太多小图，最终 PPT 需人工选 3 类

## S25 · Multi-Identity 基础进展
- **Section / status**: Multi-Identity / `MULTI_IDENTITY_FOUNDATION`
- **Core question**: 第二身份做到哪一步？
- **One-sentence conclusion**: Subject00 short canary 与 medium base-avatar pilot PASS，formal 与三服装端点仍 pending；AvatarReX 仅 preflight ready 且媒体禁止导出。
- **Main content**: Subject00: short canary PASS；Subject00: medium base-avatar PASS；formal 101245-step protocol READY; garment endpoints pending；AvatarReX: zero-copy/preflight ready; derived template/LBS missing；AvatarReX media use = 0
- **Formulas**: none
- **Primary assets**: GM-A31-SUBJECT00-SHEET
- **Secondary assets**: GM-A32-SUBJECT00-PLOT, GM-A34-AVATARREX-METADATA
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/contact_sheets/subject00/subject00.png; paper_draft/figures/plots/subject00/subject00_step0_to_medium_lpips.png; paper_draft/figures/pending/avatarrex_license_restricted/README.md
- **Allowed claims**: 仅称 base-avatar foundation 与 protocol readiness
- **Forbidden claims**: multi-identity garment editing completed；展示或导出 AvatarReX media
- **Transition**: 因此论文冻结必须把已完成核心与待验证扩展分开。
- **Timing**: 65 seconds
- **Paper eligibility**: `SUPPLEMENTARY_FOUNDATION_ONLY`
- **LOO dependency**: `NO`
- **Risks / TODO**: Subject00 图不能被解释为 garment editing

## S26 · 当前论文方法冻结状态
- **Section / status**: Method Freeze / `PROJECT_GOVERNANCE_APPENDIX`
- **Core question**: 哪些模块已冻结，哪些等待 LOO？
- **One-sentence conclusion**: Teacher Endpoint、explicit endpoint basis、Pure Endpoint 与 geometry causal 已形成核心候选；basis 的超 lookup 定位和 Figure 2 仍等待 LOO。
- **Main content**: Frozen candidate: Teacher Endpoint, explicit basis, Pure Endpoint；Core evidence: geometry causal；Supplementary: oracle Dual-Support；Negative supplementary: Controller and Headroom；Figure 2: METHOD_FREEZE_PENDING_LOO
- **Formulas**: none
- **Primary assets**: GM-A35-FIGURE-MAP
- **Secondary assets**: GM-A03-METHOD-WIREFRAME, GM-A33-LOO-PENDING
- **Backup assets**: none
- **Formal paths**: paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json; paper_draft/figures/candidates/fig02_method/layout_wireframe.svg; paper_draft/figures/pure_endpoint_refresh/pending/loo/repaired_protocol_pending.json
- **Allowed claims**: 展示当前冻结矩阵
- **Forbidden claims**: 称论文已 final；把 Controller 放回主方法
- **Transition**: 冻结矩阵对应一组明确的 reviewer risks。
- **Timing**: 65 seconds
- **Paper eligibility**: `PRESENTATION_GOVERNANCE`
- **LOO dependency**: `REQUIRED_FOR_FINAL_FREEZE`
- **Risks / TODO**: PAPER_FINAL 必须保持 0

## S27 · 当前不足与 Reviewer 风险
- **Section / status**: Method Freeze / `LIMITATION`
- **Core question**: 投稿前最可能被质疑什么？
- **One-sentence conclusion**: 最大风险是 clean closed-bank 饱和且与 hard lookup 等价，LOO 将决定 basis 是否具有独立方法贡献。
- **Main content**: closed-bank saturation；no superiority over hard lookup；rank-4 is not strong compression；no headroom for refinement；multi-identity garment editing incomplete；automatic mixed controller failed；Teacher/view-transductive boundary
- **Formulas**: none
- **Primary assets**: GM-A10-HARD-LOOKUP
- **Secondary assets**: GM-A23-SPAN-NULL, GM-A33-LOO-PENDING
- **Backup assets**: GM-A30-CONTROLLER-BUDGET
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/candidates/fig05_hard_lookup/pure_endpoint_hard_lookup_v1/pure_endpoint_hard_lookup_v1.png; paper_draft/figures/headroom_refresh/plots/headroom/span_recovery_denominator_null.png; paper_draft/figures/pure_endpoint_refresh/pending/loo/repaired_protocol_pending.json; paper_draft/figures/plots/controller/controller_budget_continuation_trajectory.png
- **Allowed claims**: 主动呈现 reviewer 风险和对应证据
- **Forbidden claims**: 用工程完成度替代科学贡献；隐去 hard lookup 等价
- **Transition**: 下一步必须按风险优先级执行，而不是继续扩大方法。
- **Timing**: 75 seconds
- **Paper eligibility**: `PAPER_RISK_DISCUSSION`
- **LOO dependency**: `REQUIRED`
- **Risks / TODO**: LOO 失败时需收缩论文主张

## S28 · 下一步：先完成决定性实验，再冻结论文
- **Section / status**: Closing / `PENDING_LOO`
- **Core question**: 今晚之后的最高优先级是什么？
- **One-sentence conclusion**: 先完成并 sealed LOO，再据此冻结方法范围、refresh Figure Bank 和重写论文；第二身份与 spatial field 仅按证据门槛推进。
- **Main content**: 1. Execute and seal LOO；2. Freeze basis claim and Figure 2；3. Refresh Figure Bank；4. Rewrite Method/Experiments；5. Subject00 formal + three-garment canary；6. Spatial field only if a new oracle headroom gate supports it
- **Formulas**: none
- **Primary assets**: GM-A35-FIGURE-MAP
- **Secondary assets**: GM-A33-LOO-PENDING
- **Backup assets**: none
- **Formal paths**: paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json; paper_draft/figures/pure_endpoint_refresh/pending/loo/repaired_protocol_pending.json
- **Allowed claims**: 按证据门槛安排 next steps
- **Forbidden claims**: 自动启动 PPTX 生成；在 LOO 前宣布论文完成
- **Transition**: 汇报结束，进入问题讨论。
- **Timing**: 55 seconds
- **Paper eligibility**: `PRESENTATION_CLOSING`
- **LOO dependency**: `REQUIRED`
- **Risks / TODO**: NEXT_TASK 仅为人工审阅 outline 与选图

# Appendix

## A01 · 数据集与服装定义
- **Section / status**: Appendix / `PROJECT_GOVERNANCE_APPENDIX`
- **Core question**: 评估对象和闭集边界是什么？
- **One-sentence conclusion**: 主实验固定 subject02 与 O01/O02/O03/O04/O08 五件 seen garments。
- **Main content**: identity；garment IDs；reference/query units
- **Formulas**: none
- **Primary assets**: GM-A04-TEACHER-ENDPOINTS
- **Secondary assets**: GM-A38-PURE-ROTATIONS
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_clean_all_garments.png; paper_protocol/reviewer_risk/pure_endpoint_rotation_manifests.json
- **Allowed claims**: 定义 closed bank
- **Forbidden claims**: unseen garment
- **Transition**: 按提问跳转。
- **Timing**: 60 seconds
- **Paper eligibility**: `APPENDIX`
- **LOO dependency**: `NO`
- **Risks / TODO**: 需最终核对展示名称

## A02 · Exact Rotation Manifests
- **Section / status**: Appendix / `PROJECT_GOVERNANCE_APPENDIX`
- **Core question**: 每个 rotation 如何划分？
- **One-sentence conclusion**: 四个 rotation 的 train/calibration/test condition folds 已封存且无重复泄漏。
- **Main content**: rotation records；fold membership；duplicate preservation
- **Formulas**: none
- **Primary assets**: GM-A38-PURE-ROTATIONS
- **Secondary assets**: none
- **Backup assets**: none
- **Formal paths**: paper_protocol/reviewer_risk/pure_endpoint_rotation_manifests.json
- **Allowed claims**: 协议可复核
- **Forbidden claims**: garment-held-out
- **Transition**: 按提问跳转。
- **Timing**: 75 seconds
- **Paper eligibility**: `APPENDIX`
- **LOO dependency**: `NO`
- **Risks / TODO**: 只展示摘要，不暴露过密 manifest

## A03 · Pure Endpoint 七方法注册表
- **Section / status**: Appendix / `PROJECT_GOVERNANCE_APPENDIX`
- **Core question**: 比较方法是否同边界？
- **One-sentence conclusion**: 七方法共享 frozen endpoint/evaluator 边界，hard lookup relation 仅描述性报告。
- **Main content**: method IDs；trainable parameters；realization rule
- **Formulas**: none
- **Primary assets**: GM-A38-PURE-ROTATIONS
- **Secondary assets**: GM-A10-HARD-LOOKUP
- **Backup assets**: none
- **Formal paths**: paper_protocol/reviewer_risk/pure_endpoint_rotation_manifests.json; paper_draft/figures/pure_endpoint_refresh/candidates/fig05_hard_lookup/pure_endpoint_hard_lookup_v1/pure_endpoint_hard_lookup_v1.png
- **Allowed claims**: 方法注册完整
- **Forbidden claims**: superiority claim
- **Transition**: 按提问跳转。
- **Timing**: 75 seconds
- **Paper eligibility**: `APPENDIX`
- **LOO dependency**: `NO`
- **Risks / TODO**: 最终 PPT 可转为小表

## A04 · 完整 Perturbation 表
- **Section / status**: Appendix / `LIMITATION`
- **Core question**: 每类扰动的分母与失败率是什么？
- **One-sentence conclusion**: single reference、mild blur 与 complete dropout 必须分别报告，不能用总平均掩盖 blur failure。
- **Main content**: denominators；endpoint flips；abstention；identity safety
- **Formulas**: none
- **Primary assets**: GM-A13-PERTURBATION
- **Secondary assets**: GM-A14-BLUR-FAILURES, GM-A15-PARITY-SAFETY
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/candidates/fig06_limitations/pure_endpoint_perturbation_v1/pure_endpoint_perturbation_v1.png; paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_mild_blur_failures.png; paper_draft/figures/pure_endpoint_refresh/candidates/supplementary/pure_endpoint_parity_safety_v1/pure_endpoint_parity_safety_v1.png
- **Allowed claims**: 完整 robustness 边界
- **Forbidden claims**: 总体鲁棒性已解决
- **Transition**: 按提问跳转。
- **Timing**: 75 seconds
- **Paper eligibility**: `APPENDIX`
- **LOO dependency**: `NO`
- **Risks / TODO**: 保持原始分母

## A05 · 历史 Rank Ablation
- **Section / status**: Appendix / `HISTORICAL_DESIGN_EVIDENCE`
- **Core question**: rank=4 为什么被保留？
- **One-sentence conclusion**: 历史 rank 1–4 结果提供设计线索，但 rank=4 的当前数学理由仍是五 endpoint centered rank 上限。
- **Main content**: historical trend；current algebraic rank
- **Formulas**: F03
- **Primary assets**: GM-A25-HIST-RANK
- **Secondary assets**: GM-A37-HIST-ALL
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/plots/historical_ablations/rank_1_4_edit_reduction.png; paper_draft/figures/plots/historical_ablations/historical_ablation_edit_reduction.png
- **Allowed claims**: 历史趋势
- **Forbidden claims**: 当前 protocol 主结果
- **Transition**: 按提问跳转。
- **Timing**: 60 seconds
- **Paper eligibility**: `APPENDIX`
- **LOO dependency**: `NO`
- **Risks / TODO**: 双重理由要分开

## A06 · Controller 诊断
- **Section / status**: Appendix / `NEGATIVE_DIAGNOSTIC`
- **Core question**: 自动 mixed controller 为什么停止？
- **One-sentence conclusion**: pair identification 未达核心门槛，最新预算诊断指向 optimizer/normalization limit，因此仅保留 supplementary diagnostic。
- **Main content**: CONTROLLER_V2_CROSSFIT_MICRO_PILOT_FAIL；OPTIMIZER_OR_NORMALIZATION_LIMIT；CONTROLLER_TRAINING_REPAIR_MODERATE_VALUE
- **Formulas**: none
- **Primary assets**: GM-A30-CONTROLLER-BUDGET
- **Secondary assets**: GM-A29-CONTROLLER-VISUALS
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/plots/controller/controller_budget_continuation_trajectory.png; paper_draft/figures/contact_sheets/controller/controller_visuals.png
- **Allowed claims**: controller 未成功
- **Forbidden claims**: controller 是当前主模块
- **Transition**: 按提问跳转。
- **Timing**: 75 seconds
- **Paper eligibility**: `APPENDIX`
- **LOO dependency**: `NO`
- **Risks / TODO**: 避免只报 AUROC 而忽略 pair gate

## A07 · Dual-Support 全 10 Pair
- **Section / status**: Appendix / `SUPPLEMENTARY_EXTENSION`
- **Core question**: 改善是否覆盖所有 pair？
- **One-sentence conclusion**: 全部 10 pair 的 aggregate 指标改善，但困难 pair 和计算开销仍存在。
- **Main content**: per-pair LPIPS；IoU/Boundary F；active Gaussian and render cost
- **Formulas**: F07
- **Primary assets**: GM-A06-DUAL-LPIPS
- **Secondary assets**: GM-A07-DUAL-SHEET
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/plots/dual_support/dual_support_all_pair_garment_lpips.png; paper_draft/figures/contact_sheets/dual_support/dual_support.png
- **Allowed claims**: oracle all-pair result
- **Forbidden claims**: automatic controller
- **Transition**: 按提问跳转。
- **Timing**: 75 seconds
- **Paper eligibility**: `APPENDIX`
- **LOO dependency**: `NO`
- **Risks / TODO**: 指出困难 pair

## A08 · Headroom λ Selection
- **Section / status**: Appendix / `NEGATIVE_DIAGNOSTIC`
- **Core question**: 超参数是否看了 test？
- **One-sentence conclusion**: λ 在规定 split 上选择，test 仅用于 sealed evaluation；120 runs 与 960 checkpoints 完整。
- **Main content**: optimize/calibration/test；λ selection；120 runs；960 checkpoints
- **Formulas**: F08
- **Primary assets**: GM-A21-HEADROOM-DISPLACEMENT
- **Secondary assets**: GM-A19-HEADROOM-GATES
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/headroom_refresh/plots/headroom/coefficient_displacement_trajectory.png; paper_draft/figures/headroom_refresh/plots/headroom/lpips_gate_summary.png
- **Allowed claims**: selection protocol closed
- **Forbidden claims**: test-selected λ
- **Transition**: 按提问跳转。
- **Timing**: 60 seconds
- **Paper eligibility**: `APPENDIX`
- **LOO dependency**: `NO`
- **Risks / TODO**: 口头解释 split

## A09 · Full-Residual Failure
- **Section / status**: Appendix / `NEGATIVE_DIAGNOSTIC`
- **Core question**: 为什么 full residual 反而更差？
- **One-sentence conclusion**: 更大自由度在有限 rendering objective 下重开 patch/cloud/mottle 与 edge scatter，说明容量不等于可泛化 headroom。
- **Main content**: artifact taxonomy；test degradation；identity safety still PASS
- **Formulas**: none
- **Primary assets**: GM-A22-FULL-RESIDUAL-ARTIFACT
- **Secondary assets**: GM-A18-HEADROOM-FOUR-METHOD
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/headroom_refresh/contact_sheets/headroom/full_residual_artifact_examples.png; paper_draft/figures/headroom_refresh/plots/headroom/four_method_test_metrics.png
- **Allowed claims**: full residual degrades
- **Forbidden claims**: 证明所有 residual optimization 必然失败
- **Transition**: 按提问跳转。
- **Timing**: 75 seconds
- **Paper eligibility**: `APPENDIX`
- **LOO dependency**: `NO`
- **Risks / TODO**: 机制解释保持假设语气

## A10 · Subject00 / AvatarReX 基础
- **Section / status**: Appendix / `MULTI_IDENTITY_FOUNDATION`
- **Core question**: 第二身份的证据与限制是什么？
- **One-sentence conclusion**: Subject00 仅有 base-avatar evidence；AvatarReX 仅 metadata preflight，媒体与 derived assets 均不可作为结果。
- **Main content**: Subject00 short/medium；formal and garment pending；AvatarReX license restricted
- **Formulas**: none
- **Primary assets**: GM-A31-SUBJECT00-SHEET
- **Secondary assets**: GM-A32-SUBJECT00-PLOT, GM-A34-AVATARREX-METADATA
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/contact_sheets/subject00/subject00.png; paper_draft/figures/plots/subject00/subject00_step0_to_medium_lpips.png; paper_draft/figures/pending/avatarrex_license_restricted/README.md
- **Allowed claims**: foundation only
- **Forbidden claims**: multi-garment completion；AvatarReX media
- **Transition**: 按提问跳转。
- **Timing**: 60 seconds
- **Paper eligibility**: `APPENDIX`
- **LOO dependency**: `NO`
- **Risks / TODO**: license audit

## A11 · Claim Boundary Matrix
- **Section / status**: Appendix / `PROJECT_GOVERNANCE_APPENDIX`
- **Core question**: 哪些措辞可以进入论文？
- **One-sentence conclusion**: 所有核心结论必须绑定 fixed subject02、closed five-garment seen-reference protocol；八类开放问题不得提前宣称。
- **Main content**: supported；descriptive only；pending；forbidden
- **Formulas**: none
- **Primary assets**: GM-A35-FIGURE-MAP
- **Secondary assets**: GM-A10-HARD-LOOKUP, GM-A33-LOO-PENDING
- **Backup assets**: none
- **Formal paths**: paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json; paper_draft/figures/pure_endpoint_refresh/candidates/fig05_hard_lookup/pure_endpoint_hard_lookup_v1/pure_endpoint_hard_lookup_v1.png; paper_draft/figures/pure_endpoint_refresh/pending/loo/repaired_protocol_pending.json
- **Allowed claims**: claim audit
- **Forbidden claims**: PAPER_FINAL
- **Transition**: 按提问跳转。
- **Timing**: 75 seconds
- **Paper eligibility**: `APPENDIX`
- **LOO dependency**: `REQUIRED`
- **Risks / TODO**: 随 LOO 更新

## A12 · 执行治理与证据完整性
- **Section / status**: Appendix / `PROJECT_GOVERNANCE_APPENDIX`
- **Core question**: 这些结论是否可追溯？
- **One-sentence conclusion**: 所有组会素材绑定 source path/SHA/branch/HEAD；本任务 GPU、训练、推理、renderer、PPTX、Figure Bank mutation 均为 0。
- **Main content**: provenance；mutation audit；LOO files_read=0；PAPER_FINAL=0
- **Formulas**: none
- **Primary assets**: GM-A35-FIGURE-MAP
- **Secondary assets**: GM-A33-LOO-PENDING
- **Backup assets**: none
- **Formal paths**: paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json; paper_draft/figures/pure_endpoint_refresh/pending/loo/repaired_protocol_pending.json
- **Allowed claims**: audit trail complete
- **Forbidden claims**: scientific work executed in planning task
- **Transition**: 按提问跳转。
- **Timing**: 60 seconds
- **Paper eligibility**: `APPENDIX`
- **LOO dependency**: `NO`
- **Risks / TODO**: Cloud sync DNS pending is infrastructure only
