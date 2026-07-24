# CanonDressGS Group Meeting Compact Outline — 2026-07-24

**Plan:** 18 independently reorganized slides, 15–18 minutes. Every slide maps to the full version.

## C01 · 研究目标与当前边界
- **Section / status**: Opening / `PAPER_CORE_METHOD_CANDIDATE`
- **Core question**: 当前到底支持什么？
- **One-sentence conclusion**: 当前支持 subject02 closed-bank seen-endpoint control，LOO 与 multi-identity 仍未完成。
- **Main content**: reference garment + target pose/camera；closed five-garment bank；PAPER_FINAL=0
- **Formulas**: none
- **Primary assets**: GM-A01-ENDPOINT-TEASER
- **Secondary assets**: GM-A35-FIGURE-MAP
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/candidates/fig01_teaser/pure_endpoint_endpoint_only_v1/contact_sheet_for_manual_adjudication.png; paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json
- **Allowed claims**: closed-bank endpoint control
- **Forbidden claims**: unseen garment completion
- **Transition**: 先解释这个任务为什么难。
- **Timing**: 50 seconds
- **Paper eligibility**: `CANDIDATE`
- **LOO dependency**: `WAIT_FOR_LOO_FOR_TITLE`
- **Risks / TODO**: 人工选 teaser
- **Full mapping**: S01, S02

## C02 · 服装控制的核心困难
- **Section / status**: Problem / `PAPER_CORE_EVIDENCE`
- **Core question**: 为什么不能直接连续插值？
- **One-sentence conclusion**: 服装耦合 geometry、visibility 和 appearance，其中无效 geometry 是中间 artifact 主因。
- **Main content**: lookup stable/discrete；interpolation flexible/artifact-prone
- **Formulas**: F06
- **Primary assets**: GM-A02-EARLY-FAILURES
- **Secondary assets**: GM-A05-GEOMETRY-CAUSAL-DATA
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/contact_sheets/limitations/early_failures.png; paper_protocol/reviewer_risk/continuous_control_causal_attribution_final_summary.json
- **Allowed claims**: 三因素与几何主因
- **Forbidden claims**: 泛化到所有场景
- **Transition**: 这推动方法路线发生了连续收缩。
- **Timing**: 55 seconds
- **Paper eligibility**: `CORE_PROBLEM`
- **LOO dependency**: `NO`
- **Risks / TODO**: 历史图需标签
- **Full mapping**: S03, S04

## C03 · 从 reference-to-residual 到 endpoint control
- **Section / status**: Research Evolution / `HISTORICAL_DESIGN_EVIDENCE`
- **Core question**: 失败如何改变方法？
- **One-sentence conclusion**: 历史 capacity 与 causal 诊断排除了错误假设，路线收缩为 Teacher endpoints、explicit basis 与最小控制器。
- **Main content**: full model failures；support capacity PASS；Teacher → basis → Pure Endpoint → Headroom → LOO
- **Formulas**: none
- **Primary assets**: GM-A02-EARLY-FAILURES
- **Secondary assets**: GM-A27-HIST-REPRESENTATION
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/contact_sheets/limitations/early_failures.png; paper_draft/figures/contact_sheets/ablations/representation_ablations.png
- **Allowed claims**: 历史负结果推动收缩
- **Forbidden claims**: 最终方案从一开始已确定
- **Transition**: 先看收缩后的当前 pipeline。
- **Timing**: 60 seconds
- **Paper eligibility**: `PRESENTATION_ONLY`
- **LOO dependency**: `NO`
- **Risks / TODO**: 协议分层
- **Full mapping**: S05

## C04 · 当前 Pipeline
- **Section / status**: Core Method / `PAPER_CORE_METHOD_CANDIDATE`
- **Core question**: reference 如何决定 Gaussian endpoint？
- **One-sentence conclusion**: Frozen F2 预测 raw coefficient，endpoint snapping 选择 valid Teacher state，再由 frozen renderer 输出。
- **Main content**: reference → F2 → aggregation → Linear(4)；raw coefficient → endpoint snapping → renderer
- **Formulas**: F04, F05
- **Primary assets**: GM-A03-METHOD-WIREFRAME
- **Secondary assets**: GM-A12-RAW-SNAPPED-SHEET
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/candidates/fig02_method/layout_wireframe.svg; paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_raw_vs_snapped.png
- **Allowed claims**: raw/snapped distinction
- **Forbidden claims**: continuous coefficient is exact
- **Transition**: 这个 pipeline 依赖稳定的 Teacher Endpoint。
- **Timing**: 65 seconds
- **Paper eligibility**: `METHOD_CANDIDATE`
- **LOO dependency**: `WAIT_FOR_LOO_FOR_FIGURE2`
- **Risks / TODO**: Figure 2 pending LOO
- **Full mapping**: S06, S11

## C05 · Teacher Endpoint 与冻结底座
- **Section / status**: Method Foundation / `PAPER_CORE_EVIDENCE`
- **Core question**: 稳定服装状态如何建立？
- **One-sentence conclusion**: 每件 seen garment 通过多视图 rendering objective 优化 canonical residual；Teacher 是 endpoint，不是 Upper Bound。
- **Main content**: frozen avatar；per-garment residual optimization；five valid endpoints
- **Formulas**: F01, F02
- **Primary assets**: GM-A04-TEACHER-ENDPOINTS
- **Secondary assets**: GM-A15-PARITY-SAFETY
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_clean_all_garments.png; paper_draft/figures/pure_endpoint_refresh/candidates/supplementary/pure_endpoint_parity_safety_v1/pure_endpoint_parity_safety_v1.png
- **Allowed claims**: Teacher endpoint
- **Forbidden claims**: Teacher upper bound
- **Transition**: 下一页给出不能插值它们的因果证据。
- **Timing**: 60 seconds
- **Paper eligibility**: `CORE_EVIDENCE`
- **LOO dependency**: `NO`
- **Risks / TODO**: 术语审计
- **Full mapping**: S06, S07

## C06 · Geometry 是中间 artifact 主因
- **Section / status**: Causal Diagnosis / `PAPER_CORE_EVIDENCE`
- **Core question**: 哪一类通道导致失败？
- **One-sentence conclusion**: geometry main effect 约 0.9959，sufficiency/necessity 均为 10/10。
- **Main content**: factor injection；0.9959；10/10 + 10/10
- **Formulas**: F06
- **Primary assets**: GM-A05-GEOMETRY-CAUSAL-DATA
- **Secondary assets**: GM-A07-DUAL-SHEET
- **Backup assets**: none
- **Formal paths**: paper_protocol/reviewer_risk/continuous_control_causal_attribution_final_summary.json; paper_draft/figures/contact_sheets/dual_support/dual_support.png
- **Allowed claims**: current-protocol geometry main effect
- **Forbidden claims**: visibility/appearance universally irrelevant
- **Transition**: 因此 basis 表示端点坐标，而不是直接插值几何。
- **Timing**: 55 seconds
- **Paper eligibility**: `CORE_EVIDENCE`
- **LOO dependency**: `NO`
- **Risks / TODO**: 使用数字 callout
- **Full mapping**: S08, S09

## C07 · Explicit Basis：endpoint 坐标系
- **Section / status**: Core Method / `PAPER_CORE_METHOD_CANDIDATE`
- **Core question**: rank-4 到底代表什么？
- **One-sentence conclusion**: 五个 centered Teacher residual 的最大 rank 本就是 4，因此它是精确 endpoint 坐标系，不是强压缩。
- **Main content**: ΔG≈μ+Bc；rank=4；adaptation value pending LOO
- **Formulas**: F03
- **Primary assets**: GM-A17-HEADROOM-PARITY
- **Secondary assets**: GM-A04-TEACHER-ENDPOINTS
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/headroom_refresh/plots/headroom/teacher_svd_parity.png; paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_clean_all_garments.png
- **Allowed claims**: endpoint coordinate system
- **Forbidden claims**: universal semantic manifold；strong compression
- **Transition**: 接着看这个坐标系上的严格 cross-fit。
- **Timing**: 55 seconds
- **Paper eligibility**: `METHOD_CANDIDATE`
- **LOO dependency**: `REQUIRED_FOR_BASIS_VALUE`
- **Risks / TODO**: LOO decides claim
- **Full mapping**: S10

## C08 · Pure Endpoint Protocol
- **Section / status**: Core Results / `PROJECT_GOVERNANCE_APPENDIX`
- **Core question**: 成功是否来自泄漏或单一划分？
- **One-sentence conclusion**: 24 runs 覆盖 4 rotations×3 seeds，并审计 7 methods、perturbation、parity 与 identity safety。
- **Main content**: 24 runs；4×3；condition folds；7 methods + safety
- **Formulas**: none
- **Primary assets**: GM-A38-PURE-ROTATIONS
- **Secondary assets**: GM-A15-PARITY-SAFETY
- **Backup assets**: none
- **Formal paths**: paper_protocol/reviewer_risk/pure_endpoint_rotation_manifests.json; paper_draft/figures/pure_endpoint_refresh/candidates/supplementary/pure_endpoint_parity_safety_v1/pure_endpoint_parity_safety_v1.png
- **Allowed claims**: closed-bank protocol
- **Forbidden claims**: garment-held-out
- **Transition**: 协议内的 clean 结果达到满分。
- **Timing**: 55 seconds
- **Paper eligibility**: `EXPERIMENT_PROTOCOL`
- **LOO dependency**: `NO`
- **Risks / TODO**: condition vs garment held-out
- **Full mapping**: S12

## C09 · Pure Endpoint 结果
- **Section / status**: Core Results / `PAPER_CORE_EVIDENCE`
- **Core question**: seen endpoint 能否可靠选择？
- **One-sentence conclusion**: clean top-1 与每 garment precision/recall 均为 1，parity PASS 且 identity contamination=0。
- **Main content**: top-1=1；precision/recall=1；parity PASS；identity contamination=0
- **Formulas**: none
- **Primary assets**: GM-A08-PURE-TOP1
- **Secondary assets**: GM-A09-ENDPOINT-MATCH, GM-A15-PARITY-SAFETY
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/clean_method_top1.png; paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/endpoint_exact_match.png; paper_draft/figures/pure_endpoint_refresh/candidates/supplementary/pure_endpoint_parity_safety_v1/pure_endpoint_parity_safety_v1.png
- **Allowed claims**: seen endpoint selection supported
- **Forbidden claims**: unseen garment supported
- **Transition**: 但满分结果必须和 hard lookup 一起解释。
- **Timing**: 55 seconds
- **Paper eligibility**: `CORE_EVIDENCE`
- **LOO dependency**: `NO`
- **Risks / TODO**: benchmark saturation
- **Full mapping**: S13

## C10 · 与 Hard Lookup 功能等价
- **Section / status**: Core Results / `PAPER_CORE_EVIDENCE`
- **Core question**: 为什么这不是 superiority？
- **One-sentence conclusion**: clean 条件下二者选择相同 endpoint 并产生等价 render，所以当前 relation 只能描述性报告。
- **Main content**: agreement；render equivalence；no superiority claim
- **Formulas**: none
- **Primary assets**: GM-A10-HARD-LOOKUP
- **Secondary assets**: GM-A36-RENDER-EQUIVALENCE
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/candidates/fig05_hard_lookup/pure_endpoint_hard_lookup_v1/pure_endpoint_hard_lookup_v1.png; paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/render_equivalence_and_cache_summary.png
- **Allowed claims**: functional equivalence
- **Forbidden claims**: better than hard lookup
- **Transition**: 真正的差异需要 out-of-bank LOO 来检验。
- **Timing**: 55 seconds
- **Paper eligibility**: `DESCRIPTIVE_ONLY`
- **LOO dependency**: `REQUIRED_FOR_VALUE_BEYOND_LOOKUP`
- **Risks / TODO**: 人工裁决 Figure 5
- **Full mapping**: S14

## C11 · Snapping 有效，但 blur 仍脆弱
- **Section / status**: Core Results / `LIMITATION`
- **Core question**: 成功来自哪里，失败发生在哪？
- **One-sentence conclusion**: 成功主要来自 endpoint snapping 而非精确 continuous regression；mild blur 会触发 endpoint flip。
- **Main content**: raw coefficient error high；snapping exact endpoint；single reference stable；blur flips; dropout abstains
- **Formulas**: F05
- **Primary assets**: GM-A11-RAW-SNAPPED-PLOT
- **Secondary assets**: GM-A13-PERTURBATION, GM-A14-BLUR-FAILURES
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/raw_vs_realized_coefficient_error.png; paper_draft/figures/pure_endpoint_refresh/candidates/fig06_limitations/pure_endpoint_perturbation_v1/pure_endpoint_perturbation_v1.png; paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_mild_blur_failures.png
- **Allowed claims**: snapping and perturbation boundary
- **Forbidden claims**: robust continuous manifold learned
- **Transition**: 这正是做 Headroom refinement 的原因。
- **Timing**: 60 seconds
- **Paper eligibility**: `CORE_EVIDENCE_AND_LIMITATION`
- **LOO dependency**: `NO`
- **Risks / TODO**: 不要弱化 blur
- **Full mapping**: S15, S16

## C12 · Headroom：连续优化是否还有空间？
- **Section / status**: Headroom / `NEGATIVE_DIAGNOSTIC`
- **Core question**: rendering loss 能否超过 Teacher？
- **One-sentence conclusion**: 实验固定 rank-4 basis，只优化 4 个系数，并用 full residual 作 headroom comparator。
- **Main content**: Teacher/SVD；refined coefficient；full residual；pre-registered gates
- **Formulas**: F08
- **Primary assets**: GM-A24-REFINED-DECOMPOSITION
- **Secondary assets**: GM-A17-HEADROOM-PARITY
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/headroom_refresh/plots/headroom/refined_lookup_decomposition.png; paper_draft/figures/headroom_refresh/plots/headroom/teacher_svd_parity.png
- **Allowed claims**: headroom hypothesis
- **Forbidden claims**: refinement is main pipeline
- **Transition**: 120 次优化给出了明确负结果。
- **Timing**: 50 seconds
- **Paper eligibility**: `SUPPLEMENTARY_NEGATIVE`
- **LOO dependency**: `NO`
- **Risks / TODO**: 只讲 sealed attempt_002
- **Full mapping**: S17, S18

## C13 · Headroom 负结果收缩了 Pipeline
- **Section / status**: Headroom / `NEGATIVE_DIAGNOSTIC`
- **Core question**: refinement 有改善吗？
- **One-sentence conclusion**: 0/5 garments 过门槛，refined coefficient 不改善 Teacher；full residual 产生 patch/cloud/mottle，因此 refinement 不进主线。
- **Main content**: 120/120 runs；0/5 pass；macro gain < 0；full residual artifacts；TEACHER_SPAN_AT_LOCAL_OPTIMUM
- **Formulas**: none
- **Primary assets**: GM-A16-HEADROOM-OVERVIEW
- **Secondary assets**: GM-A20-HEADROOM-PER-GARMENT, GM-A22-FULL-RESIDUAL-ARTIFACT
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/headroom_refresh/candidates/supplementary/headroom_negative_diagnostic_v1/supplementary_negative_diagnostic_v1.png; paper_draft/figures/headroom_refresh/plots/headroom/per_garment_teacher_minus_refined_lpips.png; paper_draft/figures/headroom_refresh/contact_sheets/headroom/full_residual_artifact_examples.png
- **Allowed claims**: negative result and design decision
- **Forbidden claims**: positive headroom；spatial field tested
- **Transition**: 只剩 LOO 能检验 basis 的独立价值。
- **Timing**: 70 seconds
- **Paper eligibility**: `SUPPLEMENTARY_NEGATIVE`
- **LOO dependency**: `NO`
- **Risks / TODO**: caption boundary
- **Full mapping**: S19, S20

## C14 · LOO 设计与当前状态
- **Section / status**: LOO Decision / `PENDING_LOO`
- **Core question**: basis 能否适配完全排除的 garment？
- **One-sentence conclusion**: 四 garment 构建 rank≤3 basis、第五件完全 held out；protocol READY，但 LOO_ATTEMPT=ABSENT。
- **Main content**: μ_-g, B_-g；K=1/K=2；hard lookup / oracle projection / full residual；READY / PENDING EXECUTION; files_read=0
- **Formulas**: F09, F10
- **Primary assets**: GM-A33-LOO-PENDING
- **Secondary assets**: GM-A28-O07-LIMITATION
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/pending/loo/repaired_protocol_pending.json; paper_draft/figures/contact_sheets/limitations/o07.png
- **Allowed claims**: design and pending status
- **Forbidden claims**: fabricated LOO result
- **Transition**: 等待 LOO 时，历史消融只能解释设计。
- **Timing**: 65 seconds
- **Paper eligibility**: `WAIT_FOR_LOO`
- **LOO dependency**: `REQUIRED`
- **Risks / TODO**: 汇报前 metadata recheck
- **Full mapping**: S21, S22

## C15 · 历史消融：设计证据，不是当前主表
- **Section / status**: Historical Evidence / `HISTORICAL_DESIGN_EVIDENCE`
- **Core question**: 旧协议还能回答什么？
- **One-sentence conclusion**: 51 个 view-transductive 实验解释 rank、mask、standardization 与 reference count 选择，但不能与 current protocol 混表。
- **Main content**: 51 runs；14,400 steps；rank/mask/mean/std/reference count
- **Formulas**: none
- **Primary assets**: GM-A37-HIST-ALL
- **Secondary assets**: GM-A25-HIST-RANK, GM-A26-HIST-REFERENCE
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/plots/historical_ablations/historical_ablation_edit_reduction.png; paper_draft/figures/plots/historical_ablations/rank_1_4_edit_reduction.png; paper_draft/figures/plots/historical_ablations/reference_count_1_3_edit_reduction.png
- **Allowed claims**: design rationale
- **Forbidden claims**: current generalization result
- **Transition**: 这些历史结果与新负诊断共同定义限制。
- **Timing**: 55 seconds
- **Paper eligibility**: `HISTORICAL_ONLY`
- **LOO dependency**: `NO`
- **Risks / TODO**: Historical 标签
- **Full mapping**: S23

## C16 · 当前限制与第二身份基础
- **Section / status**: Limitations / `LIMITATION`
- **Core question**: 投稿前还有哪些实质缺口？
- **One-sentence conclusion**: O07、blur、full-residual artifacts 与 controller failure 均未解决；Subject00 只完成 base-avatar foundation。
- **Main content**: failure modes；controller supplementary diagnostic；Subject00 base only；AvatarReX media=0
- **Formulas**: none
- **Primary assets**: GM-A28-O07-LIMITATION
- **Secondary assets**: GM-A22-FULL-RESIDUAL-ARTIFACT, GM-A31-SUBJECT00-SHEET
- **Backup assets**: GM-A29-CONTROLLER-VISUALS
- **Formal paths**: paper_draft/figures/contact_sheets/limitations/o07.png; paper_draft/figures/headroom_refresh/contact_sheets/headroom/full_residual_artifact_examples.png; paper_draft/figures/contact_sheets/subject00/subject00.png; paper_draft/figures/contact_sheets/controller/controller_visuals.png
- **Allowed claims**: current limitations
- **Forbidden claims**: multi-identity garment editing complete
- **Transition**: 因此当前方法冻结必须保持保守。
- **Timing**: 60 seconds
- **Paper eligibility**: `LIMITATION`
- **LOO dependency**: `NO`
- **Risks / TODO**: 避免把 Subject00 图写成 garment
- **Full mapping**: S24, S25

## C17 · 当前方法冻结与 Reviewer 风险
- **Section / status**: Method Freeze / `LIMITATION`
- **Core question**: 论文主线现在能冻结到哪里？
- **One-sentence conclusion**: 核心候选已收敛到 Teacher+basis+Pure Endpoint，但 hard lookup 等价和 LOO pending 阻止最终贡献冻结。
- **Main content**: core candidate；supplementary oracle；negative diagnostics；Figure 2 pending LOO；largest reviewer risk
- **Formulas**: none
- **Primary assets**: GM-A35-FIGURE-MAP
- **Secondary assets**: GM-A10-HARD-LOOKUP, GM-A33-LOO-PENDING
- **Backup assets**: none
- **Formal paths**: paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json; paper_draft/figures/pure_endpoint_refresh/candidates/fig05_hard_lookup/pure_endpoint_hard_lookup_v1/pure_endpoint_hard_lookup_v1.png; paper_draft/figures/pure_endpoint_refresh/pending/loo/repaired_protocol_pending.json
- **Allowed claims**: freeze matrix and reviewer risk
- **Forbidden claims**: PAPER_FINAL
- **Transition**: 下一步只按决定性风险排序。
- **Timing**: 65 seconds
- **Paper eligibility**: `GOVERNANCE`
- **LOO dependency**: `REQUIRED`
- **Risks / TODO**: PAPER_FINAL=0
- **Full mapping**: S26, S27

## C18 · 下一步
- **Section / status**: Closing / `PENDING_LOO`
- **Core question**: 最先做什么？
- **One-sentence conclusion**: 先 sealed LOO，再冻结 Figure 2 与论文 claim；随后人工选组会素材，不自动生成 PPTX。
- **Main content**: LOO；method/figure freeze；paper rewrite；Subject00；manual PPT asset selection
- **Formulas**: none
- **Primary assets**: GM-A33-LOO-PENDING
- **Secondary assets**: GM-A35-FIGURE-MAP
- **Backup assets**: none
- **Formal paths**: paper_draft/figures/pure_endpoint_refresh/pending/loo/repaired_protocol_pending.json; paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json
- **Allowed claims**: evidence-gated next steps
- **Forbidden claims**: automatic PPTX generation
- **Transition**: 结束并进入讨论。
- **Timing**: 50 seconds
- **Paper eligibility**: `CLOSING`
- **LOO dependency**: `REQUIRED`
- **Risks / TODO**: NEXT_TASK fixed
- **Full mapping**: S28
