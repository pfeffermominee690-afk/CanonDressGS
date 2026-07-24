# CanonDressGS Group Meeting Q&A Preparation — 2026-07-24

All answers are bounded by sealed evidence; LOO metrics must remain absent until a sealed result exists.

## Q01 · 为什么不直接用 hard lookup？
clean closed-bank 下 hard lookup 与 CanonDressGS 功能等价。继续保留 basis 的唯一科学理由是检验 out-of-bank adaptation；这由 LOO 决定。

Evidence: `GM-A10-HARD-LOOKUP, GM-A33-LOO-PENDING`

## Q02 · 当前方法和分类器有什么区别？
当前 realized behavior 确实接近 endpoint classifier 加 lookup；显式 coefficient space 的额外价值尚未由 clean benchmark 证明。

Evidence: `GM-A11-RAW-SNAPPED-PLOT`

## Q03 · rank-4 是不是伪低维？
五个 residual 居中后最大 rank 就是 4，因此不是强压缩；它是精确 endpoint coordinate system。

Evidence: `GM-A17-HEADROOM-PARITY`

## Q04 · 为什么 Teacher 不是 Upper Bound？
Teacher 是固定多视图 objective 下的局部优化 endpoint，不保证对所有 objective、视图或表示全局最优。

Evidence: `GM-A04-TEACHER-ENDPOINTS`

## Q05 · Headroom 为什么失败？
sealed 结果表明 rank-4 span 内优化没有 LPIPS gain，可能因为 Teacher/SVD 已在该 objective 的局部最优；不把机制推断写成已证明因果。

Evidence: `GM-A20-HEADROOM-PER-GARMENT`

## Q06 · full residual 为什么反而更差？
更大自由度重新打开了 patch/cloud/mottle 与 edge-scatter 解；有限 rendering objective 未提供足够正则来保证 test 泛化。

Evidence: `GM-A22-FULL-RESIDUAL-ARTIFACT`

## Q07 · LOO 为什么能回答核心问题？
它从 basis construction 中完全排除目标 garment，再用 few views 拟合系数，直接测试 basis 是否含有可迁移结构。

Evidence: `GM-A33-LOO-PENDING`

## Q08 · 如果 LOO 失败，论文还剩什么贡献？
仍有 geometry causal attribution、valid endpoint construction、closed-bank reference control、hard-lookup equivalence audit 和负诊断；但 basis 主贡献必须显著收缩。

Evidence: `GM-A05-GEOMETRY-CAUSAL-DATA, GM-A10-HARD-LOOKUP`

## Q09 · Dual-Support 是最终方法吗？
不是。它是 oracle geometry-safe mechanism，证明保留有效支持有价值；自动 mixed controller 未成功。

Evidence: `GM-A06-DUAL-LPIPS, GM-A29-CONTROLLER-VISUALS`

## Q10 · Controller 为什么停止？
核心 pair identification 与 routing gates 未达标；最新诊断是 optimizer/normalization limit，修复价值仅 moderate。

Evidence: `GM-A30-CONTROLLER-BUDGET`

## Q11 · 当前是否支持 unseen garment？
不支持。Pure Endpoint 只覆盖五个 seen endpoints，正式 unseen adaptation 等待 LOO。

Evidence: `GM-A08-PURE-TOP1, GM-A33-LOO-PENDING`

## Q12 · 是否支持第二身份？
仅支持说 Subject00 base-avatar pilot PASS；第二身份 garment endpoints 和 editing 尚未完成。

Evidence: `GM-A31-SUBJECT00-SHEET`

## Q13 · Teacher 是否使用 test views？
Teacher 是 offline endpoint construction；当前报告必须按各 sealed protocol 的 view boundary 解释，不能把 Teacher evidence 当 strict unseen-view inference。

Evidence: `GM-A38-PURE-ROTATIONS`

## Q14 · 为什么不直接做 spatial local basis？
Headroom 没有显示 global span 外存在可泛化 full-residual gain，full residual 还严重退化，因此 spatial field 只保留 contingency。

Evidence: `GM-A23-SPAN-NULL, GM-A22-FULL-RESIDUAL-ARTIFACT`

## Q15 · 论文最终主线可能有哪些版本？
LOO 成功则突出 transferable basis adaptation；LOO 失败则收缩为 endpoint control、geometry diagnosis 与 limitation-focused evidence。

Evidence: `GM-A33-LOO-PENDING, GM-A35-FIGURE-MAP`

## Q16 · 这么多负实验值得吗？
值得，因为它们分别排除了 support capacity、geometry interpolation、neural decoder、automatic controller 和 test-time refinement 等错误路线。

Evidence: `GM-A02-EARLY-FAILURES, GM-A16-HEADROOM-OVERVIEW`

## Q17 · 当前最主要投稿风险是什么？
closed-bank benchmark 饱和且与 hard lookup 等价；没有 LOO 正结果时，basis 的独立贡献不足。

Evidence: `GM-A10-HARD-LOOKUP, GM-A33-LOO-PENDING`

## Q18 · Headroom 负结果会不会否定 explicit basis？
它否定的是 seen-endpoint test-time refinement value，不否定 basis 作为精确 endpoint coordinate system；adaptation value 仍由 LOO 决定。

Evidence: `GM-A16-HEADROOM-OVERVIEW, GM-A33-LOO-PENDING`
