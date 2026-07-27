# Subject00 CommonSafe4 并发 12-run provenance 只读审计

任务 `AAAI27-SUBJECT00-COMMONSAFE4-CONCURRENT-12RUN-PROVENANCE-AUDIT-001` 对既有 CommonSafe4 方法矩阵和同级公平基线进行了严格只读、逐文件、逐检查点、逐正式测试的 post-hoc provenance 审计。

## 裁决

- 原预检状态保持为 `PREFLIGHT_STOPPED_BY_CONCURRENT_FORMAL_OUTPUT_ROOT_CREATION`，没有追溯改写为 PASS。
- 实际方法写入者为 `AAAI27-SUBJECT00-BASE60747-COMMONSAFE4-METHOD-MATRIX-001`，执行 HEAD `1d97afaafff6180062b41f16ffec6beac9dfdbbf`。
- slot04 替换由冻结 camera/target registry 独立复算为 `slot06 / cam09 / back-right`；选择规则和结果均在首个 optimizer step 之前进入 Git/config/checkpoint binding。
- 方法矩阵为 12/12 formal-valid，累计 3600 个历史 optimizer steps、72 个可解析且绑定一致的检查点。
- 独立复算 endpoint top-1 为 `0.8333333333333334`（30/36）。
- 四类公平基线均存在且 48/48 cells formal-valid；本审计没有运行任何新基线。
- Base60747、三套 Teacher、formal target、旧方法输出和新矩阵输出在审计前后均未变化。

## 方法聚合

- per-run mean/std(population): `0.8333333333333334` / `0.1666666666666667`
- min/max: `0.6666666666666666` / `1.0000000000000000`
- wall time（12 runs 合计）: `16.108335733` 秒
- peak VRAM: `18134528` bytes
- confusion: `{"O01": {"O01": 12, "O03": 0, "O04": 0}, "O03": {"O01": 0, "O03": 12, "O04": 0}, "O04": {"O01": 0, "O03": 6, "O04": 6}}`

## 公平基线

- Reference Classifier Lookup: top-1=0.8611111111111112, valid=12/12, historical optimizer steps=3600
- Nearest-Centroid Lookup: top-1=0.9722222222222222, valid=12/12, historical optimizer steps=0
- Outfit-ID Oracle: top-1=1.0000000000000000, valid=12/12, historical optimizer steps=0
- Teacher Endpoint: top-1=1.0000000000000000, valid=12/12, historical optimizer steps=0

## 限制

这是技术与 provenance 接纳，不是人工视觉或科学结论。`PAPER_FINAL=false`，论文正文未修改。下一唯一任务为 `PREPARE_SUBJECT00_COMMONSAFE4_METHOD_TEACHER_BASELINE_HUMAN_SCIENTIFIC_REVIEW_PACK`。

最终分类：`SUBJECT00_BASE60747_COMMONSAFE4_MATRIX_AND_BASELINES_VALID_POSTHOC_PROVENANCE_SEALED_PENDING_HUMAN_SCIENTIFIC_REVIEW`
