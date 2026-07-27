# AAAI-27 Paper Table and Figure Plan

所有表格采用 episode → outfit macro → seed mean/std；所有训练方法使用 seeds 0/1/2。历史数字不得填入正式主表。

## Table 1 — Seen-outfit main comparison

行固定为 B0、B1、B2、B3、B4、B5、Ours。B1 标注 `optimization upper bound`；B2 标注 `seen-only memorization upper bound`。

列固定为：Garment RGB MAE ↓、Edit reduction ↑、Target-closer ↑、Protected RGB MAE ↓、Background RGB MAE ↓、Nearest outfit accuracy ↑、Swap wins ↑、Trainable parameters ↓。

## Table 2 — Method ablation

行固定为 A1 K=1/2/3/4、A2 no mask、A3 mean only、A4 no standardization、A5 legacy supervision、A6 no pairwise geometry、A7 Kref=1/2/3。

列固定为：Coefficient RMSE ↓、Nearest accuracy ↑、Swap wins ↑、Garment MAE ↓、Protected MAE ↓。A8 不进入此统一 evaluator 表。

## Table 3 — Efficiency

列固定为：Trainable parameters、Frozen basis size、Peak VRAM、Training time、Inference time。训练/推理/渲染时间分别记录，硬件与环境随 source data 一并保存。

## Table 4 — Held-out diagnostic

固定标题：**Held-out diagnostic — FAIL**。

只包含 O07 teacher、O07 seen-basis projection、O07 reference prediction。报告 projection RMSE/cosine/support/garment MAE、prediction-to-projection coefficient/render error 和 nearest seen endpoint。不得与 Table 1 的 seen PASS 行混合。

## Figure 1 — Method overview

画出 reference RGB/masks → frozen F2 → clothing mean/masked max → set mean/max → LayerNorm + Linear(4) → coefficient de-normalization → frozen basis → frozen MMLP-Human → renderer。用清晰分界标出 reference-only prediction boundary 与 target supervision/evaluation boundary；禁止将 target 箭头接入 prediction forward。

## Figure 2 — Five-outfit qualitative comparison

- 行：O01/O02/O03/O04/O08。
- 每个 outfit 必须包含 front/back/left/right 四个固定视角。
- 列：Base、Target、Teacher、Ours、Strongest baseline。
- 不得挑选最好视角或隐藏局部伪影。

## Figure 3 — Reference swap control

固定同一 target pose/camera，展示 O01/O02/O03/O04/O08 五种 reference sets 产生的五个输出；必须同时给出 reference identity 标签和输出 endpoint，不得改变 crop/scale。

## Figure 4 — Basis rank ladder

固定 K=1/2/3/4 和相同 teacher/view，展示 K1–K3 的混合/斑驳及 K4 重建。不得只展示 K4。

## Figure 5 — Supervision collapse analysis

固定展示 old endpoint collapse、equal-logit old gradient、signed raw-logit supervision、CS-PASS result；所有曲线和数值注明历史任务及 evaluator 差异。

## Figure 6 — Held-out limitation

固定展示 O07 target/teacher、seen-basis projection、reference prediction、nearest O03 endpoint。标题或 caption 必须含 `Held-out diagnostic — FAIL`；不得隐藏 O07 失败。

## 文件与图像处理合同

- 正式图：`artifacts/paper_ready/figures/`。
- 原始数字：`artifacts/paper_ready/source_data/`。
- 调试图：`artifacts/debug/`。
- 允许：concatenate、aspect-preserving crop、label、uniform crop。
- 禁止：retouch、artifact removal、color change、successful-region selection。
- 每张正式图保存 source file list、SHA256、crop box、label map、export command 和 evaluator commit。
