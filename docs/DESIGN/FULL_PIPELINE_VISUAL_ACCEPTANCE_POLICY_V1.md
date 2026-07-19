# Full Pipeline Visual Acceptance Policy V1

任何生成 PNG 的正式 Pipeline module 都必须同时完成工程验收、视觉语义验收和跨图一致性验收。Module 总状态不得高于视觉验收状态。

## 强制产物

- `visual_acceptance_contact_sheet.png`（允许带 module 前缀）
- `visual_acceptance.json`
- `VISUAL_ACCEPTANCE.md`

Contact sheet 必须保留原始 PNG，并使用统一画布、统一色标和固定增益 diff。不得对每张图独立归一化。弱变化可同时展示原始 absolute diff 与固定增益版本，但增益图不得替代原始结果。

## 真实查看要求

视觉 PASS 必须由能够展示像素内容的工具实际打开关键 PNG。仅检查文件名、shape、统计量、哈希、PIL verify 或数值 diff 不构成视觉检查。每个关键文件必须记录具体的期望语义、观察行为、变化区域、异常区域和严重度。

无法实际显示图像时状态为 `PARTIAL`；发现严重属性语义错误、身份/相机错配、大面积轮廓破碎、漂浮 Gaussian、透明度崩溃、背景污染或颜色爆炸时状态为 `FAIL`。

## 跨图一致性

- reference 与 target 必须属于同一身份和同一条件定义；pose/camera 差异必须符合协议。
- 所有比较必须使用相同画布和显示范围。
- SHN 必须比较 degree-1 zero-SHN control 与 degree-1 perturbed-SHN，不能只与 degree-0 base 比较。
- 几何 gate 应对应轮廓/alpha 变化；appearance gate 应对应颜色变化。
- 不得使用 target RGB/mask、teacher gate 或 cloth ID 作为正式 inference 条件。

## Module 3 及后续

Module 3 验收必须生成并检查 references、clothing masks、observed/unobserved evidence、geometry/appearance gates、observed-only/diffusion/learned renders及统一 diff/overlay。必须检查跨肢体 shortcut、整人 gate、零散噪点、脸/头发/手/腿污染、轮廓破碎、透明、漂浮点和颜色爆炸。

此规则永久适用于后续所有生成 PNG 的模块。
