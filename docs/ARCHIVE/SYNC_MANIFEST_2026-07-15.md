# CanonDressGS 安全同步清单（2026-07-15）

同步方向仅允许云端→本地。本清单基于 SHA256；不包含 `.git`、`__pycache__`、`*.pyc`、`*.bak*`、`*.before_*`、`*.tar.gz`、日志、outputs、data、models、papers、smpl_model 或 checkpoint。

## 分类汇总

- A 两端一致：55 个，不需复制。
- B 仅云端存在：9 个，可进入 staging。
- C 仅本地存在：9 个，保留本地，不反向同步。
- D 两端存在且不同：3 个，冲突；只保存云端副本供人工 diff，禁止覆盖。
- E 备份/缓存/压缩包/输出：排除。

## B：仅云端存在

| 路径 | 云端 SHA256 | Git | 建议 |
|---|---|---|---|
| configs/canon_dress_gs_gate3a_overfit.yaml | acfe2e78291c5999185d8fb5c3b483a562444e3a790da417ef6b0edf96d8a4e6 | untracked | staging |
| configs/canon_dress_gs_gate3b_anchor_hardoutside100.yaml | a3c57d3afe0e0e6a6a2e822cf3c3c484108ad1405b50b35fb7f2b6767aa8e97c | untracked | staging |
| configs/canon_dress_gs_gate3b_anchor_hardoutside300.yaml | 8b0d6747ea56d787d90fba4cae4cd4c69438caf8f05d86a2ee903a75e5044359 | untracked | staging |
| configs/canon_dress_gs_gate3b_anchor_noncloth100.yaml | 5355e1c60f3f90ef080ca2a6de47e5b4a5333cb0a5b8452121cfbf73f713a092 | untracked | staging |
| configs/canon_dress_gs_gate3b_anchor_only100.yaml | 8c9cf8fe78eb3b02a50e8800c7ab37736b308604fa2113d828476f5742cefc2c | untracked | staging |
| configs/canon_dress_gs_gate3b_calibration.yaml | 43c67661479252907ef8b9c8d77de1d823843ce488c106e50266e0e23c1d11fe | untracked | staging |
| configs/canon_dress_gs_gate3b_overfit100.yaml | cc2b852e59d2a9e47d3f4e368eef428da67c9f6020723363b6a7efb92f83cb96 | untracked | staging |
| configs/canon_dress_gs_gate3b_warmup25.yaml | 25389eebf3e6a371324f87a0704a294f75a9640252cce512b1368341910d2d32 | untracked | staging |
| docs/mmlphuman_code_reading_tutorial.md | 16b36cc4c6f9bcc5f7a47163f0b8ee305bf915d99647599678688052dcac0d56 | untracked | staging |

新建的 `docs/CANONDRESSGS_PROGRESS_2026-07-15.md` 与本清单也进入 staging。

## D：冲突，禁止覆盖

| 路径 | 云端 SHA256 | 本地 SHA256 | 建议 |
|---|---|---|---|
| train_dressable.py | 72a0882c15f468a0a4be9fbbf27dac93ba5a4911b040b7f62d3a47372d54f851 | cb503239e468741dadb446acf5f49909534013a9de27bc2f9a528fcafa6a0b65 | 人工合并 |
| utils/clothing_loss_utils.py | d087ea2e834e57fbcc03a667fdaed247540e4e74e7ad280f18e7d9b7d2e35782 | f02d9fae51990cdaa07b209dea66b4a13c89475d4d8563d562a13786f1abd14b | 人工合并 |
| utils/loss_utils.py | 61ad2814634def599730c06ec9d69d24c4a9c6cf0b78a5c4f4af1c5ac1b50ca6 | 57c468f92969feb33883b4f7fdec181f153d945f7c0e689dada0947eec9d3754 | 人工合并 |

## E：明确排除

云端已发现 `canondressgs_update.tar.gz`、`*.bak_gate3b`、`*.before_resume_support`、`*.before_*`、`*.log`、Python cache，以及 outputs/data/models/papers/smpl_model。全部不进入 staging。

## 安全规则

本阶段只建立 staging，不覆盖本地仓库。正式同步前重新计算云端和 staging SHA256；三个 D 类文件必须生成 diff 并单独人工确认。
