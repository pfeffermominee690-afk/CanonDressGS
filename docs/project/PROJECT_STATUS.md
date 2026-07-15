# Project Status

更新时间：2026-07-15

## 当前状态

| 项目 | 状态 | 说明 |
|---|---|---|
| 治理基线 | PASS | 两个本地基线 commit 已完成，云端 bare remote 与干净执行 worktree 已验证 |
| legacy 云端仓库 | FROZEN | `/root/autodl-tmp/canondressgs_work/mmlphuman_code` 仅作历史实验与只读证据来源 |
| 干净云端执行目录 | PASS | `pipeline/imagecond-mvp-20260715@aa9bdaa918554fae18bd51132be023f5d9759dbe`，Git clean |
| Gate 3-C | PARTIAL | threshold 0.40 是当前-backbone 简单 baseline，不关闭 Gate 3-C |
| Gate 4 MVP | IN PROGRESS | 当前里程碑：Gate 4-A Real Image-conditioned One-batch |
| 论文证据包 | IN PROGRESS | 已建立 evidence matrix，尚无 Gate 4 端到端结果 |

## 已知阻塞与风险

1. 本地工作树包含大量既有 tracked modifications 和 untracked files，必须逐文件审核，禁止 `git add -A`。
2. 当前 checkpoint SHA256 `abbf67b5...` 与历史 Gate 指纹 `64ac7f2d...` 不一致；两者不得合并表述。
3. legacy 云端仓库 dirty，不能作为正式执行 worktree。
4. image-conditioned 代码已存在多个模块和检查脚本，但正式提交闭包、导入闭包与端到端 Gate 4 证据尚未冻结。
5. 2026-07-17 截止时间紧，功能扩张必须服从 xyz-only MVP。

## 本地基线检查

- PowerShell 项目脚本语法：3/3 PASS。
- Gate 4 核心 Python 文件 `py_compile`：PASS。
- `tools/check_real_image_conditioned_interfaces.py`：21 PASS。
- `tools/check_image_conditioned_dataset.py`：20 PASS。
- `tools/check_image_conditioned_training.py`：PASS；仅 synthetic/mock，不代表真实 MMLPHuman rendering 验收。
- `tools/check_clothing_losses.py`：10 PASS。
- `tools/check_dressable_model.py`：9 PASS。
- `git diff --check`：PASS。

## 接下来三个任务

1. 在真实 MMLP-Human、真实 reference images/poses/cameras 上完成 Gate 4-A one-batch。
2. 验证 reference sensitivity、真实 render backward、冻结梯度和 checkpoint roundtrip。
3. Gate 4-A PASS 后才进入 100/300-step overfit。

## 最近一次验收

Gate 3-C2 current-backbone fixed-offset threshold ablation：PARTIAL。0.40 相对 0.50 的 float RGB-all MAE 改善约 0.537%，但覆盖问题仍存在，不足以关闭 Gate 3-C。
