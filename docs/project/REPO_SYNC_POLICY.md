# Repository and Sync Policy

## 权威边界

- 本地权威代码：`E:\model_train\mmlphuman_code`
- legacy 云端实验仓库：`/root/autodl-tmp/canondressgs_work/mmlphuman_code`，冻结，只读审计；未经明确批准不得修改。
- 正式云端执行目录：`/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_pipeline_mvp`
- 正式分支：`pipeline/imagecond-mvp-20260715`

## 代码流

1. 所有正式代码先在本地修改和测试。
2. 逐文件审核拟 stage 清单；禁止 `git add -A`。
3. 本地 commit 后推送到明确 Git remote。
4. 干净云端 worktree 只 checkout 已提交 commit；运行前必须 clean 且 HEAD 一致。
5. legacy dirty 仓库禁止 reset、clean、restore 或整体覆盖。
6. 本地和云端不得同时编辑同一文件；云端正式 worktree 原则上只执行不编辑。

## 数据与产物流

- `data/`、`models/`、`outputs/`、checkpoint、日志和大图不进入 Git。
- 云端通过仓库外绝对路径读取数据与 backbone。
- 每个 run 写入新的唯一输出目录，禁止覆盖历史 Gate 输出。
- 小型正式产物按 allowlist 拉取到 `E:\model_train\CanonDressGS_Project\artifacts\<RunID>`。
- 拉取默认包括 manifest、metrics、Markdown、日志和 comparison PNG；大型 tensor/checkpoint 需显式批准。

## 部署门禁

- 本地 branch 正确、工作树 clean、HEAD 已提交。
- remote push 成功，remote commit 与 local commit 一致。
- 云端 checkout commit 与 local commit 一致，云端 worktree clean。
- 项目 import/compile smoke 通过。
- 部署日志记录时间、branch、local/remote/cloud commit 和 modified state。

## 安全规则

- 禁止整体文件夹复制覆盖代码。
- 禁止删除旧文件；先清单、后审批。
- 不使用 `git reset --hard`、`git clean`、`git checkout --` 或 `git restore` 处理 legacy 仓库。
- 新干净 worktree 的同步只能针对已确认 remote commit，并由部署脚本校验目标路径。

## 历史同步来源

2026-07-15 的云→本地文件分类与 SHA256 仍由 [`../SYNC_MANIFEST_2026-07-15.md`](../SYNC_MANIFEST_2026-07-15.md) 管理；本政策不改写其内容。
