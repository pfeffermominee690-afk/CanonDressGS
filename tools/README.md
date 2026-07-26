# Tools Directory

本目录放置项目工程脚本。脚本命名与最终方案文档中的工程运行命令保持一致。

计划脚本：

```text
build_dressable_index.py
check_dressable_index.py
init_base_gaussians.py
check_base_render.py
run_lhm_prior.py
render_lhm_prior_views.py
build_anchor_offsets.py
visualize_anchor_offsets.py
check_dressable_batch.py
check_tensor_shapes.py
collect_metrics.py
```

约定：

```text
1. tools 脚本只做可复用的离线处理和检查。
2. 训练主入口放在 train_dressable.py。
3. 测试主入口放在 test_dressable.py。
4. 大文件输出写入 outputs/ 或 data_dressable/，不提交 Git。
```
