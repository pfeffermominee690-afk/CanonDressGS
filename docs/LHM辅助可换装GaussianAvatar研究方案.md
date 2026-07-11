# CanonDressGS 最终落地方案

本文档是当前科研项目的最终执行版方案，用于 5 人小组直接分工落地。文档只保留最终选择的技术路线，不再保留多方案比较。

最终目标：

```text
基于真实裸模 / 白模 THuman-SMPL-X base_gs，
学习 clothing-conditioned canonical Gaussian offsets，
先在 canonical space 完成换装，
再复用 MMLPHuman 的 pose-driven deformation 与 Gaussian rendering，
实现同一身份、指定服装、指定姿态、指定视角的人体 Avatar 渲染。
```

最终方法名称建议：

```text
CanonDressGS: Canonical-space Dressable Gaussian Splatting
```

论文中可简称：

```text
CanonDressGS
```

---

## 1. 文献与工程依据

本项目最终方案基于以下调研结论收敛得到。

MMLPHuman 是我们的姿态驱动与实时渲染主干。MMLPHuman 使用 spatially distributed MLPs、Gaussian offset basis、control points 和 LBS，将 canonical Gaussians 驱动到目标姿态，并保持 novel pose / novel view 下的高质量实时渲染。因此本项目不重写姿态驱动模块，而是在 MMLPHuman 的 canonical Gaussian 输入之前增加可换装模块。

LHM 是我们的服装几何先验与教师模型。LHM 能从单张参考图前馈生成 animatable 3D Gaussian human，并保留服装几何和纹理细节。最终系统推理阶段不调用 LHM，LHM 只在训练前离线生成每套服装的 dressed Gaussian prior，用于监督 clothing offset generator。

HyperNetworks、FiLM 和 LoRA 是我们的服装条件化参数生成依据。HyperNetwork 的核心思想是用一个网络生成另一个网络的权重或权重相关参数。FiLM 使用条件信息生成 feature-wise affine modulation。LoRA 使用低秩权重增量降低参数量和显存。结合老师提出的“服装 embedding 输入网络，输出 MLP 权重系数”的要求，本项目最终采用 lightweight HyperNetwork 输出 FiLM / low-rank modulation 参数，不生成完整 MLP 权重。

PyTorch3D 支撑你们当前的数据构建流程。它提供模块化、可微分的 mesh / point 渲染能力，适合从 THuman / SMPL-X 网格渲染白模 condition 图。

参考链接：

- MMLPHuman project: https://gapszju.github.io/mmlphuman/
- MMLPHuman arXiv: https://arxiv.org/abs/2504.12909
- LHM project: https://lingtengqiu.github.io/LHM/
- LHM arXiv: https://arxiv.org/abs/2503.10625
- HyperNetworks: https://arxiv.org/abs/1609.09106
- FiLM: https://arxiv.org/abs/1709.07871
- LoRA: https://arxiv.org/abs/2106.09685
- PyTorch3D: https://arxiv.org/abs/2007.08501

---

## 2. 最终 Pipeline

整体流程固定为：

```text
Stage A: 白模姿态条件数据构建
  THuman / SMPL-X 人体模型
  -> PyTorch3D 多视角渲染灰白 condition 图
  -> 2000 张候选姿态图
  -> VLM hand-strict 筛选 300 张高质量姿态条件图
  -> GPT-image-2 生成同姿态多服装 RGB 图
  -> 保存 pose / camera / mask / cloth_id

Stage B: Base naked Gaussian avatar 构建
  THuman / SMPL-X canonical naked mesh
  -> 初始化 canonical base_gs
  -> 绑定 anchors / control points / LBS weights
  -> 使用 300 组 pose / camera / mask 验证 base_gs 姿态驱动稳定性

Stage C: LHM 服装教师先验构建
  每套服装 reference image
  -> LHM 生成 dressed Gaussian prior
  -> 对齐到 SMPL-X canonical space
  -> 汇聚到 SMPL-X body anchors
  -> 得到 anchor-level clothing offset target

Stage D: Canonical 换装模块训练
  cloth_id -> z_c
  -> lightweight HyperNetwork
  -> FiLM / low-rank modulation parameters
  -> modulate shared clothing offset MLPs
  -> predict canonical Gaussian offsets
  -> base_gs + offsets = canonical dressed Gaussians

Stage E: Pose-driven rendering
  canonical dressed Gaussians
  + target SMPL-X pose
  + camera
  -> MMLPHuman deformation / offset basis / LBS
  -> Gaussian rasterization
  -> rendered dressed avatar image
```

最核心顺序：

```text
canonical base human
        ↓
clothing-conditioned canonical offsets
        ↓
canonical dressed human
        ↓
MMLPHuman pose-driven deformation
        ↓
Gaussian rendering
```

---

## 3. 数据格式与目录规范

### 3.1 原始数据来源

你们当前训练数据集按如下方式构建：

```text
THuman / SMPL-X 人体模型
        ↓
读取 SMPL-X 参数
  - 三维人体网格
  - 骨架姿态
  - 相机信息
        ↓
PyTorch3D 多视角高分辨率渲染
        ↓
生成白色背景下的灰白人体 condition 图
        ↓
同步保存 mask / pose / camera
        ↓
生成 2000 张候选姿态图
        ↓
VLM 自动评分
  - 全身完整性
  - 手部可见性
  - 手部细节
  - 手部与身体或衣服的分离程度
  - 肢体轮廓清晰度
  - 渲染质量
        ↓
hand-strict 筛选 300 张高质量姿态条件图
        ↓
作为 GPT-image-2 同姿态多服装图像生成输入
```

### 3.2 项目数据目录

统一整理为：

```text
data_dressable/
  identity_000/
    smplx/
      canonical_mesh.obj
      canonical_params.json
      lbs_weights.npy
    conditions/
      frame_000000/
        condition.png
        mask.png
        pose.npy
        camera.json
        score.json
      frame_000001/
        condition.png
        mask.png
        pose.npy
        camera.json
        score.json
    clothes/
      cloth_000/
        reference.png
        generated_rgb/
          frame_000000.png
          frame_000001.png
        generated_mask/
          frame_000000.png
          frame_000001.png
        lhm_prior/
          gaussians.ply
          metadata.json
        offset_target/
          anchor_offsets.pt
          gaussian_offsets.pt
      cloth_001/
        reference.png
        generated_rgb/
        generated_mask/
        lhm_prior/
        offset_target/
    splits/
      train.json
      val.json
      test_novel_pose.json
      test_novel_view.json
      test_same_pose_cloth_switch.json
```

### 3.3 样本索引字段

`train.json` 中每条样本统一为：

```json
{
  "identity_id": "identity_000",
  "cloth_id": "cloth_000",
  "frame_id": "frame_000000",
  "condition_image": "conditions/frame_000000/condition.png",
  "target_rgb": "clothes/cloth_000/generated_rgb/frame_000000.png",
  "foreground_mask": "conditions/frame_000000/mask.png",
  "cloth_mask": "clothes/cloth_000/generated_mask/frame_000000.png",
  "pose": "conditions/frame_000000/pose.npy",
  "camera": "conditions/frame_000000/camera.json",
  "lhm_prior": "clothes/cloth_000/lhm_prior/gaussians.ply",
  "anchor_offset_target": "clothes/cloth_000/offset_target/anchor_offsets.pt"
}
```

---

## 4. 模块 1：Base Naked Gaussian Avatar

### 4.1 目标

构建一个同一 identity 的 canonical naked Gaussian avatar，作为所有服装偏移的共享起点。

最终使用：

```text
THuman / SMPL-X naked canonical mesh -> base_gs
```

`base_gs` 包含：

```text
base_xyz
base_scale
base_rotation
base_opacity
base_sh
base_anchor_points
base_control_points
base_lbs_weights
```

### 4.2 参考代码

直接参考 MMLPHuman：

```text
scene/scene.py
  - 加载 template mesh
  - 初始化 SMPL-X
  - 构建 LBS weight volume
  - 调用 gaussians.create_from_pcd(...)

scene/gaussian_model.py
  - GaussianModel.create_from_pcd
  - _xyz
  - _scaling
  - _rotation
  - _opacity
  - _sh0
  - _shN
  - get_Gweights
  - get_xyz
  - render

utils/smpl_utils.py
  - SMPL-X pose / transform / LBS 相关函数
```

### 4.3 实现任务

负责人和组员 A 完成：

1. 从 THuman / SMPL-X canonical naked mesh 采样 Gaussian 初始化点。
2. 建立 `base_gs` 的 canonical Gaussian 参数。
3. 使用 MMLPHuman 的 anchor sampling 逻辑创建 spatially distributed MLP anchors。
4. 使用 MMLPHuman 的 control point 机制约束位置偏移。
5. 给每个 Gaussian 绑定 LBS weights。
6. 用 300 个 hand-strict pose / camera / mask 检查 base_gs 的投影、轮廓和关节变形。

### 4.4 验收标准

```text
base_gs 可以在所有 300 个姿态条件下完成 forward render。
rendered alpha 与白模 mask 大体对齐。
手臂、腿部、躯干没有明显断裂。
关节区域没有大面积 Gaussian 爆开。
保存 base checkpoint:
  outputs/base_naked_gaussian/checkpoint_latest.pth
```

---

## 5. 模块 2：LHM 服装教师先验

### 5.1 目标

对每套服装生成一个 dressed Gaussian prior，用于监督 clothing offset generator。

LHM 只用于训练前离线处理：

```text
reference image -> LHM -> dressed Gaussian prior
```

推理阶段不调用 LHM。

### 5.2 参考输入

每套服装至少准备：

```text
reference.png
cloth_id
对应 identity 的 canonical SMPL-X 信息
```

### 5.3 输出文件

```text
data_dressable/identity_000/clothes/cloth_000/lhm_prior/
  gaussians.ply
  metadata.json
```

`metadata.json` 记录：

```json
{
  "cloth_id": "cloth_000",
  "source_image": "reference.png",
  "coordinate_system": "lhm_output",
  "scale": 1.0,
  "canonicalization": "smplx_canonical_aligned",
  "num_gaussians": 0
}
```

### 5.4 实现任务

组员 B 完成：

1. 为每套服装准备 reference image。
2. 使用 LHM 生成 dressed Gaussian prior。
3. 将 LHM 输出保存为标准 `.ply` 和 `metadata.json`。
4. 对每套服装渲染 3 个视角截图进行质量检查。
5. 记录失败样本并重新选择 reference image。

### 5.5 验收标准

```text
每套 cloth_id 都有一个 lhm_prior/gaussians.ply。
LHM prior 在正面、侧面、背面三视角可视化中人体完整。
服装轮廓与 reference image 一致。
无明显漂浮大块 Gaussian。
```

---

## 6. 模块 3：SMPL-X Surface-aware Anchor Alignment

### 6.1 目标

将 LHM dressed Gaussian prior 转换成监督信号：

```text
anchor-level clothing offset target
```

最终使用 SMPL-X surface-aware anchor alignment。

### 6.2 核心思想

LHM prior 和 base_gs 的 Gaussian 点没有一一对应关系，因此不直接做点对点相减。最终使用 SMPL-X canonical surface 作为公共中介：

```text
base_gs
  -> 投影到 SMPL-X canonical surface
  -> 得到 face_id / barycentric coordinate / nearest anchor

LHM dressed Gaussians
  -> 对齐到 SMPL-X canonical space
  -> 投影或汇聚到相同 anchor region

同一 anchor region 内:
  dressed prior - base body = clothing offset target
```

### 6.3 Offset target 内容

每个 anchor 记录：

```text
delta_xyz_anchor
delta_scale_anchor
delta_rotation_anchor
delta_opacity_anchor
delta_sh_anchor
valid_mask_anchor
cloth_region_weight
```

保存为：

```text
anchor_offsets.pt
```

结构建议：

```python
{
    "cloth_id": str,
    "anchor_xyz": Tensor[A, 3],
    "delta_xyz": Tensor[A, 3],
    "delta_scale": Tensor[A, 3],
    "delta_rotation": Tensor[A, 4],
    "delta_opacity": Tensor[A, 1],
    "delta_sh": Tensor[A, C],
    "valid_mask": Tensor[A, 1],
    "cloth_region_weight": Tensor[A, 1],
}
```

### 6.4 代码文件

新增：

```text
utils/lhm_prior_utils.py
utils/gaussian_alignment.py
```

主要函数：

```python
load_lhm_gaussians(ply_path)
load_base_gaussians(checkpoint_path)
canonicalize_lhm_to_smplx(lhm_gaussians, smplx_params)
project_gaussians_to_smplx_surface(gaussians, smplx_mesh)
aggregate_lhm_prior_to_anchors(lhm_gaussians, anchor_points)
build_anchor_offset_target(base_gs, lhm_prior, anchor_points)
save_anchor_offset_target(target, out_path)
```

### 6.5 实现任务

组员 C 完成：

1. 读取 base_gs 和 LHM prior。
2. 将 LHM prior 统一到 SMPL-X canonical 坐标系。
3. 将 base_gs 和 LHM prior 都投影到 SMPL-X surface anchors。
4. 对每个 anchor 聚合局部 LHM Gaussian 属性。
5. 计算 anchor-level offsets。
6. 保存 `anchor_offsets.pt`。
7. 输出 offset heatmap 可视化。

### 6.6 验收标准

```text
每套服装都生成 anchor_offsets.pt。
anchor offset heatmap 主要集中在衣服覆盖区域。
脸、手、脚等非服装区域 offset 较小。
宽松衣服区域允许较大 delta_xyz。
offset target 可以被训练脚本正确加载。
```

---

## 7. 模块 4：Clothing-conditioned Offset Generator

### 7.1 目标

输入 `cloth_id`，输出该服装在 canonical space 下的 Gaussian offsets。

最终结构：

```text
cloth_id
  -> learned clothing embedding z_c
  -> lightweight HyperNetwork
  -> FiLM / low-rank modulation parameters
  -> modulate shared MMLPHuman-style clothing offset MLPs
  -> anchor-level clothing coefficients
  -> interpolate to Gaussian-level offsets
```

### 7.2 模型结构

新增：

```text
scene/clothing_embedding.py
scene/clothing_hypernetwork.py
scene/dressable_gaussian_model.py
```

#### ClothingEmbedding

```python
class ClothingEmbedding(nn.Module):
    def __init__(self, num_clothes: int, dim: int = 64):
        ...

    def forward(self, cloth_id: Tensor) -> Tensor:
        ...
```

输出：

```text
z_c: [B, 64]
```

#### ClothingHyperNetwork

```python
class ClothingHyperNetwork(nn.Module):
    def __init__(self, z_dim: int, num_layers: int, hidden_dim: int, rank: int):
        ...

    def forward(self, z_c: Tensor) -> Dict[str, Tensor]:
        ...
```

输出：

```text
film_gamma_l
film_beta_l
low_rank_A_l
low_rank_B_l
coefficient_head_modulation
```

#### Clothing Offset MLP

输入：

```text
anchor positional encoding
body part encoding
optional pose-independent canonical feature
```

输出：

```text
anchor clothing coefficients
```

再通过 clothing offset basis 得到：

```text
delta_xyz
delta_scale
delta_rotation
delta_opacity
delta_sh
```

### 7.3 与 MMLPHuman 的关系

复用 MMLPHuman 的思想：

```text
anchor MLP 输出 coefficients
coefficients 线性组合 Gaussian offset basis
anchor coefficients 插值到 Gaussian
control points 约束位置 offset
```

新增部分只发生在 canonical dressing 阶段：

```text
clothing embedding z_c
  -> 调制 anchor MLP
  -> 生成服装相关 canonical offsets
```

### 7.4 实现任务

组员 D 完成：

1. 实现 `ClothingEmbedding`。
2. 实现 `ClothingHyperNetwork`。
3. 在 `dressable_gaussian_model.py` 中包装原 `GaussianModel`。
4. 实现 `compute_clothing_offsets(cloth_id)`。
5. 实现 `compute_dressed_canonical_gaussians(cloth_id)`。
6. 确保 dressed canonical Gaussian 参数可以传入 MMLPHuman 原 render 流程。
7. 输出参数量统计和显存占用记录。

### 7.5 验收标准

```text
输入 cloth_id 后可以得到稳定的 Gaussian offsets。
同一 cloth_id 多次 forward 结果一致。
不同 cloth_id 的 offset heatmap 有明显差异。
offset 后的 canonical dressed Gaussians 可视化中服装区域发生合理外扩。
render 函数可以使用 dressed canonical Gaussians 正常出图。
```

---

## 8. 模块 5：Canonical Dressed Gaussian Composition

### 8.1 目标

将 `base_gs` 和 clothing offsets 合成 canonical dressed Gaussians。

### 8.2 合成公式

```text
xyz_dressed      = xyz_base + delta_xyz_cloth
scale_dressed    = scale_base + delta_scale_cloth
rotation_dressed = compose(rotation_base, delta_rotation_cloth)
opacity_dressed  = opacity_base + delta_opacity_cloth
sh_dressed       = sh_base + delta_sh_cloth
```

约束：

```text
delta_xyz_cloth 在 canonical space 中计算。
delta_scale_cloth 使用有界激活。
delta_opacity_cloth 使用 sigmoid / clamp 约束。
delta_rotation_cloth 使用 normalized quaternion。
非服装区域使用 region weight 限制 offset 幅度。
```

### 8.3 代码函数

在 `scene/dressable_gaussian_model.py` 中实现：

```python
def compute_clothing_offsets(self, cloth_id):
    ...

def compose_dressed_gaussians(self, base_params, offsets):
    ...

def get_dressed_canonical_params(self, cloth_id):
    ...

def render(self, viewpoint_cam, cloth_id):
    dressed_params = self.get_dressed_canonical_params(cloth_id)
    return self.render_with_params(viewpoint_cam, dressed_params)
```

### 8.4 验收标准

```text
canonical dressed Gaussians 可以单独导出 ply。
不同 cloth_id 导出的 canonical dressed ply 有可见服装差异。
非服装区域没有明显漂移。
服装区域 alpha / scale / color 有合理变化。
```

---

## 9. 模块 6：Pose-driven Deformation 与 Rendering

### 9.1 目标

复用 MMLPHuman，将 canonical dressed Gaussians 变形到目标姿态并渲染。

### 9.2 复用代码

```text
scene/gaussian_model.py
  - get_xyz
  - get_Gweights
  - get_covariance
  - get_color
  - render

utils/smpl_utils.py
  - LBS 权重查询
  - rigid transform
  - SMPL-X pose transform

train.py
  - pose / camera 输入
  - loss 组合
  - optimizer step
```

### 9.3 最小侵入实现

保持原 MMLPHuman render 主体不变，在 render 前替换 canonical Gaussian 参数：

```text
base canonical params
        ↓
clothing offsets
        ↓
dressed canonical params
        ↓
原 MMLPHuman pose-driven deformation
        ↓
原 Gaussian rasterization
```

### 9.4 验收标准

```text
同一 cloth_id 可在 300 个姿态上稳定渲染。
同一 pose / camera 下切换 cloth_id 后人体姿态不变。
novel view 下服装结构稳定。
novel pose 下无大面积拉扯、断裂、漂浮。
```

---

## 10. 训练目标

### 10.1 总损失

```text
L_total =
  L_rgb
  + lambda_mask * L_mask
  + lambda_lhm * L_lhm_anchor_offset
  + lambda_smooth * L_anchor_smooth
  + lambda_region * L_noncloth_region
  + lambda_scale * L_gaussian_scale
```

### 10.2 各项损失

RGB 重建：

```text
L_rgb = L1(rendered_rgb, target_rgb)
      + lambda_ssim * (1 - SSIM(rendered_rgb, target_rgb))
      + lambda_lpips * LPIPS(rendered_rgb, target_rgb)
```

Mask：

```text
L_mask = BCE(rendered_alpha, foreground_mask)
       + Dice(rendered_alpha, foreground_mask)
```

LHM anchor offset supervision：

```text
L_lhm_anchor_offset =
  valid_mask * cloth_region_weight *
  SmoothL1(pred_anchor_offsets, lhm_anchor_offset_target)
```

Anchor smoothness：

```text
L_anchor_smooth =
  sum_neighbors || delta_anchor_i - delta_anchor_j ||
```

Non-clothing region regularization：

```text
L_noncloth_region =
  (1 - cloth_region_weight) * || predicted_offsets ||
```

Gaussian scale regularization：

```text
L_gaussian_scale =
  gaussian_scaling_loss(dressed_scale)
```

### 10.3 训练阶段

训练按固定顺序推进：

```text
Stage 0: MMLPHuman baseline 复现
  目的: 确认原项目训练、测试、渲染链路可用

Stage 1: base_gs 构建与冻结
  目的: 得到稳定 naked canonical Gaussian avatar

Stage 2: LHM prior 与 anchor offset target 生成
  目的: 为每套服装得到 geometry teacher supervision

Stage 3: Clothing HyperNetwork 训练
  目的: 学习 cloth_id -> canonical clothing offsets

Stage 4: Rendering-aware finetune
  目的: 使用 GPT-image-2 生成的同姿态多服装 RGB 图做图像级优化

Stage 5: 小范围联合微调
  目的: 微调 appearance / offset basis，保持 base_gs 主体稳定
```

### 10.4 训练权重初值

```text
lambda_ssim   = 0.2
lambda_lpips  = 0.1
lambda_mask   = 0.5
lambda_lhm    = 1.0
lambda_smooth = 0.05
lambda_region = 0.1
lambda_scale  = 0.01
```

这些权重作为第一轮训练配置写入：

```text
configs/canon_dress_gs.yaml
```

---

## 11. 代码实现路线

### 11.1 新增文件

```text
scene/dressable_dataset.py
scene/clothing_embedding.py
scene/clothing_hypernetwork.py
scene/dressable_gaussian_model.py
utils/lhm_prior_utils.py
utils/gaussian_alignment.py
train_dressable.py
test_dressable.py
configs/canon_dress_gs.yaml
```

### 11.2 文件职责

`scene/dressable_dataset.py`

```text
读取 train.json / val.json / test.json。
返回 condition image、target_rgb、mask、pose、camera、cloth_id、anchor_offset_target。
保持 MMLPHuman camera 数据结构兼容。
```

`scene/clothing_embedding.py`

```text
维护 cloth_id 到 learnable embedding z_c 的映射。
```

`scene/clothing_hypernetwork.py`

```text
输入 z_c。
输出 FiLM / low-rank modulation 参数。
这些参数调制 shared clothing offset MLP。
```

`scene/dressable_gaussian_model.py`

```text
包装 GaussianModel。
维护 base_gs。
计算 clothing offsets。
合成 dressed canonical Gaussians。
调用 MMLPHuman 原 pose-driven render。
```

`utils/lhm_prior_utils.py`

```text
读取 LHM Gaussian prior。
解析 ply 中的 xyz、scale、rotation、opacity、SH。
执行坐标归一化和 canonical 对齐。
```

`utils/gaussian_alignment.py`

```text
投影 base_gs / LHM prior 到 SMPL-X canonical surface。
聚合到 anchors。
生成 anchor_offsets.pt。
输出 offset heatmap。
```

`train_dressable.py`

```text
训练主入口。
加载 DressableDataset。
加载 base_gs。
加载 anchor offset targets。
执行 staged training。
保存 checkpoint 和可视化。
```

`test_dressable.py`

```text
测试入口。
支持 same-pose cloth switching。
支持 novel pose。
支持 novel view。
输出指标和视频。
```

### 11.3 训练入口伪代码

```python
dataset = DressableDataset(split="train", config=cfg)
model = DressableGaussianModel(cfg)
model.load_base_gaussians(cfg.base_checkpoint)

for iteration in range(cfg.train.iterations):
    batch = dataset.sample()

    cloth_id = batch["cloth_id"]
    camera = batch["camera"]
    pose = batch["pose"]
    target_rgb = batch["target_rgb"]
    mask = batch["foreground_mask"]
    anchor_target = batch["anchor_offset_target"]

    outputs = model.render(
        camera=camera,
        pose=pose,
        cloth_id=cloth_id,
    )

    loss_rgb = rgb_loss(outputs.rgb, target_rgb)
    loss_mask = mask_loss(outputs.alpha, mask)
    loss_lhm = anchor_offset_loss(outputs.anchor_offsets, anchor_target)
    loss_smooth = anchor_smooth_loss(outputs.anchor_offsets)
    loss_region = noncloth_region_loss(outputs.offsets, outputs.region_weight)
    loss_scale = gaussian_scaling_loss(outputs.dressed_scale)

    loss = (
        loss_rgb
        + cfg.lambda_mask * loss_mask
        + cfg.lambda_lhm * loss_lhm
        + cfg.lambda_smooth * loss_smooth
        + cfg.lambda_region * loss_region
        + cfg.lambda_scale * loss_scale
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 12. 五人分工

本项目共 5 人：负责人 1 人 + 组员 A/B/C/D 4 人。推荐采用“负责人控方向与验收，最强代码组员做模型主程，其他组员负责可独立验收模块”的协作方式。

### 负责人：项目负责人 / 方法负责人 / 论文主线

负责内容：

```text
1. 决定并维护最终 pipeline。
2. 固定核心接口、tensor shape、目录规范和验收标准。
3. 维护 configs/canon_dress_gs.yaml。
4. 审核各组员 PR，控制 dev-dressable 与 main 的合并节奏。
5. 每晚做集成验收，检查一个 batch 是否能跑通。
6. 统筹实验设计、结果表格、论文叙事和 method 主线。
7. 不长期陷入某一个底层模块的具体实现，避免项目无人控盘。
```

交付物：

```text
最终 pipeline 文档
核心接口与 tensor shape 规范
完整 pipeline figure
论文 method 初稿
实验计划与结果表格模板
每日进度表
```

验收重点：

```text
每个模块是否符合接口。
每个 PR 是否能跑最小样例。
每晚 dev-dressable 是否能完成一次 debug_one_batch。
论文中的方法描述是否与代码实现一致。
```

### 组员 A：数据与 base_gs

负责内容：

```text
1. 跑通 MMLPHuman 原始 train.py / test.py，记录 baseline 复现流程。
2. 整理 THuman / SMPL-X 白模 condition 数据。
3. 生成 / 检查 condition、target_rgb、mask、pose、camera 文件。
4. 生成 train / val / test JSON 索引。
5. 从 THuman / SMPL-X canonical naked mesh 初始化 base_gs。
6. 绑定 anchors / control points / LBS weights。
7. 用 hand-strict pose 验证 base_gs 渲染稳定性。
```

交付物：

```text
data_dressable/identity_000/splits/*.json
outputs/mmlphuman_reproduce/
outputs/base_naked_gaussian/checkpoint_latest.pth
docs/data_quality_report.md
docs/base_gs_render_check.md
base_gs 三视角与多姿态可视化
```

验收重点：

```text
给定 identity_id 和 pose_id，可以稳定读取 condition / mask / pose / camera。
给定 canonical SMPL-X mesh，可以得到 base_gs checkpoint。
base_gs 在多个姿态和视角下渲染不崩。
```

### 组员 B：LHM teacher prior

负责内容：

```text
1. 为每套服装准备 reference image。
2. 运行 LHM 生成 dressed human / dressed Gaussian prior。
3. 将 LHM 输出整理为统一字段格式。
4. 保存 lhm_prior/gaussians.ply 和 metadata.json。
5. 输出每套服装的 LHM prior 多视角检查图。
6. 记录 LHM 失败样例与质量筛选规则。
```

交付物：

```text
每套服装的 lhm_prior/gaussians.ply
每套服装的 lhm_prior/metadata.json
utils/lhm_prior_utils.py 中的 LHM 读取辅助函数
docs/lhm_prior_quality_report.md
LHM prior 三视角可视化
```

验收重点：

```text
给定 cloth_id，可以读取对应 LHM prior。
LHM prior 字段包含 xyz / scale / rotation / opacity / sh。
LHM prior 可视化能够看出服装几何和人体大致合理。
```

### 组员 C：Anchor alignment 与 offset target

负责内容：

```text
1. 实现 utils/lhm_prior_utils.py。
2. 实现 utils/gaussian_alignment.py。
3. 将 LHM prior 对齐到 SMPL-X canonical space。
4. 投影 / 汇聚到 SMPL-X anchors。
5. 生成 anchor_offsets.pt。
6. 输出 offset heatmap。
```

交付物：

```text
utils/lhm_prior_utils.py
utils/gaussian_alignment.py
每套服装的 offset_target/anchor_offsets.pt
docs/alignment_visualization.md
offset heatmap 图片
```

验收重点：

```text
给定 base_gs 和 LHM prior，可以输出 aligned anchor_offsets.pt。
offset target 的 shape 与文档一致。
offset heatmap 能显示服装区域的主要变化。
```

### 组员 D：模型主程 / HyperNetwork 与 clothing offsets

负责内容：

```text
1. 实现 scene/clothing_embedding.py。
2. 实现 scene/clothing_hypernetwork.py。
3. 实现 clothing offset MLP。
4. 实现 compute_clothing_offsets。
5. 实现 compose_dressed_gaussians。
6. 与负责人一起搭建 train_dressable.py 和 scene/dressable_gaussian_model.py 的核心 forward。
7. 实现 test_dressable.py 的基础批量渲染与指标入口。
8. 输出参数量、显存统计、训练速度和渲染结果。
```

交付物：

```text
scene/clothing_embedding.py
scene/clothing_hypernetwork.py
scene/dressable_gaussian_model.py 的 offset 相关部分
train_dressable.py 的模型调用主线
test_dressable.py 的基础评估入口
docs/model_parameter_report.md
canonical dressed Gaussian ply 可视化
outputs/eval/metrics.csv
outputs/eval/cloth_switching_grid.png
outputs/eval/novel_pose_video.mp4
outputs/eval/novel_view_video.mp4
```

验收重点：

```text
给定 cloth_id + pose + camera，可以完成一次 forward。
loss 可以正常 backward。
可以保存一张 rendered image。
可以批量导出 same-pose cloth switching / novel pose / novel view 可视化。
```

### 论文初稿分工

7 月 18 日前，论文初稿由负责人统一整合，但每个人必须提供对应章节素材。

| 成员 | 论文素材责任 | 7 月 18 日前交付 |
|---|---|---|
| 负责人 | Abstract、Introduction、Method 总体、Contribution、Pipeline figure、实验设计 | 完整论文 v1、方法主图、贡献点表述、主结果表和消融表模板 |
| 组员 A | MMLPHuman baseline、数据集构建、白模 condition、base_gs 构建、pose-driven rendering 复用说明 | baseline 复现描述、data construction section、base_gs 可视化、MMLPHuman 代码调用链图 |
| 组员 B | LHM prior、服装 reference、teacher prior 质量控制 | LHM prior 方法段落、LHM prior 可视化、失败样例和筛选规则 |
| 组员 C | SMPL-X surface-aware anchor alignment、offset target | alignment method 段落、anchor offset heatmap、target 数据结构说明 |
| 组员 D | Clothing embedding、HyperNetwork、FiLM / low-rank modulation、训练与评估入口 | model section 细节、网络结构图、参数量统计、基础实验结果与可视化 |

7 月 18 日论文初稿最低标准：

```text
1. 标题、摘要、Introduction 完整。
2. Related Work 有完整小节和引用占位。
3. Method 至少包含:
   - base_gs construction
   - LHM teacher prior
   - SMPL-X surface-aware anchor alignment
   - clothing-conditioned HyperNetwork
   - canonical dressed Gaussian composition
   - MMLPHuman pose-driven rendering
4. Experiments 至少包含:
   - dataset
   - baselines
   - metrics
   - implementation details
   - result table placeholders
   - ablation table placeholders
5. Pipeline figure 有第一版。
6. 所有关键图表位置都有占位图或当前中间结果。
7. Limitations 有初稿。
```

---

## 13. 团队代码协作规范

本项目代码量和中间产物较多，必须按模块边界协作。所有组员遵守以下规则。

### 13.1 分支结构

长期保留两个公共分支：

```text
main
dev-dressable
```

`main` 只保存稳定版本。`dev-dressable` 是每日集成分支，所有功能都先合并到 `dev-dressable`，阶段性稳定后再由负责人合并到 `main`。

每位成员从 `dev-dressable` 创建自己的功能分支：

```text
负责人:
  feature/integration-train

组员 A:
  feature/data-base-gs

组员 B:
  feature/lhm-prior

组员 C:
  feature/anchor-alignment

组员 D:
  feature/hypernetwork-offset
```

每日开发流：

```text
dev-dressable
        ↓
个人 feature 分支
        ↓
完成模块最小验证
        ↓
提交 PR 到 dev-dressable
        ↓
负责人 review / 合并
        ↓
晚上统一跑 train_dressable.py 一个 batch
```

### 13.2 文件所有权

每个文件有主负责人。其他人需要修改该文件时，先在组内同步，避免多人同时改核心文件。

| 文件 / 目录 | 主负责人 | 说明 |
|---|---|---|
| `configs/canon_dress_gs.yaml` | 负责人 | 全局路径、loss 权重、训练参数 |
| `train_dressable.py` | 负责人 + 组员 D | 训练主入口，由负责人控制接口、组员 D 实现模型调用主线 |
| `scene/dressable_gaussian_model.py` | 负责人 + 组员 D | 模型总装、dressed Gaussians 合成、render 接口 |
| `scene/dressable_dataset.py` | 组员 A | 多服装数据读取 |
| `scene/clothing_embedding.py` | 组员 D | `cloth_id -> z_c` |
| `scene/clothing_hypernetwork.py` | 组员 D | HyperNetwork / FiLM / low-rank modulation |
| `utils/lhm_prior_utils.py` | 组员 B + 组员 C | LHM prior 读取与坐标处理 |
| `utils/gaussian_alignment.py` | 组员 C | SMPL-X surface-aware anchor alignment |
| `test_dressable.py` | 组员 D | 测试、指标、批量渲染，负责人验收结果 |
| `docs/*.md` | 负责人 | 方案、进度、实验记录 |

原 MMLPHuman 核心文件默认不直接大改：

```text
train.py
test.py
scene/gaussian_model.py
scene/scene.py
scene/dataset.py
```

需要复用逻辑时，优先在新文件中包装或继承。确实需要改原文件时，由负责人统一合并。

### 13.3 模块接口契约

各模块必须遵守固定输入输出，不能私自改字段名。

#### Dataset 接口

`DressableDataset.__getitem__` 返回：

```python
{
    "identity_id": str,
    "cloth_id": int,
    "frame_id": str,
    "condition_image": Tensor[3, H, W],
    "target_rgb": Tensor[3, H, W],
    "foreground_mask": Tensor[1, H, W],
    "cloth_mask": Tensor[1, H, W],
    "pose": Tensor[P],
    "camera": CameraObject,
    "anchor_offset_target": Dict[str, Tensor],
}
```

#### LHM prior 接口

`load_lhm_gaussians(path)` 返回：

```python
{
    "xyz": Tensor[N, 3],
    "scale": Tensor[N, 3],
    "rotation": Tensor[N, 4],
    "opacity": Tensor[N, 1],
    "sh": Tensor[N, C],
}
```

#### Anchor offset target 接口

`anchor_offsets.pt` 固定为：

```python
{
    "cloth_id": str,
    "anchor_xyz": Tensor[A, 3],
    "delta_xyz": Tensor[A, 3],
    "delta_scale": Tensor[A, 3],
    "delta_rotation": Tensor[A, 4],
    "delta_opacity": Tensor[A, 1],
    "delta_sh": Tensor[A, C],
    "valid_mask": Tensor[A, 1],
    "cloth_region_weight": Tensor[A, 1],
}
```

#### HyperNetwork 接口

`ClothingHyperNetwork.forward(z_c)` 返回：

```python
{
    "film_gamma": List[Tensor],
    "film_beta": List[Tensor],
    "low_rank_A": List[Tensor],
    "low_rank_B": List[Tensor],
    "head_modulation": Tensor,
}
```

#### Dressable model 接口

`DressableGaussianModel.render(...)` 输入：

```python
{
    "camera": CameraObject,
    "pose": Tensor[P],
    "cloth_id": Tensor[B],
}
```

输出：

```python
{
    "rgb": Tensor[B, 3, H, W],
    "alpha": Tensor[B, 1, H, W],
    "anchor_offsets": Dict[str, Tensor],
    "dressed_scale": Tensor[N, 3],
    "region_weight": Tensor[N, 1],
}
```

### 13.4 PR 提交流程

每个功能分支合并前必须提交 PR 到 `dev-dressable`。PR 描述必须包含：

```text
模块:
  data / base_gs / lhm_prior / alignment / hypernetwork / training / evaluation

改动文件:
  ...

输入:
  ...

输出:
  ...

运行命令:
  ...

验证结果:
  ...

是否改动公共接口:
  是 / 否
```

负责人只合并满足以下条件的 PR：

```text
1. 能 import，不破坏已有代码。
2. 有最小运行命令。
3. 有明确输出文件或截图。
4. 没有提交大文件。
5. 没有私自改公共接口字段。
6. 与当前 dev-dressable 合并后能跑 train_dressable.py 一个 batch。
```

### 13.5 Commit 规范

Commit message 使用：

```text
[module] action summary
```

示例：

```text
[dataset] add dressable json loader
[lhm] parse gaussian ply prior
[alignment] build anchor offset target
[model] add clothing hypernetwork modulation
[train] connect lhm anchor loss
[eval] export cloth switching grid
[docs] update daily progress
```

### 13.6 大文件管理

以下文件不提交 Git：

```text
data_dressable/
outputs/
checkpoints/
*.pth
*.pt
*.ply
*.mp4
大量 *.png / *.jpg 渲染结果
```

Git 中只提交：

```text
代码
配置
小型 JSON 索引样例
Markdown 文档
少量必要示意图
```

大文件统一放在共享目录：

```text
shared_project/
  data_dressable/
  lhm_priors/
  checkpoints/
  eval_outputs/
```

每次实验在文档中记录路径：

```text
实验名称:
checkpoint:
metrics:
visualization:
commit:
config:
```

### 13.7 每日集成规则

每天晚上组会前，负责人执行一次集成：

```text
1. 从各组员 feature 分支合并可用 PR 到 dev-dressable。
2. 检查 configs/canon_dress_gs.yaml 路径和参数。
3. 运行 train_dressable.py 一个 batch。
4. 运行 test_dressable.py 一个小样本。
5. 保存当天最好结果图。
6. 更新每日进度表。
```

每日集成最低验收：

```text
dev-dressable 可以 import。
DressableDataset 可以返回一个 batch。
DressableGaussianModel 可以 forward。
loss 可以 backward。
能保存一张 rendered image。
```

### 13.8 冲突处理规则

冲突处理优先级：

```text
1. 公共接口以本文档为准。
2. 配置以 configs/canon_dress_gs.yaml 为准。
3. 训练入口以 train_dressable.py 为准。
4. 原 MMLPHuman 文件优先保持不动。
5. 冲突超过 15 分钟无法解决，交给负责人裁决。
```

### 13.9 组员每日汇报模板

每位组员每天发一段：

```text
姓名:
分支:
今天改动:
新增 / 修改文件:
运行命令:
输出路径:
截图 / 指标:
阻塞问题:
明天计划:
是否需要负责人合并:
```

---

## 14. 实验设计

### 14.1 主实验

训练并评估最终模型：

```text
输入:
  cloth_id
  SMPL-X pose
  camera

输出:
  rendered dressed avatar image

监督:
  GPT-image-2 同姿态多服装 RGB
  foreground mask
  LHM-derived anchor offset target
```

核心测试：

```text
同一身份 + 同一姿态 + 不同服装
同一身份 + 同一服装 + novel pose
同一身份 + 同一服装 + novel view
同一身份 + 不同服装 + novel pose
```

### 14.2 对照实验

为了论文完整性，保留以下对照实验：

```text
MMLPHuman single-cloth:
  每套服装单独训练一个 MMLPHuman avatar，用作单服装质量上限。

No LHM supervision:
  移除 LHM anchor offset loss，只用 RGB / mask 训练，用于证明 LHM teacher 的贡献。

No HyperNetwork modulation:
  移除 clothing-conditioned modulation，只保留共享 offset MLP，用于证明服装条件化调制的贡献。

No anchor alignment supervision:
  移除 anchor-level offset target，只用图像监督，用于证明 alignment teacher target 的贡献。
```

### 14.3 指标

图像指标：

```text
PSNR
SSIM
LPIPS
```

Mask 指标：

```text
Mask IoU
Alpha BCE
```

几何 / offset 指标：

```text
Anchor offset L1
Offset smoothness
Non-clothing region offset magnitude
```

效率指标：

```text
FPS
训练显存
模型参数量
每新增一套服装增加的参数量
```

### 14.4 可视化

必须生成：

```text
1. Pipeline figure
2. canonical base_gs
3. LHM prior
4. anchor offset heatmap
5. predicted canonical dressed Gaussians
6. same-pose cloth switching grid
7. novel pose rendering sequence
8. novel view rendering sequence
9. failure cases
```

---

## 15. 时间计划

以 2026 年 7 月 28 日 AAAI27 deadline 为最终提交目标；以 2026 年 7 月 18 日为论文初稿节点。7 月 18 日前必须形成一版可以发给老师讨论的论文初稿，代码和实验同步推进。

### 7 月 11 日 - 7 月 12 日

交付：

```text
MMLPHuman baseline 跑通
data_dressable 目录搭好
train / val / test JSON 初版完成
base_gs 初始化脚本跑通
论文标题、摘要、introduction 初稿完成
pipeline figure 草图完成
related work 文献列表完成
```

### 7 月 13 日 - 7 月 15 日

交付：

```text
每套服装 LHM prior 生成完成
LHM prior 三视角质量检查完成
base_gs 多姿态渲染检查完成
method section 初稿完成
data construction section 初稿完成
pipeline figure 第一版完成
实验设置表格框架完成
```

### 7 月 16 日 - 7 月 18 日

交付：

```text
SMPL-X surface-aware anchor alignment 完成
每套服装 anchor_offsets.pt 完成
offset heatmap 完成
DressableDataset 可以返回训练 batch
论文完整初稿 v1 完成
abstract / introduction / related work / method / experiments / limitations 均有内容
主结果表和消融表先用占位表格填好
所有计划放图位置用草图或当前中间结果占位
7 月 18 日晚上发给老师审阅
```

### 7 月 19 日 - 7 月 21 日

交付：

```text
ClothingEmbedding 完成
ClothingHyperNetwork 完成
DressableGaussianModel 完成
train_dressable.py 可以训练
第一批 dressed avatar 渲染结果完成
根据老师意见修改论文 v2
把第一批渲染结果替换进 qualitative figure
```

### 7 月 22 日 - 7 月 24 日

交付：

```text
最终模型主实验完成
对照实验完成
metrics.csv 完成
cloth switching / novel pose / novel view 图和视频完成
quantitative table 替换真实结果
ablation table 替换真实结果
实验分析文字完成
```

### 7 月 25 日 - 7 月 27 日

交付：

```text
论文 v3 完成
pipeline figure 精修完成
method section 精修完成
experiment section 精修完成
所有表格和可视化整理完成
limitations 完成
supplementary material 初稿完成
```

### 7 月 28 日

交付：

```text
匿名检查
引用检查
格式检查
最终 PDF
补充材料
提交
```

---

## 16. 每日验收清单

每天晚上组会前更新：

```text
1. 今天新增了哪些文件
2. 今天产出了哪些中间结果
3. 哪些模块已经能独立运行
4. 哪些模块已经接入 train_dressable.py
5. 当前最好的一组渲染结果
6. 当前最严重的失败案例
7. 明天要合并的代码
8. 每个组员当前分支名
9. 已提交 PR 列表
10. 已合并到 dev-dressable 的模块
11. train_dressable.py 一个 batch 是否通过
12. test_dressable.py 小样本是否通过
13. 今天更新了论文哪个 section
14. 今天新增或替换了哪张论文图
15. 7 月 18 日初稿缺口还剩哪些
```

---

## 17. 最终交付物

代码：

```text
scene/dressable_dataset.py
scene/clothing_embedding.py
scene/clothing_hypernetwork.py
scene/dressable_gaussian_model.py
utils/lhm_prior_utils.py
utils/gaussian_alignment.py
train_dressable.py
test_dressable.py
configs/canon_dress_gs.yaml
```

数据：

```text
data_dressable/identity_000/splits/*.json
data_dressable/identity_000/clothes/*/lhm_prior/gaussians.ply
data_dressable/identity_000/clothes/*/offset_target/anchor_offsets.pt
```

模型：

```text
outputs/base_naked_gaussian/checkpoint_latest.pth
outputs/canon_dress_gs/checkpoint_latest.pth
```

实验：

```text
outputs/eval/metrics.csv
outputs/eval/cloth_switching_grid.png
outputs/eval/novel_pose_video.mp4
outputs/eval/novel_view_video.mp4
outputs/eval/offset_heatmaps/
```

论文材料：

```text
pipeline figure
method diagram
quantitative table
qualitative comparison
ablation table
failure case figure
```

---

## 18. 附录 A：工程运行命令手册

本节给出组员本地运行和负责人每日集成时使用的统一命令。具体脚本文件在实现阶段按本文档命名创建。

### 18.1 环境检查

```powershell
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import numpy, cv2; print('basic deps ok')"
```

验收：

```text
Python 可用。
PyTorch 可 import。
CUDA 可用。
基础依赖可 import。
```

### 18.2 创建分支

负责人初始化：

```powershell
git checkout main
git pull
git checkout -b dev-dressable
git push -u origin dev-dressable
```

组员创建个人分支：

```powershell
git checkout dev-dressable
git pull
git checkout -b feature/data-base-gs
```

其他组员替换分支名：

```text
feature/lhm-prior
feature/anchor-alignment
feature/hypernetwork-offset
feature/integration-train
```

### 18.3 数据索引生成

组员 A 运行：

```powershell
python tools/build_dressable_index.py `
  --data_root data_dressable/identity_000 `
  --out_dir data_dressable/identity_000/splits `
  --num_train 240 `
  --num_val 30 `
  --num_test 30
```

输出：

```text
data_dressable/identity_000/splits/train.json
data_dressable/identity_000/splits/val.json
data_dressable/identity_000/splits/test_novel_pose.json
data_dressable/identity_000/splits/test_novel_view.json
data_dressable/identity_000/splits/test_same_pose_cloth_switch.json
```

快速检查：

```powershell
python tools/check_dressable_index.py `
  --split data_dressable/identity_000/splits/train.json
```

### 18.4 Base_gs 初始化

组员 A 运行：

```powershell
python tools/init_base_gaussians.py `
  --smplx_mesh data_dressable/identity_000/smplx/canonical_mesh.obj `
  --smplx_params data_dressable/identity_000/smplx/canonical_params.json `
  --lbs_weights data_dressable/identity_000/smplx/lbs_weights.npy `
  --out_dir outputs/base_naked_gaussian
```

输出：

```text
outputs/base_naked_gaussian/checkpoint_latest.pth
outputs/base_naked_gaussian/base_gs.ply
outputs/base_naked_gaussian/render_check/
```

渲染检查：

```powershell
python tools/check_base_render.py `
  --base_checkpoint outputs/base_naked_gaussian/checkpoint_latest.pth `
  --split data_dressable/identity_000/splits/val.json `
  --out_dir outputs/base_naked_gaussian/render_check
```

### 18.5 LHM prior 生成

组员 B 对每套服装运行：

```powershell
python tools/run_lhm_prior.py `
  --reference data_dressable/identity_000/clothes/cloth_000/reference.png `
  --out_dir data_dressable/identity_000/clothes/cloth_000/lhm_prior
```

输出：

```text
data_dressable/identity_000/clothes/cloth_000/lhm_prior/gaussians.ply
data_dressable/identity_000/clothes/cloth_000/lhm_prior/metadata.json
```

质量检查：

```powershell
python tools/render_lhm_prior_views.py `
  --ply data_dressable/identity_000/clothes/cloth_000/lhm_prior/gaussians.ply `
  --out_dir outputs/lhm_prior_check/cloth_000
```

### 18.6 Anchor offset target 生成

组员 C 运行：

```powershell
python tools/build_anchor_offsets.py `
  --base_checkpoint outputs/base_naked_gaussian/checkpoint_latest.pth `
  --lhm_prior data_dressable/identity_000/clothes/cloth_000/lhm_prior/gaussians.ply `
  --smplx_mesh data_dressable/identity_000/smplx/canonical_mesh.obj `
  --out_path data_dressable/identity_000/clothes/cloth_000/offset_target/anchor_offsets.pt
```

可视化：

```powershell
python tools/visualize_anchor_offsets.py `
  --anchor_offsets data_dressable/identity_000/clothes/cloth_000/offset_target/anchor_offsets.pt `
  --out_dir outputs/offset_heatmaps/cloth_000
```

### 18.7 Dataset batch 检查

负责人或组员 A 运行：

```powershell
python tools/check_dressable_batch.py `
  --config configs/canon_dress_gs.yaml `
  --split train `
  --num_batches 2
```

必须输出：

```text
condition_image shape
target_rgb shape
mask shape
pose shape
camera fields
cloth_id
anchor_offset_target keys
```

### 18.8 训练一个 batch

负责人运行每日集成检查：

```powershell
python train_dressable.py `
  --config configs/canon_dress_gs.yaml `
  --debug_one_batch `
  --out_dir outputs/debug_one_batch
```

验收：

```text
model forward 成功
loss 正常计算
backward 成功
保存 debug render
保存 loss log
```

### 18.9 正式训练

负责人运行：

```powershell
python train_dressable.py `
  --config configs/canon_dress_gs.yaml `
  --out_dir outputs/canon_dress_gs
```

输出：

```text
outputs/canon_dress_gs/checkpoint_latest.pth
outputs/canon_dress_gs/train_log.json
outputs/canon_dress_gs/vis/
```

### 18.10 测试与评估

组员 D 运行，负责人验收结果：

```powershell
python test_dressable.py `
  --config configs/canon_dress_gs.yaml `
  --checkpoint outputs/canon_dress_gs/checkpoint_latest.pth `
  --split test_same_pose_cloth_switch `
  --out_dir outputs/eval/cloth_switch
```

Novel pose：

```powershell
python test_dressable.py `
  --config configs/canon_dress_gs.yaml `
  --checkpoint outputs/canon_dress_gs/checkpoint_latest.pth `
  --split test_novel_pose `
  --out_dir outputs/eval/novel_pose
```

Novel view：

```powershell
python test_dressable.py `
  --config configs/canon_dress_gs.yaml `
  --checkpoint outputs/canon_dress_gs/checkpoint_latest.pth `
  --split test_novel_view `
  --out_dir outputs/eval/novel_view
```

指标汇总：

```powershell
python tools/collect_metrics.py `
  --eval_root outputs/eval `
  --out_csv outputs/eval/metrics.csv
```

---

## 19. 附录 B：核心接口与 Tensor Shape 规范

本节固定所有模块的输入输出 shape。除负责人统一修改外，组员不能私自改字段名和 shape 约定。

### 19.1 全局符号

```text
B: batch size
N: Gaussian 数量
A: anchor 数量
H: image height
W: image width
P: SMPL-X pose 向量维度
C_sh: SH feature 维度
D_c: clothing embedding 维度，固定 64
R: low-rank adapter rank，第一版固定 8
```

第一版固定：

```text
D_c = 64
R = 8
rotation = quaternion, shape [..., 4]
cloth_id = int64 tensor
image range = [0, 1]
mask range = {0, 1}
```

### 19.2 DressableDataset 输出

`DressableDataset.__getitem__` 返回：

```python
{
    "identity_id": str,
    "cloth_id": Tensor[],                 # int64 scalar
    "frame_id": str,
    "condition_image": Tensor[3, H, W],    # float32, [0, 1]
    "target_rgb": Tensor[3, H, W],         # float32, [0, 1]
    "foreground_mask": Tensor[1, H, W],    # float32, 0/1
    "cloth_mask": Tensor[1, H, W],         # float32, 0/1
    "pose": Tensor[P],                    # float32
    "camera": CameraObject,
    "anchor_offset_target": Dict[str, Tensor],
}
```

DataLoader collate 后：

```python
{
    "cloth_id": Tensor[B],
    "condition_image": Tensor[B, 3, H, W],
    "target_rgb": Tensor[B, 3, H, W],
    "foreground_mask": Tensor[B, 1, H, W],
    "cloth_mask": Tensor[B, 1, H, W],
    "pose": Tensor[B, P],
    "camera": List[CameraObject],
    "anchor_offset_target": Dict[str, Tensor[B, ...]],
}
```

### 19.3 Base Gaussian 参数

`base_gs` 中核心参数：

```python
base_xyz: Tensor[N, 3]
base_scale: Tensor[N, 3]
base_rotation: Tensor[N, 4]
base_opacity: Tensor[N, 1]
base_sh: Tensor[N, C_sh]
base_lbs_weights: Tensor[N, J]
base_anchor_indices: Tensor[N, K]
base_anchor_weights: Tensor[N, K]
```

说明：

```text
J: SMPL-X joint 数量
K: 每个 Gaussian 关联的 nearest anchors 数量
base_rotation 必须归一化为 quaternion
base_scale 存储方式与 MMLPHuman 原 GaussianModel 保持一致
```

### 19.4 LHM prior 参数

`load_lhm_gaussians` 输出：

```python
{
    "xyz": Tensor[M, 3],
    "scale": Tensor[M, 3],
    "rotation": Tensor[M, 4],
    "opacity": Tensor[M, 1],
    "sh": Tensor[M, C_sh],
}
```

其中：

```text
M: LHM 输出 Gaussian 数量
所有字段 canonicalize 后必须处于 SMPL-X canonical 坐标系
rotation 使用 quaternion
```

### 19.5 Anchor offset target

`anchor_offsets.pt` 固定为：

```python
{
    "cloth_id": str,
    "anchor_xyz": Tensor[A, 3],
    "delta_xyz": Tensor[A, 3],
    "delta_scale": Tensor[A, 3],
    "delta_rotation": Tensor[A, 4],
    "delta_opacity": Tensor[A, 1],
    "delta_sh": Tensor[A, C_sh],
    "valid_mask": Tensor[A, 1],
    "cloth_region_weight": Tensor[A, 1],
}
```

Batch 后：

```python
delta_xyz: Tensor[B, A, 3]
delta_scale: Tensor[B, A, 3]
delta_rotation: Tensor[B, A, 4]
delta_opacity: Tensor[B, A, 1]
delta_sh: Tensor[B, A, C_sh]
valid_mask: Tensor[B, A, 1]
cloth_region_weight: Tensor[B, A, 1]
```

### 19.6 ClothingEmbedding

输入：

```python
cloth_id: Tensor[B]  # int64
```

输出：

```python
z_c: Tensor[B, 64]
```

### 19.7 ClothingHyperNetwork

输入：

```python
z_c: Tensor[B, 64]
```

输出：

```python
{
    "film_gamma": List[Tensor[B, H_l]],
    "film_beta": List[Tensor[B, H_l]],
    "low_rank_A": List[Tensor[B, H_l, R]],
    "low_rank_B": List[Tensor[B, R, H_l]],
    "head_modulation": Tensor[B, D_head],
}
```

其中：

```text
H_l: 第 l 层 hidden dimension
R = 8
D_head: coefficient head modulation 维度
```

### 19.8 Clothing offset 输出

Anchor-level 输出：

```python
pred_anchor_offsets = {
    "delta_xyz": Tensor[B, A, 3],
    "delta_scale": Tensor[B, A, 3],
    "delta_rotation": Tensor[B, A, 4],
    "delta_opacity": Tensor[B, A, 1],
    "delta_sh": Tensor[B, A, C_sh],
}
```

插值到 Gaussian-level：

```python
pred_gaussian_offsets = {
    "delta_xyz": Tensor[B, N, 3],
    "delta_scale": Tensor[B, N, 3],
    "delta_rotation": Tensor[B, N, 4],
    "delta_opacity": Tensor[B, N, 1],
    "delta_sh": Tensor[B, N, C_sh],
}
```

### 19.9 Dressed Gaussian 参数

合成后：

```python
dressed_xyz: Tensor[B, N, 3]
dressed_scale: Tensor[B, N, 3]
dressed_rotation: Tensor[B, N, 4]
dressed_opacity: Tensor[B, N, 1]
dressed_sh: Tensor[B, N, C_sh]
```

### 19.10 DressableGaussianModel.render 输出

```python
{
    "rgb": Tensor[B, 3, H, W],
    "alpha": Tensor[B, 1, H, W],
    "anchor_offsets": Dict[str, Tensor[B, A, ...]],
    "gaussian_offsets": Dict[str, Tensor[B, N, ...]],
    "dressed_scale": Tensor[B, N, 3],
    "region_weight": Tensor[B, N, 1],
}
```

### 19.11 Loss 输入约定

```python
rgb_loss:
  pred: Tensor[B, 3, H, W]
  target: Tensor[B, 3, H, W]

mask_loss:
  pred_alpha: Tensor[B, 1, H, W]
  target_mask: Tensor[B, 1, H, W]

anchor_offset_loss:
  pred_anchor_offsets: Dict[str, Tensor[B, A, ...]]
  target_anchor_offsets: Dict[str, Tensor[B, A, ...]]
  valid_mask: Tensor[B, A, 1]
  cloth_region_weight: Tensor[B, A, 1]

anchor_smooth_loss:
  pred_delta_xyz: Tensor[B, A, 3]
  anchor_graph_edges: Tensor[E, 2]

noncloth_region_loss:
  pred_gaussian_offsets: Dict[str, Tensor[B, N, ...]]
  region_weight: Tensor[B, N, 1]
```

### 19.12 Shape 检查脚本

新增：

```text
tools/check_tensor_shapes.py
```

运行：

```powershell
python tools/check_tensor_shapes.py `
  --config configs/canon_dress_gs.yaml `
  --split train
```

检查项：

```text
Dataset batch shape
anchor_offsets.pt shape
ClothingEmbedding output shape
ClothingHyperNetwork output shape
DressableGaussianModel render output shape
loss input shape
```

---

## 20. 附录 C：7 月 18 日论文初稿目录模板

7 月 18 日前提交给老师的论文初稿必须使用以下结构。没有最终实验结果的位置先放占位表格和当前中间结果图。

### 20.1 Title

候选标题：

```text
CanonDressGS: Canonical Clothing-conditioned Gaussian Avatars with LHM-assisted Priors
```

标题需要体现：

```text
LHM-assisted
clothing-conditioned
canonical Gaussian avatar
pose-driven rendering
```

### 20.2 Abstract

必须回答：

```text
1. 问题是什么：同一身份下可换装、可姿态驱动的 Gaussian avatar。
2. 难点是什么：服装几何变化、服装与人体 canonical 对齐、多服装参数共享、姿态驱动稳定性。
3. 方法是什么：base_gs + LHM teacher prior + anchor alignment + HyperNetwork offsets + MMLPHuman rendering。
4. 结果是什么：支持同姿态多服装切换、novel pose、novel view。
5. 贡献是什么：canonical clothing offset representation、LHM-assisted supervision、lightweight clothing-conditioned HyperNetwork。
```

### 20.3 Introduction

段落结构：

```text
P1: 数字人 / Gaussian avatar 背景。
P2: 现有 pose-driven avatar 多数针对固定服装，难以换装。
P3: 换装的核心挑战：服装几何、canonical 对齐、参数量、姿态驱动一致性。
P4: 我们的关键想法：先在 canonical space 换衣，再复用 MMLPHuman 做动作。
P5: 我们的数据条件：THuman / SMPL-X 白模 condition + GPT-image-2 同姿态多服装图。
P6: 我们的方法概述。
P7: Contributions。
```

Contributions 写法：

```text
1. We introduce a clothing-conditioned canonical Gaussian offset representation for dressable Gaussian avatars.
2. We propose an LHM-assisted anchor-level supervision strategy to transfer clothed Gaussian priors to a naked canonical base avatar.
3. We design a lightweight clothing-conditioned HyperNetwork that modulates shared MMLPHuman-style MLPs for multi-outfit Gaussian offset prediction.
```

### 20.4 Related Work

小节：

```text
3D Gaussian Human Avatars
Animatable Human Reconstruction
Dressable / Controllable Human Generation
HyperNetworks and Conditional Modulation
```

必须引用：

```text
3D Gaussian Splatting
MMLPHuman
LHM
Animatable Gaussians
D3GA / GART
HyperNetworks
FiLM
LoRA
PyTorch3D
```

### 20.5 Method

Method 章节结构固定：

```text
4.1 Overview
4.2 White-body Pose-condition Dataset Construction
4.3 Base Naked Gaussian Avatar
4.4 LHM-assisted Clothed Gaussian Prior
4.5 SMPL-X Surface-aware Anchor Alignment
4.6 Clothing-conditioned HyperNetwork
4.7 Canonical Dressed Gaussian Composition
4.8 Pose-driven Rendering with MMLPHuman
4.9 Training Objectives
```

每节必须写清：

```text
输入是什么
输出是什么
核心公式是什么
对应代码模块是什么
该模块解决什么问题
```

### 20.6 Experiments

Experiments 章节结构固定：

```text
5.1 Dataset and Protocol
5.2 Baselines
5.3 Metrics
5.4 Implementation Details
5.5 Quantitative Results
5.6 Qualitative Results
5.7 Ablation Study
5.8 Runtime and Parameter Analysis
```

表格占位：

```text
Table 1: Main comparison on same-cloth novel pose / novel view.
Table 2: Multi-cloth shared avatar comparison.
Table 3: Ablation study.
Table 4: Runtime and parameter count.
```

图占位：

```text
Figure 1: Pipeline.
Figure 2: Data construction.
Figure 3: LHM prior and anchor offset target.
Figure 4: Same-pose cloth switching.
Figure 5: Novel pose rendering.
Figure 6: Ablations / failure cases.
```

### 20.7 Limitations

必须写：

```text
1. 第一版主要支持同一 identity 的多服装切换。
2. 未见服装泛化依赖后续 image-conditioned clothing embedding。
3. LHM prior 质量会影响 anchor offset target。
4. 宽松衣服、裙摆、大外套仍可能存在几何偏差。
5. GPT-image-2 生成监督图可能与白模 condition 存在局部不一致。
```

### 20.8 7 月 18 日初稿文件结构

```text
paper_draft/
  main.tex
  sections/
    00_abstract.tex
    01_introduction.tex
    02_related_work.tex
    03_method.tex
    04_experiments.tex
    05_limitations.tex
  figures/
    fig1_pipeline.pdf
    fig2_data_construction.pdf
    fig3_lhm_anchor_offsets.png
    fig4_cloth_switching_placeholder.png
  tables/
    table1_main_results.tex
    table2_ablation.tex
```

### 20.9 7 月 18 日初稿验收

```text
main.tex 可以编译成 PDF。
PDF 中没有空 section。
Pipeline figure 已放入。
每个表格都有表头和占位数据。
每个计划放图位置都有占位图。
Method 章节完整描述最终 pipeline。
Experiments 章节完整描述实验协议。
Limitations 已写。
```

---

## 21. 最终一句话

本项目最终路线是：

```text
用 THuman / SMPL-X 白模构建稳定裸模 base_gs，
用 LHM 离线提供每套服装的 dressed Gaussian teacher prior，
通过 SMPL-X surface-aware anchor alignment 得到服装 offset 监督，
用 clothing embedding 驱动 lightweight HyperNetwork 生成 FiLM / low-rank modulation，
预测 canonical clothing Gaussian offsets，
合成 canonical dressed Gaussians，
最后完全复用 MMLPHuman 的 pose-driven deformation 和 Gaussian rendering。
```
