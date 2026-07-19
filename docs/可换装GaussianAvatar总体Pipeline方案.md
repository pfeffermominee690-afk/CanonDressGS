# Dressable Gaussian Avatar Pipeline 方案整理

本文档只整理我们的目标 pipeline 方案，不展开完整代码阅读和实验排期。

目标是在 MMLPHuman / Animatable Gaussian Avatar 的基础上，从：

```text
一个人 + 一套固定服装 + pose-driven Gaussian Avatar
```

扩展为：

```text
同一身份 + 多套服装 + 可换装 + 可 pose 驱动的 Gaussian Avatar
```

核心表示为：

```text
body Gaussians + clothing Gaussians
```

其中：

```text
body Gaussians:
  表示稳定身份、脸、头发、手、裸露身体区域、身份相关外观。

clothing Gaussians:
  表示某套衣服的几何、纹理、褶皱、姿态相关变化。
```

## 1. 总体 Pipeline

完整 pipeline 可以描述为：

```text
输入:
  同一 identity 的多套服装多视角视频
  每帧相机参数
  每帧 SMPL-X pose
  人体前景 mask
  服装 mask / body mask
  每套衣服的 clothing condition

        ↓

数据预处理:
  前景提取
  SMPL-X registration
  相机标定整理
  LBS weight volume 生成
  canonical template 构建
  clothing condition 标记
  body / clothing 区域 mask 整理

        ↓

Canonical Gaussian 初始化:
  初始化共享 body Gaussians
  初始化 clothing Gaussians
  初始化 body anchor points / control points
  初始化 clothing anchor points / control points

        ↓

Pose-driven Body Branch:
  输入当前 SMPL-X pose theta
  body spatially distributed MLPs 输出 body basis coefficients
  通过 Gaussian offset basis 得到 body Gaussian 的姿态相关属性变化
  通过 control points 得到 body Gaussian 的位置 offset

        ↓

Cloth-conditioned Clothing Branch:
  输入当前 SMPL-X pose theta 和 clothing condition z_c
  clothing spatially distributed MLPs 输出 clothing basis coefficients
  通过 clothing Gaussian offset basis 得到衣服的姿态相关几何、纹理、透明度、尺度、旋转变化
  通过 clothing control points 约束衣服表面运动

        ↓

Pose Deformation:
  body Gaussians 使用 SMPL-X / LBS 变形到 posed space
  clothing Gaussians 通过 body LBS field 或 clothing-specific deformation 变形到 posed space

        ↓

Composition & Rendering:
  选择一个 clothing condition
  合并 body Gaussians 和 clothing Gaussians
  使用一次 Gaussian rasterization 渲染

        ↓

输出:
  同一身份
  指定服装
  指定姿态
  指定视角
  的 dressed avatar 图像
```

论文中可以概括为：

```text
We decompose a pose-driven Gaussian human avatar into identity-preserving body Gaussians and cloth-conditioned garment Gaussians. Given a target pose and a clothing condition, both branches predict pose-dependent Gaussian offsets and are jointly rasterized to render a dressable avatar.
```

## 2. Pipeline 总览图

```text
Multi-outfit multi-view videos
        |
        v
SMPL-X pose / camera / human mask / clothing mask
        |
        v
Canonical decomposition
        |
        +---------------------------+
        |                           |
        v                           v
Body Gaussians              Clothing Gaussians
        |                           |
        v                           v
Body distributed MLPs       Clothing distributed MLPs
input: pose theta           input: pose theta + cloth code z_c
        |                           |
        v                           v
Body basis coefficients     Clothing basis coefficients
        |                           |
        v                           v
Body Gaussian offsets       Clothing Gaussian offsets
        |                           |
        +-------------+-------------+
                      |
                      v
             LBS / pose deformation
                      |
                      v
        Joint Gaussian rasterization
                      |
                      v
       Dressable pose-driven avatar rendering
```

## 3. 与 MMLPHuman Pipeline 的关系

MMLPHuman 原 pipeline：

```text
multi-view video + SMPL-X pose
-> one canonical Gaussian avatar
-> spatially distributed MLPs
-> Gaussian offset basis
-> control point interpolation
-> LBS deformation
-> rasterization
```

我们的 pipeline：

```text
multi-outfit multi-view video + SMPL-X pose + clothing condition
-> body Gaussian avatar + clothing Gaussian avatar
-> body distributed MLPs + clothing distributed MLPs
-> body basis + clothing basis
-> body / clothing control point interpolation
-> body / clothing deformation
-> joint rasterization
```

所以核心扩展点是：

```text
1. 从单一 Gaussian set 扩展为 body / clothing 两套可组合 Gaussian set。
2. 从 pose-conditioned avatar 扩展为 pose + clothing conditioned avatar。
3. 从单服装重建扩展为多服装可切换组合渲染。
```

## 4. 模块 1：输入数据

### 4.1 可直接套用 MMLPHuman 的部分

MMLPHuman 已经支持：

```text
多视角图像 / 视频
相机内参 K
相机外参 w2c
人体前景 mask
SMPL-X pose
SMPL-X beta
Rh / Th
```

可直接参考的 pipeline 设计：

```text
每个训练样本 = 一帧 + 一个相机视角 + 一个 pose
```

即：

```text
sample = {
  image,
  mask,
  camera,
  pose,
  frame_id,
  cam_id
}
```

### 4.2 我们需要新增的部分

我们的训练样本还需要：

```text
cloth_id
cloth_name
body_mask
clothing_mask
optional: garment_part_mask
```

扩展后的 sample：

```text
sample = {
  image,
  human_mask,
  body_mask,
  clothing_mask,
  camera,
  pose,
  frame_id,
  cam_id,
  cloth_id,
  cloth_name
}
```

### 4.3 没有现成 pipeline 的部分：可选方案

#### 方案 A：已有服装 mask

```text
数据集中已经提供 clothing mask / parsing mask。
```

优点：

```text
实现最简单，监督最可靠。
```

缺点：

```text
依赖数据质量，不一定每个数据集都有。
```

适合：

```text
最优先选择。
```

#### 方案 B：离线人体解析模型生成 mask

```text
使用 human parsing / segmentation 模型离线生成 body_mask 和 clothing_mask。
```

优点：

```text
适用于大多数 RGB 图像数据。
```

缺点：

```text
mask 可能有错误，边界不稳定。
需要额外模型和清洗流程。
```

适合：

```text
没有现成 clothing mask，但需要快速做实验。
```

#### 方案 C：只用 human mask，弱监督分离 body / clothing

```text
不显式提供 clothing mask，让模型通过多服装条件自动分离。
```

优点：

```text
数据准备最轻。
```

缺点：

```text
很难稳定解耦。
容易出现 body 和 clothing 互相解释。
投稿风险高。
```

适合：

```text
作为 ablation 或 future direction，不建议作为主方案。
```

推荐：

```text
主方案使用 A 或 B。
```

## 5. 模块 2：Canonical Template 与 Gaussian 初始化

### 5.1 可直接套用 MMLPHuman 的部分

MMLPHuman 做法：

```text
读取或生成 template.ply
在 template mesh 上采样 Gaussian points
在 template mesh 上采样 anchor points
在 template mesh 上采样 control points
```

对应概念：

```text
Gaussian points:
  avatar 的基本几何和外观载体。

anchor points:
  spatially distributed MLPs 的位置。

control points:
  约束位置 offset，让 Gaussians 保持在表面层附近。
```

这套机制可以直接保留。

### 5.2 我们需要新增的部分

我们不再只有一套 template / Gaussian set，而是至少有：

```text
body template / body region
clothing template / clothing region
```

需要初始化：

```text
body Gaussian points
body anchor points
body control points

clothing Gaussian points
clothing anchor points
clothing control points
```

### 5.3 没有现成 pipeline 的部分：可选方案

#### 方案 A：Body 和 clothing 都从同一个 SMPL-X / template mesh 采样

做法：

```text
使用人体 template mesh。
body Gaussians 在 body region 采样。
clothing Gaussians 在 clothing-covered body region 采样。
```

优点：

```text
实现最快。
可以复用 MMLPHuman 的 template、LBS、采样逻辑。
适合 MVP。
```

缺点：

```text
宽松衣服、裙子、外套下摆表达弱。
clothing 初始点贴近身体，后续需要 offset 学出衣服厚度。
```

适合：

```text
T-shirt、裤子、紧身或中等宽松衣服。
第一版实验。
```

#### 方案 B：每套服装单独构建 clothing template

做法：

```text
为每套服装重建或准备 garment template.ply。
clothing Gaussians 在 garment template 上采样。
```

优点：

```text
服装几何初始化更准确。
适合宽松衣服。
结果可能更好。
```

缺点：

```text
预处理复杂。
每套衣服需要单独 template。
template 对齐和拓扑不一致会增加工程难度。
```

适合：

```text
如果数据里有高质量服装模板或可以稳定重建。
```

#### 方案 C：从多视角 mask / depth / point cloud 估计 clothing surface

做法：

```text
使用多视角图像和 clothing mask 融合出 clothing point cloud。
在 point cloud 或重建 mesh 上初始化 clothing Gaussians。
```

优点：

```text
不依赖现成 garment template。
能更贴近真实衣服外表面。
```

缺点：

```text
实现和调参成本最高。
点云噪声会影响训练。
```

适合：

```text
第二阶段增强，不建议第一版就做。
```

推荐：

```text
第一版使用方案 A。
如果结果明显受限，再尝试方案 B 或 C。
```

## 6. 模块 3：Body Gaussian Branch

### 6.1 可直接套用 MMLPHuman 的部分

body branch 基本可以直接使用 MMLPHuman pipeline：

```text
body Gaussians
body anchor MLPs
body Gaussian offset basis
body control points
body LBS deformation
```

MMLPHuman 中：

```text
MLP input:
  pose theta

MLP output:
  basis coefficients

Gaussian property:
  neutral property + coefficient-weighted basis offsets
```

body branch 可以保持：

```text
w_body^j = E_body^j(theta)
```

### 6.2 我们需要新增的部分

body branch 需要跨服装共享：

```text
同一个 identity 的所有服装序列共用 body Gaussians。
```

也就是说：

```text
cloth_0, cloth_1, cloth_2
都使用同一套 G_body。
```

### 6.3 没有现成 pipeline 的部分：可选方案

#### 方案 A：Shared body branch

做法：

```text
所有服装共用一套 body Gaussians 和 body MLPs。
```

优点：

```text
最符合 identity disentanglement。
论文表达清晰。
```

缺点：

```text
不同服装遮挡身体区域不同，body 可见监督不完整。
```

适合：

```text
主方案。
```

#### 方案 B：Shared body base + per-cloth residual

做法：

```text
共享 body base。
每套衣服有一个很小的 residual body correction。
```

优点：

```text
可以吸收不同服装造成的局部遮挡、阴影、边界误差。
训练更稳。
```

缺点：

```text
解耦不够纯粹。
可能让服装信息泄漏到 body branch。
```

适合：

```text
如果纯 shared body 训练困难，可作为工程折中。
```

#### 方案 C：只建 visible body，不建完整 underlying body

做法：

```text
body branch 只负责脸、手、头发、裸露皮肤等稳定可见区域。
被衣服覆盖的身体不显式建模。
```

优点：

```text
监督最直接，避免不可见身体区域的不确定性。
```

缺点：

```text
不是完整人体 body。
换装时如果衣服改变露肤区域，可能缺信息。
```

适合：

```text
第一版数据中服装露肤区域变化不大时。
```

推荐：

```text
主方案采用 A。
工程上可加入 B 作为稳定训练的 fallback。
```

## 7. 模块 4：Clothing Gaussian Branch

### 7.1 可直接套用 MMLPHuman 的部分

clothing branch 可以复用 MMLPHuman 的基本表达：

```text
clothing Gaussians
clothing anchor points
clothing distributed MLPs
clothing Gaussian offset basis
clothing control points
position-based interpolation
```

也就是说，衣服也可以像人体一样：

```text
neutral clothing properties
+ pose-dependent clothing offsets
```

### 7.2 我们需要新增的部分

clothing branch 需要额外条件：

```text
clothing condition z_c
```

因此从：

```text
w = E(theta)
```

扩展为：

```text
w_cloth = E_cloth(theta, z_c)
```

### 7.3 没有现成 pipeline 的部分：可选方案

#### 方案 A：每套衣服一套独立 clothing branch

做法：

```text
G_cloth^0, G_cloth^1, G_cloth^2 ...
每套衣服独立一套 GaussianModel。
```

推理时：

```text
选择 cloth_id
渲染对应 clothing branch
```

优点：

```text
最容易实现。
训练稳定。
每套衣服表达能力强。
```

缺点：

```text
参数不共享。
不是真正的 continuous clothing condition。
无法泛化到未见服装。
论文创新性略弱。
```

适合：

```text
第一版 MVP。
快速得到可视化结果。
```

#### 方案 B：共享 clothing branch + learnable clothing embedding

做法：

```text
每套衣服有一个 learnable code z_c。
共享一组 clothing MLPs。
MLP 输入 concat(theta, z_c)。
```

公式：

```text
w_cloth^j = E_cloth^j(theta, z_c)
```

优点：

```text
论文表述更自然。
能体现 cloth-conditioned。
参数共享。
有机会做 clothing interpolation。
```

缺点：

```text
不同衣服差异大时，共享模型可能容量不足。
训练更难。
需要改 MLP 输入和 checkpoint。
```

适合：

```text
主论文方法。
在方案 A 跑通后升级。
```

#### 方案 C：共享 clothing branch + garment part embedding

做法：

```text
不同服装部件有不同 code。
例如 upper / lower / shoes / coat。
MLP 输入 theta + cloth_code + part_code。
```

优点：

```text
更细粒度。
适合上衣、裤子、鞋子组合式换装。
```

缺点：

```text
需要 garment part mask。
数据和实现复杂。
```

适合：

```text
多部件服装组合的扩展版。
```

#### 方案 D：reference-image clothing encoder

做法：

```text
输入一张或多张目标服装参考图。
通过 encoder 得到 z_c。
再驱动 clothing branch。
```

优点：

```text
更接近开放式换装。
可以向未见服装泛化。
```

缺点：

```text
难度最高。
需要大量服装数据。
不是当前 19 天 deadline 的优先目标。
```

适合：

```text
future work 或下一篇。
```

推荐：

```text
工程第一版:
  方案 A

论文主方法:
  尽量推进到方案 B

扩展讨论:
  方案 C / D
```

## 8. 模块 5：Pose-driven Deformation

### 8.1 可直接套用 MMLPHuman 的部分

MMLPHuman 使用：

```text
SMPL-X pose
LBS weight volume
per-Gaussian skinning weights
rigid joint transforms
```

body branch 可以直接使用。

### 8.2 我们需要新增的部分

clothing Gaussians 也要随 pose 动。

问题是：

```text
clothing Gaussians 不一定贴在身体表面。
它们应该用什么 LBS 权重？
```

### 8.3 没有现成 pipeline 的部分：可选方案

#### 方案 A：复用 body LBS weight field

做法：

```text
clothing Gaussian 根据自己的 3D 位置查询同一个 lbs_weights_grid。
```

优点：

```text
最简单。
直接复用 MMLPHuman。
对贴身衣服有效。
```

缺点：

```text
宽松衣服、裙摆、外套下摆运动可能不自然。
```

适合：

```text
第一版主方案。
```

#### 方案 B：基于最近 body surface 的 LBS 权重

做法：

```text
clothing Gaussian 找最近的 body/template surface point。
继承该点的 LBS 权重。
```

优点：

```text
比直接空间插值更稳定。
避免远离身体的 clothing point 查询到异常权重。
```

缺点：

```text
需要最近点查询。
对宽松区域仍不完全准确。
```

适合：

```text
宽松程度中等的衣服。
```

#### 方案 C：学习 clothing-specific skinning correction

做法：

```text
基础 LBS 权重来自 body field。
再学习一个 clothing-specific weight residual 或 deformation residual。
```

优点：

```text
能适应衣服相对身体的非刚性运动。
```

缺点：

```text
实现和正则更复杂。
可能不稳定。
```

适合：

```text
增强版方法或 ablation。
```

#### 方案 D：单独的 cloth deformation network

做法：

```text
为 clothing branch 设计额外 deformation MLP。
输入 pose、cloth_code、canonical position，输出 non-rigid deformation。
```

优点：

```text
表达能力最强。
```

缺点：

```text
速度下降。
偏离 MMLPHuman 的轻量实时优势。
论文对比时需要解释为什么仍实时。
```

适合：

```text
如果主打质量而不是实时，可考虑。
当前不推荐。
```

推荐：

```text
第一版采用方案 A。
如果衣服远离身体导致伪影，再加入方案 B 或 C。
```

## 9. 模块 6：Gaussian Offset Basis 与 Coefficient Interpolation

### 9.1 可直接套用 MMLPHuman 的部分

MMLPHuman 的关键贡献之一是：

```text
MLP 输出 coefficients，而不是直接输出 Gaussian property offsets。
Gaussian property offsets 由 coefficients 线性组合 per-Gaussian basis 得到。
```

这个机制应该完整保留。

body branch：

```text
w_body = interpolate(E_body(theta))
delta_body = basis_body × w_body
```

clothing branch：

```text
w_cloth = interpolate(E_cloth(theta, z_c))
delta_cloth = basis_cloth × w_cloth
```

### 9.2 我们需要新增的部分

clothing branch 需要独立或条件化的：

```text
clothing offset basis
clothing coefficients
clothing anchor interpolation
```

### 9.3 没有现成 pipeline 的部分：可选方案

#### 方案 A：body 和 clothing 各自独立 basis

做法：

```text
body 有 body basis。
每套 clothing 有 clothing basis。
```

优点：

```text
表达能力强。
实现简单。
```

缺点：

```text
参数多。
不同衣服之间不共享 basis。
```

适合：

```text
第一版。
```

#### 方案 B：共享 clothing basis + cloth-conditioned coefficients

做法：

```text
所有衣服共享一套 clothing basis。
cloth_code 只影响 coefficients。
```

优点：

```text
参数共享。
更符合条件化建模。
```

缺点：

```text
不同衣服几何差异大时，共享 basis 可能不够。
```

适合：

```text
服装类型比较接近时。
```

#### 方案 C：shared basis + per-cloth residual basis

做法：

```text
clothing basis = shared basis + small per-cloth residual basis。
```

优点：

```text
兼顾共享和表达能力。
```

缺点：

```text
设计复杂一点。
需要 residual 正则。
```

适合：

```text
论文增强版。
```

推荐：

```text
第一版使用 A。
如果想强化论文方法，可升级 C。
```

## 10. 模块 7：Body-Clothing Composition

### 10.1 可直接套用 MMLPHuman 的部分

MMLPHuman 最终使用：

```text
gsplat rasterization
```

我们继续使用这一渲染器。

### 10.2 我们需要新增的部分

MMLPHuman 只有一套 Gaussian：

```text
G
```

我们需要组合：

```text
G = G_body ∪ G_cloth^c
```

然后一次性 rasterize。

### 10.3 没有现成 pipeline 的部分：可选方案

#### 方案 A：拼接 body 和 clothing Gaussians 后一次 rasterization

做法：

```text
means = concat(body_means, cloth_means)
covars = concat(body_covars, cloth_covars)
colors = concat(body_colors, cloth_colors)
opacities = concat(body_opacities, cloth_opacities)
一次 rasterization
```

优点：

```text
遮挡关系自然。
最符合 3D Gaussian rendering。
```

缺点：

```text
需要改 render 接口，拆出 render tensors。
```

推荐：

```text
主方案。
```

#### 方案 B：body 和 clothing 分别渲染，再 alpha blending

做法：

```text
render body image
render clothing image
后处理合成
```

优点：

```text
实现表面上更简单。
可以单独可视化。
```

缺点：

```text
遮挡关系错误。
深度排序不自然。
不推荐作为最终方法。
```

适合：

```text
只用于 debug visualization。
```

#### 方案 C：分层 rasterization + depth-aware composition

做法：

```text
分别渲染 body 和 clothing，同时输出 depth / alpha。
再根据 depth 做合成。
```

优点：

```text
比简单 alpha blend 好。
可以分析各层贡献。
```

缺点：

```text
复杂，且仍不如一次性 rasterization 纯粹。
```

适合：

```text
debug 或需要显式分层监督时。
```

推荐：

```text
论文主方法使用方案 A。
方案 B / C 可作为可视化工具。
```

## 11. 模块 8：训练目标 Loss

### 11.1 可直接套用 MMLPHuman 的部分

MMLPHuman 使用：

```text
L1 image reconstruction loss
LPIPS perceptual loss
control point smoothness loss
Gaussian scale regularization
```

这些都可以保留。

### 11.2 我们需要新增的部分

因为多了 body / clothing 解耦，需要额外监督：

```text
body 区域监督
clothing 区域监督
body-clothing 分离约束
clothing mask / alpha 约束
identity consistency
collision / penetration 约束
```

### 11.3 没有现成 pipeline 的部分：可选方案

#### 方案 A：图像重建 + mask 分区重建

loss：

```text
L = L_rgb
  + λ_body L_body_rgb
  + λ_cloth L_cloth_rgb
  + λ_lpips L_lpips
  + λ_ctrl L_ctrl
  + λ_scale L_scale
```

其中：

```text
L_body_rgb:
  body_mask 内的重建误差。

L_cloth_rgb:
  clothing_mask 内的重建误差。
```

优点：

```text
简单可靠。
实现成本低。
```

缺点：

```text
只能弱约束解耦。
body / clothing 仍可能互相解释部分区域。
```

推荐：

```text
第一版主方案。
```

#### 方案 B：增加 alpha / mask supervision

loss：

```text
L_alpha_body
L_alpha_cloth
```

要求：

```text
body Gaussians 的 alpha 主要落在 body_mask。
clothing Gaussians 的 alpha 主要落在 clothing_mask。
```

优点：

```text
更直接约束 body/clothing 分工。
```

缺点：

```text
需要分别渲染 body alpha 和 clothing alpha，或在 rasterizer 前后做额外统计。
```

适合：

```text
第二步增强。
```

#### 方案 C：identity consistency loss

做法：

```text
不同服装序列下，body branch 的 canonical appearance / rendered visible body 应保持一致。
```

优点：

```text
强化 shared identity。
```

缺点：

```text
需要设计可比较区域。
如果不同服装露肤区域差异大，监督困难。
```

适合：

```text
有稳定脸、手、头发区域时。
```

#### 方案 D：body-cloth collision / penetration loss

做法：

```text
约束 clothing Gaussians 不要进入 body surface 内部。
```

优点：

```text
改善穿模。
尤其对宽松衣服有帮助。
```

缺点：

```text
需要 body surface SDF 或 inside/outside 判断。
实现复杂。
```

适合：

```text
如果出现明显穿模，再加入。
```

推荐：

```text
第一版:
  方案 A

增强版:
  A + B + C

如果有穿模:
  再加 D
```

## 12. 模块 9：训练策略

### 12.1 可直接套用 MMLPHuman 的部分

MMLPHuman 采用阶段式训练：

```text
先训练 neutral Gaussian properties
之后打开 dxyz basis
之后打开 Gaussian property basis
之后提升 SH degree
```

这个思路可以保留。

### 12.2 我们需要新增的部分

body / clothing 两个分支可能需要分阶段训练。

### 12.3 没有现成 pipeline 的部分：可选方案

#### 方案 A：联合训练

做法：

```text
body branch 和 clothing branch 从一开始一起训练。
```

优点：

```text
流程简单。
```

缺点：

```text
容易互相抢解释权。
body 和 clothing 解耦不稳定。
```

适合：

```text
有强 mask 监督时。
```

#### 方案 B：先训练 body，再训练 clothing，最后联合微调

做法：

```text
Stage 1:
  只用 body visible regions 训练 body branch。

Stage 2:
  固定或弱更新 body，训练 clothing branch。

Stage 3:
  body + clothing 联合微调。
```

优点：

```text
解耦更稳定。
训练逻辑清晰。
```

缺点：

```text
训练流程更长。
需要设计 body-only 数据或 mask。
```

推荐：

```text
主方案优先考虑。
```

#### 方案 C：先训练每套完整 MMLPHuman，再蒸馏/分解 body-clothing

做法：

```text
每套衣服先训练一个完整 avatar。
再从多个 avatar 中分解 shared body 和 clothing residual。
```

优点：

```text
每套衣服都有强 baseline。
分解过程可以利用已有模型。
```

缺点：

```text
整体流程复杂。
训练成本高。
```

适合：

```text
如果已有多套训练好的 MMLPHuman checkpoint。
```

#### 方案 D：alternate optimization

做法：

```text
一个 iteration 更新 body。
下一个 iteration 更新 clothing。
或者按 cloth_id 分组交替训练。
```

优点：

```text
减少互相干扰。
```

缺点：

```text
调参复杂。
```

适合：

```text
联合训练不稳定时尝试。
```

推荐：

```text
第一版如果时间紧:
  方案 A + strong mask loss

更稳的论文版本:
  方案 B
```

## 13. 模块 10：测试与推理

### 13.1 可直接套用 MMLPHuman 的部分

MMLPHuman 支持：

```text
training pose / training camera rendering
novel pose rendering
novel view rendering
speed test
viewer interaction
PCA projection for novel poses
```

这些可以继续复用。

### 13.2 我们需要新增的部分

推理时额外输入：

```text
clothing condition
```

即：

```text
render(identity, pose, camera, cloth_id)
```

### 13.3 没有现成 pipeline 的部分：可选方案

#### 方案 A：指定离散 cloth_id

```text
--cloth_id 0
--cloth_id 1
```

优点：

```text
实现最简单。
结果最明确。
```

缺点：

```text
只能切换训练见过的衣服。
```

推荐：

```text
第一版主方案。
```

#### 方案 B：指定 clothing embedding

```text
输入 z_c。
可以做 embedding 插值。
```

优点：

```text
更符合 conditional representation。
能展示服装插值效果。
```

缺点：

```text
需要共享 conditional clothing branch。
```

适合：

```text
方案 B clothing branch 跑通后。
```

#### 方案 C：输入 reference image 生成 clothing code

```text
输入一张目标衣服图像。
encoder 生成 z_c。
```

优点：

```text
更接近开放世界换装。
```

缺点：

```text
当前不建议做。
```

推荐：

```text
先做 A。
如时间允许展示 B。
```

## 14. 模块 11：可视化 Pipeline

### 14.1 可直接套用 MMLPHuman 的部分

viewer 原本支持：

```text
相机旋转
pose 选择
novel pose 加载
背景色
scaling modifier
FPS 显示
```

### 14.2 我们需要新增的部分

增加：

```text
clothing selector
```

viewer 发送：

```text
cloth_id
```

模型端根据 `cloth_id` 选择 clothing branch 或 clothing code。

### 14.3 没有现成 pipeline 的部分：可选方案

#### 方案 A：简单下拉菜单选择 cloth_id

优点：

```text
实现最快。
```

推荐：

```text
主方案。
```

#### 方案 B：服装缩略图选择

优点：

```text
展示效果更好。
适合 demo。
```

缺点：

```text
GUI 工程成本更高。
```

#### 方案 C：支持 body-only / cloth-only / composed 三种视图

优点：

```text
非常适合论文 demo 和消融可视化。
```

推荐：

```text
如果时间允许，优先做这个。
```

## 15. 推荐主方案

为了兼顾 deadline、可实现性和论文表达，推荐主方案如下：

```text
数据:
  同一身份至少 2-3 套服装
  每套服装多视角图像/视频
  SMPL-X pose
  human mask
  clothing mask / body mask

初始化:
  body Gaussians 从 shared body/template region 采样
  clothing Gaussians 第一版从 clothing-covered SMPL/template region 采样

body branch:
  直接复用 MMLPHuman 的 pose-conditioned distributed MLP pipeline
  所有服装共享 body branch

clothing branch:
  第一版每套服装一套 clothing branch
  若时间允许，升级为 shared clothing branch + learnable cloth embedding

deformation:
  第一版 clothing Gaussians 复用 body LBS weight field
  后续加入 nearest-surface LBS 或 clothing-specific residual

composition:
  body Gaussians + selected clothing Gaussians 拼接
  一次 gsplat rasterization

loss:
  RGB reconstruction
  LPIPS
  body masked RGB loss
  clothing masked RGB loss
  control smoothness
  scale regularization

testing:
  指定 cloth_id
  novel pose
  novel view
  viewer 实时切换衣服
```

这套方案的论文 claim 可以是：

```text
We propose a compositional Gaussian avatar representation that disentangles identity-preserving body Gaussians and cloth-conditioned garment Gaussians, enabling real-time pose-driven rendering with switchable outfits.
```

## 16. MVP 版本

如果时间非常紧，MVP 可以只做：

```text
1. shared body branch
2. per-cloth clothing branch
3. body + clothing Gaussian concatenation
4. cloth_id 切换
5. mask-based loss
6. novel pose / novel view rendering
```

MVP 不做：

```text
未见衣服泛化
reference image clothing encoder
复杂 cloth physics
跨身份换装
多 garment part 组合
SDF collision loss
```

MVP 论文表述要收敛：

```text
目标是 reconstruct and switch observed outfits for the same identity。
不是 arbitrary virtual try-on。
```

## 17. 进阶版本

如果 MVP 跑通且还有时间，可以升级：

```text
1. shared clothing branch + learnable cloth embedding
2. clothing embedding interpolation
3. body-only / cloth-only 分层可视化
4. alpha / mask supervision
5. identity consistency loss
6. clothing-specific deformation residual
```

这些可以变成论文中的：

```text
method enhancement
ablation
supplementary experiment
```

## 18. 最需要进一步讨论的开放点

下面这些没有 MMLPHuman 现成 pipeline，需要结合数据再定：

```text
1. clothing mask 是否已有？如果没有，使用什么 human parsing 模型？

2. body Gaussians 是否表示完整 underlying body，还是只表示 visible identity regions？

3. clothing Gaussians 从 SMPL region 初始化，还是从每套衣服 template / point cloud 初始化？

4. clothing condition 第一版用 per-cloth branch，还是直接做 shared branch + cloth embedding？

5. clothing Gaussians 使用 body LBS field 是否足够？是否需要 nearest-surface LBS 或 residual deformation？

6. 是否需要 body-cloth collision loss？

7. 定量评价是否有同 pose 不同服装的 GT？如果没有，如何设计可接受的评价？

8. 论文 claim 是“observed outfit switching”还是“generalizable dressing”？这会决定方法设计和实验要求。
```

当前最稳妥的 claim 是：

```text
observed outfit-switchable pose-driven Gaussian avatar for the same identity
```

也就是：

```text
同一身份、已观测多套服装之间的高质量实时换装与动作驱动。
```

