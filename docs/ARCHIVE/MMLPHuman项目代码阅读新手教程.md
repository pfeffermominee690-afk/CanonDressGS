# MMLPHuman 代码阅读新手教程：从论文 Pipeline 对应到工程文件

本文档的目标是帮你把论文 **Real-time High-fidelity Gaussian Human Avatars with Position-based Interpolation of Spatially Distributed MLPs** 中的 pipeline 和当前工程代码对应起来。

它不是逐行代码注释，而是告诉你：

```text
论文里的每个模块
在代码中对应哪个文件
训练 / 测试 / 可视化时数据如何流动
应该按什么顺序读代码
```

如果你刚读完论文但还没读懂代码，建议按本文档顺序看。

## 1. 先记住论文中的核心 Pipeline

论文方法可以简化成下面这条链路：

```text
多视角视频 + SMPL-X pose + mask
        ↓
构建 canonical template mesh
        ↓
在 template mesh 上采样:
  1. Gaussian points
  2. anchor points for spatially distributed MLPs
  3. control points
        ↓
每个 anchor MLP 输入 pose θ
        ↓
输出 basis coefficients
        ↓
把 anchor coefficients 插值到每个 Gaussian
        ↓
用 coefficients 线性组合 Gaussian offset basis
        ↓
得到 pose-dependent Gaussian properties
        ↓
通过 control points 得到 position offset
        ↓
通过 LBS 把 canonical Gaussians 变形到当前 pose
        ↓
gsplat rasterization
        ↓
渲染图像，与 GT 图像算 loss
```

如果用代码语言概括：

```text
Dataset
  提供 image / mask / camera / SMPL pose

Scene
  准备 template、采样点、LBS 权重体，初始化 GaussianModel

GaussianModel
  实现 spatially distributed MLP、basis、control points、LBS、render

train.py
  组织训练循环

test.py / visualize.py
  加载模型并渲染
```

## 2. 工程目录总览

当前仓库结构可以这样理解：

```text
mmlphuman_code/
├─ train.py
│  训练入口，负责完整训练循环
│
├─ test.py
│  测试 / 渲染入口，负责加载 checkpoint 后输出图像或测速
│
├─ visualize.py
│  可视化入口，加载训练好的模型并等待 viewer 连接
│
├─ config/
│  每个 subject / dataset 的训练参数
│
├─ scene/
│  核心算法目录
│
├─ utils/
│  SMPL、loss、图像、网络、PLY、球谐函数等工具
│
├─ viewer/
│  实时 GUI viewer 客户端
│
├─ script/
│  数据预处理脚本，比如 LBS 权重体生成
│
└─ template/
   示例 template mesh
```

真正需要精读的主线文件只有几个：

```text
train.py
scene/scene.py
scene/dataset.py
scene/gaussian_model.py
scene/mlp.py
utils/smpl_utils.py
utils/loss_utils.py
test.py
scene/net_vis.py
viewer/net_viewer.py
```

## 3. 论文模块与代码文件对应表

| 论文模块 | 代码位置 | 你要理解什么 |
|---|---|---|
| Multi-view images / masks / cameras / SMPL-X pose | `scene/dataset.py` | 一个训练样本如何被读出来 |
| Dataset type detection | `scene/dataset.py::get_dataset_type` | 如何区分 AvatarReX / THuman / ActorsHQ |
| Canonical template mesh | `scene/scene.py` | 如何读取或生成 `template.ply` |
| Initial Gaussian points | `scene/scene.py` | 如何在 mesh 上采样 `xyz` |
| Anchor points for MLPs | `scene/scene.py` | 如何采样 `xyz_ft` |
| Control points | `scene/scene.py` | 如何采样 `xyz_vt` |
| LBS weight volume | `script/gen_weight_volume.py`, `utils/smpl_utils.py` | 每个 Gaussian 如何得到骨骼权重 |
| Spatially distributed MLPs | `scene/gaussian_model.py`, `scene/mlp.py` | 多个 anchor MLP 如何输入 pose 输出系数 |
| Gaussian offset basis | `scene/gaussian_model.py` | 每个 Gaussian 的 basis 参数在哪里 |
| Coefficient interpolation | `scene/gaussian_model.py::prepare_interpolating_weights` | anchor 输出如何插值到 Gaussian |
| Control point interpolation | `scene/gaussian_model.py::get_dxyz_vt`, `get_dxyz` | Gaussian position offset 如何来自 control points |
| LBS pose deformation | `scene/gaussian_model.py::get_Gweights`, `get_xyz` | canonical Gaussian 如何变到当前 pose |
| Gaussian rasterization | `scene/gaussian_model.py::render` | 最终如何调用 gsplat 渲染 |
| Training losses | `train.py`, `utils/loss_utils.py` | L1 / LPIPS / smooth / scale loss |
| Testing novel pose / novel view | `test.py` | checkpoint 加载后如何渲染 |
| Real-time viewer | `visualize.py`, `scene/net_vis.py`, `viewer/net_viewer.py` | GUI 如何传 pose/camera，模型如何回图 |

## 4. 推荐代码阅读顺序

不要一开始就读 `gaussian_model.py`，它最核心也最难。建议按下面顺序来：

```text
Step 1: README.md
Step 2: config/*.yaml
Step 3: train.py
Step 4: scene/dataset.py
Step 5: scene/scene.py
Step 6: scene/gaussian_model.py
Step 7: scene/mlp.py
Step 8: utils/smpl_utils.py
Step 9: utils/loss_utils.py
Step 10: test.py
Step 11: visualize.py + viewer/
```

每一步只需要先读懂“它在 pipeline 中负责什么”，不用一口气理解所有细节。

## 5. 从训练命令开始理解整个工程

README 中的训练命令是：

```shell
python train.py --config ./config/{DATASET}.yaml --data_dir {DATASET_DIR} --out_dir {MODEL_DIR}
```

这条命令包含三个关键信息：

```text
--config
  训练超参数，比如迭代次数、学习率、相机 id、帧范围

--data_dir
  数据集路径，里面有图像、mask、相机、SMPL-X pose、LBS 权重体

--out_dir
  模型输出路径，保存 config、poses.json、cameras.json、checkpoint
```

主入口在：

```text
train.py::__main__
```

它做的事：

```text
1. 读取命令行参数
2. 用 OmegaConf 加载 config
3. 写入 data_dir / out_dir / ip / port
4. 保存 config 到 out_dir
5. 设置随机种子和 CUDA
6. 调用 training(args)
```

然后真正进入：

```text
train.py::training(args)
```

## 6. `train.py` 对应论文中的训练过程

`train.py::training(args)` 是训练总导演。

最关键的初始化：

```python
gaussians = GaussianModel()
scene = Scene(args, gaussians)
gaussians.training_setup(args, scene.scene_scale)
```

对应论文：

```text
GaussianModel()
  创建 avatar 表示的空壳

Scene(args, gaussians)
  构建 canonical template
  采样 Gaussians / anchors / control points
  读取 LBS weight volume
  初始化 Gaussian avatar

training_setup
  设置需要学习的参数和 optimizer
```

训练循环核心：

```python
cam = data_to_cam(cam)

gaussians.smpl_poses = cam['pose']
gaussians.Th, gaussians.Rh = cam['Th'], cam['Rh']

image, alpha, info = gaussians.render(cam, background=bg)
```

对应论文：

```text
输入当前帧 pose θ 和当前相机
-> 生成当前 pose 下的 Gaussian properties
-> LBS 变形
-> rasterize
-> 得到 rendered image
```

loss 部分：

```python
l1loss = l1_loss(image, image_gt)
dxyzsmoothloss = dxyz_smooth_loss(gaussians) * args.lambda_dxyz_smooth
lpipsloss = lpips_loss(...) * args.lambda_lpips
scaling_loss = args.lambda_scaling * gaussian_scaling_loss(...)
```

对应论文：

```text
L1 reconstruction loss
LPIPS perceptual loss
control point smoothness loss
Gaussian scale regularization
```

注意代码中的 `dxyz_smooth_loss` 对应论文里的 control point smoothness 思想。

## 7. `config/*.yaml` 对应论文中的实验设置

以 `config/subject02.yaml` 为例。

你需要重点看：

```yaml
train_cam_ids
num_train_frame
begin_ith_frame
frame_interval
image_scaling
```

对应论文：

```text
训练用哪些相机
训练用哪些帧
图像是否下采样
```

优化参数：

```yaml
iterations
position_lr
opacity_lr
scaling_lr
rotation_lr
color_lr
encoder_lr
```

对应论文：

```text
训练 800K iterations
不同 Gaussian property 和 MLP 使用不同学习率
```

结构参数：

```yaml
init_num_gs: 200_000
num_verts: 10000
num_features: 300
```

对应论文：

```text
N = 200K Gaussians
C = 10000 control points
F = 300 spatially distributed MLPs / anchor points
```

阶段开关：

```yaml
iteration_dxyz_basis: 2000
iteration_gsparam_basis: 2000
iteration_sh_degree: 250000
```

对应代码：

```text
train.py 中打开 dxyz basis / Gaussian property basis / SH degree
```

## 8. `scene/dataset.py` 对应论文中的输入数据

论文说输入是：

```text
multi-view videos
foreground masks
camera parameters
SMPL-X registrations
```

代码中这些都在：

```text
scene/dataset.py
```

### 8.1 数据集类型判断

函数：

```python
get_dataset_type(datadir)
```

判断逻辑：

```text
calibration.json       -> ThumanDataset
calibration_full.json  -> AVRexDataset
calibration.csv        -> ActorsHQDataset
```

也就是说，代码通过数据目录里的标定文件判断数据集类型。

### 8.2 一个训练样本长什么样

每个 Dataset 的 `__getitem__` 最后都会返回一个 dict：

```text
K
  相机内参

w2c
  world-to-camera 外参

image
  GT RGB 图像

mask
  人体前景 mask

mask_boundary
  mask 边界区域，训练时会特殊处理

pose
  当前帧 SMPL-X pose

Rh / Th
  全局旋转和平移

beta
  SMPL-X shape

height / width
  图像尺寸

frame_id / cam_id / idx
  帧号、相机号、样本编号
```

`train.py` 中变量名叫 `cam`，但它其实不是单纯相机，而是一个完整样本。

### 8.3 `data_to_cam`

函数：

```python
data_to_cam(data, non_blocking=True)
```

作用：

```text
把 Dataset 返回的 numpy / tensor 数据整理到 CPU 或 CUDA
```

关键设计：

```text
K / w2c
  放到 CUDA

image / mask / mask_boundary
  放到 CUDA

pose / beta / Rh / Th
  保留 CPU tensor，同时需要时再转 CUDA

height / width / frame_id / cam_id
  转成普通 Python int
```

## 9. `scene/scene.py` 对应论文中的初始化

论文中的这部分：

```text
obtain canonical template mesh
sample Gaussian points
sample anchor points
sample control points
load skinning weights
```

主要都在：

```text
scene/scene.py::Scene.__init__
```

### 9.1 初始化 SMPL-X

代码：

```python
init_smpl(args.smpl_pkl_path)
```

对应论文：

```text
register SMPL-X model
obtain joints and skeleton hierarchy
```

### 9.2 创建 Dataset 和 DataLoader

代码：

```python
DatasetType = get_dataset_type(args.data_dir)
trainset = DatasetType(...)
testset = DatasetType(...)
trainloader = DataLoader(...)
```

对应论文：

```text
读取多视角训练图像、mask、camera、pose
```

### 9.3 导出 viewer 需要的 pose 和 camera

代码会保存：

```text
out_dir/poses.json
out_dir/cameras.json
```

这不是论文核心算法，而是为了 viewer 能选择训练姿态和训练相机。

### 9.4 读取 LBS 权重体

代码：

```python
weights_grid_path = path.join(args.data_dir, 'gaussian/lbs_weights_grid.npz')
grid_info = dict(np.load(weights_grid_path, allow_pickle=True))
```

对应论文：

```text
让任意 Gaussian point 都能插值得到骨骼权重
```

这个文件由：

```text
script/gen_weight_volume.py
```

生成。

### 9.5 准备 template mesh

代码：

```python
temp_path = path.join(args.data_dir, 'gaussian/template.ply')
```

如果没有 template：

```text
使用 SMPL-X big pose mesh 生成 template.ply
```

对应论文：

```text
canonical template mesh
```

注意：对于宽松衣服，论文和 README 都建议使用重建出来的 template，而不是裸 SMPL-X mesh。

### 9.6 采样 Gaussian / anchor / control points

代码：

```python
xyz = rand_point_on_mesh(verts, faces, pts_num=args.init_num_gs)
xyz_ft = rand_point_on_mesh(verts, faces, pts_num=args.num_features, init_factor=7)
xyz_vt = rand_point_on_mesh(verts, faces, pts_num=args.num_verts, init_factor=7)
```

对应论文：

```text
xyz:
  Gaussian points

xyz_ft:
  anchor points / spatially distributed MLP locations

xyz_vt:
  control points
```

### 9.7 初始化 GaussianModel

代码：

```python
gaussians.create_from_pcd(
    xyz=xyz,
    t_joints=t_joints,
    joint_parents=smpl.model.parents,
    lbs_weights_grid_info=grid_info,
    all_poses=all_poses,
    xyz_ft=xyz_ft,
    xyz_vt=xyz_vt,
)
```

这是 `Scene` 和 `GaussianModel` 的交接点。

## 10. `scene/gaussian_model.py` 对应论文核心方法

这是最重要、也最需要耐心读的文件。

你可以把 `GaussianModel` 理解成论文中的 avatar representation。

### 10.1 模型中存了什么

在 `GaussianModel.__init__` 中，有几类变量。

#### Neutral Gaussian properties

对应论文：

```text
x0, r0, s0, o0, c0
```

代码变量：

```python
self._xyz
self._rotation
self._scaling
self._opacity
self._sh0
self._shN
```

含义：

```text
_xyz:
  canonical neutral position

_rotation:
  Gaussian rotation

_scaling:
  Gaussian scale

_opacity:
  Gaussian opacity

_sh0 / _shN:
  spherical harmonics color coefficients
```

#### Control point position offsets

代码变量：

```python
self.dxyz_vt
self.dxyz_bs
```

含义：

```text
dxyz_vt:
  control point neutral position offset

dxyz_bs:
  control point position offset basis
```

对应论文：

```text
control point interpolation for Gaussian position offset
```

#### Gaussian property offset basis

代码变量：

```python
self.sh0_bs
self.shN_bs
self.scaling_bs
self.rotation_bs
self.opacity_bs
```

对应论文：

```text
Gaussian offset basis:
  delta rotation basis
  delta scale basis
  delta opacity basis
  delta color basis
```

注意：position offset 的 basis 是 control point 上的 `dxyz_bs`，不是直接给每个 Gaussian 自由学习。

#### Spatially distributed MLPs

代码变量：

```python
self.encoder_feat_params
self.encoder_feat_model_meta
```

对应论文：

```text
F spatially distributed MLPs located at anchor points
```

这里代码不是用一个普通 `ModuleList` 保存很多 MLP，而是用 `torch.func.stack_module_state` 把多个 MLP 的参数堆叠起来，方便批量前向。

### 10.2 `create_from_pcd`

函数：

```python
GaussianModel.create_from_pcd(...)
```

对应论文中的初始化：

```text
initialize Gaussians
initialize neutral properties
initialize distributed MLPs
initialize Gaussian offset basis
initialize control point basis
prepare interpolation relationship
```

这里会做：

```text
1. 把 xyz 作为 Gaussian 初始位置
2. 根据 KNN 距离初始化 scale
3. 初始化 rotation / opacity / color
4. 保存 SMPL joints 和 skeleton parents
5. 保存所有训练 poses
6. 保存 LBS weight grid
7. 创建 F 个 MLP
8. 创建 basis 参数
9. 创建 control points
10. 调用 prepare_interpolating_weights
11. 初始化 body pose transform
```

重点看这一行：

```python
models = [MLP(layers_size_list=[63, 512, 256, 256, 256, self.num_basis+self.num_vt_basis]) for i in range(len(xyz_ft))]
```

对应论文：

```text
每个 anchor point 一个 MLP
输入 pose vector
输出:
  Gaussian property basis coefficients
  control point position basis coefficients
```

这里：

```text
63 = 21 body joints * 3 axis-angle
self.num_basis = Gaussian property basis number, 默认 15
self.num_vt_basis = control point basis number, 默认 15
```

### 10.3 `prepare_interpolating_weights`

函数：

```python
GaussianModel.prepare_interpolating_weights(xyz_ft, xyz_vt)
```

对应论文中的：

```text
position-based interpolation
```

它预先计算几种 KNN 关系：

```text
Gaussian -> nearest control points
control point -> nearby control points
Gaussian -> nearest anchor points
control point -> nearest anchor points
```

代码变量：

```python
self.nbr_gs
self.nbr_gs_invdist
self.nbr_vt
self.nbr_gsft
self.nbr_gsft_wght
self.nbr_vtft
self.nbr_vtft_wght
```

怎么理解：

```text
nbr_gs:
  每个 Gaussian 附近的 control points

nbr_vt:
  每个 control point 附近的 control points，用于 smooth loss

nbr_gsft:
  每个 Gaussian 附近的 anchor MLPs

nbr_vtft:
  每个 control point 附近的 anchor MLPs
```

这些关系在训练开始前算好，后面每一帧只需要查表插值。

### 10.4 `get_joint_features`

函数：

```python
GaussianModel.get_joint_features
```

对应论文：

```text
MLP input pose vector θ
```

训练时：

```python
features = self.smpl_poses_cuda[3:3*22]
```

也就是取 SMPL-X 的 body joints，不包括全局 orient，也不包括手指。

测试时：

```text
会用 PCA 把 novel pose 投影到训练 pose space
```

对应论文 Testing 部分：

```text
use PCA to project novel poses to the space of training poses
```

### 10.5 `get_encoded_feature`

函数：

```python
GaussianModel.get_encoded_feature
```

对应论文：

```text
w_a^j = E_j(theta)
```

含义：

```text
把当前 pose θ 输入所有 spatially distributed MLPs
得到每个 anchor point 的 coefficients
```

代码中：

```python
features = vmap_mlp(self.encoder_feat_params, features)
```

真正的 MLP 批量前向在：

```text
scene/mlp.py::vmap_mlp
```

### 10.6 `get_encoded_feature_gsparam_weight`

函数：

```python
GaussianModel.get_encoded_feature_gsparam_weight
```

对应论文：

```text
把 anchor coefficients 插值到每个 Gaussian
```

代码思想：

```python
features = self.get_encoded_feature[...,:self.num_basis]
features = torch.einsum('nrc,nr->nc', features[self.nbr_gsft], self.nbr_gsft_wght)
```

含义：

```text
每个 Gaussian 找附近 3 个 anchor MLP
根据距离权重插值得到自己的 basis coefficients
```

### 10.7 `get_dxyz_vt` 和 `get_dxyz`

函数：

```python
GaussianModel.get_dxyz_vt
GaussianModel.get_dxyz
```

对应论文：

```text
control point interpolation for Gaussian position offset
```

`get_dxyz_vt`：

```text
先得到 control point 的 position offset
```

`get_dxyz`：

```text
再把 control point offset 插值到每个 Gaussian
```

这就是论文强调的：

```text
不要让每个 Gaussian 的位置自由乱动
而是通过 control points 约束在表面层附近运动
```

### 10.8 `get_cano_xyz`

函数：

```python
GaussianModel.get_cano_xyz
```

对应论文：

```text
canonical Gaussian position after position offset
```

代码：

```python
xyz = self._xyz + self.get_dxyz + torch.tanh(self.xyz_offset) * 0.008
```

含义：

```text
初始位置
+ control point 插值得到的位置 offset
+ 一个很小的自由 offset
```

最后一项是代码实现中的 trick，让 Gaussian 在很小范围内自由调整。

### 10.9 Gaussian property basis

例如 `get_cano_scaling`：

```python
features = self.get_encoded_feature_gsparam_weight
dscaling = torch.einsum('nc,ncl->nl', features, self.scaling_bs)
scaling = self._scaling + dscaling
```

对应论文：

```text
Gaussian property offset = coefficients × Gaussian offset basis
```

类似逻辑还在：

```text
get_cano_rotation
get_opacity
get_sh
```

也就是说，论文中的：

```text
delta r, delta s, delta o, delta c
```

在代码中分别通过 basis 组合得到。

### 10.10 `get_Gweights`

函数：

```python
GaussianModel.get_Gweights
```

对应论文：

```text
LBS deformation
```

这里做的是：

```text
1. 根据当前 SMPL pose 算每个骨骼的 rigid transform
2. 每个 Gaussian 根据自己的 skinning weights 混合这些 transform
3. 得到每个 Gaussian 的变换矩阵
```

Gaussian 的 skinning weights 来自：

```python
self.get_weights
```

而 `get_weights` 会调用：

```python
interpolate_skinningfield(self.weights_grid_info, xyz)
```

这个函数在：

```text
utils/smpl_utils.py
```

### 10.11 `get_xyz`

函数：

```python
GaussianModel.get_xyz
```

对应论文：

```text
transform canonical Gaussians to posed space
```

逻辑：

```text
canonical position
-> LBS transform
-> global rotation Rh
-> global translation Th
-> posed position
```

这就是最终送进 rasterizer 的 Gaussian 位置。

### 10.12 `get_covariance`

函数：

```python
GaussianModel.get_covariance
```

对应论文：

```text
transform Gaussian rotation and scale under pose
```

它把：

```text
canonical rotation
canonical scale
LBS rotation
```

组合成 rasterization 需要的 covariance。

### 10.13 `get_color`

函数：

```python
GaussianModel.get_color(cam_pos)
```

对应论文：

```text
view-dependent color by spherical harmonics
```

代码中会根据相机位置和 Gaussian 位置计算 view direction，再用：

```python
spherical_harmonics(...)
```

得到颜色。

### 10.14 `render`

函数：

```python
GaussianModel.render(cam, override_color=None, scaling_modifier=1.0, background=None)
```

对应论文最终一步：

```text
LBS & Rasterization
```

核心调用：

```python
image, alpha, info = rasterization(
    means=self.get_xyz,
    opacities=self.get_opacity,
    colors=override_color,
    viewmats=cam['w2c'][None],
    Ks=cam['K'][None],
    covars=covars,
)
```

这是工程中真正调用 gsplat 的地方。

## 11. `scene/mlp.py` 对应 Spatially Distributed MLPs

文件：

```text
scene/mlp.py
```

类：

```python
class MLP(nn.Module)
```

对应论文：

```text
spatially distributed MLP
```

函数：

```python
vmap_mlp(params, x, meta=None)
```

对应论文：

```text
对所有 anchor MLP 同时前向，得到所有 anchor coefficients
```

代码中 `USE_VMAP = False`，实际使用手写 einsum 版本，作者注释说比 vmap 稍快。

你只需要理解：

```text
它不是一个 MLP 给所有 Gaussian 用
而是每个 anchor point 有一个 MLP
这些 MLP 的参数被 stack 起来批量计算
```

## 12. `utils/smpl_utils.py` 对应 LBS 和 SMPL-X

文件：

```text
utils/smpl_utils.py
```

重点函数：

```python
init_smpl
init_smpl_pose
interpolate_skinningfield
rigid_transform_tensor
rigid_transform_numba
```

对应论文：

```text
SMPL-X pose
LBS transformation
skinning weights
```

### 12.1 `init_smpl`

加载 SMPL-X 模型：

```python
smpl.model = smplx.SMPLX(...)
```

### 12.2 `init_smpl_pose`

定义：

```text
T pose
big pose
```

big pose 用于 canonical 初始化和 body transform。

### 12.3 `interpolate_skinningfield`

输入：

```text
LBS weight volume
3D points
```

输出：

```text
每个点的 skinning weights
```

对应论文：

```text
Gaussian points are deformed by LBS
```

### 12.4 `rigid_transform_numba`

根据 SMPL pose 和 skeleton hierarchy 计算每个 joint 的 rigid transform。

## 13. `script/gen_weight_volume.py` 对应预处理

论文中不会重点展开工程预处理，但代码必须有这一步。

文件：

```text
script/gen_weight_volume.py
```

作用：

```text
生成 gaussian/lbs_weights_grid.npz
```

为什么需要：

```text
Gaussian points 不是 SMPL-X 顶点
所以不能直接用 SMPL-X 顶点的 LBS weights
需要一个空间中的 weight field
让任意 3D Gaussian point 都能查询自己的 LBS weights
```

训练时：

```text
scene/scene.py 读取 lbs_weights_grid.npz
gaussian_model.py::get_weights 查询这个 field
```

## 14. `utils/loss_utils.py` 对应训练损失

文件：

```text
utils/loss_utils.py
```

重点函数：

```python
psnr
ssim_loss
lpips_loss
dxyz_smooth_loss
gaussian_scaling_loss
```

对应论文：

```text
L1
LPIPS
Lctrl
Lscale
```

注意：

```text
L1 loss 直接从 torch.nn.functional import l1_loss
```

而 control point smoothness：

```python
dxyz_smooth_loss(gaussians)
```

用的是：

```text
control point offset 和邻居 control point offset 的差异
```

scale loss：

```python
gaussian_scaling_loss(...)
```

限制 Gaussian 不要变太大。

## 15. `test.py` 对应论文 Testing

文件：

```text
test.py
```

入口：

```text
test.py::__main__
test.py::testing(args)
```

对应论文：

```text
novel view rendering
novel pose rendering
speed testing
PCA for novel pose
```

### 15.1 加载模型

代码：

```python
gaussians = load_model(args.model_dir)
gaussians.is_test = args.test.is_test
gaussians.prepare_test()
```

`load_model` 在：

```text
scene/net_vis.py
```

`prepare_test` 在：

```text
scene/gaussian_model.py
```

对应论文：

```text
Testing 时用 PCA 将 novel pose 投影到 training pose space
```

### 15.2 两种测试模式

如果提供：

```text
--cam_path
--pose_path
```

则：

```text
使用自定义 camera 和 novel pose 渲染
```

否则：

```text
用 data_dir 中的测试集相机和训练/测试帧渲染
```

### 15.3 speed test

参数：

```shell
--test_speed
```

对应论文中的 FPS 评估。

## 16. `visualize.py` 和 `viewer/` 对应实时可视化

这一部分不是论文核心算法，但对理解系统很有帮助。

### 16.1 模型端

文件：

```text
visualize.py
scene/net_vis.py
utils/net_utils.py
```

作用：

```text
加载模型
启动 WebSocket 服务
接收 viewer 发来的 pose / camera / background
调用 GaussianModel.render
把渲染图传回 viewer
```

### 16.2 Viewer 端

文件：

```text
viewer/net_viewer.py
viewer/net.py
```

作用：

```text
打开 DearPyGUI 窗口
鼠标控制相机
滑块选择 frame / camera
发送当前参数给模型端
接收图片并显示
```

### 16.3 通信内容

viewer 发送：

```text
w2c
K
fovx
height / width
pose
Rh / Th
background
scaling_modifier
is_test
```

模型端返回：

```text
image_bytes
gaussian_num
camera_list
pose_list
```

## 17. 训练时数据流完整串起来

下面是一次训练 iteration 的完整代码流：

```text
train.py::training
  |
  | 从 scene.trainloader 取一个 sample
  v
scene/dataset.py::__getitem__
  |
  | 返回 image / mask / K / w2c / pose / Rh / Th
  v
scene/dataset.py::data_to_cam
  |
  | 把数据搬到 CPU / CUDA
  v
train.py
  |
  | 设置 gaussians.smpl_poses / Rh / Th
  v
scene/gaussian_model.py::render
  |
  +--> get_encoded_feature
  |      pose -> distributed MLP -> anchor coefficients
  |
  +--> get_encoded_feature_gsparam_weight
  |      anchor coefficients -> Gaussian coefficients
  |
  +--> get_dxyz_vt / get_dxyz
  |      control point offset -> Gaussian position offset
  |
  +--> get_cano_xyz
  |      canonical Gaussian position
  |
  +--> get_Gweights / get_xyz
  |      LBS deformation
  |
  +--> get_covariance / get_color / get_opacity
  |      Gaussian render properties
  |
  +--> gsplat.rasterization
  |      render image
  v
train.py
  |
  | 计算 loss
  | loss.backward()
  | gaussians.optimizer_step()
  v
checkpoint / TensorBoard / viewer
```

这条链路读懂，就基本读懂了整个工程。

## 18. 测试时数据流完整串起来

测试时没有反向传播。

```text
test.py::testing
  |
  | load_model
  v
scene/net_vis.py::load_model
  |
  | 读取 checkpoint
  | GaussianModel.restore
  v
gaussians.prepare_test
  |
  | PCA fitting on training poses
  v
选择测试模式:
  |
  +--> dataset camera / pose
  |
  +--> custom cam_path + pose_path
  |
  v
设置 gaussians.smpl_poses / Rh / Th
  |
  v
gaussians.render
  |
  v
保存 png 或统计 FPS
```

## 19. 论文图 Figure 2 和代码的对应关系

论文 Figure 2 中有几个关键块。

### 19.1 Template mesh

代码：

```text
scene/scene.py
```

对应：

```text
template.ply
verts / faces
```

### 19.2 Anchor point

代码：

```python
xyz_ft = rand_point_on_mesh(...)
```

存储：

```python
self.xyz_ft
```

### 19.3 Spatially distributed MLP

代码：

```text
scene/gaussian_model.py::create_from_pcd
scene/mlp.py
```

变量：

```python
self.encoder_feat_params
```

### 19.4 Gaussian point

代码：

```python
xyz = rand_point_on_mesh(...)
self._xyz
```

### 19.5 Gaussian offset basis

代码：

```python
self.sh0_bs
self.shN_bs
self.scaling_bs
self.rotation_bs
self.opacity_bs
```

### 19.6 Control point interpolation

代码：

```python
self.xyz_vt
self.dxyz_vt
self.dxyz_bs
get_dxyz_vt
get_dxyz
```

### 19.7 LBS & Rasterization

代码：

```python
get_Gweights
get_xyz
get_covariance
render
```

## 20. 最容易混淆的几个点

### 20.1 `cam` 不是单纯相机

训练代码里变量叫 `cam`：

```python
cam = next(trainloader_iter)
```

但它其实包含：

```text
相机 + 图像 + mask + pose + frame 信息
```

你可以在脑子里把它叫做 `sample`。

### 20.2 `xyz_ft` 不是 Gaussian 点

```text
xyz:
  Gaussian points

xyz_ft:
  anchor points for MLPs

xyz_vt:
  control points
```

### 20.3 MLP 不直接输出 Gaussian 属性

论文的重点就是：

```text
MLP 不直接输出 delta scale / delta color 等
MLP 输出 coefficients
然后 coefficients 组合 Gaussian-specific basis
```

对应代码：

```text
get_encoded_feature
-> get_encoded_feature_gsparam_weight
-> basis einsum
```

### 20.4 Gaussian position offset 不完全自由学习

位置 offset 主要来自：

```text
control points
```

对应代码：

```text
get_dxyz_vt
get_dxyz
```

但代码额外有一个小自由项：

```python
torch.tanh(self.xyz_offset) * 0.008
```

### 20.5 `is_test` 会影响 pose 输入

测试时：

```python
gaussians.is_test = args.test.is_test
gaussians.prepare_test()
```

如果 `is_test=True`，`get_joint_features` 会用 PCA 处理 novel pose。

### 20.6 `is_dxyz_bs` 和 `is_gsparam_bs` 是阶段开关

训练中：

```python
gaussians.is_dxyz_bs = True
gaussians.is_gsparam_bs = True
```

含义：

```text
是否启用 position basis
是否启用 Gaussian property basis
```

它们在训练到指定 iteration 后打开。

## 21. 读完之后你应该能回答的问题

如果你读完这份教程和对应代码，应该能回答：

```text
1. 一个训练样本从哪里来？
2. SMPL-X pose 是在哪里读入和设置的？
3. template.ply 在哪里用？
4. 200K Gaussian points 是在哪里采样的？
5. 300 个 MLP anchor points 是在哪里采样的？
6. 10000 个 control points 是在哪里采样的？
7. MLP 的输入和输出维度是什么？
8. anchor coefficients 如何插值到 Gaussian？
9. Gaussian offset basis 在代码里是什么变量？
10. position offset 为什么要通过 control points？
11. LBS 权重从哪里来？
12. 当前 pose 下的 Gaussian 位置在哪里计算？
13. 最终 gsplat rasterization 在哪里调用？
14. 训练 loss 对应论文哪几个项？
15. checkpoint 保存了哪些东西？
16. test.py 如何做 novel pose / novel view？
17. viewer 是怎么和模型通信的？
```

这些问题都能答出来，就说明你已经把论文 pipeline 和代码主线对应起来了。

## 22. 最后建议：如何边读边做笔记

建议你准备一张自己的对照表：

```text
论文符号          代码变量 / 函数
--------------------------------
θ pose           smpl_poses / get_joint_features
x0               _xyz
r0               _rotation
s0               _scaling
o0               _opacity
c0               _sh0 / _shN
anchor points    xyz_ft
control points   xyz_vt
MLPs             encoder_feat_params
coefficients     get_encoded_feature
Gaussian coeff   get_encoded_feature_gsparam_weight
δx control       get_dxyz_vt / get_dxyz
LBS              get_Gweights / get_xyz
rasterization    render
```

这张表一旦补全，后面做你的 `body Gaussians + clothing Gaussians` 扩展时，就知道该改哪里了。
