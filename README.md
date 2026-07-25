# CanonDressGS

**Geometry-Safe Reference-Controlled Garment Editing for Animatable Gaussian Avatars**

CanonDressGS studies post-hoc garment editing for a pretrained animatable
Gaussian avatar. Given one or more clothing-masked garment reference images,
the system realizes a registered canonical garment state while keeping the
identity, deformation model, and renderer frozen.

This repository is based on
[MMLP-Human](https://gapszju.github.io/mmlphuman/) and retains its base-avatar
training and rendering code.

## Method Overview

CanonDressGS separates garment representation from reference control:

1. **Teacher Endpoint Bank** optimizes one canonical Gaussian residual field
   for each registered garment on a frozen avatar.
2. **Reference-Controlled Garment Realization** extracts frozen spatial
   features from masked references, predicts coefficients in an explicit
   garment residual basis, and realizes the nearest verified endpoint.
3. **Geometry-Safe Composition** keeps two endpoint geometries intact and
   weights their effective opacity instead of interpolating Gaussian position,
   scale, or rotation.

The current verified research setting is deliberately narrow:

- one fixed THuman4.0 `subject02` identity;
- a closed wardrobe of five seen garments (`O01`, `O02`, `O03`, `O04`, and
  `O08`);
- registered pose and camera conditions;
- reference-controlled endpoint realization;
- user- or oracle-specified Dual-Support composition.

The current results do **not** establish unseen-garment generation,
cross-identity transfer, arbitrary real-world virtual try-on, or an automatic
mixed-reference controller.

## Repository Status

This is an active research repository.

| Component | Status |
| --- | --- |
| MMLP-Human base-avatar training and rendering | Available |
| Dataset preprocessing and LBS-volume generation | Available |
| CanonDressGS paper/configuration scaffold | Available |
| Frozen endpoint protocol and final evaluation scripts | Being consolidated for release |
| Pretrained checkpoints and derived benchmark data | Not included |

`configs/canon_dress_gs.yaml` is an early project scaffold and is not the
frozen paper protocol. Reproducible CanonDressGS commands and assets will be
documented here when the release package is finalized.

## Installation

The tested base environment uses Python 3.10, CUDA 12.1, and PyTorch 2.4.1.

```bash
conda create -n canondressgs python=3.10
conda activate canondressgs

pip install torch==2.4.1 torchvision "numpy<2.0" \
  --index-url https://download.pytorch.org/whl/cu121
pip install iopath ninja jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu121
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html
pip install imageio numba omegaconf open3d opencv-python scipy smplx \
  scikit-image tensorboardx tensorboard trimesh websockets torchmetrics \
  websocket-client dearpygui plyfile torch_pca
```

Download the [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php) neutral model
and place it at:

```text
smpl_model/smplx/SMPLX_NEUTRAL.npz
```

## Data Preparation

The base avatar code supports
[AvatarReX](https://github.com/lizhe00/AnimatableGaussians/blob/master/AVATARREX_DATASET.md),
[ActorsHQ](https://actors-hq.com/), and
[THuman4.0](https://github.com/ZhengZerong/THUman4.0-Dataset).
Users must obtain each dataset and SMPL-X under their respective licenses.

Generate the LBS weight volume after compiling `PointInterpolant` as described
by [AnimatableGaussians](https://github.com/lizhe00/AnimatableGaussians/blob/master/gen_data/GEN_DATA.md#preprocessing):

```bash
cd script
python gen_weight_volume.py \
  --data_dir /path/to/dataset \
  --smpl_path ../smpl_model/smplx/SMPLX_NEUTRAL.npz
```

For loose clothing, place a template mesh at
`DATASET_DIR/gaussian/template.ply`. A subject02 template is included under
`template/subject02/`.

## Base Avatar Training

```bash
python train.py \
  --config ./config/subject02.yaml \
  --data_dir /path/to/subject02 \
  --out_dir /path/to/base_avatar
```

Large datasets, generated assets, model checkpoints, and experiment outputs
are intentionally excluded from Git.

## Rendering and Viewer

Start the viewer:

```bash
cd viewer
python net_viewer.py
```

Connect a trained model:

```bash
python visualize.py --model_dir /path/to/model --ip 127.0.0.1 --port 6009
```

Render evaluation images:

```bash
python test.py \
  --config ./config/subject02.yaml \
  --model_dir /path/to/model \
  --out_dir /path/to/renders \
  --data_dir /path/to/subject02
```

## Project Layout

```text
config/        MMLP-Human dataset and avatar configurations
configs/       CanonDressGS research configuration scaffolds
scene/         Gaussian avatar models and scene components
script/        Dataset preprocessing and evaluation utilities
tools/         Research checks and experiment utilities
paper_draft/   Manuscript scaffold
viewer/        Interactive viewer
```

## Acknowledgements

CanonDressGS builds on
[MMLP-Human](https://github.com/gapszju/mmlphuman),
[gsplat](https://github.com/nerfstudio-project/gsplat),
[AnimatableGaussians](https://github.com/lizhe00/AnimatableGaussians), and
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
Please follow the licenses and citation requirements of the upstream projects
and datasets.

## Citation

The CanonDressGS manuscript is in preparation. Please cite the MMLP-Human
backbone when using the current base-avatar implementation:

```bibtex
@inproceedings{zhan2025realtime,
  title     = {Real-time High-fidelity Gaussian Human Avatars with Position-based Interpolation of Spatially Distributed MLPs},
  author    = {Zhan, Youyi and Shao, Tianjia and Yang, Yin and Zhou, Kun},
  booktitle = {CVPR},
  year      = {2025}
}
```

## License

The repository is released under the [MIT License](LICENSE). Third-party code,
models, and datasets remain subject to their original licenses.
