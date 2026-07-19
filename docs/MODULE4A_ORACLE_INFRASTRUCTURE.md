# Module 4A — Full-Attribute Representation Oracle Infrastructure

Status: **PASS_WITH_CUDA_NONDETERMINISM**

Final adjudication commit basis: `15457ffaa3753854f19fbc2ad35f0c150067af75`

The Oracle method and infrastructure pass. Checkpoint model/optimizer/RNG/
sampler/global-step restoration, graph/interpolation/config fingerprints,
pre-raster tensors, and the no-raster optimizer step are exact. Functional
interrupted resume therefore passes.

The current gsplat/CUDA raster backward contains nondeterministic reductions,
so independent processes are not expected to reproduce an identical multi-step
parameter trajectory. The first observed difference occurs in CUDA raster
backward attribute gradients; independent control processes show the same
single-step noise scale as resumed execution. This limitation must not be
described as approximate checkpoint restore, optimizer restore error, or resume
failure.

Historical PARTIAL reports and their original numeric measurements remain valid
audit evidence. This final adjudication supersedes only their status label.

The representation oracles optimize one outfit-level canonical residual shared by
every pose and camera. They are capacity upper bounds, not inference methods:

```yaml
oracle_type: representation_capacity_upper_bound
```

Neither oracle accepts reference images, image embeddings, cloth IDs, Module 2
predictions, Module 3 gates, or frame/camera-specific residuals.

## Oracle paths

- `GaussianResidualOracle` stores bounded six-channel residuals and two soft
  gates directly at the base Gaussian count.
- `AnchorResidualOracle` stores the same contract at canonical anchors and uses
  `interpolate_anchor_clothing_residuals` / `interpolate_anchor_field` from the
  formal Module 2 implementation.
- Both paths use `compose_canonical_gaussian_overrides` from Module 1.

Geometry gates affect xyz, log scaling, and rotation. Appearance gates affect
SH0 and SHN. Opacity uses the maximum of the two gates. All raw residuals start
at zero and gate probability starts at 0.05.

## Stages

0. Base evaluation.
1. Appearance: SH0, opacity, appearance gate.
2. Geometry: xyz, scaling, rotation, opacity, SH0, both gates.
3. Joint refinement of the Stage 2 channels.
4. Optional SHN, only with explicit configuration.

The default configuration keeps SHN disabled and effective SH degree zero.

## Formal commands

```bash
python train_full_attribute_oracle.py \
  --manifest /path/full_manifest.json \
  --outfit-id O01 \
  --oracle-type gaussian \
  --split /path/oracle_split_v1.json \
  --output /path/O01/gaussian

python train_full_attribute_oracle.py \
  --manifest /path/full_manifest.json \
  --outfit-id O01 \
  --oracle-type anchor \
  --split /path/oracle_split_v1.json \
  --output /path/O01/anchor
```

Generate the deterministic 140/30/30 split first:

```bash
python tools/build_oracle_condition_split.py \
  --manifest /path/full_manifest.json \
  --outfit-id O01 \
  --output /path/oracle_split_v1.json
```

The same commands apply to O00 and O10 after each outfit passes the Full Dataset
Checker. Synthetic fixture results must remain labelled `fixture_only: true`.
