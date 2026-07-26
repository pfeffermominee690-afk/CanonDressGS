# Subject00 24-Cell Mask Generation Execution Contract

## Authority and immutable input

- Preflight task: `AAAI27-SUBJECT00-24-CELL-MASK-GENERATION-PREFLIGHT-001`
- Accepted source: `research/subject00-24-cell-accepted-promotion-20260726` at `fdbe9e74104641b242d1c33242a362f8822c113f`
- Accepted registry: `E:\model_train\canondressgs_subject00_24_cell_mask_generation_preflight\paper_protocol\reviewer_risk\subject00_global_accepted_cell_registry_24of24_20260726.json`
- Accepted cells: `24` (`O01/O03/O04 = 8/8/8`)
- Accepted with disclosed limitations: `9`
- Raw/source files and SHA256 values: frozen per execution manifest
- Raw resizing or modification: forbidden

## Selected method

The unique method is
`SUBJECT02_SEGFORMER_V3A_STRATEGY_C_SUBJECT00_SPECIALIZATION_V1`:
`HUMAN_PARSING_WITH_GARMENT_LABEL_MAPPING`, specialized from the successful
Subject02 SegFormer V3-A Strategy-C pipeline.

- Executor: `E:\model_train\canondressgs_subject00_24_cell_mask_generation_preflight\tools\datasets\execute_subject00_24_cell_masks_from_frozen_contract.py`
- Executor SHA256: `b602df821f30046941bd98956d5afc552f36c7c72f1fe4af5c58621713a36693`
- Model: `mattmdjaga/segformer_b2_clothes` revision `584abc1e1d260e23c0fc627c5217a09b2b461046`
- Weights: `E:\data_pre\audit_subject02_layered_composite_v3\mask_backend_closure_v3a\m\model.safetensors`
- Weights SHA256: `8f86fd90c567afd4370b3cc3a7e81ed767a632b2832a738331af660acc0c4c68`
- Environment: `D:\miniconda3\envs\garment-mask-v3a\python.exe`
- Execution device: CUDA GPU, batch size 1, estimated minimum VRAM 4 GiB
- Downloads, installation, network access and external APIs: forbidden

Person foreground is the largest complete human component from parser labels
1-15 and 17. Background and bag/handheld-object label 16 are excluded.
Garment is Strategy-C probability >= 0.35 over labels 4/5/6/7/8/17, intersected
with person foreground and stripped of protected/non-garment labels. Therefore
`GARMENT_MASK ⊆ PERSON_MASK` is mandatory.

The two documented registration failures do not drive generation: neither
person nor garment masks are propagated from source registration. Their source
person masks remain review-only priors and both cells require dedicated human
review.

## Output and order

- Attempt namespace: `attempt_001_subject00_24_cell_person_garment_masks`
- Attempt root: `E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-MASKS-001\attempt_001_subject00_24_cell_person_garment_masks`
- Expected formal masks: `24 person + 24 garment = 48`
- Format: 8-bit single-channel `L` PNG, strict values `0/255`, lossless,
  no soft alpha or antialiasing, exact accepted-raw native width/height
- Request order: garment `O01`, `O03`, `O04`; slot `00` through `07`
- Exact paths and request bindings: `E:\model_train\canondressgs_subject00_24_cell_mask_generation_preflight\paper_protocol\reviewer_risk\subject00_24_cell_mask_generation_execution_manifest_20260726.json`

The attempt root is absent and was not created by preflight. Execution may
create only the frozen directory layout. No `attempt_002` may be created.

## QA and review

Person, garment, pair and cross-view gates are frozen in `E:\model_train\canondressgs_subject00_24_cell_mask_generation_preflight\paper_protocol\reviewer_risk\subject00_24_cell_mask_quality_gate_registry_20260726.json`.
Subject02 distributions are reference/escalation evidence; thresholds may not
be retuned from the 24 Subject00 images. Every cell remains unaccepted until
the three human decisions are filled. Review assets include 72 per-cell views,
three 8-view contact sheets, high-risk/override pages and a decision matrix.

## Failure, retry and recovery

Each cell is independent. A technical failure is recorded with traceback and
is never retried automatically. No previous-cell mask may be substituted.
Partial successful masks are retained. Resume skips only records whose stored
person/garment SHA256 values still match; failed or tampered records are not
recomputed without a new audit/authorization. The method may not switch, raw
may not resize, and `attempt_002` is forbidden.

## Teacher boundary and authorization

`MASK_GENERATION_AUTHORIZED=false`,
`TEACHER_TARGET_CREATION_AUTHORIZED=false`, and `TEACHER_TARGET_COUNT=0`.
This preflight performed no inference and created no mask. A separate exact
authorization `EXECUTE_SUBJECT00_24_CELL_MASK_GENERATION_FROM_FROZEN_CONTRACT` is required to run the executor. Mask generation
does not itself authorize Teacher-target creation, Teacher Endpoint
optimization or CanonDressGS training. The execution manifest freezes the
future Teacher registry fields for raw/person/garment paths and SHA256 values,
camera, slot, garment, direction, accepted source binding and human acceptance.

Supporting registries:

- semantics: `E:\model_train\canondressgs_subject00_24_cell_mask_generation_preflight\paper_protocol\reviewer_risk\subject00_24_cell_mask_semantics_registry_20260726.json`
- feasibility: `E:\model_train\canondressgs_subject00_24_cell_mask_generation_preflight\paper_protocol\reviewer_risk\subject00_24_cell_mask_pipeline_feasibility_audit_20260726.json`
- quality: `E:\model_train\canondressgs_subject00_24_cell_mask_generation_preflight\paper_protocol\reviewer_risk\subject00_24_cell_mask_quality_gate_registry_20260726.json`
- storage: `E:\model_train\canondressgs_subject00_24_cell_mask_generation_preflight\paper_protocol\reviewer_risk\subject00_24_cell_mask_generation_storage_estimate_20260726.json`

Final classification: `SUBJECT00_24_CELL_MASK_GENERATION_CONTRACT_READY_FOR_EXECUTION`.
Next task: `EXECUTE_SUBJECT00_24_CELL_MASK_GENERATION_FROM_FROZEN_CONTRACT`.
