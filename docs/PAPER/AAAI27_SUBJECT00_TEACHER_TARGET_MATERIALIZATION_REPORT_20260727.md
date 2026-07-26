# Subject00 Teacher-target materialization report

Task `AAAI27-SUBJECT00-TEACHER-TARGET-MATERIALIZATION-WITH-QUARANTINE-001` completed Stage A only. The immutable dataset contains 24 provenance
records, 22 camera-safe training/evaluation records, and two review-only camera
quarantines. No optimizer step, training, generation, mask inference, Formal Base
resume, checkpoint mutation, or paper-body modification occurred.

## Materialized dataset

- Windows root: `E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-TEACHER-TARGETS-001\attempt_001_subject00_24_cell_teacher_targets`
- Cloud root: `/root/autodl-tmp/canondressgs_work/teacher_targets/SUBJECT00-24CELL-001/attempt_001`
- Schema: `canondressgs.full_dataset.v1`
- Training coverage: O01 7, O03 7, O04 8
- Review-only quarantine: `subject00_O03_slot04_canary_attempt004_cand00`,
  `subject00_O01_slot04_remaining_attempt005_cand00`
- Archive: `E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-TEACHER-TARGETS-001\attempt_001_subject00_24_cell_teacher_targets\12_portable_archive\subject00_24cell_teacher_targets_attempt001.tar.zst`
- Archive SHA256: `d7079aad688a0c74a7f9ed97d000968a184c041e12db5b914c4d9699b0e9ea81`
- Cloud loader smoke: `PASS_ZERO_OPTIMIZER_CPU_LOADER_SMOKE`

## Scientific boundaries

The two quarantined records retain accepted raw/masks and provenance but have
`camera_model=null`, `K_target=null`, and zero camera-derived targets. O03 is
explicitly provisional and uses the camera-safe slots 00, 01, 02, 03, 05, 06,
and 07 only (`PROVISIONAL_O03_USES_CAMERA_SAFE_7_OF_8_VIEWS`); it is not paper
eligible.

The V5.3-compatible task wrapper uses the frozen radius 5. It constructs the
base RGB/foreground by the frozen source-to-target similarity, protects the
accepted target person-minus-garment region, and uses an explicit conservative
empty revealed-skin mask because new parser/mask inference was forbidden.

## Result

`SUBJECT00_TEACHER_TARGET_MATERIALIZATION_PASS_22_TRAINING_2_QUARANTINED`

Only next task: `RUN_SUBJECT00_O03_PROVISIONAL_TEACHER_CAMERA_SAFE_7VIEW_RERUN`. It was not started.
