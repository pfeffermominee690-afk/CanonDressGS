# AvatarReX LBN1 Plan B Targeted Extraction Report

Status: BLOCKED BEFORE EXTRACTION

## Classification

`AVATARREX_PLAN_B_EXTRACTION_CONTRACT_FAIL`

The frozen Plan B allowlist is valid and unique, but the fixed authorized cloud source path was not reachable from this environment. WSL reported `/root/autodl-tmp/avatarrex_lbn1.7z` missing, and `ssh autodl` failed hostname resolution. The verified Windows mirror was checked read-only and matched the expected bytes and SHA-256, but it was not used for extraction because the task binds the source archive to `/root/autodl-tmp/avatarrex_lbn1.7z`.

## Final Fields

1. TASK_ID: `AAAI27-AVATARREX-LBN1-TARGETED-EXTRACTION-PLAN-B-001`
2. SOURCE_BRANCH: `research/external-baseline-feasibility-from-bundle-rerun-20260726`
3. SOURCE_HEAD: `572637f08aef21fde35dcd7f8879ff74549c2afc`
4. NEW_BRANCH: `research/avatarrex-lbn1-targeted-extraction-plan-b-20260726`
5. WINDOWS_WORKTREE: `E:\model_train\canondressgs_avatarrex_lbn1_targeted_extraction_plan_b`
6. CLOUD_WORKTREE: `/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_avatarrex_lbn1_targeted_extraction_plan_b`
7. ARCHIVE_PATH: `/root/autodl-tmp/avatarrex_lbn1.7z`
8. ARCHIVE_BYTES: `12569755256`
9. ARCHIVE_SHA256: `531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1`
10. ARCHIVE_TEST_STATUS: `NOT_RUN_CLOUD_ARCHIVE_SOURCE_UNREACHABLE`
11. PLAN_ID: `PLAN_B_CANARY_001`
12. STAGING_ROOT: `/root/autodl-tmp/datasets/avatarrex_lbn1_staging/PLAN_B_CANARY_001/attempt_001`
13. ATTEMPT_ID: `attempt_001`
14. IDENTITY: `implicit_single_identity_lbn1`
15. SEQUENCE: `implicit_single_sequence`
16. CAMERA_IDS: `22053908, 22053926, 22010708, 22010710, 22010716, 22010714, 22070935, 22053923`
17. FRAME_RANGE_OR_SET: `100 exact frozen frame IDs in avatarrex_lbn1_plan_b_allowlist.txt`
18. ALLOWLIST_COUNT: `1602`
19. EXPECTED_UNCOMPRESSED_BYTES: `501198133`
20. CLOUD_FREE_BYTES_BEFORE: `NOT_OBTAINED_CLOUD_SOURCE_UNREACHABLE`
21. STORAGE_GATE_STATUS: `NOT_RUN_CLOUD_SOURCE_UNREACHABLE`
22. EXTRACTION_CALLS: `0`
23. EXTRACTED_FILE_COUNT: `0`
24. EXTRACTED_TOTAL_BYTES: `0`
25. RGB_COUNT: `0`
26. PHA_COUNT: `0`
27. RGB_PHA_PAIRING_STATUS: `NOT_RUN_EXTRACTION_NOT_STARTED`
28. CALIBRATION_STATUS: `NOT_RUN_EXTRACTION_NOT_STARTED`
29. BODY_PARAMETER_STATUS: `NOT_RUN_EXTRACTION_NOT_STARTED`
30. MISSING_FILE_COUNT: `NOT_COMPUTED_EXTRACTION_NOT_STARTED`
31. UNEXPECTED_FILE_COUNT: `NOT_COMPUTED_EXTRACTION_NOT_STARTED`
32. EMPTY_FILE_COUNT: `NOT_COMPUTED_EXTRACTION_NOT_STARTED`
33. SHA_REGISTRY_PATH: `NOT_CREATED_EXTRACTION_NOT_STARTED`
34. INVENTORY_PATH: `NOT_CREATED_EXTRACTION_NOT_STARTED`
35. ARCHIVE_MUTATIONS: `0`
36. FORMAL_DATA_ROOT_STATUS: `NONE`
37. PREPROCESSING_STEPS: `0`
38. TRAINING_STEPS: `0`
39. GENERATION_CALLS: `0`
40. PAPER_MODIFICATIONS: `0`
41. TEST_RESULT: `MANIFEST_CHECK_PASS; CONTRACT_TESTS_PASS`
42. COMMIT_HEAD: `RECORDED_IN_FINAL_RESPONSE_AFTER_COMMIT`
43. FINAL_REPORTING_HEAD: `RECORDED_IN_FINAL_RESPONSE_AFTER_COMMIT`
44. ORIGIN_SYNC_STATUS: `NOT_ATTEMPTED_BEFORE_COMMIT`
45. CLOUD_GIT_SYNC_STATUS: `NOT_ATTEMPTED_CLOUD_HOST_UNREACHABLE`
46. WORKTREE_CLEAN_STATUS: `PENDING_COMMIT`
47. PAPER_FINAL: `false`
48. FINAL_CLASSIFICATION: `AVATARREX_PLAN_B_EXTRACTION_CONTRACT_FAIL`
49. NEXT_TASK: `USER_RESTORE_CLOUD_ACCESS_AND_RERUN_AVATARREX_PLAN_B_PREFLIGHT`

## Boundaries

- ARCHIVE_MUTATIONS = `0`
- FORMAL_DATA_ROOT_STATUS = `NONE`
- PREPROCESSING_STEPS = `0`
- TRAINING_STEPS = `0`
- GENERATION_CALLS = `0`
- PAPER_MODIFICATIONS = `0`
- PAPER_FINAL = `false`

No formal raw-data root was created. No preprocessing, template generation, LBS grid generation, Base Avatar training, garment generation, API call, accepted set creation, Teacher target generation, or paper modification was performed.

## Created Control Artifacts

- `tools/execute_avatarrex_lbn1_plan_b.py`
- `paper_protocol/avatarrex_lbn1_plan_b/manifests/avatarrex_lbn1_plan_b_allowlist.txt`
- `paper_protocol/avatarrex_lbn1_plan_b/manifests/avatarrex_lbn1_plan_b_allowlist.json`
- `paper_protocol/avatarrex_lbn1_plan_b/avatarrex_lbn1_plan_b_manifest_check.json`
- `paper_protocol/avatarrex_lbn1_plan_b/avatarrex_lbn1_plan_b_blocked_summary.json`
- `project_control_handoff/avatarrex_lbn1_plan_b_handoff.json`
- `tests/test_avatarrex_lbn1_plan_b_extraction_contract.py`
