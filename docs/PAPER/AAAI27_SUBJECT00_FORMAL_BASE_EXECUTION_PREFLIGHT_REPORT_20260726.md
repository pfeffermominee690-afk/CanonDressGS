# AAAI27 Subject00 Formal Base Execution Preflight Report

Task ID: `AAAI27-SUBJECT00-FORMAL-BASE-EXECUTION-PREFLIGHT-001`

The Subject00 formal-base execution contract is frozen and ready for user authorization. No training was launched. No output root, attempt directory, checkpoint, dataset file, or paper manuscript was created or mutated.

## Result

- Final classification: `SUBJECT00_FORMAL_BASE_EXECUTION_CONTRACT_READY_PENDING_USER_AUTHORIZATION`
- Next task: `USER_AUTHORIZE_SUBJECT00_FORMAL_BASE_101245_STEP_EXECUTION`
- Training authorized: `false`
- Optimizer steps: `0`
- Output root created: `false`

## Hard Gates

- Source branch/head: PASS, `research/subject00-storage-reclamation-execution-20260725` at `d541e46a7963502d1e6613cbdfa477fdb4e82d6c`
- Cloud identity: PASS, `autodl-container-ef19489c10-464381bb` as `root`
- Project root: PASS, `/root/autodl-tmp/canondressgs_work`
- Output root absent: PASS, `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-FORMAL-BASE-101245-001`
- GPU: PASS, RTX 4090, 24081 MiB free, no compute processes
- Runtime env: PASS, `/root/autodl-tmp/conda_envs/mmlphuman`

## Frozen Assets

- Data manifest SHA256: `e61ca6061f0a85be9c5c6e1d9163341c10749bc1a4ff1b14d27f64b5705aa99e`
- Data tree SHA256: `2c0f894f70d944fa8d6ebd48ab1188b78cb19b673f0b7c006f1922a592b1ea7b`
- RGB/mask pairing: PASS, `59704` pairs, `296` identical official missing entries
- Calibration SHA256: `4b2d98d20740da03be209040851fab7e8801e5670937ed9ad290a20264d1dde7`
- `smpl_params.npz` SHA256: `ac2738c308ad1a9cc02e7b63323c75e0eab28bddc88c57bb5887e07bf8ea28d2`
- Template SHA256: `f10a3b516e2b3a2ad38dc4924a3692b2f3e72a6cc9e66f3c0063c4e9cd210031`
- Surface LBS weights SHA256: `56aa68a9d4baade67621fa2bfac462ac88074eeaf7c9bfdbe86f2360b91261a1`
- Surface attachment manifest SHA256: `de6cd51fe3f81e29139b45494860b186038b75a378b101e7428252b3b66c1af8`

## Initialization

Only legal initialization is the Subject00 short-canary step-0 checkpoint:

- Path: `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/checkpoints/step_000000.pth`
- Step: `0`
- Bytes: `701938720`
- SHA256: `29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a`

## Storage

Live free bytes are `46166728704`, above the formal pre-attempt gate `32212254720`. Conservative peak-write projection is tight:

- Steady-state increment: `12280653758`
- Peak-write increment: `14231374378`
- Projected final free bytes: `33886074946`
- Projected minimum free bytes: `31935354326`
- Status: `PASS_LIVE_FREE_BYTES_EXCEED_FORMAL_PRE_ATTEMPT_GATE_WITH_TIGHT_CONSERVATIVE_PEAK_MARGIN`

## Outputs

- Execution contract: `paper_protocol/subject00_formal_base/SUBJECT00_FORMAL_BASE_101245_EXECUTION_CONTRACT_20260726.md`
- Execution manifest: `paper_protocol/subject00_formal_base/subject00_formal_base_101245_execution_manifest_draft_20260726.json`
- Preflight summary: `paper_protocol/subject00_formal_base/subject00_formal_base_101245_preflight_summary_20260726.json`
- Preflight tests: `paper_protocol/subject00_formal_base/subject00_formal_base_101245_preflight_tests_20260726.json`
- Handoff: `paper_protocol/subject00_formal_base/subject00_formal_base_101245_handoff_20260726.json`
- Validator: `tools/second_identity/validate_subject00_formal_base_preflight.py`
- Pytest wrapper: `tests/test_subject00_formal_base_preflight_contract.py`
