# Subject00 Checkpoint Reclamation Plan

Task: `AAAI27-SUBJECT00-CHECKPOINT-RECLAMATION-ADJUDICATION-001`

## Guardrails

- Checkpoint deletion/movement/compression in this task: `0 / 0 / 0`
- Scientific output, dataset, attempt and manifest mutation: `0`
- Formal Base execution and optimizer steps: `0`
- Renderer/API calls: `0 / 0`
- PAPER_FINAL: `false`
- Cloud free bytes at inventory: `17104633856`
- Windows E: free bytes: `115592388608`
- Contract/safety/45-GiB targets: `32212254720 / 40802189312 / 48318382080`

## Inventory

- Checkpoints: `3733`
- Logical checkpoint bytes: `72582423760`
- Duplicate SHA groups: `40`
- Short-canary path: `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/checkpoints/step_000000.pth`
- Short-canary SHA256: `29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a`

| Classification | Count | Bytes |
|---|---:|---:|
| `CRITICAL_ACTIVE_KEEP` | 670 | 6475388881 |
| `SEALED_PROVENANCE_KEEP` | 2344 | 28044766919 |
| `UNIQUE_FINAL_KEEP` | 15 | 1835148753 |
| `DUPLICATE_SAFE_DELETE_CANDIDATE` | 70 | 293877819 |
| `ARCHIVE_THEN_DELETE_CANDIDATE` | 11 | 9735410800 |
| `REGENERABLE_DELETE_CANDIDATE` | 0 | 0 |
| `TEMPORARY_OPTIMIZER_STATE_CANDIDATE` | 1 | 219104837 |
| `UNKNOWN_KEEP` | 622 | 25978725751 |

## Recovered-Report Boundary

The recovered report classifies `/root/autodl-tmp/canondressgs_work/outputs` as protected class C and `pipeline_full` as sealed scientific provenance with zero safe migration/deletion bytes. This checkpoint plan does not override that conclusion. The archive candidates below are outside that protected root.

## Archive Candidates

| Cloud source | Windows destination | Files | Logical bytes | Allocated bytes |
|---|---|---:|---:|---:|
| `/root/autodl-tmp/outputs/diffusion_piper_50k` | `E:\canondressgs_archive\cloud_checkpoints\diffusion_piper_50k` | 12 | 3334819751 | 3334852608 |
| `/root/autodl-tmp/outputs/diffusion_30k` | `E:\canondressgs_archive\cloud_checkpoints\diffusion_30k` | 8 | 3200315309 | 3200339968 |
| `/root/autodl-tmp/outputs/irregular_diffusion_30k` | `E:\canondressgs_archive\cloud_checkpoints\irregular_diffusion_30k` | 8 | 3200315345 | 3200331776 |

Every source must remain untouched until rsync exits zero, file counts and logical bytes match, all relative-path SHA256 values match, a manifest/seal is committed, and the user authorizes the exact retirement path.

## Plans

| Plan | Checkpoints | Archive bytes | Delete bytes | Projected free | Contract margin | Safety margin | Reaches 45 GiB |
|---|---:|---:|---:|---:|---:|---:|---|
| `PLAN_CKPT_A` | 70 | 0 | 293877819 | 17398511675 | -14813743045 | -23403677637 | false |
| `PLAN_CKPT_B` | 70 | 0 | 293877819 | 17398511675 | -14813743045 | -23403677637 | false |
| `PLAN_CKPT_C` | 11 | 9735450405 | 9735524352 | 26840158208 | -5372096512 | -13962031104 | false |
| `PLAN_CKPT_D` | 82 | 9735450405 | 10248507008 | 27353140864 | -4859113856 | -13449048448 | false |

`PLAN_CKPT_D` is the recommended checkpoint-only combination, but it still misses the contract, safety, and 45-GiB targets. No checkpoint-only execution can unblock Formal Base under the present conservative evidence. The separately recovered original AvatarReX storage plan remains unchanged and is not silently folded into this checkpoint plan.

## Classification

- Final classification: `SUBJECT00_CHECKPOINT_SAFE_DELETE_SPACE_INSUFFICIENT`
- NEXT_TASK: `EXECUTE_USER_SELECTED_SUBJECT00_CHECKPOINT_RECLAMATION_PLAN`
- Automatic execution authorized: `false`
