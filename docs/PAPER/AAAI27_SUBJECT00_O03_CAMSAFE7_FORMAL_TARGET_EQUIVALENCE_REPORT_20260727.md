# Subject00 O03 camera-safe7 formal-target equivalence audit

Task: `AAAI27-SUBJECT00-O03-CAMSAFE7-FORMAL-TARGET-EQUIVALENCE-AUDIT-001`

## Outcome

The completed camera-safe7 rerun **cannot be positively bound** to the full
formal materialized O03 Teacher-target contract.  The result is class **D -
FORMAL_TARGET_SCIENTIFIC_FIELD_MISMATCH**.

The seven accepted raw images, seven person masks, seven garment masks, request
order, denominator, slot04 exclusion, and all canonical camera matrices are
exact.  However, the rerun loader/loss consumes only
`raw/person/garment/protected/boundary` plus live Base60747 render/alpha values,
whereas the formal `canondressgs.full_dataset.v1` loader also requires
precomputed base targets and six additional edit/preserve supervision masks.
Those are scientific training inputs, not display-only metadata.

## Exact shared evidence

| Evidence | Result |
|---|---:|
| Raw SHA / pixels | 7/7 / 7/7 |
| Person-mask SHA / pixels | 7/7 / 7/7 |
| Garment-mask SHA / pixels | 7/7 / 7/7 |
| Canonical camera records | 7/7 |
| Camera max absolute / relative difference | 0.0 / 0.0 |
| Protected tensors | 7/7 exact |
| Runtime 3x3 garment boundary from shared garment mask | 7/7 exact |
| Runtime garment-masked RGB | 7/7 exact |
| Request order and denominator | exact / 7 |
| slot04/cam11 | absent; sample count 0 |

## Scientific mismatches

Eight formal fields are absent from or differently bound by the rerun:

- `target_base_rgb`
- `target_edit_mask`
- `target_edit_core_mask`
- `target_preserve_mask`
- `target_transition_mask`
- `target_base_foreground_mask`
- `target_old_clothing_mask`
- `target_revealed_skin_mask`

`target_base_rgb` and `target_base_foreground_mask` are especially decisive:
the formal dataset binds precomputed, camera-warped source-image targets, while
the rerun loss binds a live Base60747 render and alpha.  The other six fields
are present in the formal loader but absent from the rerun loader/loss
contract.  The formal `target_transition_mask` is also not an alias of the
rerun 3x3 garment boundary (0/7 exact; every view has pixel differences).

## Checkpoint and review-pack discovery

All five checkpoints at steps 0/300/600/900/1200 are byte-verified and remain
fully bound to the sealed rerun snapshot.  That evidence is complete, so the
special checkpoint-evidence failure classification does not apply.

The 11-page human-review PDF exists at:

`/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-CAMSAFE7-001/attempt_001/review/human_review_upload_pack_20260727/06_indexes/subject00_O03_camsafe7_provisional_teacher_human_review_pages_20260727.pdf`

SHA256: `f24b1010426343401a9552875898158a8cf91f3b0220e9fb4da92e8bc97f8c8f`.  Its cloud reporting artifacts exist but are
still uncommitted in the separate review-pack worktree; this audit did not
modify or regenerate them.

## Safety and disposition

- Optimizer steps: 0
- GPU forward calls: 0
- Target/mask/camera/checkpoint mutations: 0
- Formal Base: `USER_AUTHORIZED_PAUSED`, durable step 60747, resume unauthorized
- Paper eligibility: false
- Scientific pass: null
- Paper body modifications: 0
- Positive binding overlay: not created (the required overlay file records the rejection)

Final classification:
`SUBJECT00_O03_CAMSAFE7_RERUN_TARGET_MISMATCH_REQUIRES_CLEAN_FORMAL_TARGET_RERUN`

Next unique task:
`RUN_SUBJECT00_O03_PROVISIONAL_TEACHER_FROM_FORMAL_MATERIALIZED_7VIEW_TARGETS`
