# AAAI27 Dual-Support Controller Training Contract Correction

**RESEARCH METHOD CONTRACT — NOT PAPER FINAL**

## Pre-result stop

The first formal training request stopped at source HEAD `174655ce6aabcae5d60ce45f3b4be319eb71e914` with `CONTROLLER_TRAINING_CONTRACT_INCOMPLETE`. No formal branch, local/cloud formal worktree, output root, or attempt was created. Training, forward training batches, backward, optimizer creation/step, scheduler step, checkpoint write, render, evaluation, visual review, and formal-output mutation were all zero. No scientific result existed when this repair began.

The missing fields were the 320/20 record roles, five-record batch grouping, pair and assignment order, formal-pure role, 300-step wrap rule, and a unique scientific data-order hash. This correction changes only those pre-result contract fields. Controller architecture, soft labels, loss, optimizer hyperparameters, fallback thresholds, evaluator thresholds, teacher bank, renderer, F2, historical outputs, and the sealed training manifest remain unchanged.

## Training and evaluation scopes

Formal training uses all 320 protocol query records: 80 protocol-pure and 240 mixed. The 80 consistent duplicate protocol records remain independent record IDs with their original pair/fold provenance. They are not deduplicated, merged, reweighted, or sampled less often.

The 20 `formal_pure_endpoint_episodes` are `FORMAL_PURE_EVAL_ONLY`. Their record IDs never enter the 64-batch cycle or 300-step schedule, their training exposure and optimizer-target use are zero, and they cannot select a checkpoint or tune a threshold. Record-level evaluation isolation is therefore PASS. Their reference images/logical inputs overlap the closed seen-wardrobe protocol training assets (`20/20` logical inputs overlap), so this report does not claim image-level independence.

The 320-record mixed classification task remains a closed-wardrobe protocol-fit evaluation. It is not unseen-reference, novel-view, cross-identity, or unseen-garment generalization.

## Frozen ordering and balanced cycle

Outfit order is `O01,O02,O03,O04,O08`. Pair order is `O01_O02,O01_O03,O01_O04,O01_O08,O02_O03,O02_O04,O02_O08,O03_O04,O03_O08,O04_O08`. Fold order is `cond_000000,cond_000318,cond_000017,cond_000347`.

Dominant outfit is the unique argmax of the soft target. Each dominant outfit has a 64-record queue, divided into four 16-record folds. Within a fold, counterparts follow frozen outfit order excluding the dominant. Each counterpart contributes `PURE_DOMINANT`, then mixed assignment positions `0,1,2`. For pair-side A this is `AAA,AAB0,AAB1,AAB2`; for pair-side B it is `BBB,ABB0,ABB1,ABB2`.

For `slot=0..15` and `fold=0..3`, `batch_index=4*slot+fold`. Each batch takes queue index `slot` from that fold for all five dominant outfits, in frozen outfit order. Thus every batch has five records, one per dominant outfit, and one target-view fold. The 64-batch cycle covers all 320 protocol record IDs exactly once; every fold has 16 batches/80 records.

## Frozen 300-step wrap

For 1-based step `s`, `cycle_index=(s-1)//64` and `batch_index=(s-1)%64`. Steps 1–256 are four complete cycles. Steps 257–300 are batch indices 0–43 of cycle 4. Total scheduled exposure is 1,500 samples. The 220 records in batches 0–43 appear five times; the 100 records in batches 44–63 appear four times. This fixed asymmetry is preregistered and must not be reordered after results.

All three seeds use the identical schedule and record/member/fold order; only random model initialization may vary. DataLoader shuffle, per-seed sampler, per-seed augmentation, and seed-dependent wrap offsets are forbidden.

## Scientific hashes

Canonical scientific JSON is UTF-8 with LF, sorted keys, no NaN, fixed JSON floating representation, and no absolute machine paths. Runtime absolute paths are stored separately and do not participate in either scientific hash.

- Training cycle SHA256: `f60b0ee64ff5ce2b6693f53dd7eb665cc6aee5aa310f314175d2932307d57c77`.
- 300-step data-order SHA256: `63902a2f36ccf864bfe6028654be0b162417c3adc6b99e1455b30fdab8a2bacf`.
- Seed data-order hash unique count: `1`.

Protocol SHA changed from `675cd131e8ab38ad0a348145598aba14aac5f3eee1c9e07b1036892e5fc07e2e` to `44ef0c53f7a5ec4fe19733371ffefb1be14d5b6902e7f4060707edd4660d9f49`. Training-manifest SHA remained `a0dbfe98a25b7f82f198405cf41c1b8625b6faa63499396a78683d5d326ce8e3`.

## Verification and decision

The schedule module uses only Python standard-library data operations. It imports no model, optimizer, renderer, PyTorch, gsplat, or PyTorch3D module. The repair and original controller-design tests passed `45/45` before archive sealing. JSON/YAML parse, Python compilation, Markdown path, source-archive immutability, and Git whitespace checks are final gates.

`CONTROLLER_TRAINING_CONTRACT=REPAIRED_AND_FROZEN`, `DATA_SCOPE=PASS`, `BALANCED_BATCH_CONTRACT=PASS`, `PAIR_ASSIGNMENT_ORDER=PASS`, `FORMAL_PURE_ISOLATION=PASS`, `300_STEP_SCHEDULE=PASS`, `DATA_ORDER_HASH=PASS`, and `NO_FORMAL_EXECUTION_GATE=PASS`.

Training steps executed=0, forward training batches=0, backward=0, optimizer created/steps=0, scheduler steps=0, checkpoint writes=0, formal renders/metrics/visual reviews=0, frozen mutation=0, and `PAPER_FINAL=0`.

The next task is `TRAIN_AND_EVALUATE_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER_FROM_REPAIRED_CONTRACT`. It has not been started.
