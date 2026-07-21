# Reproducibility Checklist Skeleton

This is a truth-preserving draft. `CONFIRMED` means supported by checked-in code or frozen artifacts; `PENDING` and `TODO` must remain unresolved until evidence exists.

| Topic | Status | Current information | Required completion |
| --- | --- | --- | --- |
| Method identity | CONFIRMED / FINAL CHOICE PENDING | Ours-v2 candidate is frozen F2 set mean/max, LayerNorm, zero-initialized Linear(4), and standardized coefficient SmoothL1 only. | Replace candidate wording only after P0 adjudication. |
| Ours-v2 replicates | CONFIRMED DEFINITION | Three deterministic protocol replicate indices (0/1/2); initial trainable state and output are bitwise identical across configured seeds. | Report completed replicate outcomes without independent-random-initialization language. |
| Random-seed methods | CONFIRMED DEFINITION | B6, M3, and M4 use seeds 0/1/2; M3/M4 share an initial trunk within each seed. | Record final seed-specific commands and output hashes. |
| Fixed method | CONFIRMED DEFINITION | B7 is non-trainable, has no optimizer or seed, and yields one fixed result. | Preserve fold construction manifests and target-exclusion proof. |
| Run count | TODO | Static candidate plan: 13 entries (12 trainable, 1 fixed). No completed formal run count is asserted in this manuscript branch. | Fill completed/failed/sealed counts after authorized execution; include ablations separately. |
| Hardware | CONFIRMED FOR PROTOCOL AUDIT / FINAL RUN PENDING | Audited environment reports one NVIDIA GeForce RTX 4090 with PyTorch 2.4.1+cu121 and CUDA 12.1. | Verify and report the hardware/software record from every formal run; do not assume it matches the audit host. |
| Frozen base and renderer | CONFIRMED | Base/MMLP-Human checkpoint, base fingerprint, renderer source bundle, and fixed Gaussian count are listed in the frozen asset manifest. | Include manifest hash and artifact availability statement in the final package. |
| Frozen teachers | CONFIRMED | Five seen teacher checkpoints and the held-out O07 diagnostic teacher are fingerprinted; seen teachers are stored at step 1,200. | Provide licensed/releasable artifact route or deterministic regeneration instructions. |
| Frozen basis/normalization | CONFIRMED | Selected rank-4 explicit basis and seen-train-only coefficient mean/std are fingerprinted. | Export a portable schema description and verification command. |
| Frozen features | CONFIRMED | Reference feature cache is derived from the frozen F2 backbone; target inputs are excluded from coefficient prediction. | Document cache regeneration and checksum verification. |
| Data split | CONFIRMED | Seen wardrobe: O01/O02/O03/O04/O08. Four fixed conditions. O07 is held-out failure-only; it is excluded from basis, normalization, fitting, and seen aggregation. | Document data provenance, license, preprocessing, and any release restrictions. |
| View protocol | CONFIRMED | Every correct episode uses three target-disjoint references, but teachers/basis/predictor fitting cover all four conditions; classification is VIEW-TRANSDUCTIVE. | Repeat this boundary in captions, release README, and evaluator documentation. |
| Teacher training steps | CONFIRMED HISTORICAL ASSET | 1,200 Adam steps per frozen teacher, four-view round robin, fixed teacher objective and protected guard. | Preserve exact teacher config/commit and note any pre-existing teachers reused. |
| Candidate training steps | PENDING EXECUTION | Runtime plan allocates 300 steps per trainable candidate with Adam, learning rate 0.02, and no weight decay. | Record executed steps, interruptions, scheduler, gradient clipping, checkpoint hashes, and failure states. |
| Losses | CONFIRMED CONTRACT | Ours-v2/M3: standardized coefficient SmoothL1 only. B6: CrossEntropy. M4: frozen legacy endpoint contract. B7: none. | Verify run metadata matches the declared adapter loss before aggregation. |
| Correct/swap records | CONFIRMED CONTRACT | Exactly 20 unique correct episodes and 80 unique cross-outfit swaps per method replicate/seed result. | Provide raw record manifests and evaluator acceptance report. |
| Robustness records | CONFIRMED CONTRACT | Permutation, single reference, two-reference dropout, zero replacement, and base replacement are required. | Freeze actual record counts and hashes after evaluation. |
| Metrics | PARTIAL | Existing evaluator schema has 27 coefficient/residual/render/reference/efficiency slots. LPIPS, IoU, boundary, protected perceptual, and spatial diagnostics have separate preregistered gates. | Report resource availability, exact versions, thresholds, units, and every N/A reason. |
| Aggregation | CONFIRMED | Episode -> equal-weight outfit mean -> five-outfit macro -> replicate/seed summary; no best-run selection and no O07 mixing. | Publish per-episode, per-outfit, per-run, and aggregate tables. |
| Target leakage | CONFIRMED CONTRACT | Coefficient prediction allows reference RGB/masks/features/validity only. Target RGB/masks are supervision/evaluation-only; target pose/camera are downstream render-only. | Run and archive the final forward-boundary audit for every method. |
| Initialization | CONFIRMED CONTRACT | Standardized zero maps to coefficient mean/raw origin and then to the basis mean residual: mean-garment prediction, not base avatar. | Include initialization hashes and step-zero output checks in formal metadata. |
| Limitations | CONFIRMED DRAFT | One identity, five seen garments, held-out O07 failure, view-transductive basis, teacher onboarding, complete-rank K=4, lookup/color/spatial risks, synthetic-only targets, no real-world validation. | Reconcile wording after P0 outcomes without deleting negative evidence. |
| Code package | TODO | Candidate code is checked in on an engineering branch; no anonymous release bundle is prepared. | Prepare anonymous source snapshot, environment lock, commands, license, and checksum manifest. |
| Data/artifact package | TODO | Frozen manifest identifies assets, but a distributable package and access policy are absent. | Define release/access route, licenses, privacy review, storage sizes, and verification instructions. |
| PDF/source build | PENDING | TeX structure is checked in environments without a TeX engine. | Compile with the final venue template, archive logs, inspect warnings, and visually audit the PDF. |

## Final checklist guard

Do not convert `PENDING`, `TODO`, or conditional rows to affirmative answers merely to complete a submission form. Every affirmative answer must point to a frozen artifact, command, log, manifest, or reviewed manuscript location.
