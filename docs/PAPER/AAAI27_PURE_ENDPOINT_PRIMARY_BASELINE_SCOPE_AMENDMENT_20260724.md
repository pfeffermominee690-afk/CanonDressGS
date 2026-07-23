# Pure Endpoint Primary Baseline Scope Amendment

Task: `AAAI27-PURE-ENDPOINT-BASELINE-ADJUDICATION-001`

This prospective amendment freezes the matched primary registry at seven
methods, in this exact order:

1. Base Avatar
2. Teacher Endpoint
3. Outfit-ID Oracle
4. Reference Classifier Lookup
5. Nearest-Centroid Lookup
6. Linear Coefficient Predictor
7. CanonDressGS-Endpoint

Base Avatar has no garment residual. Teacher Endpoint is the non-deployable
full frozen teacher reference. Outfit-ID Oracle is the non-deployable rank-4
GT outfit-ID upper bound. Reference Classifier Lookup is the learned hard
lookup baseline. Nearest-Centroid Lookup uses train-fold-only reference
feature centroids. Linear Coefficient Predictor renders its continuous rank-4
prediction. CanonDressGS-Endpoint is the primary method and snaps the same
shared predictor output to a frozen endpoint.

All seven methods use the same test records and frozen renderer. The table
must disclose deployability, ground-truth information, protocol-weighted
metrics, and unique-query metrics. Historical V7 is absent from this primary
registry but remains present in the supplementary historical registry.

The model, coefficient loss, data partitions, unique-query training schedule,
four rotations, seeds `[0,1,2]`, 300-step budget, checkpoints
`[0,20,50,100,200,300]`, evaluator, test denominator, perturbations, and
success gates are unchanged. The independent training families are the
Reference Classifier and the predictor shared by Linear Coefficient Predictor
and CanonDressGS-Endpoint: 24 runs, 7,200 optimizer steps, and 144 checkpoint
writes.

No scientific execution occurred during this amendment. `PAPER_FINAL=false`.
