# AAAI27 Pure and Mixed Task Semantics Audit

Status: `PASS`

Protocol semantic classification: `LATENT_SECONDARY_NOT_INPUT_IDENTIFIABLE`.

Pure query records contain one visible garment in all three reference slots. Their stored `pair_id` retains a second garment from the original pair template even though that garment is absent from every Controller input. Across each fixed observable input, four equiprobable latent secondary labels remain possible; entropy is `2.0` bits and Bayes-optimal exact-secondary accuracy is `0.25`.

The hidden secondary does not enter the garment target, mixedness target, pure weight loss, routing denominator, calibration pair objective, or visual target content. It did enter historical all-record top-2 correctness through the stored `pair_id`.

| Audit question | Result |
|---|---|
| Pure input contains only one garment | yes |
| Stored pair contains an absent second garment | yes |
| Secondary source | original pair template retained in `pair_id` |
| Participates in garment/mixedness/pure-weight loss | no |
| Participates in historical all-record pair accuracy | yes |
| Participates in calibration/routing/visual target | no |
| Varies across observational duplicates | yes |

Protocol counts are 320 pair-query records: 80 pure and 240 mixed, with 20 pure and 60 mixed per fold. All pure visible sets have cardinality one and all mixed visible sets have cardinality two. Compatibility manifests for all four rotations and ten pairs were validated without changing them.

The 80 pure records form 20 observable equivalence classes. Every class has four equiprobable latent secondary candidates, entropy `2.000000` bits, latent-label duplicate consistency `0.000000`, and uniform/majority/Bayes exact-secondary accuracy `0.250000`.

Correct evaluator scopes:

- Pure: visible-garment top-1 and endpoint safety; no exact pair metric.
- Mixed AAB/ABB: unordered pair, ordering, weight, and routing.
- Formal-pure 20: `SECONDARY_ENDPOINT_SAFETY_ONLY`.

Historical overall classification remains `CONTROLLER_V2_CROSSFIT_MICRO_PILOT_FAIL`.
