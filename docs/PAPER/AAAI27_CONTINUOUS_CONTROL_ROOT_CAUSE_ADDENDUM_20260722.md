# AAAI27 Continuous-Control Root-Cause Addendum

**RESEARCH DIAGNOSTIC — NOT PAPER FINAL**

This addendum records a limitation in the channel-attribution design archived at source HEAD
`68623c36eee70c0b41aefb480f8f85e668eae201`. It does not modify, delete, or replace the
previous report, machine results, renders, visual review, or attempt history.

## Frozen adjudication

- `PREVIOUS_CHANNEL_ATTRIBUTION = SCIENTIFICALLY_INCONCLUSIVE_DUE_MIDPOINT_DEGENERACY`
- `PREVIOUS_ENGINEERING_EXECUTION = PASS`
- `PREVIOUS_SUPPORT_MISMATCH_EVIDENCE = NOT_SUPPORTED`
- `PREVIOUS_MULTIPLE_FACTORS_CLASSIFICATION = PRESERVED_AS_HISTORICAL_BUT_NOT_FINAL`

At `alpha=0.5`, the previous selected channels and non-selected channels were all assigned the
same pair midpoint. Consequently, every channel variant became the identical six-channel FULL
midpoint. The observed 10/10 reproduction for all six variants was therefore a necessary result
of the construction and could not identify a unique channel cause.

The previous execution remains valid as an engineering record: 720/720 requested channel renders
completed, 440/440 FULL renders were reused without regeneration, 82/82 visual items were opened,
the repaired no-training gate passed, and frozen assets were unchanged. The limitation is solely
scientific identifiability of the channel-causal contribution.

The support analysis is also preserved: exclusive-support visual alignment was 0/10 and no
preregistered support/conflict metric met the strong-evidence rule. It therefore does not support
a support-aware micro-pilot at this stage.

The endpoint-anchored factorial diagnostic frozen in
`paper_protocol/reviewer_risk/continuous_control_causal_attribution_protocol.yaml` is a new,
read-only diagnostic. It keeps non-selected channels at the source endpoint and separately tests
geometry, visibility, appearance, their pairwise interactions, and their three-way interaction.
No training, tuning, result selection, or automatic micro-pilot is authorized.
