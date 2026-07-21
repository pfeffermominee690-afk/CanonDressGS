# Historical Three-Seed Interpretation Addendum

Date: 2026-07-21  
Task: `AAAI27-DETERMINISTIC-INITIALIZATION-PROTOCOL-001`

## Scope and preservation

This addendum corrects the interpretation of sealed historical runs. It does not change their registry status, raw metrics, checkpoints, output files, or earlier reports. The formal registry was parsed and each executable method was mapped through the actual formal model factory; the list was not reconstructed from memory.

The old `SEED-PROPAGATION-FAIL` remains the true result of its preregistered uniqueness gate. The new root-cause classification is `DETERMINISTIC_INITIALIZATION_CONTRACT_MISMATCH`: the gate expected seed-unique final initialization from a constructor whose random default Linear initialization is immediately overwritten by explicit zeros.

## Corrected method classification

The following formal methods used deterministic zero initialization and must not be described as three independent random initializations:

- `Ours_Seen_Outfit_Explicit_Basis_V1`
- `A1_Basis_Rank_1`, `A1_Basis_Rank_2`, `A1_Basis_Rank_3`, `A1_Basis_Rank_4`
- `A2_No_Clothing_Mask`
- `A3_Mean_Only`
- `A4_No_Coefficient_Standardization`
- `A5_Legacy_Endpoint_Supervision`
- `A6_No_Pairwise_Coefficient_Geometry`
- `A7_Reference_Count_1`, `A7_Reference_Count_2`, `A7_Reference_Count_3`
- `B3_Global_Reference_Feature`
- `B4_Clothing_Mean_Only`

For each of these 15 method groups, the three step-0 model-state hashes are identical, the first-20-step trajectory hashes are identical, and the final model-state hashes are identical. The sealed evaluated metric payloads have three distinct hashes, so the conservative machine classification is `PARTIALLY_STOCHASTIC_SHARED_INITIALIZATION`, not `INDEPENDENT_RANDOM_INITIALIZATIONS`.

The accurate language is **shared deterministic initialization with seed-dependent training/evaluation outcomes**. Where the entire observed protocol is identical, the narrower phrases **three deterministic protocol replicates** or **three runs under a deterministic zero-initialization contract** are valid. Neither formulation claims independent random initialization.

`B5_Legacy_Complex_Fusion_RF_F` is different. Its actual factory is `LegacyRFFRank4`, its policy is `RANDOM_SEEDED_INITIALIZATION`, and it has three unique initial-state, early-trajectory, final-state, and final-metric hashes. It remains `INDEPENDENT_RANDOM_INITIALIZATIONS` and is retained only as off-matrix historical evidence.

## Factory evidence

The parsed mapping is:

- `B5_*` -> `LegacyRFFRank4`
- `A5_*` -> `BoundedVectorControl`
- every other trainable formal method -> `MultiOutfitLinearCoefficientControl`

The historical constructors and outputs were not modified. Full per-seed checkpoint, model-state, trajectory, and metrics hashes are preserved in `paper_protocol/reviewer_risk/historical_three_seed_interpretation_audit.json`.

Final classification: `HISTORICAL_THREE_SEED_INTERPRETATION = CORRECTED`.
