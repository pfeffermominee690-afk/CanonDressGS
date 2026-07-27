# Multi-Identity Garment Benchmark Foundation Plan

Task: `AAAI27-MULTI-IDENTITY-GARMENT-DATA-PREPARATION-001`

## Scope

This task freezes foundations for identity-conditioned synthetic closed-wardrobe garment endpoint banks for THuman4.0 `subject00` and AvatarReX `avatarrex_lbn1`. It does not create a captured multi-outfit dataset, cross-identity garment ground truth, generated images, avatars, garment teachers, or `PAPER_FINAL` evidence.

The shared wardrobe is exactly `O01/O02/O03/O04/O08`. Each identity uses eight semantic pose/view slots, four cardinal garment references, eight accepted endpoints per garment, and a matched 1200-step teacher budget per garment. Raw frame IDs are identity-specific deterministic assignments and are never forced to match across identities.

## Gates

Subject00 is blocked on its formal step-101245 final checkpoint. AvatarReX is blocked on upstream license confirmation and base-avatar preparation after zero-copy standardization. Generation requires a pinned provider, model revision, complete parameters, deterministic seed, append-only provenance, and two-reviewer screening.

Overall classification: `MULTI_IDENTITY_GARMENT_BENCHMARK_FOUNDATIONS_READY`. This classification covers protocol and adapter readiness only.
