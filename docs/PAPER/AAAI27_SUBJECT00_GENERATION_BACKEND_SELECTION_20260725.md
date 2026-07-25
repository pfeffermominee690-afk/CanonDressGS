# Subject00 Generation Backend Selection

Task: `AAAI27-SUBJECT00-DATA-PREPARATION-GENERATION-READY-001`

## Decision

`BACKEND_SELECTION_REQUIRED`. No provider was selected and no connectivity or paid request was made.

The local evidence covers three historical surfaces: the platform-managed Codex image-edit route, the Sublyx OpenAI-compatible proxy, and the 78Code proxy. Historical executions establish that image-edit workflows existed, but they do not establish one current provider/model/revision contract. The platform route does not expose its exact model. The proxy routes lack current capability, image-count, mask, seed, price, rate-limit, data-processing, content-policy, and paper-use verification. Provider selection also carries user payment and credential-risk decisions.

## Reuse

Append-only provenance, deterministic naming, technical-only retry policy, output normalization, two blinded initial human reviews, human adjudication, and the no-secret-persistence policy are reusable. Subject00 paths/prompts must be rebound. Subject02 identity pixels, condition IDs, accepted targets, unpinned backend labels, and historical credential fingerprint logging are not reusable.

The only persisted credential field is the prospective environment-variable name `MULTI_IDENTITY_GENERATION_API_KEY`. This task did not read, print, hash, fingerprint, or store any credential value.
