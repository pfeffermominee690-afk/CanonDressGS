# AAAI-27 Subject00 Generation Backend Adjudication

Task: `AAAI27-SUBJECT00-MINIMAL-SECOND-IDENTITY-DATASET-CONTRACT-001`

Decision: `BACKEND_SELECTION_REQUIRED`

No provider or model was selected, and no connectivity or generation request was made. The comparison below is limited to frozen local historical provenance; it is not a claim of current availability.

| Historical surface | Image conditioning | Identity/mask/pose support | Reproducibility | Price/rate | Resolution/batch | License and policy |
|---|---|---|---|---|---|---|
| Codex platform-managed direct image edit | Historically used | Identity preservation, mask support, and separate pose support not verified | Model/revision hidden; seed not recorded | `PRICE_NOT_VERIFIED`; rate not verified | Historical expected 1024 x 1536; batch not verified | Not verified |
| Sublyx OpenAI-compatible `/v1/images/edits` | Historical identity plus condition image list | Identity quality not verified; mask not historically used; pose only through condition image | Request provenance only; seed absent | `PRICE_NOT_VERIFIED`; no explicit historical client limit | Historical 1536 x 1024, `n=1`, concurrency 1 | Provider moderation only; license not verified |
| 78code OpenAI-compatible `/v1/images/edits` | Historically implemented | Identity/mask/pose capability not verified | Historical seed support explicitly false | `PRICE_NOT_VERIFIED`; rate not verified | Historical 1536 x 1024, `n=1`; current batch behavior not verified | Not verified |

None can be selected under the zero-external-call boundary because current provider/model availability, exact revision, image count/format, response mode, seed behavior, price, rate limit, identity preservation, mask/pose conditioning, policy, data processing, license, and paper-use boundary all remain unverified.

Before even one connectivity request, the user must select a provider and exact model/revision policy, freeze the base URL and wire schema, accept payment and data-processing risk, and separately authorize a minimal probe. Batch generation requires another explicit authorization after a probe passes.

Only the credential environment variable name `MULTI_IDENTITY_GENERATION_API_KEY` may be persisted. Credential value, presence, length, prefix, hash/fingerprint, authorization headers, and unsafe raw response dumps are forbidden. Dry-run code must not read the credential.

The planning budget is two valid candidates per 24 slots: 48 `n=1` calls before retries. This is a count forecast, not authorization. API price and total API cost remain `PRICE_NOT_VERIFIED`.
