# Multi-Identity Generation Backend Contract

Task: `AAAI27-MULTI-IDENTITY-GENERATION-BACKEND-001`

## Decision

The backend classification is `GENERATION_BACKEND_SELECTION_REQUIRED`. No provider, paid endpoint, model alias, or model revision is selected by this task. The historical record contains multiple incompatible execution surfaces and cannot prove current capability without a real connectivity request.

The prospective normalized artifact is a 1024 x 1536 portrait PNG. The provider wire-size spelling, quality field, provider-side background behavior, response mode, model ID, base URL identifier, endpoint, and seed behavior remain null until the user selects a provider/model and separately authorizes one minimal connectivity request.

## Historical execution provenance

The frozen five-garment wardrobe records `CODEX_IMAGE_GENERATION_SKILL` using `CODEX_PLATFORM_MANAGED_DIRECT_IMAGE_EDIT`. The backend model and base URL were not exposed, seed was not recorded, and only the expected 1024 x 1536 dimensions are known. This remains `LIMITED_GARMENT_GENERATION_PROVENANCE`; it is historical evidence, not a prospective reproducible contract.

`E:/data_pre/scripts/run_gpt_image2_pose_outfit_sheets.py` is a separate OpenAI-compatible client. Its defaults were `https://api.sublyx.org/`, model `gpt-image-2`, endpoint `/v1/images/edits`, size `1536x1024`, two retries, a 300-second timeout, and `OPENAI_API_KEY`. It called `client.images.with_raw_response.edit` with model, image list, prompt, `n=1`, and size. It accepted base64 or URL responses and retained HTTP/request metadata. It also exposed connectivity, model listing, a ten-ID model probe, and optional raw debug dumps.

`E:/data_pre/scripts/run_jay_coverage_supplement_v1.py` later changed the historical base URL to `https://www.78code.cc/v1`, required a CLI model, and explicitly recorded historical seed support as false. This provider migration prevents a unique provider inference. Historical Jay/Rose manifests contain 915/1,989 records respectively, but lack a complete provider/model/revision/seed record for prospective reproduction.

## Credential boundary

Only `credential_env_name` may be persisted. The prospective name is `MULTI_IDENTITY_GENERATION_API_KEY`; no dry-run code reads it. Credential value, presence, length, hash/fingerprint/prefix, authorization headers, and unsafe raw responses are forbidden in source, configuration, manifests, prompts, logs, and reports.

Both audited external scripts printed credential length and a SHA-256 prefix. That historical diagnostic must not be reused. Any scan finding is classified `API_CREDENTIAL_LEAK_RISK`; processing stops without printing the matched text.

## Prospective request contract

Every logical candidate is identified by identity, garment, semantic slot, and candidate index. Each attempt has a deterministic request ID and shares a deterministic parent request ID. Inputs are condition RGB, condition alpha/mask, identity manifest, garment prompt hash, and negative prompt hash. Source paths and SHA-256 values must be retained once formal inputs exist.

The append-only response event adds request/response timestamps, HTTP status, provider request ID, output SHA-256, error class, retry reason, billing-known state, moderation state, and safe response metadata. Secrets and raw unsafe responses are excluded.

The current dry-runs are blocked because `subject00` lacks the formal base and `avatarrex_lbn1` lacks the authorized base avatar. Their source SHA fields are null and marked `FORMAL_BASE_PENDING`; no placeholder SHA was invented.

## Selection gate

Before the first connectivity request, the user must select one provider and exact model ID/revision policy, accept its payment and data-processing risk, freeze one base URL and wire schema, and decide how unsupported seed behavior is represented. The next task is `USER_SELECT_GENERATION_PROVIDER_AND_MODEL`; it does not authorize connectivity or batch generation.

Execution in this task: connectivity calls 0, image API calls 0, generated images 0, retries 0, training 0, formal rendering 0, raw mutation 0, and `PAPER_FINAL=0`.
