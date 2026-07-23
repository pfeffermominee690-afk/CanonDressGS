# Multi-Identity Generation Failure and Retry Contract

Task: `AAAI27-MULTI-IDENTITY-GENERATION-BACKEND-001`

## Retry rule

One logical candidate may have an initial attempt and at most two retries. Retries are allowed only for `NETWORK`, `TIMEOUT`, `HTTP_5XX`, or a provider-declared transient error. The frozen backoff is 3 seconds then 9 seconds. A retry retains the same parent request ID, source hashes, prompt hashes, parameters, candidate index, and seed if verified seed support later exists; each attempt receives its own deterministic request ID and attempt index.

Authentication, quota/credit, non-transient 4xx, content policy, moderation, unsupported parameters, malformed responses, and empty images are not automatically retried. A scientifically poor but decodable image is a screening rejection, never a transport retry. Failed requests do not count as valid candidates and retry attempts do not increase N.

## Append-only preservation

`REQUEST_PREPARED` is appended before an authorized request. Exactly one terminal event follows: `REQUEST_SUCCEEDED`, `REQUEST_FAILED`, or `REQUEST_MODERATED`. Events are hash chained. Existing JSONL bytes are never rewritten, and a terminal request cannot receive later events.

Every failure retains the request manifest SHA, request and response timestamps, attempt index, retry reason, billing-known state, moderation and timeout states, HTTP status, provider request ID when safely available, error class/code, malformed-response state, and empty-image state. Partial outputs and error records cannot be overwritten. Provider debug bodies are not retained unless a future provider-specific safety review defines a strict allowlist.

`CONTENT_POLICY`, `MODERATION_REJECTION`, and provider failures remain distinct. This prevents silent recoding of policy blocks as quality failures and prevents paid failed requests from disappearing from the denominator.

## Budget interaction

Valid-candidate options are MINIMAL N=1 (80), STANDARD N=2 (160), and ROBUST N=4 (320). STANDARD is the sole recommendation because provider cost is unknown and N=2 gives one planned alternative without result-driven identity-specific expansion. If no planned candidate passes a slot, the result is `GARMENT_SLOT_GENERATION_INCOMPLETE`; more generation requires explicit user authorization.

This task executed zero requests and zero retries.
