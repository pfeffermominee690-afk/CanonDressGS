# AAAI27 Reference-Conditioned Dual-Support Interface

**RESEARCH EVALUATION — NOT PAPER FINAL**

This is an interface design only. No controller is trained or declared final.

## Required controller outputs

- A softmax garment-endpoint distribution over the frozen garment bank.
- Top-1 and top-2 garment IDs with normalized mixture weights.
- Confidence plus entropy and endpoint-compatibility diagnostics.
- A confidence-based single-endpoint fallback that cannot silently delete a support after rendering.

## Integration boundary

The current Ours-v2 predictor emits four-dimensional basis coefficients; it does not directly select or weight dual-support endpoint branches. A future controller must map reference evidence to endpoint selection before the frozen opacity-gating renderer contract is applied.

Candidate mechanisms are: softmax garment distribution, top-2 endpoint mixture, confidence-based single-endpoint fallback, and an entropy/compatibility gate. Their thresholds must be preregistered before any controller evaluation.

This document does not claim arbitrary garments, unseen garments, novel poses/views, cross-identity generalization, or a final method.
