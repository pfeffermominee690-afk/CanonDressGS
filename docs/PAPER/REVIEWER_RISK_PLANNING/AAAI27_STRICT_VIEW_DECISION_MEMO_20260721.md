# AAAI-27 Strict-View Decision Memo

Date: 2026-07-21
Task: `AAAI27-REVIEWER-RISK-PAPER-MATERIALS-001`

## Decision

`STRICT_VIEW_ONE_FOLD_CANARY = BLOCKED_PENDING_AUTHORIZATION`

No strict-view teacher, basis, predictor, render, or evaluation is authorized by this memo. The current manuscript must use **VIEW-TRANSDUCTIVE** throughout.

## What the canary would test

The proposed canary would hold out the back view (`cond_000318`) end to end. Five fold-specific teachers would use only the remaining three views; a fold-specific basis would be constructed from those teachers; a fold-specific predictor would then be trained and evaluated only on the held-out back view for the five registered garments. One fold is a canary, not a complete four-fold protocol.

## Value, cost, and dependency comparison

| Dimension | Assessment | Consequence |
| --- | --- | --- |
| Reviewer value | High for correcting the most obvious transductive-view concern, but limited because a single fold cannot establish complete view robustness. | Useful appendix evidence only; it cannot retroactively relabel the main results. |
| GPU cost | Existing planning estimate: 6,300 optimizer steps, approximately 0.5 RTX-4090 GPU-hours, 5–10 minutes evaluation, and about 1.5 GiB storage. This is an estimate, not a measurement or authorization. | Material but smaller than the P0 main closure; still creates a separate five-teacher/basis/predictor evidence chain. |
| Writing benefit | A positive canary would show one strictly isolated view is feasible. A negative canary would strengthen the limitation and prevent overstatement. | Either outcome is publishable only if clearly isolated from the transductive main table. |
| Main dependency | Ours-v2 must be confirmed and its final evaluator/visual gate must pass. | Do not spend the canary budget on an unconfirmed paper candidate. |
| Hard-lookup dependency | If B6/B7 fully match the proposed method and force a retrieval/control framing, strict-view evidence has less marginal value. | Lower priority when the central contribution is already substantially narrowed. |
| M1–M4 dependency | Architecture/supervision causality must be closed first. | P0 causal validity outranks a new evaluation axis. |
| Color dependency | If reference control is color-dominated, strict-view isolation does not repair the semantic-control concern. | Complete and interpret color counterfactuals before authorization. |
| Manuscript dependency | Problem Setup, Method, main tables, objections, and limitations must be near stable. | Authorize only if the result can be integrated without displacing required P0 writing and checks. |

## Authorization gate

Recommend authorization of the one-fold canary only when **all** conditions are true:

1. Ours-v2 is confirmed by the unified evaluator and manual visual adjudication.
2. Hard lookup does not fully null the contribution under the preregistered decision rule.
3. The complete M1–M4 matrix is finished and contract-clean.
4. The manuscript schedule permits a separate canary evidence chain without delaying MUST HAVE sections and final artifacts.

Color results are an additional priority modifier: a color-dominated outcome lowers the value of strict-view evidence and should normally keep the canary blocked unless reviewers are expected to make view isolation decisive.

## Required safeguards if later authorized

- Use a new registry entry, new output root, and fold-specific asset manifest.
- Rebuild all five teachers and the basis from only the three allowed views.
- Do not mix canary metrics with transductive main-table aggregates.
- Do not tune on the held-out view or select the fold after inspecting results.
- Preserve a negative or failed run and label it honestly.
- Describe a positive result as a one-fold canary, never as a complete strict-view study.

## Current manuscript action

Keep the canary out of the critical path. Put the precise transductive boundary in Figure 1, Problem Setup, Experiments, and Limitations. `PAPER_FINAL` remains zero.
