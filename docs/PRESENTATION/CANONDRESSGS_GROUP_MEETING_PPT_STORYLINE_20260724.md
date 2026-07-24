# CanonDressGS Group Meeting PPT Storyline — 2026-07-24

## Governance

- Task: `CANONDRESSGS-GROUP-MEETING-PRESENTATION-PLAN-20260724`
- Source: `research/paper-figure-bank-headroom-refresh-20260724@7e2d8efb3925880da4a9b8b85c0ebdce2437bd37`
- Target: `research/group-meeting-presentation-plan-20260724`
- LOO: `READY / PENDING EXECUTION`; `LOO_ATTEMPT=ABSENT`; files read = 0
- Boundaries: GPU=0, training=0, inference=0, renderer=0, new scientific images=0, image API=0, PPTX=0, paper mutation=0, Figure Bank mutation=0, `PAPER_FINAL=0`

## Story Arc

The presentation follows the actual contraction of the research rather than a retrospective success story:

1. **Problem** — reference garment + target pose/camera should produce the same identity with a specified garment.
2. **Early failures** — full reference-to-residual routes fail despite adequate support capacity.
3. **Causal diagnosis** — geometry interpolation is the dominant source of invalid intermediate support.
4. **Valid states** — per-garment Teacher Endpoints provide stable canonical residual states.
5. **Explicit coordinates** — centered SVD gives a rank-4 endpoint coordinate system, not yet a universal manifold.
6. **Pure Endpoint** — a minimal reference predictor plus endpoint snapping reliably selects five seen states.
7. **Hard lookup audit** — clean closed-bank behavior is functionally equivalent to hard lookup.
8. **Headroom negative result** — coefficient refinement adds no gain; full residual reintroduces artifacts.
9. **Decisive pending test** — LOO is the remaining test of basis value beyond lookup.
10. **Scope freeze** — core, supplementary, negative, pending, and multi-identity foundation evidence remain visibly separated.

## Section Structure

| Section | Full slides | Purpose |
|---|---:|---|
| Opening and problem | S01–S04 | Task, difficulty, and research questions |
| Research evolution and foundation | S05–S10 | Contraction, Teacher, causal geometry, Dual-Support, basis |
| Core method and evidence | S11–S16 | Pipeline, protocol, result, lookup relation, snapping, robustness |
| Headroom decision | S17–S20 | Hypothesis, protocol, negative result, design consequence |
| LOO decision | S21–S22 | Decisive design and honest pending status |
| Historical, limitations, identity | S23–S25 | Design evidence and unresolved scope |
| Freeze and next steps | S26–S28 | Paper state, reviewer risks, priorities |

## Visual Rules for the Later PPTX

- 16:9, white/light-gray background, one conclusion per slide.
- Use existing sealed scientific assets only; do not beautify or regenerate scientific pixels.
- Keep fixed visual labels for Main, Historical, Negative, Pending, and Foundation.
- Keep method/baseline/Teacher colors stable after manual asset selection.
- Show source IDs in notes, not as dense Git metadata on the visible slide.
- Headroom caption must state that refinement did not improve Teacher and full residual produced patch/cloud/mottle artifacts.
