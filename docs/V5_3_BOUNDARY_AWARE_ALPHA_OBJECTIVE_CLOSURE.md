# V5.3 Boundary-Aware Alpha Objective Closure

## Frozen run contract

- Formal run commit: `170990ed5b711ef0ed16d3815f0b912e20e79111`
- Outfit: `O05`
- References: `cond_000318`, `cond_000000`
- Target: `cond_000347`
- Sample index / seed / LR / steps: `11` / `20260717` / `1e-5` / `120`
- V5.2 initial-state SHA256: `e9d1d3f8b544fee8e2fda8a19002b4b4e7774357777c5ed1df4f0d37e4005a18`
- Formal output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-DUAL-TARGET-V5-3-001/attempt_002`
- `attempt_001` is retained as a pre-training diagnostic-tool failure; it performed zero optimizer steps. No second candidate configuration was run.

## Root cause and method

The legacy narrow-ring alpha term was reproduced at the exact V5.2 step-0 state. Its raw BCE/Dice values were `4.046646118` / `0.649221659`; their shared-trunk gradient norms were `0.398258535` / `0.047484484`. Both conflicted with edit and clothing RGB gradients. The transition ring contained `20,475` pixels, with edit-target foreground/background ratios `42.5788%` / `57.4212%`; `45.9634%` disagreed between edit and base alpha.

V5.3 defines a continuous target using fixed unit-pixel Euclidean distances:

`w_edit = d_base / (d_edit + d_base + 1e-6)`, `w_base = 1 - w_edit`, and `A_transition = w_edit A_edit + w_base A_base`.

Protected pixels are forced to base-only. Transition alpha uses SmoothL1 (`beta=1`) with no Dice. Edit/base alpha remain BCE+Dice. The new raw transition trunk norm was `0.0146086825`. The cap formula produced `0.260281585`, so the original effective coefficient `0.125` remained the frozen coefficient; its applied norm `0.00182608531` is below half the RGB norm (`0.00380237103`).

## Result

- Objective: `1.540736675` → `1.514210939`
- Edit first-10 / last-20: `0.283117893` / `0.281152658` (`0.694140%` reduction)
- Clothing first-10 / last-20: `0.284818670` / `0.282571809` (`0.788874%` reduction)
- Last-40 edit/clothing slopes: `-2.83519e-5` / `-3.27100e-5`
- Protected / preserve / alpha-base: `0.003170817`→`0.003295059`; `0.000221681`→`0.000234482`; `0.015840583`→`0.016447140`
- White-shoe final float RGB MAE to base/raw: `0.004722449` / `0.384340256`
- Frozen base maximum change: exactly `0` for xyz, scaling, rotation, opacity, SH0, and SHN
- Frozen image backbone: bitwise unchanged; frozen base/backbone gradient counts are zero
- Protected RGB/alpha flip: all affected loss deltas exactly zero

All six residual heads and the trunk, HyperNetwork, completer, aggregator, and encoder projection received finite nonzero gradients. Local and cloud interface/dataset/clothing/dressable/training/V5.2/dual-target regression suites passed.

The V5.3 contact sheets and steps 0/20/40/80/120 were actually opened. Clothing-region movement was subtle but coherent; face, hair, hand, white shoes, and background remained stable, with no visible alpha halo or opacity explosion. A complete long coat was not formed at 120 steps and was not a V5.3 requirement.

## Adjudication

**PASS.** The preregistered optimization-direction and protection criteria are satisfied. V5.3 permits a separately authorized Module 4B micro-pilot; this closure does not start it.
