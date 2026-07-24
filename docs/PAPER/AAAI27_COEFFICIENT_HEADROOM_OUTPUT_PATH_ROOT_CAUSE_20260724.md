# Coefficient Headroom Output Path Root Cause

Classification: `PRE_OPTIMIZER_OUTPUT_PATH_ENGINEERING_FAILURE`.

The preserved `attempt_001` completed exactly two in-memory renderer calls for the first `O01/cond_000000` Teacher/SVD parity cell and then raised `FileNotFoundError` while saving `/root/autodl-tmp/canondressgs_work/outputs/COEFFICIENT-HEADROOM-001/attempt_001/02_static_parity/renders/O01_cond_000000_teacher_rgb.png`. Its missing parent was `/root/autodl-tmp/canondressgs_work/outputs/COEFFICIENT-HEADROOM-001/attempt_001/02_static_parity/renders`.

The exact caller chain was `run_parity -> save_render_tensor -> torchvision.utils.save_image -> PIL.Image.Image.save`. `materialize` created `02_static_parity` but not `02_static_parity/renders`; the imported writer did not close its parent. The old preflight had neither a complete output path plan nor a PNG write smoke, so it did not detect the missing nested parent before renderer call 1.

This is not basis parity, Teacher capacity, rendering-objective, optimization, OOM, convergence, or data-leakage evidence. Optimizer creations/steps, forward/backward calls, checkpoints, completed runs, metric rows, and visual sheets were all zero. `attempt_001` remains sealed and byte-identical.
