# Coefficient Headroom Failure Analysis

- execution HEAD: `11206393153d710a62b351ff1d136d82154d9e85`
- final scientific evidence HEAD: `1275d79bde07fdd798987024310d822826aaf135`
- failure code: `COEFFICIENT_HEADROOM_PARITY_OUTPUT_DIRECTORY_MISSING`
- stage: `02_static_parity`
- completed renderer calls before failure: 2
- persisted render outputs: 0
- optimizer creations / steps / checkpoints: 0 / 0 / 0
- attempt preserved: `attempt_001`
- retry performed: no
- `attempt_002` created: no

The runner rendered Teacher and SVD for `O01/cond_000000`, then `save_render_tensor` raised `FileNotFoundError` because the parity render directory had not been created. Creating the directory and retrying would leave two unregistered renderer calls and violate the frozen expected-count contract. The required next task is `REPAIR_COEFFICIENT_HEADROOM_EXECUTION_FAILURE`. LOO was not started. `PAPER_FINAL=false`.
