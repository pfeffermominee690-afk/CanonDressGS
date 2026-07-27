# Repaired Controller V2 Efficiency

The scientific execution used 24 fresh training processes: 12 V2 and 12
matched V1. V2 contains 9,232 trainable parameters and matched V1 contains
3,589. Summed run wall time was 84.029 seconds for V2 and 30.451 seconds for
matched V1. Peak recorded VRAM was 68,389,376 bytes for V2 and 68,238,336
bytes for matched V1.

Controller evaluation performed 6,240 unique forwards and reused 6,240 frozen
feature rows. Clean calibration/test/formal controller time was 4.446 seconds;
perturbation controller time was 3.612 seconds; total controller time was
8.057 seconds. Failed inference count was zero.

Rendering consumed 1,214.919 seconds wall time for 5,280 logical records. The
cache reduced this to 1,035 new physical render signatures with 4,245 current
attempt exact-signature reuses. There were 1,879 unique metric records and
zero failed renders. Frozen historical oracle/target/endpoint outputs were
reused for all 5,280 logical records.

Exact scientific counts were:

| Operation | Count |
|---|---:|
| training steps | 3,600 |
| backward calls | 3,600 |
| optimizer creations | 24 |
| optimizer steps | 3,600 |
| scheduler steps | 3,600 |
| checkpoint writes | 144 |
| clean forward batches | 3,600 |
| V2 augmented forward batches | 1,800 |
| clean record forwards | 18,000 |
| augmented record forwards | 9,000 |
| controller inference | 6,240 |
| logical render records | 5,280 |
| new physical render signatures | 1,035 |
| exact-signature render reuses | 4,245 |
| failed inference/render | 0 / 0 |

The two preserved step-zero startup attempts each contain one startup
checkpoint and are not included in the 144 scientific checkpoint count.
`PAPER_FINAL=0`.
