# Controller V2 calibration and evaluator contract

For every rotation and seed, V2 calibration searches the frozen 9x9 grid:
tau_mix={0.10,...,0.90} and
tau_pair={0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50}. Pair confidence is
TOP2_VS_TOP3_PROBABILITY_MARGIN. The fixed objective first minimizes pure
false-mixed, wrong-pair DUAL, and incompatible-pair DUAL; then maximizes
correct-compatible DUAL and correct-incompatible HARD_GEOMETRY_SOFT_VA; then
minimizes mixed false-SINGLE. Exact ties choose larger tau_pair, then larger
tau_mix, then the lexicographically larger pair. No threshold is selected in
this repair task.

Primary gates use PROTOCOL_WEIGHTED_MACRO with duplicates retained. The full
UNIQUE_QUERY_MACRO keyed by logical_input_sha256 is mandatory and cannot replace
the primary aggregation. Each test fold has 80 protocol records / 65 unique
queries: 20/5 pure and 60/60 mixed. The formal-pure 20 set is reported
separately.

Twenty perturbation representatives per rotation are frozen before results by
sorting logical query ID, assignment position, and record ID within every
pair/composition cell. The same records define 240 main visual sheets. The
historical 0=none, 1=minor, 2=moderate, 3=severe scale and all artifact
categories are unchanged.
