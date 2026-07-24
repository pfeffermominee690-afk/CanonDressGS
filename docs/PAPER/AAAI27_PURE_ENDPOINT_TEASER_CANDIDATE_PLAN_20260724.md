# AAAI27 Pure Endpoint Teaser Candidate Plan

## Status

- Candidate: `FIGURE1_ENDPOINT_ONLY_TEASER_CANDIDATE_V1`
- Status: `REQUIRES_MANUAL_ADJUDICATION`
- Substatus: `PURE_ENDPOINT_ENDPOINT_ONLY_CANDIDATE`
- `PAPER_FINAL=0`

## Frozen Selection

The sealed `selection_manifest.json` does not uniquely identify one teaser
query for each garment. No five-image teaser was selected. The generated review
candidate retains all 60 main sheets:

- garments: O01, O02, O03, O04, O08;
- rotation 0: `cond_000347`;
- rotation 1: `cond_000000`;
- rotation 2: `cond_000318`;
- rotation 3: `cond_000017`;
- seeds: 0, 1, 2.

This is the complete frozen garment/rotation/query/seed product represented by
the main-sheet manifest. Manual adjudication may choose the final five columns,
but it may not introduce a new query, camera, pose, seed, or garment.

## Intended Layout

Columns are O01, O02, O03, O04, and O08. Intended rows are Garment Reference,
Base Avatar, CanonDressGS-Endpoint, and Teacher Endpoint. Clean hard-lookup
methods are excluded from the teaser because they select the same endpoint and
would duplicate the visual result; their relationship is handled in Figure 5.

## Claim Boundary

Safe caption core: "CanonDressGS provides reliable reference-controlled
selection of seen garment endpoints in a fixed closed wardrobe."

Required qualifiers: fixed Subject02, closed five-garment wardrobe, seen
garment references, and discrete endpoint realization. The candidate does not
support unseen-garment, Headroom, LOO, strict-novel-view, continuous-control,
or hard-lookup superiority claims.
