# Subject00 Managed Output Resolution Audit

- Task: `AAAI27-SUBJECT00-MANAGED-OUTPUT-RESOLUTION-ADJUDICATION-001`
- Source: `research/subject00-codex-managed-generation-20260725` at `8717118c9fd23c2cc92f65c5e13eff7ec6f15b86`
- Dataset root: `E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_001`
- Registry consistency: `PASS`
- Actual PNG parse: 48/48
- A exact 1024x1536: 5
- B exact 2:3 portrait resize candidate: 0
- C near 2:3 portrait: 0
- D incompatible portrait: 5
- E landscape: 38
- F square: 0
- G parse/metadata failure: 0
- Technical PASS: 5
- Pending user resize contract: 0
- Technical FAIL: 43
- Visual review queue: 48
- Original image mutations: 0
- Resize/crop/pad/rotate operations: 0/0/0/0
- Generation calls: 0
- Formal Base: `PENDING`
- PAPER_FINAL: `false`
- Contract tests: `PASS` (30/30)

The dominant actual dimension is `1349x1166` (34/48); 38 outputs are landscape. This is an observed repeated pattern, not a causal attribution. No visual quality field was filled by Codex.
