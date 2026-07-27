"""Recompute FORMAL-002 frozen-asset fingerprints without scientific work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--yaml-root", type=Path)
    parser.add_argument("--yaml-only", action="store_true")
    args = parser.parse_args()
    if args.yaml_root is not None:
        import yaml

        yaml_files = sorted(args.yaml_root.glob("dual_support_controller*.yaml"))
        for path in yaml_files:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        print(json.dumps({"yaml_parse_count": len(yaml_files)}, sort_keys=True))
        if args.yaml_only:
            return
    repository = args.repository.resolve()
    sys.path.insert(0, str(repository))
    from tools.paper.run_reference_conditioned_dual_support_controller_formal import tree_snapshots

    preflight = json.loads(
        (args.output_root / "attempt_001/audits/preflight.json").read_text(encoding="utf-8")
    )
    after = tree_snapshots(args.asset_root.resolve())
    keys = sorted(set(preflight["frozen_before"]) | set(after))
    changed = [
        key for key in keys if preflight["frozen_before"].get(key) != after.get(key)
    ]
    result = {
        "schema_version": "canondressgs.research.dual_support_controller_frozen_immutability.v1",
        "task_id": "AAAI27-REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-FORMAL-002",
        "status": "PASS" if not changed else "FAIL",
        "frozen_mutation_count": len(changed),
        "changed_keys": changed,
        "before": preflight["frozen_before"],
        "after": after,
        "paper_final": False,
        "paper_final_count": 0,
    }
    target = args.output_root / "attempt_004/audits/frozen_after_immutability.json"
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "frozen_mutation_count": len(changed),
                      "path": str(target)}, sort_keys=True))


if __name__ == "__main__":
    main()
