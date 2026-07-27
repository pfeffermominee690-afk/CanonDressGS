from __future__ import annotations

import argparse
import json
from pathlib import Path

SPLITS = {"train": [f"O{i:02d}" for i in range(8)], "val": ["O08", "O09"], "test": ["O10", "O11"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CanonDressGS full dataset v1 manifest")
    parser.add_argument("--condition-index", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-map", help="optional JSON object outfit_id -> teacher descriptor")
    parser.add_argument("--regression", action="store_true", help="allow non-12x200 Gate-4 regression only")
    args = parser.parse_args()
    index_path, root = Path(args.condition_index).resolve(), Path(args.dataset_root).resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    conditions = index["conditions"]
    outfit_ids = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("O"))
    expected = [f"O{i:02d}" for i in range(12)]
    if not args.regression and (outfit_ids != expected or len(conditions) != 200):
        raise ValueError("official build requires O00..O11 and exactly 200 conditions")
    teachers = {} if not args.teacher_map else json.loads(Path(args.teacher_map).read_text(encoding="utf-8"))
    outfits = []
    for outfit_id in outfit_ids:
        observations = []
        for condition in conditions:
            cid = condition["condition_id"]
            paths = {key: root / outfit_id / key / f"{cid}.png" for key in ("rgb", "foreground_mask", "clothing_mask")}
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"{outfit_id}/{cid} missing {missing}")
            observations.append({"condition_id": cid, **{key: str(path.resolve()) for key, path in paths.items()}})
        outfit = {"outfit_id": outfit_id, "metadata": {"official": not args.regression}, "observations": observations}
        if outfit_id in teachers: outfit["teacher"] = teachers[outfit_id]
        outfits.append(outfit)
    output = {"schema_version": "canondressgs.full_dataset.v1", "dataset_kind": "regression" if args.regression else "official_12x200", "conditions": conditions, "splits": SPLITS if not args.regression else {"train": outfit_ids, "val": [], "test": []}, "outfits": outfits}
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
