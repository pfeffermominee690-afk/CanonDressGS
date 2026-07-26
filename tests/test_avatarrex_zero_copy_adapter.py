from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "avatarrex_zero_copy_adapter",
    ROOT / "tools/datasets/avatarrex_zero_copy_adapter.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class AvatarReXAdapterContractTests(unittest.TestCase):
    def test_camera_holdout_contract_is_quarter_rig(self) -> None:
        self.assertEqual(list(range(0, module.EXPECTED_CAMERA_COUNT, 4)), [0, 4, 8, 12])

    def test_pose_contract_matches_subject00_buffer_principle(self) -> None:
        self.assertEqual(module.TEMPORAL_BUFFER_RADIUS, 5)
        self.assertEqual(round(module.EXPECTED_FRAME_COUNT * 0.05), 95)

    def test_mask_layout_matches_audited_avatarrex_source(self) -> None:
        self.assertEqual(module.MASK_RELATIVE_DIRECTORY, Path("mask") / "pha")

    def test_canonical_hash_is_order_invariant_for_objects(self) -> None:
        self.assertEqual(
            module.canonical_sha256({"b": 2, "a": 1}),
            module.canonical_sha256({"a": 1, "b": 2}),
        )

    def test_adapter_declares_no_generation_or_copy_api(self) -> None:
        source = (ROOT / "tools/datasets/avatarrex_zero_copy_adapter.py").read_text(encoding="utf-8")
        forbidden = ("requests.post", "openai", "shutil.copy", "copyfile", "optimizer.step", "renderer")
        self.assertFalse(any(token in source for token in forbidden))


if __name__ == "__main__":
    unittest.main()
