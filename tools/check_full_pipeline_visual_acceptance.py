from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from build_full_pipeline_visual_acceptance import build


def image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 48), color).save(path)


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        for module, names in {
            "module1": ["base_rgb.png", "zero_override_rgb.png", "post_restore_rgb.png", "xyz_rgb.png", "scaling_rgb.png", "rotation_rgb.png", "opacity_rgb.png", "sh0_rgb.png", "shN_degree1_zero_control_rgb.png", "shN_rgb.png", "attribute_comparison.png"],
            "module2": ["teacher_reference_contact_sheet.png", "teacher_target_rgb.png", "base_rgb.png", "predicted_rgb.png", "predicted_alpha.png", "xyz_only.png", "scaling_only.png", "rotation_only.png", "opacity_only.png", "sh0_only.png", "shN_degree1_zero_control.png", "shN_only.png", "all_channels.png", "channel_comparison.png"],
            "module3": ["reference_1.png", "reference_2.png", "reference_clothing_mask_1.png", "reference_clothing_mask_2.png", "observed_probability_S1.png", "observed_probability_S2.png", "observed_probability_S12.png", "geometry_gate_S1.png", "geometry_gate_S2.png", "geometry_gate_S12.png", "appearance_gate_S1.png", "appearance_gate_S2.png", "appearance_gate_S12.png", "teacher_gate.png", "observed_only_gate.png", "diffusion_gate.png", "learned_completed_gate.png", "teacher_gate_render.png", "observed_only_render.png", "diffusion_render.png", "learned_gate_render.png", "non_clothing_leakage_overlay.png", "rotation_non_clothing_leakage_overlay.png", "render_comparison.png", "feature_holdout_comparison.png", "loss_curve.png"],
        }.items():
            directory = root / module; directory.mkdir()
            for index, name in enumerate(names): image(directory / name, (index, index, index))
            build(module, directory, directory, None)
            payload = json.loads((directory / "visual_acceptance.json").read_text())
            assert payload["visual_acceptance_status"] == "PARTIAL"
            assert payload["images_actually_opened"] is False
            assert (directory / f"{module}_visual_acceptance_contact_sheet.png").is_file()
    print("full pipeline visual acceptance checks: PASS")


if __name__ == "__main__": main()
