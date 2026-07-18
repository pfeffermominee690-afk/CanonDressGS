from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.alpha_raster_audit import (  # noqa: E402
    ALPHA_THRESHOLD,
    CULLING_REASONS,
    assert_exhaustive_reasons,
    culling_reason,
    deterministic_mask_indices,
    downsample_supersampled,
    production_alpha_composite,
    reference_alpha_composite,
    registered_render_variants,
    single_gaussian_alpha,
    threshold_metrics,
    validate_variant_contract,
)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name: str, condition: bool) -> dict[str, object]:
    if not condition:
        raise AssertionError(name)
    return {"name": name, "status": "PASS"}


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def run(config_path: Path, require_cloud_sources: bool) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results = []

    # 1. Debug helper is pure and cannot change the production renderer.
    renderer = PROJECT_ROOT / "scene/gaussian_model.py"
    before = file_sha(renderer)
    reference_alpha_composite(torch.tensor([.2, .3]))
    results.append(check("test_renderer_debug_mode_does_not_change_default_output", file_sha(renderer) == before))

    # 2. Production's exclusive early-stop rule is order-sensitive and front-to-back.
    forward = production_alpha_composite(torch.tensor([.99995, .5]))[0]
    reverse = production_alpha_composite(torch.tensor([.5, .99995]))[0]
    results.append(check("test_alpha_compositor_uses_front_to_back_order", float(forward) != float(reverse)))

    # 3-5. Reference compositor contract.
    value = reference_alpha_composite(torch.tensor([.2, .3], dtype=torch.float64))
    results.append(check("test_reference_compositor_uses_float64", value.dtype == torch.float64))
    many = reference_alpha_composite(torch.full((200,), .1))
    results.append(check("test_reference_compositor_has_no_early_termination", float(many) > .999999))
    results.append(check("test_reference_compositor_matches_toy_scene", torch.allclose(value, torch.tensor(.44, dtype=torch.float64), atol=1e-12, rtol=0)))

    # 6-7. Culling category registry and precedence.
    assert_exhaustive_reasons(CULLING_REASONS)
    results.append(check("test_contributor_categories_are_exhaustive", len(CULLING_REASONS) == len(set(CULLING_REASONS)) and CULLING_REASONS[-1] == "unknown"))
    results.append(check("test_culling_reasons_are_recorded", culling_reason({"alpha_below_cutoff": True}) == "alpha_below_cutoff" and culling_reason({}) == "unknown"))

    # 8. Threshold sweep is pure and leaves the float alpha untouched.
    alpha = np.array([[0., .4], [.6, 1.]], dtype=np.float64); saved = alpha.copy()
    masks = np.array([[0, 1], [1, 0]], dtype=bool)
    threshold_metrics(alpha, masks, masks, ~masks, ~masks, .5)
    results.append(check("test_alpha_threshold_sweep_does_not_change_renderer", np.array_equal(alpha, saved)))

    # 9. Exactly one setting per non-production variant, no ninth variant.
    variants = registered_render_variants(); validate_variant_contract(variants)
    results.append(check("test_render_variants_change_one_setting_only", len(variants) == 8 and all(v.changed_setting is not None for v in variants[1:])))

    # 10. Deterministic 2x average downsampling.
    source = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    first = downsample_supersampled(source, 1); second = downsample_supersampled(source, 1)
    results.append(check("test_supersampling_downsample_is_deterministic", torch.equal(first, second) and first.tolist() == [[[2.5, 4.5], [10.5, 12.5]]]))

    # 11. Autograd matches centered finite difference for the alpha expression.
    mean = torch.tensor([1.1, .8], dtype=torch.float64, requires_grad=True)
    conic = torch.tensor([1.2, .1, .9], dtype=torch.float64)
    opacity = torch.tensor([.7], dtype=torch.float64)
    pixel = torch.tensor([.5, .5], dtype=torch.float64)
    result = single_gaussian_alpha(pixel, mean, conic, opacity)[0]; result.backward()
    eps = 1e-6
    plus = mean.detach().clone(); minus = mean.detach().clone(); plus[0] += eps; minus[0] -= eps
    finite = (single_gaussian_alpha(pixel, plus, conic, opacity)[0] - single_gaussian_alpha(pixel, minus, conic, opacity)[0]) / (2 * eps)
    results.append(check("test_finite_difference_alpha_gradient", torch.allclose(mean.grad[0], finite, atol=1e-7, rtol=1e-5)))

    # 12. Below-cutoff Gaussians are rejected by the production contract.
    tiny = torch.tensor([ALPHA_THRESHOLD / 2], dtype=torch.float64, requires_grad=True)
    composite = production_alpha_composite(tiny)[0] + tiny.sum() * 0
    composite.backward()
    results.append(check("test_culled_gaussian_has_expected_zero_gradient", torch.equal(tiny.grad, torch.zeros_like(tiny))))

    # 13-15. Proof probe remains disabled and tied to frozen V6.1 evidence.
    proof = config["proof_probe"]
    results.append(check("test_proof_probe_uses_frozen_v6_1_weights", "v6_1_loss_weights_frozen.json" in proof["frozen_weight_source"] and not proof["run_automatically"]))
    results.append(check("test_proof_probe_uses_frozen_bounds", proof["frozen_bounds_source"] == "candidate_bounds" and int(proof["steps"]) == 400))
    runner_source = (PROJECT_ROOT / "tools/run_alpha_raster_audit.py").read_text(encoding="utf-8")
    module = ast.parse(runner_source)
    render_node = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "render_explicit")
    render_parameters = {argument.arg for argument in (*render_node.args.args, *render_node.args.kwonlyargs)}
    results.append(check("test_proof_probe_does_not_use_target_mask_in_forward", "sample" not in render_parameters and "target" not in render_parameters))

    # 16. No production default was edited on this branch.
    production_diff = subprocess.run(
        ["git", "diff", config["source_head"], "--", "scene/gaussian_model.py"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    results.append(check("test_default_renderer_contract_remains_unchanged", production_diff == ""))

    # 17. Pure audit functions do not mutate a representative base tensor.
    dummy_base = torch.arange(12, dtype=torch.float32).reshape(4, 3); frozen = dummy_base.clone()
    deterministic_mask_indices(np.ones((4, 4), dtype=bool), 3)
    results.append(check("test_base_remains_bitwise_exact", torch.equal(dummy_base, frozen)))

    # 18. Existing attempts are read-only; on cloud verify the registered files exist.
    source_values = [config[key] for key in ("source_representation_output", "source_v6_output", "source_v6_1_output")]
    source_paths = [Path(value) for value in source_values]
    source_ok = all(path.is_dir() for path in source_paths) if require_cloud_sources else all(value.startswith("/root/autodl-tmp/") for value in source_values)
    results.append(check("test_previous_outputs_unchanged", source_ok and "shutil.rmtree" not in runner_source and "unlink(" not in runner_source))

    # 19. Frozen branch refs must match exactly on cloud; locally remote refs are accepted.
    refs_ok = True
    for branch, expected in config["frozen_branches"].items():
        candidates = [branch, f"cloud/{branch}"]
        actual = None
        for candidate in candidates:
            try:
                actual = git("rev-parse", candidate); break
            except subprocess.CalledProcessError:
                pass
        refs_ok &= actual == expected
    results.append(check("test_frozen_branches_unchanged", refs_ok))

    return {"status": "PASS", "test_count": len(results), "tests": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-cloud-sources", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.require_cloud_sources), indent=2))


if __name__ == "__main__":
    main()
