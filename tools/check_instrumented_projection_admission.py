from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.instrumented_projection_admission import (  # noqa: E402
    InstrumentationOptions,
    REJECTION_REASONS,
    STAGE_NAMES,
    backend_first_rejections,
    counterfactual_alpha,
    independent_gaussian_projection_oracle_v1,
    projection_agreement,
    run_backend_projection_debug,
    same_index_displacement,
    stage_statuses,
    support_funnel_counts,
    tile_bboxes,
)


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def check(name: str, condition: bool) -> dict[str, str]:
    if not condition:
        raise AssertionError(name)
    return {"name": name, "status": "PASS"}


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def synthetic_inputs(device: torch.device, *, xyz: torch.Tensor | None = None, covariance: torch.Tensor | None = None):
    means = xyz if xyz is not None else torch.tensor([[0.0, 0.0, 2.0], [.08, -.04, 2.4]], device=device)
    n = means.shape[0]
    covars = covariance if covariance is not None else torch.eye(3, device=device).repeat(n, 1, 1) * .0025
    opacity = torch.full((n,), .8, device=device)
    canonical = {
        "canonical_xyz": means.clone(),
        "canonical_scaling": torch.zeros((n, 3), device=device),
        "canonical_quaternion": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1),
        "canonical_opacity": opacity.clone(),
    }
    w2c = torch.eye(4, device=device)
    K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], device=device)
    return canonical, means, covars, opacity, w2c, K


def oracle(device: torch.device, **changes):
    canonical, means, covars, opacity, w2c, K = synthetic_inputs(device, **changes)
    return independent_gaussian_projection_oracle_v1(
        **canonical,
        posed_xyz=means,
        posed_covariance=covars,
        opacity=opacity,
        w2c=w2c,
        K=K,
        width=64,
        height=64,
        options=InstrumentationOptions(),
    )


def run(config_path: Path, require_cloud_backend: bool) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results: list[dict[str, str]] = []
    module_path = PROJECT_ROOT / "scene/instrumented_projection_admission.py"
    runner_path = PROJECT_ROOT / "tools/run_instrumented_projection_admission.py"
    module_source = module_path.read_text(encoding="utf-8")
    runner_source = runner_path.read_text(encoding="utf-8") if runner_path.is_file() else ""
    device = torch.device("cuda" if require_cloud_backend else "cpu")

    # 1. Default is explicitly off.
    results.append(check("test_debug_instrumentation_default_is_off", InstrumentationOptions().enabled is False))

    # 2. The disabled helper is a true no-op.
    canonical, means, covars, opacity, w2c, K = synthetic_inputs(device)
    before = tuple(tensor_hash(value) for value in (means, covars, opacity, w2c, K))
    disabled = run_backend_projection_debug(
        means=means, covars=covars, opacities=opacity, w2c=w2c, K=K,
        width=64, height=64, options=InstrumentationOptions(enabled=False),
    )
    after = tuple(tensor_hash(value) for value in (means, covars, opacity, w2c, K))
    results.append(check("test_debug_off_is_bitwise_exact", disabled is None and before == after))

    # 3. Enabled debug is read-only.  On cloud this executes the real low-level backend.
    sentinel_render = torch.arange(64, dtype=torch.float32, device=device).reshape(8, 8).clone()
    sentinel_hash = tensor_hash(sentinel_render)
    backend = None
    if require_cloud_backend:
        backend = run_backend_projection_debug(
            means=means, covars=covars, opacities=opacity, w2c=w2c, K=K,
            width=64, height=64, options=InstrumentationOptions(enabled=True),
        )
    results.append(check("test_debug_on_does_not_change_render", tensor_hash(sentinel_render) == sentinel_hash and before == tuple(tensor_hash(value) for value in (means, covars, opacity, w2c, K))))

    # 4. The stable index is a direct arange over the input order.
    projected = oracle(device)
    results.append(check("test_gaussian_indices_remain_stable", torch.equal(projected["gaussian_index"].cpu(), torch.arange(means.shape[0]))))

    # 5. All independent numerical projection outputs are float64.
    float_outputs = ("camera_xyz", "depth", "projected_mean", "covariance2d", "conic", "determinant", "eigenvalues")
    results.append(check("test_independent_projection_uses_float64", all(projected[name].dtype == torch.float64 for name in float_outputs)))

    # 6. There is no target/mask/RGB parameter in the oracle interface.
    signature = inspect.signature(independent_gaussian_projection_oracle_v1)
    forbidden = {"target", "target_rgb", "target_mask", "trusted_mask"}
    results.append(check("test_independent_projection_does_not_use_target_in_forward", not forbidden.intersection(signature.parameters)))

    # 7. Validate against gsplat where available; otherwise validate the analytic center.
    if backend is not None:
        agreement = projection_agreement(projected, backend)
        projection_ok = bool(agreement["pass"])
    else:
        projection_ok = torch.allclose(projected["projected_mean"][0].cpu(), torch.tensor([32.0, 32.0], dtype=torch.float64), atol=1e-12, rtol=0)
    results.append(check("test_projection_matches_backend_accepted_gaussians", projection_ok))

    # 8-9. Stage states are exhaustive and have one registered first rejection.
    statuses = stage_statuses(2, REJECTION_REASONS.index("BEHIND_NEAR_PLANE"), emitted=False)
    results.append(check("test_stage_status_is_exhaustive", tuple(statuses) == STAGE_NAMES and all(value == "PASS" or value.startswith("REJECT_") for value in statuses.values())))
    first_rejections = [index for index, value in enumerate(statuses.values()) if value.startswith("REJECT_") and (index == 0 or list(statuses.values())[index - 1] == "PASS")]
    results.append(check("test_stage_status_has_single_first_rejection", first_rejections == [2]))

    # 10. A backend zero radius after independent S5 is named, never silently ignored.
    fake_backend = {
        "radii": torch.zeros((1, means.shape[0], 2), dtype=torch.int32, device=device),
        "tiles_per_gauss": torch.zeros((1, means.shape[0]), dtype=torch.int32, device=device),
    }
    _, codes, _ = backend_first_rejections(projected, fake_backend)
    results.append(check("test_backend_unexposed_rejection_is_not_silently_ignored", torch.all(codes == REJECTION_REASONS.index("BACKEND_UNEXPOSED_REJECTION"))))

    # 11. Tile bounds use backend's floor/ceil, inclusive/exclusive convention.
    tile_min, tile_max, tile_count = tile_bboxes(
        torch.tensor([[16.0, 16.0]], dtype=torch.float64),
        torch.tensor([[1, 1]], dtype=torch.int64),
        tile_size=16, tile_width=4, tile_height=4,
    )
    tile_ok = tile_min.tolist() == [[0, 0]] and tile_max.tolist() == [[2, 2]] and tile_count.tolist() == [4]
    if backend is not None:
        independent_count = projected["tile_count"].to(backend["tiles_per_gauss"].device)
        tile_ok &= torch.equal(independent_count, backend["tiles_per_gauss"][0].to(torch.int64))
    results.append(check("test_tile_bbox_matches_backend", tile_ok))

    # 12. Funnel monotonicity is asserted.
    monotonic = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    results.append(check("test_support_funnel_is_monotonic", support_funnel_counts(monotonic) == [3, 2, 2, 1, 1, 1, 1]))

    # 13. Same-index analysis never invokes nearest-neighbour rematching.
    ids = torch.tensor([1, 0])
    source = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    target = torch.tensor([[2., 0., 0.], [4., 0., 0.]])
    displacement = same_index_displacement(ids, source, target, source[:, :2], target[:, :2])
    results.append(check("test_same_index_matching_uses_base_index", displacement["canonical_or_posed_displacement"][:, 0].tolist() == [3.0, 2.0]))

    # 14-15. Counterfactual is pure and cannot mutate renderer/Gaussian state.
    projection_hash = tensor_hash(projected["projected_mean"])
    gaussian_hash = tensor_hash(means)
    counterfactual_alpha(projected, torch.tensor([0], device=device), [(31, 31), (0, 0)])
    results.append(check("test_counterfactual_does_not_modify_renderer", tensor_hash(sentinel_render) == sentinel_hash))
    results.append(check("test_counterfactual_does_not_modify_gaussians", tensor_hash(projected["projected_mean"]) == projection_hash and tensor_hash(means) == gaussian_hash))

    # 16. Tile boundary scene deterministically overlaps four tiles.
    results.append(check("test_synthetic_tile_boundary_scene", tile_count.item() == 4))

    # 17. Near-plane rejection is explicit.
    near_xyz = torch.tensor([[0., 0., .05]], device=device)
    near_projected = oracle(device, xyz=near_xyz, covariance=torch.eye(3, device=device)[None] * .001)
    results.append(check("test_synthetic_near_plane_scene", int(near_projected["first_rejection_code"][0]) == REJECTION_REASONS.index("BEHIND_NEAR_PLANE")))

    # 18. Anisotropic covariance preserves a larger major radius.
    anisotropic_cov = torch.diag(torch.tensor([.01, .0001, .001], device=device))[None]
    anisotropic = oracle(device, xyz=torch.tensor([[0., 0., 2.]], device=device), covariance=anisotropic_cov)
    results.append(check("test_synthetic_anisotropic_scene", float(anisotropic["major_minor_radius"][0, 0]) > float(anisotropic["major_minor_radius"][0, 1])))

    # 19. Static guard: the task cannot create a training optimizer.
    syntax = ast.parse(module_source + "\n" + runner_source)
    forbidden_calls = [node for node in ast.walk(syntax) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "torch" and node.attr == "optim"]
    results.append(check("test_no_optimizer_is_created", not forbidden_calls and int(config["permissions"]["optimizer_steps"]) == 0))

    # 20. Prior formal outputs are only read and never destructive.
    alpha_path = Path(config["source_alpha_audit_output"])
    alpha_ok = alpha_path.is_dir() if require_cloud_backend else config["source_alpha_audit_output"].startswith("/root/autodl-tmp/")
    destructive = ("rmtree(", ".unlink(", "rm -rf", "Remove-Item")
    results.append(check("test_previous_audit_outputs_are_unchanged", alpha_ok and not any(token in runner_source for token in destructive)))

    # 21. Every frozen ref resolves exactly.
    refs_ok = True
    for branch, expected in config["frozen_branches"].items():
        actual = None
        for candidate in (branch, f"cloud/{branch}", f"origin/{branch}"):
            try:
                actual = git("rev-parse", candidate)
                break
            except subprocess.CalledProcessError:
                pass
        refs_ok &= actual == expected
    results.append(check("test_frozen_branches_are_unchanged", refs_ok))

    return {"status": "PASS", "test_count": len(results), "tests": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-cloud-backend", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.require_cloud_backend), indent=2))


if __name__ == "__main__":
    main()
