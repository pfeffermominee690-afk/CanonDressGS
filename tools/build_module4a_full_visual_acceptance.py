from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_dressable as training
from scene.full_attribute_oracle import AnchorResidualOracle, GaussianResidualOracle
from scene.gaussian_clothing_residuals import interpolate_anchor_field
from utils.mmlphuman_state_utils import mmlphuman_state_transaction


ATTRS = (
    "geometry_gate", "appearance_gate", "max_gate", "delta_xyz",
    "delta_log_scaling", "delta_rotvec", "delta_opacity_logit", "delta_sh0", "delta_shN",
)
ABS_MAX = {
    "geometry_gate": 1.0, "appearance_gate": 1.0, "max_gate": 1.0,
    "delta_xyz": .05, "delta_log_scaling": .35, "delta_rotvec": .2617993878,
    "delta_opacity_logit": 2.0, "delta_sh0": .25, "delta_shN": .10,
}


def magnitude(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().reshape(value.shape[0], -1).norm(dim=1)


def values(output, level: str):
    residual = output.gaussian_residuals if level == "gaussian" else output.gated_residuals
    geometry = output.gaussian_geometry_gate[:, 0] if level == "gaussian" else output.geometry_gate[:, 0]
    appearance = output.gaussian_appearance_gate[:, 0] if level == "gaussian" else output.appearance_gate[:, 0]
    result = {"geometry_gate": geometry, "appearance_gate": appearance, "max_gate": torch.maximum(geometry, appearance)}
    for name in ATTRS[3:]: result[name] = magnitude(getattr(residual, name))
    return result


def statistics(value: torch.Tensor):
    x = value.detach().float().cpu()
    q = torch.quantile(x, torch.tensor([.5, .9, .95, .99]))
    return {"min": float(x.min()), "mean": float(x.mean()), "median": float(q[0]), "p90": float(q[1]), "p95": float(q[2]), "p99": float(q[3]), "max": float(x.max()), "active_count_gt_1e-6": int((x > 1e-6).sum()), "count": x.numel()}


def scatter_panel(xyz, score, title, output, vmax, views=("front", "back", "left", "right")):
    xyz, score = xyz.detach().cpu().numpy(), score.detach().cpu().numpy()
    if len(xyz) > 50000:
        index = np.linspace(0, len(xyz) - 1, 50000).astype(np.int64); xyz, score = xyz[index], score[index]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=120)
    pairs = [(0, 1), (0, 1), (2, 1), (2, 1)]
    signs = [(1, 1), (-1, 1), (1, 1), (-1, 1)]
    last = None
    for ax, name, pair, sign in zip(axes, views, pairs, signs):
        last = ax.scatter(xyz[:, pair[0]] * sign[0], xyz[:, pair[1]] * sign[1], c=score, s=.8, cmap="magma", vmin=0, vmax=vmax, linewidths=0)
        ax.set_title(name); ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle(title); fig.colorbar(last, ax=axes, fraction=.02, pad=.01)
    fig.savefig(output, bbox_inches="tight"); plt.close(fig)


def overlay(points, score, request, base, output_path, background_image):
    device = base._xyz.device
    with mmlphuman_state_transaction(base, request["target_pose"].to(device), request["target_Rh"].to(device), request["target_Th"].to(device)):
        posed = base.get_xyz.detach().cpu()
    w2c = request["target_camera"]["w2c"].float()
    K = request["target_camera"]["K"].float()
    homogeneous = torch.cat((posed, torch.ones(posed.shape[0], 1)), 1)
    camera = homogeneous @ w2c.T
    z = camera[:, 2]
    pixels = camera[:, :3] @ K.T
    uv = pixels[:, :2] / pixels[:, 2:3].clamp_min(1e-6)
    image = np.array(Image.open(background_image).convert("RGB"), copy=True)
    height, width = image.shape[:2]
    u, v = uv[:, 0].long(), uv[:, 1].long()
    valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    heat = np.zeros((height, width), dtype=np.float32)
    np.maximum.at(heat, (v[valid].numpy(), u[valid].numpy()), score.detach().cpu()[valid].numpy())
    vmax = max(float(torch.quantile(score.detach().float().cpu(), .99)), 1e-8)
    heat = np.clip(heat / vmax, 0, 1)
    color = plt.get_cmap("magma")(heat)[..., :3] * 255
    mask = heat > 0
    image[mask] = .45 * image[mask] + .55 * color[mask]
    Image.fromarray(image.astype(np.uint8)).save(output_path)


def labeled_grid(paths, output, columns=5, width=420, height=340):
    cells = []
    for label, path in paths:
        image = Image.open(path).convert("RGB"); image.thumbnail((width, height - 35))
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(image, ((width - image.width) // 2, 30))
        ImageDraw.Draw(canvas).text((8, 8), label, fill="black")
        cells.append(canvas)
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for index, cell in enumerate(cells): sheet.paste(cell, ((index % columns) * width, (index // columns) * height))
    sheet.save(output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline-config", required=True, type=Path); p.add_argument("--oracle-config", required=True, type=Path)
    p.add_argument("--request", required=True, type=Path); p.add_argument("--teacher-anchor-residuals", required=True, type=Path)
    p.add_argument("--gate7-root", required=True, type=Path); p.add_argument("--output-dir", required=True, type=Path); p.add_argument("--device", default="cuda")
    args = p.parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    for name in ("gaussian_canonical_panels", "anchor_canonical_panels", "target_overlays"): (out / name).mkdir(exist_ok=True)
    config = training.load_config(args.pipeline_config); oc = yaml.safe_load(args.oracle_config.read_text(encoding="utf-8")); device = torch.device(args.device)
    base = training.load_frozen_mmlphuman_base(config["base"]["model_dir"], config["base"]["checkpoint_path"], device)
    request = torch.load(args.request, map_location="cpu", weights_only=False)
    indices = base.nbr_gs.to(device=device, dtype=torch.long); weights = base.nbr_gs_invdist.to(device=device, dtype=base._xyz.dtype); weights /= weights.sum(1, keepdim=True)
    common = dict(bounds=oc["bounds"], initial_gate_probability=oc["initial_gate_probability"], enable_shn=False)
    models = {
        "gaussian": GaussianResidualOracle(base, **common).to(device),
        "anchor": AnchorResidualOracle(base, base.xyz_vt.shape[0], indices, weights, graph_edges=training.build_anchor_knn_edges(base.xyz_vt.to(device), 4), **common).to(device),
    }
    stats, panels = {}, []
    for kind, model in models.items():
        checkpoint = torch.load(args.gate7_root / f"{kind}_oracle/last_checkpoint.pth", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
        with torch.no_grad(): output = model(base)
        levels = ("gaussian",) if kind == "gaussian" else ("anchor", "gaussian")
        stats[kind] = {}
        for level in levels:
            xyz = base._xyz if level == "gaussian" else base.xyz_vt
            fields = values(output, level); stats[kind][level] = {}
            directory = out / f"{kind}_canonical_panels"
            for name, score in fields.items():
                stats[kind][level][name] = statistics(score)
                absolute = directory / f"{level}_{name}_absolute.png"
                fixed = directory / f"{level}_{name}_fixed_gain.png"
                scatter_panel(xyz, score, f"{kind} {level} {name} absolute", absolute, ABS_MAX[name])
                fixed_max = max(stats[kind][level][name]["p99"], 1e-8)
                scatter_panel(xyz, score, f"{kind} {level} {name} fixed gain (p99={fixed_max:.6g})", fixed, fixed_max)
                if level == ("gaussian" if kind == "gaussian" else "anchor"): panels.append((f"{kind} {name}", fixed))
            gaussian_fields = values(output, "gaussian")
            for name in ("geometry_gate", "appearance_gate", "delta_xyz", "delta_sh0"):
                path = out / "target_overlays" / f"{kind}_{name}_overlay.png"
                overlay(base._xyz, gaussian_fields[name], request, base, path, args.gate7_root / f"{kind}_oracle/renders/target.png")
                panels.append((f"{kind} target {name}", path))
    for kind in ("gaussian", "anchor"):
        for name in ("base", "target", "prediction", "clothing_mask", "diff", "alpha"):
            panels.insert(0, (f"{kind} {name}", args.gate7_root / f"{kind}_oracle/renders/{name}.png"))
    (out / "residual_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    labeled_grid(panels, out / "module4a_full_visual_acceptance_contact_sheet.png")
    shn_zero = all(stats[kind][level]["delta_shN"]["max"] == 0 for kind in stats for level in stats[kind])
    summary = {"images_actually_opened": False, "inspection_method": "pending manual image inspection", "shn_disabled_strict_zero": shn_zero, "visual_acceptance_status": "PENDING", "statistics": stats}
    (out / "visual_acceptance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "VISUAL_ACCEPTANCE.md").write_text("# Module 4A.1 Visual Acceptance\n\nStatus: PENDING actual image inspection.\n", encoding="utf-8")


if __name__ == "__main__": main()
