from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import make_grid, save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_dressable as training
from scene.full_attribute_oracle import AnchorResidualOracle, GaussianResidualOracle
from scene.gaussian_clothing_residuals import (
    AnchorClothingResiduals,
    compose_canonical_gaussian_overrides,
    interpolate_anchor_clothing_residuals,
)
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.full_training_checkpoint_utils import load_full_training_checkpoint, save_full_training_checkpoint
from utils.mmlphuman_state_utils import mmlphuman_state_transaction
from utils.oracle_loss_utils import OracleLossWeights, mask_iou, oracle_rendering_loss


def chw(value, channels):
    if value.ndim == 3 and value.shape[0] == channels: return value
    if value.ndim == 3 and value.shape[-1] == channels: return value.permute(2, 0, 1).contiguous()
    raise ValueError("ambiguous render shape")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline-config", required=True, type=Path)
    p.add_argument("--oracle-config", required=True, type=Path)
    p.add_argument("--request", required=True, type=Path)
    p.add_argument("--teacher-anchor-residuals", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if not 20 <= args.steps <= 50: raise ValueError("fixture steps must be in [20,50]")
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    config = training.load_config(args.pipeline_config)
    oc = yaml.safe_load(args.oracle_config.read_text(encoding="utf-8"))
    device = torch.device(args.device); torch.manual_seed(20260715)
    base = training.load_frozen_mmlphuman_base(config["base"]["model_dir"], config["base"]["checkpoint_path"], device)
    request = torch.load(args.request, map_location="cpu", weights_only=False)
    teacher_anchor = AnchorClothingResiduals(**torch.load(args.teacher_anchor_residuals, map_location=device, weights_only=True))
    indices = base.nbr_gs.to(device=device, dtype=torch.long)
    weights_i = base.nbr_gs_invdist.to(device=device, dtype=base._xyz.dtype)
    weights_i = weights_i / weights_i.sum(1, keepdim=True)
    teacher_gaussian = interpolate_anchor_clothing_residuals(teacher_anchor, indices, weights_i, base)
    teacher_overrides = compose_canonical_gaussian_overrides(base, teacher_gaussian)
    background = torch.tensor(config["render"]["background"], device=device, dtype=base._xyz.dtype)
    conditions = [
        {"id": "reference_0", "pose": request["reference_poses"][0], "Rh": request["reference_Rh"][0], "Th": request["reference_Th"][0], "camera": request["reference_cameras"][0]},
        {"id": "reference_1", "pose": request["reference_poses"][1], "Rh": request["reference_Rh"][1], "Th": request["reference_Th"][1], "camera": request["reference_cameras"][1]},
        {"id": "target", "pose": request["target_pose"], "Rh": request["target_Rh"], "Th": request["target_Th"], "camera": request["target_camera"]},
    ]
    height, width = int(request["target_height"]), int(request["target_width"])

    def render(condition, overrides):
        camera = build_mmlphuman_camera(condition["camera"], height, width, device)
        with mmlphuman_state_transaction(base, condition["pose"].to(device), condition["Rh"].to(device), condition["Th"].to(device)):
            return base.render(camera, background=background, canonical_overrides=None if overrides is None else overrides.as_dict())

    fixture = []
    with torch.no_grad():
        for condition in conditions:
            br, ba, _ = render(condition, None); tr, ta, _ = render(condition, teacher_overrides)
            br, ba, tr, ta = chw(br, 3), chw(ba, 1), chw(tr, 3), chw(ta, 1)
            clothing = (((tr - br).abs().mean(0, keepdim=True) > .005) | ((ta - ba).abs() > .005)).float() * (ta > .01)
            fixture.append({**condition, "base_rgb": br, "base_alpha": ba, "target_rgb": tr, "target_alpha": ta, "clothing": clothing})
    manifest = {"fixture_only": True, "oracle_type": "representation_capacity_upper_bound", "conditions": [x["id"] for x in fixture], "split": {"train": ["reference_0", "reference_1"], "validation": ["target"], "test": ["target"]}}
    (out / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "oracle_split.json").write_text(json.dumps(manifest["split"], indent=2), encoding="utf-8")
    results = {}
    for kind in ("gaussian", "anchor"):
        root = out / f"{kind}_oracle"; root.mkdir(exist_ok=True); (root / "renders").mkdir(exist_ok=True); (root / "gates").mkdir(exist_ok=True)
        common = dict(bounds=oc["bounds"], initial_gate_probability=oc["initial_gate_probability"], enable_shn=False)
        if kind == "gaussian":
            oracle = GaussianResidualOracle(base, **common).to(device); target_residual = teacher_gaussian
        else:
            edges = training.build_anchor_knn_edges(base.xyz_vt.to(device), 4)
            oracle = AnchorResidualOracle(base, base.xyz_vt.shape[0], indices, weights_i, graph_edges=edges, **common).to(device); target_residual = teacher_anchor
        lrs = {"geometry_residuals": .03, "appearance_residuals": .05, "geometry_gate": .03, "appearance_gate": .03}
        groups, names = oracle.parameter_groups(lrs); optimizer = torch.optim.Adam(groups)
        loss_weights = OracleLossWeights.from_mapping({**oc["loss"], "lpips": 0.0})
        initial = evaluate(oracle, fixture, render, base, loss_weights)
        records=[]; started=time.perf_counter()
        for step in range(args.steps):
            stage = 1 if step < 5 else 2 if step < 10 else 3
            oracle.configure_stage(stage); sample=fixture[step % 2]
            optimizer.zero_grad(set_to_none=True); output=oracle(base); pred=render(sample, output.canonical_overrides)
            parts=oracle_rendering_loss(prediction_rgb=pred[0],prediction_alpha=pred[1],target_rgb=sample["target_rgb"],target_foreground_mask=sample["target_alpha"],target_clothing_mask=sample["clothing"],base_rgb=sample["base_rgb"],base_alpha=sample["base_alpha"],oracle_output=output,weights=loss_weights)
            supervision=sum(F.smooth_l1_loss(getattr(output.raw_residuals,n),getattr(target_residual,n)) for n in ("delta_xyz","delta_log_scaling","delta_rotvec","delta_opacity_logit","delta_sh0","delta_shN"))
            active=((target_residual.delta_xyz.reshape(target_residual.delta_xyz.shape[0],-1).abs().sum(1)+target_residual.delta_sh0.reshape(target_residual.delta_sh0.shape[0],-1).abs().sum(1))>1e-8).float().reshape(-1,1)
            gate_loss=F.binary_cross_entropy(output.geometry_gate.clamp(1e-6,1-1e-6),active)+F.binary_cross_entropy(output.appearance_gate.clamp(1e-6,1-1e-6),active)
            total=parts["total"]+5*supervision+.1*gate_loss
            if not torch.isfinite(total): raise FloatingPointError("nonfinite fixture loss")
            total.backward(); grad=float(torch.nn.utils.clip_grad_norm_(oracle.parameters(),1.0)); optimizer.step()
            records.append({"step":step+1,"stage":stage,"total":float(total),"render_total":float(parts["total"]),"supervision":float(supervision),"gate":float(gate_loss),"gradient_norm":grad})
        final=evaluate(oracle,fixture,render,base,loss_weights)
        with (root/"training_log.jsonl").open("w",encoding="utf-8") as f:
            for row in records: f.write(json.dumps(row)+"\n")
        method={"oracle_type":oracle.oracle_type,"kind":kind}; data={"fixture_only":True,"conditions":[x["id"] for x in fixture]}
        save_full_training_checkpoint(root/"last_checkpoint.pth",model=oracle,optimizer=optimizer,optimizer_group_names=names,scheduler=None,scaler=None,training_state={"global_step":args.steps,"stage":oracle.stage},data_state=data,method_state=method)
        save_full_training_checkpoint(root/"best_checkpoint.pth",model=oracle,optimizer=optimizer,optimizer_group_names=names,scheduler=None,scaler=None,training_state={"global_step":args.steps,"stage":oracle.stage},data_state=data,method_state=method)
        before=oracle(base).canonical_overrides.xyz.detach().cpu(); fresh=(GaussianResidualOracle(base,**common) if kind=="gaussian" else AnchorResidualOracle(base,base.xyz_vt.shape[0],indices,weights_i,graph_edges=training.build_anchor_knn_edges(base.xyz_vt.to(device),4),**common)).to(device); fresh.configure_stage(oracle.stage); fg,fn=fresh.parameter_groups(lrs); fo=torch.optim.Adam(fg)
        load_full_training_checkpoint(root/"last_checkpoint.pth",model=fresh,optimizer=fo,optimizer_group_names=fn,scheduler=None,scaler=None,expected_method_state=method,expected_data_state=data)
        parity=float((before-fresh(base).canonical_overrides.xyz.detach().cpu()).abs().max())
        diagnostics={"initial":initial,"final":final,"loss_drop_ratio":1-final["total"]/initial["total"],"clothing_rgb_improvement_ratio":1-final["clothing_rgb"]/initial["clothing_rgb"],"checkpoint_roundtrip_max_abs":parity,"steps":args.steps,"training_seconds":time.perf_counter()-started,"parameter_count":sum(p.numel() for p in oracle.parameters()),"base_gradient_count":sum(p.grad is not None for p in base.parameters()),"shared_canonical_residual":True,"no_image_conditioning":True}
        (root/"metrics.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8"); (root/"residual_diagnostics.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
        save_visuals(root,oracle,fixture,render,base); results[kind]=diagnostics
    comparison=out/"comparison"; comparison.mkdir(exist_ok=True); (comparison/"metric_comparison.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    panels=[]
    for kind in ("gaussian","anchor"):
        panels.extend([torch.load(out/f"{kind}_oracle"/"renders"/f"{name}.pt",weights_only=True) for name in ("base","target","prediction","diff")])
    save_image(make_grid(panels,nrow=4),comparison/"render_comparison.png")
    status="PASS" if all(v["loss_drop_ratio"]>=.30 and v["clothing_rgb_improvement_ratio"]>=.25 and v["final"]["mask_iou"]>=v["initial"]["mask_iou"] and v["checkpoint_roundtrip_max_abs"]==0 and v["base_gradient_count"]==0 for v in results.values()) else "PARTIAL"
    (out/"GATE_ACCEPTANCE.md").write_text(f"# Module 4A Oracle Infrastructure\n\nStatus: {status}\n\nSynthetic fixture only; no real-outfit capacity claim.\n",encoding="utf-8")
    print(json.dumps({"status":status,"results":results},indent=2))


def evaluate(oracle,fixture,render,base,weights):
    values={"total":0.0,"clothing_rgb":0.0,"mask_iou":0.0,"non_clothing_rgb":0.0}; oracle.eval()
    with torch.no_grad():
        output=oracle(base)
        for sample in fixture:
            pred=render(sample,output.canonical_overrides); parts=oracle_rendering_loss(prediction_rgb=pred[0],prediction_alpha=pred[1],target_rgb=sample["target_rgb"],target_foreground_mask=sample["target_alpha"],target_clothing_mask=sample["clothing"],base_rgb=sample["base_rgb"],base_alpha=sample["base_alpha"],oracle_output=output,weights=weights)
            values["total"]+=float(parts["total"]); values["clothing_rgb"]+=float(parts["clothing_rgb"]); values["mask_iou"]+=float(mask_iou(pred[1],sample["target_alpha"])); values["non_clothing_rgb"]+=float(parts["non_clothing_rgb"])
    return {k:v/len(fixture) for k,v in values.items()}


def save_visuals(root,oracle,fixture,render,base):
    sample=fixture[-1]; output=oracle(base)
    with torch.no_grad(): pred=render(sample,output.canonical_overrides)
    rgb=chw(pred[0],3).detach().cpu().clamp(0,1); alpha=chw(pred[1],1).detach().cpu().clamp(0,1); target=sample["target_rgb"].cpu(); base_rgb=sample["base_rgb"].cpu(); diff=(rgb-target).abs()
    for name,tensor in (("base",base_rgb),("target",target),("prediction",rgb),("alpha",alpha),("clothing_mask",sample["clothing"].cpu()),("diff",diff)):
        torch.save(tensor,root/"renders"/f"{name}.pt"); save_image(tensor,root/"renders"/f"{name}.png")
    for name,tensor in (("geometry_gate",output.geometry_gate),("appearance_gate",output.appearance_gate)):
        flat=tensor.detach().cpu().flatten(); width=500 if flat.numel()%500==0 else 100; image=flat.reshape(1,-1,width); save_image(image,root/"gates"/f"{name}.png")


if __name__=="__main__": main()
