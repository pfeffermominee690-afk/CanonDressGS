from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

import train_dressable as training
from scene.gaussian_clothing_residuals import AnchorClothingResiduals, CHANNELS
from tools.infer_module3_online_completion import _InferenceModelContract, _num_clothes
from utils.clothing_completion_loss_utils import completion_losses, deterministic_feature_holdout
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.full_training_checkpoint_utils import (
    capture_random_state, load_full_training_checkpoint, mark_legacy_model_only_checkpoint,
    random_state_fingerprint, save_full_training_checkpoint, validate_full_training_checkpoint,
)
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter
from utils.mmlphuman_state_utils import mmlphuman_state_transaction


def sha(path: str | Path) -> str: return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(value: torch.Tensor) -> str: return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def chw(value, channels): return value if value.shape[0] == channels else value.permute(2, 0, 1)


class Fixture:
    def __init__(self, args):
        self.args=args; self.device=torch.device(args.device)
        self.c4=training.load_config(args.gate4_config); self.c2=training.load_config(args.module2_config); self.c3=training.load_config(args.module3_config); self.cfg=self.c3["online_completion"]
        random.seed(self.cfg["seed"]); np.random.seed(self.cfg["seed"]); torch.manual_seed(self.cfg["seed"]); torch.cuda.manual_seed_all(self.cfg["seed"])
        self.legacy=torch.load(args.legacy_checkpoint,map_location=self.device,weights_only=True)
        self.request=torch.load(args.inference_request,map_location="cpu",weights_only=True)
        self.base=training.load_frozen_mmlphuman_base(self.c4["base"]["model_dir"],self.c4["base"]["checkpoint_path"],self.device)
        self.model,_,_,_=training.create_image_conditioned_components(_InferenceModelContract(_num_clothes(self.legacy["model"])),self.c4,self.base,self.device)
        self.model.dressable_model.configure_six_channel_decoder(self.c2["dressable_channels"])
        gi=self.legacy["graph_indices"].to(self.device); gw=self.legacy["graph_weights"].to(device=self.device,dtype=self.base._xyz.dtype)
        md=self.legacy["metadata"]; self.model.initialize_online_completion(gi,gw,int(md["feature_completion_hidden_dim"]),int(md["feature_completion_blocks"])); self.model.load_state_dict(self.legacy["model"],strict=True)
        self.gi,self.gw=gi,gw; self.adapter=MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(self.base,canonical_anchors=self.model.canonical_anchors,lbs_grid_path=self.c4["base"]["lbs_grid_path"])
        self.background=torch.tensor(self.c4["render"]["background"],device=self.device,dtype=self.base._xyz.dtype)
        episode={k:v for k,v in self.request.items() if k.startswith("reference_")}; geometry=training.prepare_real_reference_geometry(self.base,self.adapter,episode,self.background)
        self.full_inputs=self.inputs([0,1],geometry)
        teacher=AnchorClothingResiduals(**torch.load(Path(args.module2_output)/"teacher_anchor_residuals.pt",map_location=self.device,weights_only=True)).validate(10000,self.base)
        self.gt=((teacher.delta_xyz.abs().sum(1)+teacher.delta_log_scaling.abs().sum(1)+teacher.delta_rotvec.abs().sum(1))>1e-8).to(self.base._xyz.dtype).reshape(-1,1)
        self.at=((teacher.delta_sh0.abs().sum(1)+teacher.delta_shN.abs().sum(1))>1e-8).to(self.base._xyz.dtype).reshape(-1,1)
        for p in self.model.parameters(): p.requires_grad_(False)
        for p in self.model.canonical_clothing_completer.parameters(): p.requires_grad_(True)
        self.model.eval()
        with torch.no_grad():
            self.evidence={name:self.model.encode_clothing_online(**self.inputs(ids,geometry)) for name,ids in {"S1":[0],"S2":[1],"S12":[0,1]}.items()}
        observed=self.evidence["S12"]["observed"]; self.hold=deterministic_feature_holdout(observed["clothing_observed_mask"],float(self.cfg["feature_holdout_fraction"]),self.cfg["seed"]); self.target=observed["observed_clothing_feature"].detach()
        self.optimizer=torch.optim.Adam(self.model.canonical_clothing_completer.parameters(),lr=float(self.cfg["learning_rate"]))
        self.scheduler=None; self.scaler=None
        self.data_state={"fixture_fingerprint":md["dataset_fixture_fingerprint"],"manifest_fingerprint":md["dataset_fixture_fingerprint"],"reference_subset":["f000_c018","f1000_c000"],"target_id":"f2000_c009","sampler_state":None,"deterministic_batch_fingerprint":sha(args.inference_request),"feature_holdout_mask_fingerprint":tensor_sha(self.hold)}
        self.method_state={"graph_fingerprint":md["graph_fingerprint"],"graph_version":md["graph_version"],"gate_source":md["gate_source"],"module1_residual_contract":md["module2_decoder_metadata"]["residual_contract_version"],"module2_decoder_metadata":md["module2_decoder_metadata"],"active_sh_degree":md["module2_decoder_metadata"]["effective_sh_degree"],"base_checkpoint_sha256":md["base_checkpoint_sha256"],"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"config_fingerprint":sha(args.module3_config)}

    def inputs(self,ids,geometry):
        r=self.request
        ep={"reference_Rh":r["reference_Rh"][ids],"reference_Th":r["reference_Th"][ids]}
        return {"reference_images":r["reference_images"][ids].to(self.device),"reference_cloth_masks":r["reference_cloth_masks"][ids].to(self.device),"reference_foreground_masks":r["reference_foreground_masks"][ids].to(self.device),"reference_poses":r["reference_poses"][ids].to(self.device),"reference_cameras":[r["reference_cameras"][i] for i in ids],"reference_valid_mask":r["reference_valid_mask"][ids].to(self.device),"deformation_fn":training.build_episode_deformation_fn(self.adapter,ep),"surface_depth_maps":geometry["surface_depth_maps"][ids],"surface_alpha_maps":geometry["surface_alpha_maps"][ids],"require_depth_visibility":True}

    def complete(self,name,holdout=False):
        result=self.evidence[name]; obs=result["observed"]; clothing=obs["observed_clothing_feature"]; coverage=obs["observation_coverage"]; probability=obs["observed_clothing_probability"]
        if holdout: clothing=clothing.clone(); coverage=coverage.clone(); probability=probability.clone(); clothing[self.hold]=0; coverage[self.hold]=0; probability[self.hold]=0
        anchors=self.model.canonical_anchors; pos=(anchors-anchors.mean(0))/(anchors-anchors.mean(0)).abs().amax().clamp_min(1e-8)
        return self.model.canonical_clothing_completer(result["global_clothing_embedding"].detach(),obs["observed_surface_feature"].detach(),clothing.detach(),probability.detach(),coverage.detach(),pos,self.gi,self.gw)

    def loss(self):
        outputs={n:self.complete(n) for n in ("S1","S2","S12")}; hold=self.complete("S12",True); parts=completion_losses(hold,self.gt,self.at,self.gi,self.gw,self.hold,self.target)
        common=(outputs["S1"].observation_coverage>0)&(outputs["S2"].observation_coverage>0)
        parts["subset_feature_consistency"]=F.smooth_l1_loss(outputs["S1"].completed_anchor_features[common[:,0]],outputs["S2"].completed_anchor_features[common[:,0]])
        parts["subset_geometry_gate_consistency"]=F.l1_loss(outputs["S1"].geometry_gate,outputs["S12"].geometry_gate); parts["subset_appearance_gate_consistency"]=F.l1_loss(outputs["S2"].appearance_gate,outputs["S12"].appearance_gate)
        total=sum(float(self.cfg["loss_weights"].get(k,0))*v for k,v in parts.items()); return total,parts,outputs["S12"]

    def render(self,completion):
        raw,bounded,gated=self.model.dressable_model.compute_film_anchor_residuals(completion.gate_bundle(),clothing_embedding=self.evidence["S12"]["global_clothing_embedding"],anchor_clothing_features=completion.completed_anchor_features)
        gaussian=self.model.dressable_model.interpolate_anchor_clothing_residuals(gated); overrides=self.model.dressable_model.compose_canonical_gaussian_overrides(gaussian,CHANNELS); r=self.request; cam=build_mmlphuman_camera(r["target_camera"],int(r["target_height"]),int(r["target_width"]),self.device); old=int(self.base.sh_degree)
        try:
            self.base.sh_degree=int(self.legacy["metadata"]["module2_decoder_metadata"]["effective_sh_degree"])
            with mmlphuman_state_transaction(self.base,r["target_pose"].to(self.device),r["target_Rh"].to(self.device),r["target_Th"].to(self.device)): rgb,alpha,_=self.base.render(cam,background=self.background,canonical_overrides=overrides.as_dict())
        finally:self.base.sh_degree=old
        return raw,bounded,gated,gaussian,chw(rgb,3),chw(alpha,1)

    def step(self,global_step):
        pre_rng=random_state_fingerprint(capture_random_state()); self.optimizer.zero_grad(set_to_none=True); total,parts,completion=self.loss(); rendered=self.render(completion); pre=self.pack(total,parts,completion,rendered)
        total.backward(); grads={n:p.grad.detach().cpu().clone() for n,p in self.model.named_parameters() if p.requires_grad and p.grad is not None}; base_grad=sum(getattr(self.base,n).grad is not None for n in ("_xyz","_scaling","_rotation","_opacity","_sh0","_shN")); before={n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if p.requires_grad}; base_before={n:getattr(self.base,n).detach().clone() for n in ("_xyz","_scaling","_rotation","_opacity","_sh0","_shN")}; self.optimizer.step()
        with torch.no_grad(): post_total,post_parts,post_completion=self.loss(); post_rendered=self.render(post_completion); post=self.pack(post_total,post_parts,post_completion,post_rendered)
        updates={n:(p.detach().cpu()-before[n]) for n,p in self.model.named_parameters() if p.requires_grad}
        base_max=max(float((getattr(self.base,n)-value).abs().max()) for n,value in base_before.items())
        return {"global_step_before":global_step,"global_step_after":global_step+1,"pre_rng":pre_rng,"post_rng":random_state_fingerprint(capture_random_state()),"pre":pre,"post":post,"gradients":grads,"updates":updates,"base_grad_count":base_grad,"base_parameter_max_diff":base_max}

    def pack(self,total,parts,c,r):
        raw,bounded,gated,gaussian,rgb,alpha=r
        d={"total_loss":total.detach().cpu(),"loss_components":{k:v.detach().cpu() for k,v in parts.items()},"completed_features":c.completed_anchor_features.detach().cpu(),"geometry_gate":c.geometry_gate.detach().cpu(),"appearance_gate":c.appearance_gate.detach().cpu(),"confidence":c.confidence.detach().cpu(),"RGB":rgb.detach().cpu(),"alpha":alpha.detach().cpu()}
        for stage,obj in (("raw",raw),("bounded",bounded),("anchor",gated),("Gaussian",gaussian)):
            for n in CHANNELS:d[f"{stage}.{n}"]=getattr(obj,n).detach().cpu()
        return d


def compare(a,b,path="",rgb=False):
    rows={}
    if isinstance(a,dict):
        for k in a: rows.update(compare(a[k],b[k],f"{path}.{k}" if path else k,k in ("RGB","alpha")))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b): rows[path]={"equal":False}
        else:
            for i,(left,right) in enumerate(zip(a,b)): rows.update(compare(left,right,f"{path}[{i}]",rgb))
    elif isinstance(a,torch.Tensor):
        diff=(a.float()-b.float()).abs(); atol=1e-6 if rgb else 1e-7; rtol=1e-5 if rgb else 1e-6
        rows[path]={"max_abs":float(diff.max()),"mean_abs":float(diff.mean()),"allclose":bool(torch.allclose(a,b,atol=atol,rtol=rtol)),"exact":bool(torch.equal(a,b))}
    elif isinstance(a,(int,float,str,bool)) or a is None: rows[path]={"equal":a==b}
    return rows


def save_result(path,data): torch.save(data,path)


def run_worker(args):
    f=Fixture(args); ck=load_full_training_checkpoint(args.resume_source,model=f.model,optimizer=f.optimizer,optimizer_group_names=["canonical_clothing_completer"],scheduler=None,scaler=None,expected_method_state=f.method_state,expected_data_state=f.data_state); result=f.step(int(ck["training_state"]["global_step"])); save_result(args.worker_output,{"step":result,"model":f.model.state_dict(),"optimizer":f.optimizer.state_dict(),"training_state":{"global_step":result["global_step_after"]}}); return


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--gate4-config",required=True); ap.add_argument("--module2-config",required=True); ap.add_argument("--module3-config",required=True); ap.add_argument("--module2-output",required=True); ap.add_argument("--legacy-checkpoint",required=True); ap.add_argument("--inference-request",required=True); ap.add_argument("--output-dir"); ap.add_argument("--device",default="cuda"); ap.add_argument("--worker",action="store_true"); ap.add_argument("--resume-source"); ap.add_argument("--worker-output"); args=ap.parse_args()
    if args.worker:return run_worker(args)
    out=Path(args.output_dir)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True); f=Fixture(args)
    legacy=mark_legacy_model_only_checkpoint(args.legacy_checkpoint)
    warm=f.step(97)
    source=out/"checkpoint_resume_source.pth"
    source_payload=save_full_training_checkpoint(source,model=f.model,optimizer=f.optimizer,optimizer_group_names=["canonical_clothing_completer"],scheduler=None,scaler=None,training_state={"global_step":98,"optimizer_step":1,"epoch":None,"batch_index":0,"gradient_accumulation_position":0,"best_metric":None,"selected_step":97,"initialization_source":"module3_model_weights_only"},data_state=f.data_state,method_state=f.method_state)
    control=f.step(98); control_path=out/"control_after_next_step.pth"; save_result(control_path,{"step":control,"model":f.model.state_dict(),"optimizer":f.optimizer.state_dict(),"training_state":{"global_step":99}})
    resumed_path=out/"resumed_after_next_step.pth"; command=[sys.executable,str(Path(__file__).resolve()),"--worker","--gate4-config",args.gate4_config,"--module2-config",args.module2_config,"--module3-config",args.module3_config,"--module2-output",args.module2_output,"--legacy-checkpoint",args.legacy_checkpoint,"--inference-request",args.inference_request,"--resume-source",str(source),"--worker-output",str(resumed_path),"--device",args.device]; proc=subprocess.run(command,capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(proc.stderr)
    resumed=torch.load(resumed_path,map_location="cpu",weights_only=False); control_saved=torch.load(control_path,map_location="cpu",weights_only=False)
    output_cmp=compare(control_saved["step"],resumed["step"]); param_cmp=compare(control_saved["model"],resumed["model"]); opt_cmp=compare(control_saved["optimizer"],resumed["optimizer"])
    rng_cmp={"pre_equal":control["pre_rng"]==resumed["step"]["pre_rng"],"post_equal":control["post_rng"]==resumed["step"]["post_rng"],"holdout_fingerprint":f.data_state["feature_holdout_mask_fingerprint"],"batch_fingerprint":f.data_state["deterministic_batch_fingerprint"]}
    mutations={}
    tests=[("graph_fingerprint","method_state","graph_fingerprint","bad"),("base_checkpoint_sha256","method_state","base_checkpoint_sha256","bad"),("fixture_fingerprint","data_state","fixture_fingerprint","bad"),("effective_sh_degree","method_state","active_sh_degree",99)]
    for name,section,key,value in tests:
        bad=copy.deepcopy(source_payload); bad[section][key]=value
        try:validate_full_training_checkpoint(bad,model=f.model,optimizer=f.optimizer,optimizer_group_names=["canonical_clothing_completer"],expected_method_state=f.method_state,expected_data_state=f.data_state)
        except ValueError:mutations[name]="REJECTED"
        else:mutations[name]="NOT_REJECTED"
    for name,mutator in [("checkpoint_version",lambda x:x.__setitem__("training_checkpoint_version",99)),("optimizer_parameter_group",lambda x:x["optimizer_state"]["parameter_groups"][0].__setitem__("fingerprint","bad")),("parameter_shape",lambda x:x["optimizer_state"]["parameter_groups"][0]["parameter_shapes"][0].__setitem__(0,999))]:
        bad=copy.deepcopy(source_payload);mutator(bad)
        try:validate_full_training_checkpoint(bad,model=f.model,optimizer=f.optimizer,optimizer_group_names=["canonical_clothing_completer"],expected_method_state=f.method_state,expected_data_state=f.data_state)
        except ValueError:mutations[name]="REJECTED"
        else:mutations[name]="NOT_REJECTED"
    # Inactive top-k diagnostic.
    with torch.no_grad(): c=f.complete("S12")
    active=f.gt[:,0]>=.5; inactive=~active; anchors=f.model.canonical_anchors.detach(); scores=c.geometry_gate[:,0].detach(); idx=torch.nonzero(inactive).flatten(); order=idx[torch.argsort(scores[idx],descending=True)[:50]]; active_xyz=anchors[active]
    distances=[]
    for x in anchors[order]:distances.append(float(torch.linalg.vector_norm(active_xyz-x,dim=1).min()))
    rows=[{"rank":i+1,"anchor_index":int(j),"geometry_gate":float(scores[j]),"appearance_gate":float(c.appearance_gate[j,0]),"canonical_xyz":anchors[j].cpu().tolist(),"nearest_teacher_active_distance":distances[i],"graph_neighbors":f.gi[j].cpu().tolist(),"body_region":"unavailable"} for i,j in enumerate(order)]
    diag={"geometry_gt_0_5":{"count":int((scores[inactive]>.5).sum()),"ratio":float((scores[inactive]>.5).float().mean())},"geometry_gt_0_9":{"count":int((scores[inactive]>.9).sum()),"ratio":float((scores[inactive]>.9).float().mean())},"appearance_gt_0_5":{"count":int((c.appearance_gate[:,0][inactive]>.5).sum()),"ratio":float((c.appearance_gate[:,0][inactive]>.5).float().mean())},"appearance_gt_0_9":{"count":int((c.appearance_gate[:,0][inactive]>.9).sum()),"ratio":float((c.appearance_gate[:,0][inactive]>.9).float().mean())},"top50":rows,"normal_scale_visible_contamination":False,"fixed_gain_leakage_warning":True}
    plt.figure(figsize=(7,7)); a=anchors.cpu(); plt.scatter(a[:,0],a[:,2],s=1,c="lightgray"); plt.scatter(a[active.cpu(),0],a[active.cpu(),2],s=2,c="steelblue"); plt.scatter(a[order.cpu(),0],a[order.cpu(),2],s=18,c="red"); plt.axis("equal"); plt.tight_layout(); plt.savefig(out/"inactive_gate_topk_overlay.png",dpi=180);plt.close()
    allclose=all(v.get("allclose",v.get("equal",True)) for v in output_cmp.values()) and all(v.get("allclose",v.get("equal",True)) for v in param_cmp.values()) and all(v.get("allclose",v.get("equal",True)) for v in opt_cmp.values())
    metrics={"status":"PASS" if allclose and all(v=="REJECTED" for v in mutations.values()) and rng_cmp["pre_equal"] and rng_cmp["post_equal"] and control["base_grad_count"]==0 and resumed["step"]["base_grad_count"]==0 and control["base_parameter_max_diff"]==0 and resumed["step"]["base_parameter_max_diff"]==0 else "FAIL","warmup_step":{"before":97,"after":98},"compare_step":{"before":98,"after":99},"independent_process_command":command,"pre_step_parity":{k:v for k,v in output_cmp.items() if k.startswith("pre.")},"gradient_parity":{k:v for k,v in output_cmp.items() if k.startswith("gradients.")},"post_step_parity":{k:v for k,v in output_cmp.items() if k.startswith("post.")},"base_grad_count_control":control["base_grad_count"],"base_grad_count_resumed":resumed["step"]["base_grad_count"],"base_parameter_max_diff_control":control["base_parameter_max_diff"],"base_parameter_max_diff_resumed":resumed["step"]["base_parameter_max_diff"]}
    contracts={"training_checkpoint_version":1,"legacy_checkpoint":legacy,"optimizer":{"class":"torch.optim.Adam","groups":source_payload["optimizer_state"]["parameter_groups"]},"scheduler":source_payload["scheduler_state"],"amp":source_payload["amp_state"],"fields":sorted(source_payload)}
    for name,value in [("checkpoint_contract.json",contracts),("checkpoint_resume_metrics.json",metrics),("optimizer_state_comparison.json",opt_cmp),("rng_state_comparison.json",rng_cmp),("parameter_update_comparison.json",param_cmp),("output_parity.json",output_cmp),("mismatch_rejection_tests.json",mutations),("inactive_gate_topk_diagnostics.json",diag)]: (out/name).write_text(json.dumps(value,indent=2),encoding="utf-8")
    (out/"config_resolved.yaml").write_text(yaml.safe_dump({"gate4":f.c4,"module2":f.c2,"module3":f.c3},sort_keys=False),encoding="utf-8"); shutil_text=Path("docs/MODULE3_RESUME_AUDIT.md").read_text(encoding="utf-8");(out/"resume_audit.md").write_text(shutil_text,encoding="utf-8");(out/"forward.log").write_text(json.dumps(metrics,indent=2),encoding="utf-8");(out/"GATE_ACCEPTANCE.md").write_text(f"# Module 3.1 Resume Closure\n\nStatus: **{metrics['status']}**\n",encoding="utf-8");print(json.dumps(metrics,indent=2))


if __name__=="__main__":main()
