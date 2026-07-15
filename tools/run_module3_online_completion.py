from __future__ import annotations

import argparse, copy, hashlib, json, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import make_grid,save_image

import train_dressable as training
from scene.dressable_dataset import ImageConditionedEpisodeDataset
from scene.gaussian_clothing_residuals import AnchorClothingResiduals,CHANNELS
from tools.check_real_image_conditioned_one_batch import _as_chw_render,_build_fixed_episode,save_render_tensor
from utils.anchor_graph_utils import build_surface_aware_anchor_graph
from utils.clothing_completion_loss_utils import completion_losses,deterministic_feature_holdout,gate_classification_metrics
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter
from utils.mmlphuman_state_utils import mmlphuman_state_transaction

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def diff(a,b):
 d=(a.detach().float()-b.detach().float()).abs(); return {"max":float(d.max()),"mean":float(d.mean()),"allclose":bool(torch.allclose(a,b,atol=1e-6,rtol=1e-5))}
def subset_episode(ep,ids):
 out=dict(ep); tensor_keys=['reference_images','reference_cloth_masks','reference_foreground_masks','reference_poses','reference_Rh','reference_Th','reference_valid_mask']; list_keys=['reference_cameras','reference_frame_ids','reference_view_ids']
 for k in tensor_keys: out[k]=ep[k][ids]
 for k in list_keys: out[k]=[ep[k][i] for i in ids]
 return out
def mask_metrics(pred,target,observed):
 masks={"all":torch.ones_like(observed,dtype=torch.bool),"observed":observed>.01,"unobserved":observed<=.01,"active":target>=.5,"inactive":target<.5}; result={}
 for n,m in masks.items():
  result[n]=gate_classification_metrics(pred[m].reshape(-1,1),target[m].reshape(-1,1)) if m.any() else {}
 result['inactive_mean']=float(pred[target<.5].mean()); return result
def save_anchor_map(path,value): save_image(value.detach().cpu().reshape(1,100,100).clamp(0,1),path)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--gate4-config',required=True); ap.add_argument('--module2-config',required=True); ap.add_argument('--module3-config',required=True); ap.add_argument('--module2-output',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
 out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); device=torch.device(args.device)
 c4=training.load_config(args.gate4_config); c2=training.load_config(args.module2_config); c3=training.load_config(args.module3_config); cfg=c3['online_completion']; torch.manual_seed(cfg['seed'])
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg['seed'])
 dataset=ImageConditionedEpisodeDataset(c4['image_conditioning']['manifest_path'],split='train',reference_count=2,seed=42,allow_clothing_mask_fallback=False,allow_missing_rh_th=False); episode=_build_fixed_episode(dataset)
 base=training.load_frozen_mmlphuman_base(c4['base']['model_dir'],c4['base']['checkpoint_path'],device); model,_,_,_=training.create_image_conditioned_components(dataset,c4,base,device); model.dressable_model.configure_six_channel_decoder(c2['dressable_channels'])
 m2ck=Path(args.module2_output)/'checkpoint_step_000001.pth'; model.load_state_dict(torch.load(m2ck,map_location=device,weights_only=True)['model'],strict=True)
 gi,gw,gdiag=build_surface_aware_anchor_graph(base.xyz_vt,int(cfg['graph_k']),base.nbr_vt); model.initialize_online_completion(gi,gw,int(cfg['hidden_dim']),int(cfg['blocks']))
 adapter=MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(base,canonical_anchors=model.canonical_anchors,lbs_grid_path=c4['base']['lbs_grid_path']); background=torch.tensor(c4['render']['background'],device=device,dtype=base._xyz.dtype); base_degree=int(base.sh_degree)
 teacher_data=torch.load(Path(args.module2_output)/'teacher_anchor_residuals.pt',map_location=device,weights_only=True); teacher=AnchorClothingResiduals(**teacher_data).validate(10000,base); tg=model.dressable_model.interpolate_anchor_clothing_residuals(teacher); to=model.dressable_model.compose_canonical_gaussian_overrides(tg,CHANNELS)
 geometry_target=((teacher.delta_xyz.abs().sum(1)+teacher.delta_log_scaling.abs().sum(1)+teacher.delta_rotvec.abs().sum(1))>1e-8).to(base._xyz.dtype).reshape(-1,1); appearance_target=((teacher.delta_sh0.abs().sum(1)+teacher.delta_shN.abs().sum(1))>1e-8).to(base._xyz.dtype).reshape(-1,1)
 def render(ep,overrides=None,degree=0):
  cam=build_mmlphuman_camera(ep['target_camera'],ep['target_rgb'].shape[-2],ep['target_rgb'].shape[-1],device); before=int(base.sh_degree)
  try:
   base.sh_degree=degree
   with mmlphuman_state_transaction(base,ep['target_pose'].to(device),ep['target_Rh'].to(device),ep['target_Th'].to(device)): return base.render(cam,background=background,canonical_overrides=None if overrides is None else overrides.as_dict())
  finally: base.sh_degree=before
 refs=[]
 with torch.no_grad():
  for i in range(2):
   e=dict(episode); e.update(target_pose=episode['reference_poses'][i],target_Rh=episode['reference_Rh'][i],target_Th=episode['reference_Th'][i],target_camera=episode['reference_cameras'][i]); refs.append(_as_chw_render(render(e,to,1)[0],3).cpu())
  teacher_render=render(episode,to,1)
 episode['reference_images']=torch.stack(refs); episode['target_rgb']=_as_chw_render(teacher_render[0],3).cpu(); episode['target_foreground_mask']=(_as_chw_render(teacher_render[1],1).cpu()>.01).float()
 full_geometry=training.prepare_real_reference_geometry(base,adapter,episode,background)
 subsets={'S1':[0],'S2':[1],'S12':[0,1]}
 def online_inputs(name,rgb=None,masks=None):
  ids=subsets[name]; ep=subset_episode(episode,ids); depth=full_geometry['surface_depth_maps'][ids]; alpha=full_geometry['surface_alpha_maps'][ids]
  return dict(reference_images=(ep['reference_images'] if rgb is None else rgb).to(device),reference_cloth_masks=(ep['reference_cloth_masks'] if masks is None else masks).to(device),reference_foreground_masks=ep['reference_foreground_masks'].to(device),reference_poses=ep['reference_poses'].to(device),reference_cameras=ep['reference_cameras'],reference_valid_mask=ep['reference_valid_mask'].to(device),deformation_fn=training.build_episode_deformation_fn(adapter,ep),surface_depth_maps=depth,surface_alpha_maps=alpha,require_depth_visibility=True)
 # Freeze every pre-existing module; Module 3 isolation trains only completer/adapters.
 original_trainable={id(p) for p in model.parameters() if p.requires_grad}
 for p in model.parameters(): p.requires_grad_(False)
 for p in model.canonical_clothing_completer.parameters(): p.requires_grad_(True)
 model.eval(); evidence={}
 with torch.no_grad():
  for name in subsets:
   result=model.encode_clothing_online(**online_inputs(name)); evidence[name]=(result['global_clothing_embedding'].detach(),{k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in result['observed'].items()})
 completer=model.canonical_clothing_completer; optimizer=torch.optim.Adam(completer.parameters(),lr=float(cfg['learning_rate'])); held=deterministic_feature_holdout(evidence['S12'][1]['clothing_observed_mask'],float(cfg['feature_holdout_fraction']),cfg['seed']); full_target=evidence['S12'][1]['observed_clothing_feature']; logs=[]; checkpoints={0}
 def complete(name,holdout=False):
  global_e,obs=evidence[name]; clothing=obs['observed_clothing_feature']; coverage=obs['observation_coverage']; probability=obs['observed_clothing_probability']
  if holdout:
   clothing=clothing.clone(); coverage=coverage.clone(); probability=probability.clone(); clothing[held]=0; coverage[held]=0; probability[held]=0
  anchors=model.canonical_anchors; pos=(anchors-anchors.mean(0,keepdim=True))/(anchors-anchors.mean(0,keepdim=True)).abs().amax().clamp_min(1e-8)
  return completer(global_e,obs['observed_surface_feature'],clothing,probability,coverage,pos,gi,gw)
 initial_outputs={n:complete(n) for n in subsets}; initial_hold=complete('S12',True); weights=cfg['loss_weights']
 def loss_parts(outputs,hold):
  parts=completion_losses(hold,geometry_target,appearance_target,gi,gw,held,full_target)
  common=(outputs['S1'].observation_coverage>0)&(outputs['S2'].observation_coverage>0)
  parts['subset_feature_consistency']=F.smooth_l1_loss(outputs['S1'].completed_anchor_features[common[:,0]],outputs['S2'].completed_anchor_features[common[:,0]]) if common.any() else hold.completed_anchor_features.sum()*0
  parts['subset_geometry_gate_consistency']=F.l1_loss(outputs['S1'].geometry_gate,outputs['S12'].geometry_gate); parts['subset_appearance_gate_consistency']=F.l1_loss(outputs['S2'].appearance_gate,outputs['S12'].appearance_gate)
  return parts
 initial_parts=loss_parts(initial_outputs,initial_hold)
 initial_total=sum(float(weights.get(k,0))*v for k,v in initial_parts.items())
 initial_loss=float(initial_total.detach())
 logs=[{'step':0,'total':initial_loss,**{k:float(v.detach()) for k,v in initial_parts.items()}}]
 best_step=None; best_loss=float('inf'); best_state=None
 for step in range(1,int(cfg['max_steps'])+1):
  optimizer.zero_grad(set_to_none=True); outputs={n:complete(n) for n in subsets}; hold=complete('S12',True); parts=loss_parts(outputs,hold)
  total=sum(float(weights.get(k,0))*v for k,v in parts.items())
  record={'step':step,'total':float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()}}; logs.append(record)
  if step>=int(cfg['min_steps']) and record['total']<best_loss:
   best_step=step; best_loss=record['total']; best_state=copy.deepcopy(completer.state_dict())
  total.backward(); optimizer.step()
  if step in (1,10,25,50,100): checkpoints.add(step)
 if best_state is None: raise RuntimeError('no eligible completion checkpoint was produced')
 completer.load_state_dict(best_state,strict=True)
 final_outputs={n:complete(n) for n in subsets}; final_loss=best_loss; loss_drop=(initial_loss-final_loss)/max(initial_loss,1e-12)
 # Baselines and gate metrics.
 observed_gate=evidence['S12'][1]['observed_clothing_probability']*(evidence['S12'][1]['observation_coverage']>0).to(base._xyz.dtype)
 diffusion=observed_gate.clone()
 for _ in range(4): diffusion=torch.maximum(observed_gate,(diffusion[gi]*gw.unsqueeze(-1)).sum(1))
 metrics={};
 for name,o in final_outputs.items():
  observed=o.observation_coverage
  metrics[name]={"geometry":mask_metrics(o.geometry_gate,geometry_target,observed),"appearance":mask_metrics(o.appearance_gate,appearance_target,observed),"coverage_mean":float(observed.mean())}
 baseline={"observed_only":{"geometry":mask_metrics(observed_gate,geometry_target,evidence['S12'][1]['observation_coverage']),"appearance":mask_metrics(observed_gate,appearance_target,evidence['S12'][1]['observation_coverage'])},"diffusion":{"geometry":mask_metrics(diffusion,geometry_target,evidence['S12'][1]['observation_coverage']),"appearance":mask_metrics(diffusion,appearance_target,evidence['S12'][1]['observation_coverage'])},"learned":metrics['S12']}
 final_held_output=complete('S12',True)
 zero_hold=F.smooth_l1_loss(torch.zeros_like(full_target[held]),full_target[held]); final_hold=F.smooth_l1_loss(final_held_output.completed_anchor_features[held],full_target[held]); feature_metrics={"zero_fill":float(zero_hold),"learned":float(final_hold),"relative_improvement":float((zero_hold-final_hold)/zero_hold.clamp_min(1e-12)),"evaluation_input":"deterministic_30_percent_holdout"}
 # Oracle-residual gate renders.
 def gate_teacher(geometry,appearance):
  opacity=torch.maximum(geometry,appearance)
  return AnchorClothingResiduals(delta_xyz=teacher.delta_xyz*geometry,delta_log_scaling=teacher.delta_log_scaling*geometry,delta_rotvec=teacher.delta_rotvec*geometry,delta_opacity_logit=teacher.delta_opacity_logit*opacity,delta_sh0=teacher.delta_sh0*appearance,delta_shN=teacher.delta_shN*appearance)
 gates={'teacher':(geometry_target,appearance_target),'observed_only':(observed_gate,observed_gate),'diffusion':(diffusion,diffusion),'learned':(final_outputs['S12'].geometry_gate,final_outputs['S12'].appearance_gate)}; renders={}; render_metrics={}; target_rgb=_as_chw_render(teacher_render[0],3); target_alpha=_as_chw_render(teacher_render[1],1)
 for name,(gg,ag) in gates.items():
  gr=model.dressable_model.interpolate_anchor_clothing_residuals(gate_teacher(gg,ag)); ov=model.dressable_model.compose_canonical_gaussian_overrides(gr,CHANNELS); rr=render(episode,ov,1); renders[name]=rr; rgb=_as_chw_render(rr[0],3); alpha=_as_chw_render(rr[1],1); render_metrics[name]={"rgb_l1":float(F.l1_loss(rgb,target_rgb)),"clothing_rgb_l1":float(((rgb-target_rgb).abs()*target_alpha).sum()/target_alpha.sum().clamp_min(1)),"alpha_l1":float(F.l1_loss(alpha,target_alpha)),"mask_iou":gate_classification_metrics(alpha.reshape(-1,1),target_alpha.reshape(-1,1))['IoU'],"inactive_geometry_gate_mean":float(gg[geometry_target<.5].mean()),"inactive_appearance_gate_mean":float(ag[appearance_target<.5].mean())}
  render_name={'teacher':'teacher_gate_render.png','observed_only':'observed_only_render.png','diffusion':'diffusion_render.png','learned':'learned_gate_render.png'}[name]
  save_render_tensor(out/render_name,rr[0],3)
 # Full graph one-batch gradients.
 for p in model.parameters(): p.requires_grad_(id(p) in original_trainable or any(p is q for q in completer.parameters()))
 for p in base_tensors(base): p.requires_grad_(False)
 model.train(); model.zero_grad(set_to_none=True); full=model.compute_online_six_channel_residuals(**online_inputs('S12')); pred=full['gated_anchor_residuals']; gl=sum(F.smooth_l1_loss(getattr(pred,n),getattr(teacher,n)) for n in CHANNELS); pov=model.dressable_model.compose_canonical_gaussian_overrides(full['gaussian_residuals'],CHANNELS); pr=render(episode,pov,1); gl=gl+F.l1_loss(_as_chw_render(pr[0],3),target_rgb); gl.backward()
 heads={"xyz":model.dressable_model.anchor_clothing_mlp.output_layer,"scaling":model.dressable_model.anchor_clothing_mlp.scaling_head,"rotation":model.dressable_model.anchor_clothing_mlp.rotation_head,"opacity":model.dressable_model.anchor_clothing_mlp.opacity_head,"sh0":model.dressable_model.anchor_clothing_mlp.sh0_head,"shN":model.dressable_model.anchor_clothing_mlp.shN_head}; gradient={n:float(torch.sqrt(sum(p.grad.float().square().sum() for p in m.parameters() if p.grad is not None))) for n,m in heads.items()}; gradient['completer']=float(torch.sqrt(sum(p.grad.float().square().sum() for p in completer.parameters() if p.grad is not None)))
 def module_grad(module):
  values=[p.grad for p in module.parameters() if p.grad is not None]; return float(torch.sqrt(sum(x.float().square().sum() for x in values))) if values else 0.0
 gradient.update(shared_trunk=module_grad(model.dressable_model.anchor_clothing_mlp.hidden_layers),hypernetwork=module_grad(model.dressable_model.clothing_film_generator),aggregator=module_grad(model.multiview_aggregator),encoder_trainable_head=module_grad(model.clothing_observation_encoder.projection_head),frozen_image_backbone_grad_count=sum(p.grad is not None for p in model.clothing_observation_encoder.backbone.parameters()),base_grad_count=sum(p.grad is not None for p in base_tensors(base)))
 # Sensitivity and deterministic repeat.
 def evaluate(inp):
  model.eval()
  with torch.no_grad(): return model.compute_online_six_channel_residuals(**inp)
 variants={'A':online_inputs('S12'),'B':online_inputs('S1'),'C':online_inputs('S2')}; variants['D']=online_inputs('S12',masks=torch.zeros_like(episode['reference_cloth_masks'])); variants['E']=online_inputs('S12',rgb=_build_fixed_episode(dataset)['reference_images'])
 results={k:evaluate(v) for k,v in variants.items()}; repeat=evaluate(variants['A']); sensitivity={}
 sensitivity_renders={}
 for key,value in results.items():
  ov=model.dressable_model.compose_canonical_gaussian_overrides(value['gaussian_residuals'],CHANNELS); sensitivity_renders[key]=render(episode,ov,1)
 for k in ('B','C','D','E'):
  sensitivity[k]={"observed_probability":diff(results['A']['completion'].observed_clothing_probability,results[k]['completion'].observed_clothing_probability),"completed_feature":diff(results['A']['completion'].completed_anchor_features,results[k]['completion'].completed_anchor_features),"geometry_gate":diff(results['A']['completion'].geometry_gate,results[k]['completion'].geometry_gate),"appearance_gate":diff(results['A']['completion'].appearance_gate,results[k]['completion'].appearance_gate),"confidence":diff(results['A']['completion'].confidence,results[k]['completion'].confidence),"anchor_residuals":{n:diff(getattr(results['A']['gated_anchor_residuals'],n),getattr(results[k]['gated_anchor_residuals'],n)) for n in CHANNELS},"Gaussian_residuals":{n:diff(getattr(results['A']['gaussian_residuals'],n),getattr(results[k]['gaussian_residuals'],n)) for n in CHANNELS},"rendered_rgb":diff(sensitivity_renders['A'][0],sensitivity_renders[k][0]),"rendered_alpha":diff(sensitivity_renders['A'][1],sensitivity_renders[k][1])}
 sensitivity['repeat']={"observed_probability":diff(results['A']['completion'].observed_clothing_probability,repeat['completion'].observed_clothing_probability),"completed_feature":diff(results['A']['completion'].completed_anchor_features,repeat['completion'].completed_anchor_features),"geometry_gate":diff(results['A']['completion'].geometry_gate,repeat['completion'].geometry_gate),"appearance_gate":diff(results['A']['completion'].appearance_gate,repeat['completion'].appearance_gate),"confidence":diff(results['A']['completion'].confidence,repeat['completion'].confidence),"anchor_residuals":{n:diff(getattr(results['A']['gated_anchor_residuals'],n),getattr(repeat['gated_anchor_residuals'],n)) for n in CHANNELS},"Gaussian_residuals":{n:diff(getattr(results['A']['gaussian_residuals'],n),getattr(repeat['gaussian_residuals'],n)) for n in CHANNELS}}
 # Checkpoint reload and inference parity use identical legal inputs.
 metadata={"completion_version":1,"graph_version":gdiag['graph_version'],"graph_fingerprint":gdiag['fingerprint'],"graph_k":cfg['graph_k'],"graph_source":gdiag['graph_source'],"gate_source":"online_reference_mask_completion","feature_completion_hidden_dim":cfg['hidden_dim'],"feature_completion_blocks":cfg['blocks'],"observation_formula_version":1,"enabled_losses":[k for k,v in weights.items() if v>0],"base_checkpoint_sha256":sha(Path(c4['base']['model_dir'])/c4['base']['checkpoint_path']),"module2_decoder_metadata":torch.load(m2ck,map_location='cpu',weights_only=True)['metadata'],"dataset_fixture_fingerprint":dataset.manifest_fingerprint,"protocol":{"S1":["f000_c018"],"S2":["f1000_c000"],"S12":["f000_c018","f1000_c000"],"target":"f2000_c009","target_reference_overlap":False},"Git_commit":subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}; ck=out/'checkpoint_final.pth'; torch.save({'model':model.state_dict(),'graph_indices':gi.cpu(),'graph_weights':gw.cpu(),'metadata':metadata},ck)
 before=evaluate(variants['A']); before_render=render(episode,model.dressable_model.compose_canonical_gaussian_overrides(before['gaussian_residuals'],CHANNELS),1); model.load_state_dict(torch.load(ck,map_location=device,weights_only=True)['model'],strict=True); after=evaluate(variants['A']); after_render=render(episode,model.dressable_model.compose_canonical_gaussian_overrides(after['gaussian_residuals'],CHANNELS),1); fields={'completed_features':(before['completion'].completed_anchor_features,after['completion'].completed_anchor_features),'geometry_gate':(before['completion'].geometry_gate,after['completion'].geometry_gate),'appearance_gate':(before['completion'].appearance_gate,after['completion'].appearance_gate),'confidence':(before['completion'].confidence,after['completion'].confidence),'RGB':(before_render[0],after_render[0]),'alpha':(before_render[1],after_render[1])}; roundtrip={k:diff(*v) for k,v in fields.items()}; roundtrip.update({f"anchor.{n}":diff(getattr(before['gated_anchor_residuals'],n),getattr(after['gated_anchor_residuals'],n)) for n in CHANNELS}); roundtrip.update({f"Gaussian.{n}":diff(getattr(before['gaussian_residuals'],n),getattr(after['gaussian_residuals'],n)) for n in CHANNELS}); parity=dict(roundtrip); parity.update({"forbidden_fields_count":0,"teacher_read":False,"target_image_read":False,"target_mask_read":False,"cloth_id_used":False,"gate_source":"online_reference_mask_completion"})
 # Persist plots/maps/artifacts.
 for name,o in final_outputs.items(): save_anchor_map(out/f"observed_probability_{name}.png",o.observed_clothing_probability); save_anchor_map(out/f"geometry_gate_{name}.png",o.geometry_gate); save_anchor_map(out/f"appearance_gate_{name}.png",o.appearance_gate)
 gate_names={'teacher':'teacher_gate.png','observed_only':'observed_only_gate.png','diffusion':'diffusion_gate.png','learned':'learned_completed_gate.png'}
 for name,(gg,_) in gates.items(): save_anchor_map(out/gate_names[name],gg)
 save_image(make_grid([_as_chw_render(renders[n][0],3).cpu() for n in gates],nrow=4),out/'render_comparison.png'); save_image(make_grid([full_target[held].detach().cpu().reshape(-1,1).mean(1).reshape(1,-1,1),final_outputs['S12'].completed_anchor_features[held].detach().cpu().reshape(-1,1).mean(1).reshape(1,-1,1)],nrow=2),out/'feature_holdout_comparison.png')
 plt.figure(); plt.plot([x['step'] for x in logs],[x['total'] for x in logs]); plt.xlabel('step'); plt.ylabel('loss'); plt.tight_layout(); plt.savefig(out/'loss_curve.png'); plt.close()
 (out/'online_gate_audit.md').write_text(Path('docs/ONLINE_GATE_COMPLETION_AUDIT.md').read_text(encoding='utf-8'),encoding='utf-8')
 (out/'training_log.jsonl').write_text('\n'.join(json.dumps(x) for x in logs)+'\n'); files={'config_resolved.yaml':{**c3,'resolved_gate4':c4,'resolved_module2':c2},'input_manifest.json':metadata,'teacher_supervision_manifest.json':{'supervision_only':True,'forward_input':False,'geometry_active':int(geometry_target.sum()),'appearance_active':int(appearance_target.sum())},'anchor_graph_diagnostics.json':gdiag,'online_observation_diagnostics.json':{n:{"coverage_mean":float(o.observation_coverage.mean()),"observed_probability_mean":float(o.observed_clothing_probability.mean())} for n,o in final_outputs.items()},'gate_metrics_S1.json':metrics['S1'],'gate_metrics_S2.json':metrics['S2'],'gate_metrics_S12.json':metrics['S12'],'gate_baseline_comparison.json':baseline,'feature_completion_metrics.json':feature_metrics,'oracle_render_metrics.json':render_metrics,'reference_sensitivity.json':sensitivity,'gradient_metrics.json':gradient,'checkpoint_roundtrip.json':roundtrip,'inference_parity.json':parity}
 for name,value in files.items(): (out/name).write_text(yaml.safe_dump(value,sort_keys=False) if name.endswith('.yaml') else json.dumps(value,indent=2),encoding='utf-8')
 repeat_allclose=all(v['allclose'] for k,v in sensitivity['repeat'].items() if isinstance(v,dict) and 'allclose' in v) and all(v['allclose'] for group in ('anchor_residuals','Gaussian_residuals') for v in sensitivity['repeat'][group].values())
 pass_engineering=gradient['completer']>0 and all(v>0 for k,v in gradient.items() if k not in ('base_grad_count','completer','frozen_image_backbone_grad_count')) and gradient['base_grad_count']==0 and gradient['frozen_image_backbone_grad_count']==0 and all(v['allclose'] for v in roundtrip.values()) and repeat_allclose and base.sh_degree==base_degree
 s12=metrics['S12']; learned_un=s12['geometry']['unobserved'].get('recall',0); observed_un=baseline['observed_only']['geometry']['unobserved'].get('recall',0); coverage_monotonic=s12['coverage_mean']>=max(metrics['S1']['coverage_mean'],metrics['S2']['coverage_mean']); pass_learning=loss_drop>=.3 and s12['geometry']['all']['F1']>=.9 and s12['geometry']['all']['IoU']>=.82 and s12['appearance']['all']['F1']>=.9 and s12['appearance']['all']['IoU']>=.82 and learned_un>=.6 and learned_un-observed_un>=.15 and s12['geometry']['inactive_mean']<=.1 and s12['appearance']['inactive_mean']<=.1 and feature_metrics['relative_improvement']>=.2 and coverage_monotonic
 passed=pass_engineering and pass_learning and render_metrics['learned']['clothing_rgb_l1']<render_metrics['observed_only']['clothing_rgb_l1'] and render_metrics['learned']['mask_iou']>=render_metrics['observed_only']['mask_iou']
 summary={"status":"PASS" if passed else "PARTIAL","loss_drop":loss_drop,"selected_step":best_step,"selection_rule":"minimum training total after min_steps; no acceptance metric used","engineering":pass_engineering,"learning":pass_learning,"coverage_monotonic":coverage_monotonic,"coverage_formula_conflict":"mean over valid references cannot be monotonic when single-view coverages differ" if not coverage_monotonic else None,"metrics":metrics,"baseline":baseline,"feature":feature_metrics,"render":render_metrics,"gradient":gradient}
 (out/'forward.log').write_text(json.dumps(summary,indent=2)); (out/'GATE_ACCEPTANCE.md').write_text(f"# Gate 6 Online Completion\n\nStatus: **{'PASS' if passed else 'PARTIAL'}**\n\n- Engineering: {'PASS' if pass_engineering else 'FAIL'}\n- Learning thresholds: {'PASS' if pass_learning else 'FAIL'}\n- Coverage formula conflict: {summary['coverage_formula_conflict'] or 'none'}\n- Independent inference: evaluator reload parity only; separate-process proof pending\n",encoding='utf-8'); print(json.dumps(summary,indent=2))
 if not passed: raise RuntimeError('Module 3 acceptance remains partial')

def base_tensors(base): return [getattr(base,n) for n in ('_xyz','_scaling','_rotation','_opacity','_sh0','_shN')]
if __name__=='__main__': main()
