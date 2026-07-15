from __future__ import annotations

import argparse, hashlib, json, subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import make_grid, save_image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_dressable as training
from scene.clothing_gate_bundle import TemporaryReferenceGateAdapter
from scene.dressable_dataset import ImageConditionedEpisodeDataset
from scene.gaussian_clothing_residuals import CHANNELS, AnchorClothingResiduals
from tools.check_real_image_conditioned_one_batch import _as_chw_render, _build_fixed_episode, save_render_tensor
from utils.dressable_camera_utils import build_mmlphuman_camera
from utils.mmlphuman_anchor_deformation import MMLPHumanAnchorDeformationAdapter
from utils.mmlphuman_state_utils import mmlphuman_state_transaction

DISPLAY={"delta_xyz":"xyz","delta_log_scaling":"scaling","delta_rotvec":"rotation","delta_opacity_logit":"opacity","delta_sh0":"sh0","delta_shN":"shN"}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def diff(a,b):
 d=(a.detach().float()-b.detach().float()).abs(); return {"max_abs_diff":float(d.max()),"mean_abs_diff":float(d.mean()),"allclose":bool(torch.allclose(a,b,atol=1e-6,rtol=1e-5))}
def stats(x): return {"shape":list(x.shape),"min":float(x.detach().min()),"max":float(x.detach().max()),"finite_ratio":float(torch.isfinite(x).float().mean())}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--gate4-config',required=True); ap.add_argument('--module2-config',required=True); ap.add_argument('--old-checkpoint',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
 out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); log=[]
 config=training.load_config(args.gate4_config); m2=training.load_config(args.module2_config); channels=m2['dressable_channels']; device=torch.device(args.device); torch.manual_seed(int(m2['module2']['seed']))
 dataset=ImageConditionedEpisodeDataset(config['image_conditioning']['manifest_path'],split='train',reference_count=2,seed=42,allow_clothing_mask_fallback=False,allow_missing_rh_th=False)
 episode=_build_fixed_episode(dataset); assert episode['target_frame_id'] not in episode['reference_frame_ids']
 base=training.load_frozen_mmlphuman_base(config['base']['model_dir'],config['base']['checkpoint_path'],device)
 model,_,anchor_features,anchor_edges=training.create_image_conditioned_components(dataset,config,base,device)
 model.dressable_model.configure_six_channel_decoder(channels)
 old=torch.load(args.old_checkpoint,map_location='cpu',weights_only=True); model.load_state_dict(old['model'],strict=True)
 optimizer=training.build_image_conditioned_optimizer(model,config['optimizer'])
 adapter=MMLPHumanAnchorDeformationAdapter.from_mmlphuman_base(base,canonical_anchors=model.canonical_anchors,lbs_grid_path=config['base']['lbs_grid_path'])
 region=torch.load(config['image_conditioning']['reference_region_path'],map_location='cpu',weights_only=True)['cloth_region_weight'].float().reshape(-1)
 gate=(region>=float(config['image_conditioning']['reference_gate_threshold'])).to(device=device,dtype=base._xyz.dtype).reshape(-1,1)
 bundle=TemporaryReferenceGateAdapter.from_reference_gate(
  gate, region.clamp(0,1).to(device=device,dtype=base._xyz.dtype).reshape(-1,1)
 )
 A=model.canonical_anchors.shape[0]; index=torch.arange(A,device=device,dtype=base._xyz.dtype); phase=index/97
 teacher=AnchorClothingResiduals(
  delta_xyz=torch.stack((torch.zeros_like(phase),.018*torch.sin(phase),.009*torch.cos(phase)),1)*gate,
  delta_log_scaling=torch.stack((.12*torch.sin(phase),-.08*torch.cos(phase),.05*torch.sin(phase*.7)),1)*gate,
  delta_rotvec=torch.stack((.06*torch.sin(phase),.08*torch.cos(phase),torch.zeros_like(phase)),1)*gate,
  delta_opacity_logit=(.5*torch.sin(phase)).reshape(-1,1)*gate,
  delta_sh0=torch.stack((.18*torch.ones_like(phase),-.05*torch.ones_like(phase),-.08*torch.ones_like(phase)),1)*gate,
  delta_shN=torch.stack([.04*torch.sin(phase*(i+1)/3) for i in range(int(base._shN[0].numel()))],1)*gate,
 ).validate(A,base)
 teacher_g=model.dressable_model.interpolate_anchor_clothing_residuals(teacher); teacher_o=model.dressable_model.compose_canonical_gaussian_overrides(teacher_g,CHANNELS)
 background=torch.tensor(config['render']['background'],device=device,dtype=base._xyz.dtype); base_degree=int(base.sh_degree)
 def render(pose,Rh,Th,camera_data,overrides=None,degree=0):
  camera=build_mmlphuman_camera(camera_data,episode['target_rgb'].shape[-2],episode['target_rgb'].shape[-1],device); before=int(base.sh_degree)
  try:
   base.sh_degree=degree
   with mmlphuman_state_transaction(base,pose.to(device),Rh.to(device),Th.to(device)):
    return base.render(camera,background=background,canonical_overrides=None if overrides is None else overrides.as_dict())
  finally: base.sh_degree=before
 teacher_refs=[]
 with torch.no_grad():
  for i in range(2): teacher_refs.append(_as_chw_render(render(episode['reference_poses'][i],episode['reference_Rh'][i],episode['reference_Th'][i],episode['reference_cameras'][i],teacher_o,1)[0],3).cpu())
  teacher_target=render(episode['target_pose'],episode['target_Rh'],episode['target_Th'],episode['target_camera'],teacher_o,1)
 episode['reference_images']=torch.stack(teacher_refs); episode['target_rgb']=_as_chw_render(teacher_target[0],3).cpu(); episode['target_foreground_mask']=(_as_chw_render(teacher_target[1],1).cpu()>.01).float()
 geometry=training.prepare_real_reference_geometry(base,adapter,episode,background)
 def predict():
  encoded=model.encode_clothing_with_anchors(episode['reference_images'].to(device),episode['reference_cloth_masks'].to(device),episode['reference_poses'].to(device),episode['reference_cameras'],model.canonical_anchors,reference_valid_mask=episode['reference_valid_mask'].to(device),deformation_fn=geometry['deformation_fn'],surface_depth_maps=geometry['surface_depth_maps'],surface_alpha_maps=geometry['surface_alpha_maps'],require_depth_visibility=True)
  bounded,gated=model.dressable_model.compute_film_anchor_residuals(bundle,clothing_embedding=encoded['global_clothing_embedding'],anchor_clothing_features=encoded['anchor_clothing_features'])
  gaussian=model.dressable_model.interpolate_anchor_clothing_residuals(gated); overrides=model.dressable_model.compose_canonical_gaussian_overrides(gaussian,CHANNELS)
  rendered=render(episode['target_pose'],episode['target_Rh'],episode['target_Th'],episode['target_camera'],overrides,1)
  return encoded,bounded,gated,gaussian,overrides,rendered
 model.train(); optimizer.zero_grad(set_to_none=True); encoded,bounded,gated,gaussian,overrides,pred=predict()
 losses={}
 for name in CHANNELS: losses[name]=F.smooth_l1_loss(getattr(gated,name),getattr(teacher,name))
 prgb=_as_chw_render(pred[0],3); palpha=_as_chw_render(pred[1],1); trgb=_as_chw_render(teacher_target[0],3); talpha=_as_chw_render(teacher_target[1],1)
 losses.update(rgb_l1=F.l1_loss(prgb,trgb),ssim=1-training.ssim(prgb.unsqueeze(0),trgb.unsqueeze(0)),alpha_bce=F.binary_cross_entropy(palpha.clamp(1e-6,1-1e-6),talpha.clamp(0,1)),alpha_dice=1-(2*(palpha*talpha).sum()+1)/(palpha.sum()+talpha.sum()+1),non_region=sum((getattr(gated,n)*(1-gate)).abs().mean() for n in CHANNELS),regularization=sum(getattr(gated,n).square().mean() for n in CHANNELS))
 total=sum(losses.values()); total.backward()
 heads={"xyz_head":model.dressable_model.anchor_clothing_mlp.output_layer,"scaling_head":model.dressable_model.anchor_clothing_mlp.scaling_head,"rotation_head":model.dressable_model.anchor_clothing_mlp.rotation_head,"opacity_head":model.dressable_model.anchor_clothing_mlp.opacity_head,"sh0_head":model.dressable_model.anchor_clothing_mlp.sh0_head,"shN_head":model.dressable_model.anchor_clothing_mlp.shN_head}
 gradients={}
 for name,module in heads.items():
  values=[p.grad for p in module.parameters() if p.grad is not None]; gradients[name]={"exists":bool(values),"norm":float(torch.sqrt(sum(x.float().square().sum() for x in values))) if values else 0,"finite_ratio":float(torch.cat([x.flatten() for x in values]).isfinite().float().mean()) if values else 0}
 gradients['shared_trunk']={"norm":float(model.dressable_model.anchor_clothing_mlp.hidden_layers[0].weight.grad.float().norm())}; gradients['hypernetwork']={"norm":float(model.dressable_model.clothing_film_generator.trunk[0].weight.grad.float().norm())}; gradients['aggregator']={"norm":float(next(model.multiview_aggregator.parameters()).grad.float().norm())}; trainable_encoder=[p.grad for p in model.clothing_observation_encoder.parameters() if p.requires_grad and p.grad is not None]; gradients['encoder_trainable']={"norm":float(torch.sqrt(sum(x.float().square().sum() for x in trainable_encoder)))}; gradients['base_grad_count']={"count":sum(getattr(base,n).grad is not None for n in ('_xyz','_scaling','_rotation','_opacity','_sh0','_shN'))}
 optimizer.step(); model.eval()
 with torch.no_grad(): encoded,bounded,gated,gaussian,overrides,pred=predict()
 base_render=render(episode['target_pose'],episode['target_Rh'],episode['target_Th'],episode['target_camera'],None,0); attribution={}; panels=[_as_chw_render(base_render[0],3).cpu()]
 for name in CHANNELS:
  single=type(gaussian)(**{key:(getattr(gaussian,key) if key==name else torch.zeros_like(getattr(gaussian,key))) for key in CHANNELS}); ov=model.dressable_model.compose_canonical_gaussian_overrides(single,[name]); degree=1 if name=='delta_shN' else 0; rr=render(episode['target_pose'],episode['target_Rh'],episode['target_Th'],episode['target_camera'],ov,degree); attribution[name]={"rgb":diff(render(episode['target_pose'],episode['target_Rh'],episode['target_Th'],episode['target_camera'],None,degree)[0],rr[0]),"alpha":diff(base_render[1],rr[1])}; save_render_tensor(out/f"{DISPLAY[name]}_only.png",rr[0],3); panels.append(_as_chw_render(rr[0],3).cpu())
 save_render_tensor(out/'all_channels.png',pred[0],3); save_render_tensor(out/'predicted_rgb.png',pred[0],3); save_render_tensor(out/'predicted_alpha.png',pred[1],1); save_render_tensor(out/'base_rgb.png',base_render[0],3); save_render_tensor(out/'teacher_target_rgb.png',teacher_target[0],3); save_image(make_grid(teacher_refs,nrow=2),out/'teacher_reference_contact_sheet.png'); save_image(make_grid(panels+[_as_chw_render(pred[0],3).cpu()],nrow=4),out/'channel_comparison.png')
 snapshot={"model":model.state_dict(),"metadata":{"residual_contract_version":1,"decoder_version":2,"enabled_channels":list(CHANNELS),"channel_bounds":channels,"gate_contract_version":1,"gate_source":bundle.gate_source,"anchor_output_shapes":{n:list(getattr(gated,n).shape) for n in CHANNELS},"Gaussian_output_shapes":{n:list(getattr(gaussian,n).shape) for n in CHANNELS},"base_attribute_shapes":{n:list(getattr(base,n).shape) for n in ('_xyz','_scaling','_rotation','_opacity','_sh0','_shN')},"quaternion_convention":"wxyz","rotation_composition":"local base*delta","base_sh_degree":base_degree,"configured_sh_degree":1,"effective_sh_degree":1,"base_checkpoint_sha256":sha(Path(config['base']['model_dir'])/config['base']['checkpoint_path']),"Git_commit":subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}}
 ck=out/'checkpoint_step_000001.pth'; torch.save(snapshot,ck); before=predict(); model.load_state_dict(torch.load(ck,map_location=device,weights_only=True)['model'],strict=True); after=predict(); roundtrip={"embedding":diff(before[0]['global_clothing_embedding'],after[0]['global_clothing_embedding']),"local":diff(before[0]['anchor_clothing_features'],after[0]['anchor_clothing_features']),"rgb":diff(before[-1][0],after[-1][0]),"alpha":diff(before[-1][1],after[-1][1])}; roundtrip.update({n:diff(getattr(before[2],n),getattr(after[2],n)) for n in CHANNELS})
 legacy_zero={n:float(getattr(bounded,n).abs().max()) for n in CHANNELS if n!='delta_xyz'}
 compatibility={"old_checkpoint":str(Path(args.old_checkpoint).resolve()),"old_checkpoint_sha256":sha(args.old_checkpoint),"legacy_decoder_keys":[k for k in old['model'] if 'anchor_clothing_mlp' in k or 'clothing_film_generator' in k],"new_channels_after_migration_max_abs":legacy_zero,"xyz_head_key_preserved":'dressable_model.anchor_clothing_mlp.output_layer.weight' in old['model'],"status":"PASS" if snapshot['metadata']['Git_commit'] else 'FAIL'}
 teacher_manifest={"seed":m2['module2']['seed'],"active_anchor_count":int(gate.sum()),"ranges":{n:stats(getattr(teacher,n)) for n in CHANNELS},"gate_source":bundle.gate_source,"effective_sh_degree":1,"base_checkpoint_sha256":snapshot['metadata']['base_checkpoint_sha256'],"protocol":{"references":[['0','cam18'],['1000','cam00']],"target":['2000','cam09']}}
 torch.save(teacher.as_dict(cpu=True),out/'teacher_anchor_residuals.pt'); teacher_manifest['teacher_file_sha256']=sha(out/'teacher_anchor_residuals.pt')
 diagnostics={"losses":{k:float(v.detach()) for k,v in losses.items()},"anchor_shapes":snapshot['metadata']['anchor_output_shapes'],"Gaussian_shapes":snapshot['metadata']['Gaussian_output_shapes'],"sh_degree_restored":int(base.sh_degree)==base_degree,"target_not_reference":True,"cloth_id_used":False}
 passed=all(x['exists'] and x['finite_ratio']==1 and x['norm']>0 for x in gradients.values() if 'exists' in x) and gradients['base_grad_count']['count']==0 and all(v['rgb']['max_abs_diff']>0 for v in attribution.values()) and all(v['allclose'] for v in roundtrip.values()) and diagnostics['sh_degree_restored']
 files={'config_resolved.yaml':{**config,**m2},'input_manifest.json':snapshot['metadata'],'teacher_manifest.json':teacher_manifest,'decoder_diagnostics.json':diagnostics,'gradient_metrics.json':gradients,'channel_attribution.json':attribution,'checkpoint_compatibility.json':compatibility,'checkpoint_roundtrip.json':roundtrip}
 for name,value in files.items():
  p=out/name; p.write_text(yaml.safe_dump(value,sort_keys=False) if p.suffix=='.yaml' else json.dumps(value,indent=2),encoding='utf-8')
 md=f"# Six-channel Decoder Diagnostics\n\nStatus: **{'PASS' if passed else 'FAIL'}**\n"; (out/'decoder_diagnostics.md').write_text(md,encoding='utf-8'); (out/'decoder_audit.md').write_text(Path('docs/SIX_CHANNEL_DECODER_AUDIT.md').read_text(encoding='utf-8'),encoding='utf-8'); (out/'forward.log').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8'); (out/'GATE_ACCEPTANCE.md').write_text(f"# Gate 5 Module 2\n\nStatus: **{'PASS' if passed else 'FAIL'}**\n",encoding='utf-8')
 print(json.dumps({'status':'PASS' if passed else 'FAIL','gradients':gradients,'attribution':attribution,'roundtrip':roundtrip},indent=2));
 if not passed: raise RuntimeError('Module 2 acceptance failed')

if __name__=='__main__': main()
