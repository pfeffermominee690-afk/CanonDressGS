from __future__ import annotations

import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene.anchor_clothing_mlp import AnchorClothingMLP
from scene.clothing_gate_bundle import ClothingGateBundle, TemporaryReferenceGateAdapter
from scene.gaussian_clothing_residuals import interpolate_anchor_field


CONFIG = {
 "xyz":{"enabled":True,"max_abs":.05}, "log_scaling":{"enabled":True,"max_abs":.35},
 "rotation":{"enabled":True,"max_angle_rad":.2617993878}, "opacity_logit":{"enabled":True,"max_abs":2},
 "sh0":{"enabled":True,"max_abs":.25}, "shN":{"enabled":True,"max_abs":.1,"sh_degree":1},
}

def main():
 torch.manual_seed(1); A,N,K=11,17,3
 mlp=AnchorClothingMLP(5,4,8,3); mlp.initialize_local_feature_adapter(6); mlp.configure_six_channel_decoder(CONFIG,9)
 with torch.no_grad():
  for head in (mlp.output_layer,mlp.scaling_head,mlp.rotation_head,mlp.opacity_head,mlp.sh0_head,mlp.shN_head): head.weight.fill_(.01)
 anchor=torch.randn(A,5); local=torch.randn(A,6); gamma=[torch.zeros(1,8),torch.zeros(1,8)]; beta=[x.clone() for x in gamma]
 raw_residual,residual=mlp.forward_film_six_channel_outputs(anchor,gamma,beta,local)
 assert all(x.shape==y.shape for x,y in zip(raw_residual.as_dict().values(),residual.as_dict().values()))
 assert [tuple(x.shape) for x in residual.as_dict().values()]==[(A,3),(A,3),(A,3),(A,1),(A,3),(A,9)]
 zero=torch.zeros(A,1); one=torch.ones(A,1)
 g=ClothingGateBundle(zero,one).apply(residual); assert not g.delta_sh0.count_nonzero() == 0 and g.delta_xyz.count_nonzero()==0 and g.delta_opacity_logit.count_nonzero()>0
 g=ClothingGateBundle(one,zero).apply(residual); assert g.delta_sh0.count_nonzero()==0 and g.delta_xyz.count_nonzero()>0 and g.delta_opacity_logit.count_nonzero()>0
 g=ClothingGateBundle(zero,zero).apply(residual); assert g.delta_opacity_logit.count_nonzero()==0
 g=TemporaryReferenceGateAdapter.from_reference_gate(one).apply(residual)
 for a,b in zip(residual.as_dict().values(),g.as_dict().values()): assert torch.equal(a,b)
 for bad in (torch.ones(A),torch.full((A,1),1.1)):
  try: ClothingGateBundle(bad,one).apply(residual); raise AssertionError('invalid gate accepted')
  except ValueError: pass
 idx=torch.randint(0,A,(N,K)); weights=torch.rand(N,K); weights/=weights.sum(1,keepdim=True)
 field=torch.randn(A,7,requires_grad=True); out=interpolate_anchor_field(field,idx,weights); out.sum().backward()
 assert out.shape==(N,7) and field.grad is not None and torch.isfinite(field.grad).all()
 raw=torch.zeros(A,3,requires_grad=True); mlp.rotation_head.weight.data.zero_(); mlp.rotation_head.bias.data.zero_()
 rot=mlp.forward_film_six_channel(anchor,gamma,beta,local).delta_rotvec; assert torch.equal(rot,torch.zeros_like(rot)); rot.sum().backward(); assert mlp.rotation_head.bias.grad is not None and torch.isfinite(mlp.rotation_head.bias.grad).all()
 legacy=AnchorClothingMLP(5,4,8,3); state=legacy.state_dict(); migrated=AnchorClothingMLP(5,4,8,3); migrated.initialize_six_channel_heads(9); migrated.load_state_dict(state,strict=True)
 assert all(torch.count_nonzero(getattr(migrated,n).weight)==0 for n in ('scaling_head','rotation_head','opacity_head','sh0_head','shN_head'))
 print('six-channel decoder unit checks: PASS')

if __name__=='__main__': main()
