from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scene.canonical_clothing_completion import CanonicalClothingCompleter
from scene.multiview_clothing_aggregator import MultiViewClothingAggregator
from utils.anchor_graph_utils import build_surface_aware_anchor_graph
from utils.clothing_completion_loss_utils import gate_classification_metrics,deterministic_feature_holdout

def main():
 torch.manual_seed(3); A,K,C=32,3,6; xyz=torch.randn(A,3); topology=torch.stack([torch.tensor([i,(i-1)%A,(i+1)%A]) for i in range(A)])
 indices,weights,d1=build_surface_aware_anchor_graph(xyz,8,topology); i2,w2,d2=build_surface_aware_anchor_graph(xyz,8,topology)
 assert torch.equal(indices,i2) and torch.equal(weights,w2) and d1['fingerprint']==d2['fingerprint'] and torch.allclose(weights.sum(1),torch.ones(A))
 agg=MultiViewClothingAggregator(C,8); features=torch.randn(K,A,C); vis=torch.rand(K,A,1); cloth=torch.rand(K,A,1); valid=torch.tensor([1.,1.,0.])
 obs=agg.aggregate_online_observations(features,vis,cloth,valid); assert obs['observed_surface_feature'].shape==(A,8)
 try: agg.aggregate_online_observations(features,vis,cloth,torch.zeros(K)); raise AssertionError('all invalid accepted')
 except ValueError: pass
 comp=CanonicalClothingCompleter(8,5,3,16,4); output=comp(torch.randn(1,5),obs['observed_surface_feature'],obs['observed_clothing_feature'],obs['observed_clothing_probability'],obs['observation_coverage'],xyz,indices,weights)
 assert output.completed_anchor_features.shape==(A,8)
 for value in (output.geometry_gate,output.appearance_gate,output.confidence): assert value.shape==(A,1) and torch.isfinite(value).all() and ((value>=0)&(value<=1)).all()
 mask=deterministic_feature_holdout(torch.ones(A,1,dtype=torch.bool),.3,9); assert torch.equal(mask,deterministic_feature_holdout(torch.ones(A,1,dtype=torch.bool),.3,9))
 assert gate_classification_metrics(torch.ones(A,1),torch.ones(A,1))['F1']==1
 output.geometry_gate.mean().backward(); assert comp.geometry_gate_head.weight.grad is not None and torch.isfinite(comp.geometry_gate_head.weight.grad).all()
 print('online completion unit checks: PASS')
if __name__=='__main__': main()
