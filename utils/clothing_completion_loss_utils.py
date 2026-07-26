from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_dice_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prediction,target=_pair(prediction,target)
    return 1-(2*(prediction*target).sum()+eps)/(prediction.sum()+target.sum()+eps)


def graph_smoothness(value: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if value.ndim!=2 or indices.shape!=weights.shape or indices.shape[0]!=value.shape[0]:
        raise ValueError("graph smoothness shapes mismatch")
    return ((value[:,None]-value[indices]).abs()*weights.unsqueeze(-1)).sum(1).mean()


def deterministic_feature_holdout(observed_mask: torch.Tensor, fraction: float, seed: int) -> torch.Tensor:
    mask=observed_mask.reshape(-1).bool(); candidates=torch.nonzero(mask).reshape(-1)
    if candidates.numel()==0: return mask.new_zeros(mask.shape)
    generator=torch.Generator(device='cpu').manual_seed(int(seed))
    count=max(1,int(candidates.numel()*fraction)); chosen=candidates.cpu()[torch.randperm(candidates.numel(),generator=generator)[:count]]
    result=torch.zeros_like(mask); result[chosen.to(mask.device)]=True; return result


def gate_classification_metrics(prediction: torch.Tensor, target: torch.Tensor, threshold: float = .5) -> dict[str,float]:
    prediction,target=_pair(prediction,target); pred=prediction>=threshold; truth=target>=threshold
    tp=(pred&truth).sum().item(); fp=(pred&~truth).sum().item(); fn=(~pred&truth).sum().item(); tn=(~pred&~truth).sum().item()
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); f1=2*precision*recall/max(precision+recall,1e-12); iou=tp/max(tp+fp+fn,1)
    return {"TP":tp,"FP":fp,"FN":fn,"TN":tn,"precision":precision,"recall":recall,"F1":f1,"IoU":iou}


def completion_losses(output, geometry_target, appearance_target, graph_indices, graph_weights,
                      held_out_mask=None, held_out_target=None) -> dict[str,torch.Tensor]:
    geometry_inactive=geometry_target < 0.5
    appearance_inactive=appearance_target < 0.5
    losses={
        "geometry_gate_bce":F.binary_cross_entropy(output.geometry_gate.clamp(1e-6,1-1e-6),geometry_target),
        "geometry_gate_dice":binary_dice_loss(output.geometry_gate,geometry_target),
        "appearance_gate_bce":F.binary_cross_entropy(output.appearance_gate.clamp(1e-6,1-1e-6),appearance_target),
        "appearance_gate_dice":binary_dice_loss(output.appearance_gate,appearance_target),
        "observed_reprojection":F.smooth_l1_loss(output.geometry_gate*output.observation_coverage,output.observed_clothing_probability*output.observation_coverage),
        "gate_graph_smoothness":graph_smoothness(output.geometry_gate,graph_indices,graph_weights)+graph_smoothness(output.appearance_gate,graph_indices,graph_weights),
        "feature_graph_smoothness":graph_smoothness(output.completed_anchor_features,graph_indices,graph_weights),
        "gate_sparsity":(
            _safe_masked_mean(output.geometry_gate,geometry_inactive)
            + _safe_masked_mean(output.appearance_gate,appearance_inactive)
        ),
    }
    if held_out_mask is not None and held_out_mask.any():
        losses["feature_reconstruction"]=F.smooth_l1_loss(output.completed_anchor_features[held_out_mask],held_out_target[held_out_mask])
    else: losses["feature_reconstruction"]=output.completed_anchor_features.sum()*0
    return losses


def _pair(prediction,target):
    if prediction.shape!=target.shape or prediction.ndim!=2 or prediction.shape[1]!=1:
        raise ValueError("gate tensors must have matching [A,1] shapes")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all(): raise ValueError("gate contains NaN or Inf")
    return prediction,target


def _safe_masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return value[mask].mean() if mask.any() else value.sum()*0
