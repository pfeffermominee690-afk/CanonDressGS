from __future__ import annotations

from typing import Any, Mapping

from .B0_base_avatar import B0BaseAvatarAdapter
from .B1_direct_teacher import B1DirectTeacherAdapter
from .B2_outfit_id_lookup import B2OutfitIdLookupAdapter
from .B3_global_reference import B3GlobalReferenceAdapter
from .B4_clothing_mean_only import B4ClothingMeanOnlyAdapter
from .B5_legacy_complex_fusion import B5LegacyComplexFusionAdapter
from .ablations import AblationAdapter
from .ours_explicit_basis_v1 import OursExplicitBasisV1Adapter


ADAPTER_TYPES = (
    B0BaseAvatarAdapter,
    B1DirectTeacherAdapter,
    B2OutfitIdLookupAdapter,
    B3GlobalReferenceAdapter,
    B4ClothingMeanOnlyAdapter,
    B5LegacyComplexFusionAdapter,
    OursExplicitBasisV1Adapter,
    AblationAdapter,
)


def adapter_for(experiment: Mapping[str, Any], config: Mapping[str, Any]):
    if not experiment.get("executable", False):
        raise ValueError("historical A8 evidence has no executable adapter")
    method = experiment["method"]
    for adapter_type in ADAPTER_TYPES:
        if adapter_type.supports(method):
            return adapter_type(experiment, config)
    raise KeyError(f"no paper adapter for method: {method}")


__all__ = ["ADAPTER_TYPES", "adapter_for"]
