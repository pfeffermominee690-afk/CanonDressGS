from __future__ import annotations

import re

from .base import PaperMethodAdapter


class AblationAdapter(PaperMethodAdapter):
    prefixes = ("A1_", "A2_", "A3_", "A4_", "A5_", "A6_", "A7_")

    def validate_contract(self):
        report = super().validate_contract()
        method = self.experiment["method"]
        report["ablation"] = method.split("_", 1)[0]
        if method.startswith("A1_"):
            rank = int(re.search(r"Rank_(\d+)", method).group(1))
            if rank not in self.config["ablations"]["A1"]["values"]:
                raise ValueError("invalid A1 rank")
            report["basis_rank"] = rank
        elif method.startswith("A7_"):
            count = int(re.search(r"Count_(\d+)", method).group(1))
            if count not in self.config["ablations"]["A7"]["Kref"]:
                raise ValueError("invalid A7 reference count")
            report["reference_count"] = count
        return report
