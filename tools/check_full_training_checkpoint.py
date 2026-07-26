from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.full_training_checkpoint_utils import (
    load_full_training_checkpoint, save_full_training_checkpoint, validate_full_training_checkpoint,
)


def main() -> None:
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 3)).square().mean(); loss.backward(); optimizer.step()
    method = {"graph_fingerprint": "graph", "base_checkpoint_sha256": "base", "effective_sh_degree": 1}
    data = {"fixture_fingerprint": "fixture", "holdout_mask_fingerprint": "holdout"}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.pth"
        saved = save_full_training_checkpoint(
            path, model=model, optimizer=optimizer, optimizer_group_names=["trainable"],
            scheduler=None, scaler=None, training_state={"global_step": 1}, data_state=data, method_state=method,
        )
        clone = copy.deepcopy(model); clone_optimizer = torch.optim.Adam(clone.parameters(), lr=1e-3)
        loaded = load_full_training_checkpoint(
            path, model=clone, optimizer=clone_optimizer, optimizer_group_names=["trainable"],
            scheduler=None, scaler=None, expected_method_state=method, expected_data_state=data,
        )
        assert loaded["scheduler_state"] == {"scheduler_enabled": False, "class": None, "state_dict": None}
        assert all(torch.equal(a, b) for a, b in zip(model.parameters(), clone.parameters()))
        for mutation, message in [
            (("training_checkpoint_version", 99), "version"),
            (("method_state", {**method, "graph_fingerprint": "bad"}), "method"),
            (("data_state", {**data, "fixture_fingerprint": "bad"}), "data"),
        ]:
            bad = copy.deepcopy(saved); bad[mutation[0]] = mutation[1]
            try:
                validate_full_training_checkpoint(
                    bad, model=clone, optimizer=clone_optimizer, optimizer_group_names=["trainable"],
                    expected_method_state=method, expected_data_state=data,
                )
            except ValueError: pass
            else: raise AssertionError(f"{message} mismatch was not rejected")
        wrong = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Linear(5, 1))
        wrong_optimizer = torch.optim.Adam(wrong.parameters(), lr=1e-3)
        try:
            validate_full_training_checkpoint(
                saved, model=wrong, optimizer=wrong_optimizer, optimizer_group_names=["trainable"],
                expected_method_state=method, expected_data_state=data,
            )
        except ValueError: pass
        else: raise AssertionError("parameter shape mismatch was not rejected")
    print("full training checkpoint checks: PASS")


if __name__ == "__main__": main()
