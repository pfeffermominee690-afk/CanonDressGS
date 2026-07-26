from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence


SCHEDULER_STATE_VERSION = 1


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OutfitBalancedEpisodeScheduler:
    """Deterministic condition-major, outfit-interleaved episode scheduler.

    The scheduler is deliberately independent of optimizer and dataset classes.  Its
    index points to the *next* episode, so restoring a checkpoint cannot replay the
    episode that produced the checkpoint.
    """

    def __init__(
        self,
        episodes: Sequence[Mapping[str, Any]],
        *,
        outfit_order: Sequence[str] | None = None,
        target_order: Sequence[str] | None = None,
        seed: int = 0,
    ) -> None:
        if not episodes:
            raise ValueError("episodes must not be empty")
        copied = [dict(item) for item in episodes]
        required = {"episode_id", "outfit_id", "target_condition_id"}
        for item in copied:
            missing = required.difference(item)
            if missing:
                raise ValueError(f"episode lacks scheduler fields: {sorted(missing)}")
        if len({item["episode_id"] for item in copied}) != len(copied):
            raise ValueError("episode_id values must be unique")

        discovered_outfits = list(dict.fromkeys(item["outfit_id"] for item in copied))
        discovered_targets = list(dict.fromkeys(item["target_condition_id"] for item in copied))
        self.outfit_order = list(discovered_outfits if outfit_order is None else outfit_order)
        self.target_order = list(discovered_targets if target_order is None else target_order)
        if set(self.outfit_order) != set(discovered_outfits) or len(self.outfit_order) != len(discovered_outfits):
            raise ValueError("outfit_order must contain every outfit exactly once")
        if set(self.target_order) != set(discovered_targets) or len(self.target_order) != len(discovered_targets):
            raise ValueError("target_order must contain every target condition exactly once")

        by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
        for item in copied:
            key = (item["outfit_id"], item["target_condition_id"])
            if key in by_key:
                raise ValueError(f"duplicate outfit/target episode: {key}")
            by_key[key] = item
        expected = {(outfit, target) for target in self.target_order for outfit in self.outfit_order}
        if set(by_key) != expected:
            missing = sorted(expected.difference(by_key))
            extra = sorted(set(by_key).difference(expected))
            raise ValueError(f"balanced scheduler requires a Cartesian episode grid; missing={missing}, extra={extra}")

        self._cycle = [deepcopy(by_key[(outfit, target)]) for target in self.target_order for outfit in self.outfit_order]
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.index = 0
        self.outfit_update_counts = Counter({value: 0 for value in self.outfit_order})
        self.view_update_counts = Counter({value: 0 for value in self.target_order})
        self.episode_update_counts = Counter({item["episode_id"]: 0 for item in self._cycle})
        self.schedule_fingerprint = _fingerprint({
            "episode_order": [item["episode_id"] for item in self._cycle],
            "outfit_order": self.outfit_order,
            "target_order": self.target_order,
            "seed": self.seed,
        })

    @property
    def cycle_length(self) -> int:
        return len(self._cycle)

    @property
    def cycle_count(self) -> int:
        return self.index // self.cycle_length

    @property
    def cycle_position(self) -> int:
        return self.index % self.cycle_length

    @property
    def episode_order(self) -> list[str]:
        return [item["episode_id"] for item in self._cycle]

    def peek(self) -> dict[str, Any]:
        return deepcopy(self._cycle[self.cycle_position])

    def next_episode(self) -> dict[str, Any]:
        episode = self.peek()
        self.index += 1
        self.outfit_update_counts[episode["outfit_id"]] += 1
        self.view_update_counts[episode["target_condition_id"]] += 1
        self.episode_update_counts[episode["episode_id"]] += 1
        return episode

    def take(self, count: int) -> list[dict[str, Any]]:
        if count < 0:
            raise ValueError("count must be non-negative")
        return [self.next_episode() for _ in range(count)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler_state_version": SCHEDULER_STATE_VERSION,
            "schedule_fingerprint": self.schedule_fingerprint,
            "scheduler_index": self.index,
            "outfit_update_counts": dict(self.outfit_update_counts),
            "per_view_update_counts": dict(self.view_update_counts),
            "episode_update_counts": dict(self.episode_update_counts),
            "epoch_or_cycle_count": self.cycle_count,
            "current_cycle_position": self.cycle_position,
            "rng_state": self._rng.getstate(),
            "seed": self.seed,
            "episode_order": self.episode_order,
            "outfit_order": list(self.outfit_order),
            "target_order": list(self.target_order),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("scheduler_state_version") != SCHEDULER_STATE_VERSION:
            raise ValueError("unsupported balanced scheduler state version")
        if state.get("schedule_fingerprint") != self.schedule_fingerprint:
            raise ValueError("balanced episode schedule fingerprint mismatch")
        if list(state.get("episode_order", ())) != self.episode_order:
            raise ValueError("balanced episode order mismatch")
        index = int(state["scheduler_index"])
        if index < 0:
            raise ValueError("scheduler_index must be non-negative")
        expected_cycle, expected_position = divmod(index, self.cycle_length)
        if int(state["epoch_or_cycle_count"]) != expected_cycle:
            raise ValueError("cycle count is inconsistent with scheduler index")
        if int(state["current_cycle_position"]) != expected_position:
            raise ValueError("cycle position is inconsistent with scheduler index")

        expected_outfit = Counter({value: 0 for value in self.outfit_order})
        expected_view = Counter({value: 0 for value in self.target_order})
        expected_episode = Counter({item["episode_id"]: 0 for item in self._cycle})
        for step in range(index):
            item = self._cycle[step % self.cycle_length]
            expected_outfit[item["outfit_id"]] += 1
            expected_view[item["target_condition_id"]] += 1
            expected_episode[item["episode_id"]] += 1
        if Counter(state["outfit_update_counts"]) != expected_outfit:
            raise ValueError("per-outfit update counts are inconsistent with scheduler index")
        if Counter(state["per_view_update_counts"]) != expected_view:
            raise ValueError("per-view update counts are inconsistent with scheduler index")
        if Counter(state["episode_update_counts"]) != expected_episode:
            raise ValueError("per-episode update counts are inconsistent with scheduler index")

        self.index = index
        self.outfit_update_counts = expected_outfit
        self.view_update_counts = expected_view
        self.episode_update_counts = expected_episode
        self._rng.setstate(state["rng_state"])

    def balance_report(self) -> dict[str, Any]:
        outfit_values = list(self.outfit_update_counts.values())
        view_values = list(self.view_update_counts.values())
        return {
            "scheduler_index": self.index,
            "cycle_count": self.cycle_count,
            "cycle_position": self.cycle_position,
            "outfit_update_counts": dict(self.outfit_update_counts),
            "per_view_update_counts": dict(self.view_update_counts),
            "outfit_count_spread": max(outfit_values) - min(outfit_values),
            "view_count_spread": max(view_values) - min(view_values),
            "episode_order": self.episode_order,
        }


def scheduler_from_manifest(
    manifest: Mapping[str, Any], *, split: str = "seen", seed: int = 0
) -> OutfitBalancedEpisodeScheduler:
    split_outfits = list(manifest["protocol"].get("splits", {}).get(split, ()))
    if not split_outfits:
        raise ValueError(f"manifest split {split!r} contains no outfits")
    selected = [item for item in manifest["episodes"] if item["outfit_id"] in split_outfits]
    return OutfitBalancedEpisodeScheduler(
        selected,
        outfit_order=split_outfits,
        target_order=manifest["protocol"]["conditions"],
        seed=seed,
    )
