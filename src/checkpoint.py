"""Disk checkpoints so paper runs can resume after API/network failures."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "results" / "checkpoints"


def config_fingerprint(config: dict[str, Any]) -> str:
    """Stable id for resume; ignore ephemeral flags."""
    keys = [
        "seed",
        "n_agents",
        "n_tasks",
        "skill_dim",
        "ks",
        "spam_frac",
        "api_tasks",
        "claude_spam_max",
        "paper",
        "with_anthropic",
        "claude_spam_rewrite",
    ]
    payload = {k: config.get(k) for k in keys if k in config}
    # paper mode uses api_tasks as n_tasks
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class CheckpointStore:
    def __init__(self, name: str = "paper", config: dict[str, Any] | None = None):
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.path = CKPT_DIR / f"{name}.json"
        self.config = config or {}
        self.fp = config_fingerprint(self.config)
        self.data = self._load_or_init()

    def _load_or_init(self) -> dict[str, Any]:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("fingerprint") == self.fp:
                print(f"[ckpt] Resuming from {self.path} (fp={self.fp})")
                return data
            bak = CKPT_DIR / f"{self.name}.stale_{int(time.time())}.json"
            self.path.rename(bak)
            print(f"[ckpt] Config changed; archived old checkpoint → {bak.name}")
        return {
            "fingerprint": self.fp,
            "config": self.config,
            "regimes": {},
            "updated_at": time.time(),
        }

    def save(self) -> None:
        self.data["updated_at"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    def get_method(self, regime: str, k: int | str, method: str) -> dict | None:
        return (
            self.data.get("regimes", {})
            .get(regime, {})
            .get("by_k", {})
            .get(str(k), {})
            .get(method)
        )

    def set_method(self, regime: str, k: int | str, method: str, stats: dict) -> None:
        regimes = self.data.setdefault("regimes", {})
        block = regimes.setdefault(regime, {"by_k": {}, "api": {}})
        by_k = block.setdefault("by_k", {})
        row = by_k.setdefault(str(k), {})
        row[method] = stats
        self.save()
        print(f"[ckpt] saved {regime} k={k} {method} success={stats.get('success_rate')}")

    def get_regime_api(self, regime: str) -> dict:
        return self.data.get("regimes", {}).get(regime, {}).get("api", {})

    def set_regime_api(self, regime: str, api_meta: dict) -> None:
        regimes = self.data.setdefault("regimes", {})
        block = regimes.setdefault(regime, {"by_k": {}, "api": {}})
        block["api"] = {**block.get("api", {}), **api_meta}
        self.save()

    def cards_path(self, regime: str) -> Path:
        safe = regime.replace(":", "").replace(" ", "_")
        return CKPT_DIR / f"cards_{self.name}_{safe}.json"

    def load_cards(self, regime: str) -> list[str] | None:
        path = self.cards_path(regime)
        if path.exists():
            return json.loads(path.read_text())
        return None

    def save_cards(self, regime: str, cards: list[str]) -> None:
        path = self.cards_path(regime)
        path.write_text(json.dumps(cards))
        print(f"[ckpt] saved {len(cards)} cards → {path.name}")

    def get_task_progress(self, regime: str, k: int | str, method: str) -> dict | None:
        return (
            self.data.get("regimes", {})
            .get(regime, {})
            .get("task_progress", {})
            .get(str(k), {})
            .get(method)
        )

    def set_task_progress(
        self,
        regime: str,
        k: int | str,
        method: str,
        done_indices: list[int],
        successes: list[float],
        sizes: list[int],
    ) -> None:
        regimes = self.data.setdefault("regimes", {})
        block = regimes.setdefault(regime, {"by_k": {}, "api": {}, "task_progress": {}})
        prog = block.setdefault("task_progress", {})
        by_k = prog.setdefault(str(k), {})
        by_k[method] = {
            "done_indices": done_indices,
            "successes": successes,
            "sizes": sizes,
        }
        self.save()

    def clear_task_progress(self, regime: str, k: int | str, method: str) -> None:
        try:
            del self.data["regimes"][regime]["task_progress"][str(k)][method]
            self.save()
        except KeyError:
            pass

    def regime_complete(self, regime: str, ks: list[int], methods: list[str]) -> bool:
        block = self.data.get("regimes", {}).get(regime, {})
        by_k = block.get("by_k", {})
        for k in ks:
            row = by_k.get(str(k), {})
            for m in methods:
                if m not in row:
                    return False
        return True

    def export_payload(self, regime_keys: dict[str, str]) -> dict[str, Any]:
        """regime_keys: logical_name -> checkpoint regime name"""
        out = {"config": self.config, "fingerprint": self.fp}
        for logical, regime in regime_keys.items():
            out[logical] = self.data["regimes"].get(regime, {"regime": regime, "by_k": {}})
            out[logical]["regime"] = regime
        return out
