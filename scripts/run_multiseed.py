"""Multi-seed stats matching the paper protocol (offline methods, T=24)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.discover import (
    select_complementary_greedy,
    select_complementary_reputation,
    select_oracle_true,
    select_random,
    select_similarity,
)
from src.env import apply_churn, apply_spam, make_registry, reset_honest_ads, sample_tasks
from src.metrics import evaluate_method


def run_one(seed: int, regime: str, k: int, n_agents: int = 200, n_tasks: int = 24) -> dict[str, float]:
    spam = regime != "honest"
    churn = regime == "churn"
    reg = make_registry(n_agents, 12, seed=seed)
    reset_honest_ads(reg)
    if spam:
        apply_spam(reg, spam_frac=0.4, inflate=4)
    if churn:
        apply_churn(reg, leave_frac=0.15, join_frac=0.05)
    tasks = sample_tasks(n_tasks, 12, seed=seed + 7)
    out = {}
    for name, fn, upd, ep, pr in [
        ("random", select_random, False, 1, 0),
        ("similarity", select_similarity, False, 1, 0),
        ("complementary", select_complementary_greedy, False, 1, 0),
        ("comp+rep", select_complementary_reputation, True, 5, 80),
        ("oracle", select_oracle_true, False, 1, 0),
    ]:
        reg_m = make_registry(n_agents, 12, seed=seed)
        reg_m.true_skills = reg.true_skills.copy()
        reg_m.advertised = reg.advertised.copy()
        reg_m.alive = reg.alive.copy()
        reg_m.reputation = np.full(n_agents, 0.5, dtype=np.float64)
        reg_m.verified = np.zeros((n_agents, 12), dtype=np.float64)
        st = evaluate_method(
            reg_m,
            tasks,
            k,
            fn,
            update_rep=upd,
            epochs=ep,
            probes_per_epoch=pr,
            rng=np.random.default_rng(seed + k),
        )
        out[name] = st["success_rate"]
    return out


def main() -> None:
    seeds = list(range(5))
    regimes = ["honest", "spam", "churn"]
    ks = [3, 5]
    all_r = {r: {str(k): {} for k in ks} for r in regimes}
    for regime in regimes:
        for k in ks:
            rows = {m: [] for m in ["random", "similarity", "complementary", "comp+rep", "oracle"]}
            for seed in seeds:
                one = run_one(seed, regime, k)
                for m, v in one.items():
                    rows[m].append(v)
            summary = {}
            for m, vals in rows.items():
                summary[m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": vals,
                }
            all_r[regime][str(k)] = summary
    out_path = Path("results/multiseed_paper_protocol.json")
    out_path.write_text(json.dumps(all_r, indent=2))
    print(f"Wrote {out_path}")
    for regime in regimes:
        print(f"\n=== {regime} ===")
        for k in ks:
            bits = []
            for m in ["similarity", "complementary", "comp+rep", "oracle"]:
                s = all_r[regime][str(k)][m]
                bits.append(f"{m}={s['mean']:.2f}±{s['std']:.2f}")
            print(f" k={k}: " + ", ".join(bits))


if __name__ == "__main__":
    main()
