"""Evaluation loop and plotting (with resume checkpoints)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from .checkpoint import CheckpointStore
from .discover import (
    select_complementary_greedy,
    select_complementary_reputation,
    select_random,
    select_similarity,
)
from .env import (
    Registry,
    apply_churn,
    apply_spam,
    make_registry,
    reset_honest_ads,
    run_capability_probes,
    sample_tasks,
    team_covers,
    update_reputation,
)


def evaluate_method(
    reg: Registry,
    tasks: np.ndarray,
    k: int,
    select_fn: Callable,
    update_rep: bool = False,
    epochs: int = 1,
    probes_per_epoch: int = 0,
    rng: np.random.Generator | None = None,
    ckpt: CheckpointStore | None = None,
    regime: str = "",
    method: str = "",
    checkpoint_tasks: bool = False,
) -> dict:
    rng = rng or np.random.default_rng(0)
    epoch_rates: list[float] = []
    last_sizes: list[float] = []

    # Task-level resume only for single-epoch API methods (claude).
    resume = None
    if checkpoint_tasks and ckpt is not None and epochs == 1:
        resume = ckpt.get_task_progress(regime, k, method)

    for _ep in range(epochs):
        if probes_per_epoch > 0:
            run_capability_probes(reg, n_probes=probes_per_epoch)

        if resume and _ep == 0:
            done = set(resume["done_indices"])
            successes = list(resume["successes"])
            sizes = list(resume["sizes"])
            print(f"  [ckpt] resume {method} k={k}: {len(done)}/{tasks.shape[0]} tasks done")
        else:
            done = set()
            successes = []
            sizes = []

        order = list(rng.permutation(tasks.shape[0])) if not resume else list(range(tasks.shape[0]))
        # Keep deterministic remaining order when resuming
        if resume and _ep == 0:
            order = [i for i in range(tasks.shape[0]) if i not in done]

        for t in order:
            if t in done:
                continue
            req = tasks[t]
            if select_fn is select_random:
                team = select_fn(reg, req, k, rng)
            else:
                team = select_fn(reg, req, k)
            ok = team_covers(reg.true_skills, team, req)
            successes.append(float(ok))
            sizes.append(len(team))
            done.add(int(t))
            if update_rep:
                update_reputation(reg, team, ok, req=req)
            if checkpoint_tasks and ckpt is not None and epochs == 1:
                # Save every task so a mid-method crash can resume.
                ckpt.set_task_progress(
                    regime,
                    k,
                    method,
                    done_indices=sorted(done),
                    successes=successes,
                    sizes=sizes,
                )

        epoch_rates.append(float(np.mean(successes)) if successes else 0.0)
        last_sizes = sizes
        resume = None  # only first epoch resumes

    if checkpoint_tasks and ckpt is not None:
        ckpt.clear_task_progress(regime, k, method)

    return {
        "success_rate": epoch_rates[-1],
        "success_rate_first_epoch": epoch_rates[0],
        "epoch_rates": epoch_rates,
        "avg_team_size": float(np.mean(last_sizes)) if last_sizes else 0.0,
        "n": tasks.shape[0],
        "epochs": epochs,
    }


def run_regime(
    name: str,
    n_agents: int,
    n_tasks: int,
    skill_dim: int,
    ks: list[int],
    spam_frac: float,
    churn: bool,
    use_anthropic: bool,
    seed: int,
    claude_spam_rewrite: bool = False,
    claude_spam_max: int = 30,
    ckpt: CheckpointStore | None = None,
) -> dict:
    from .anthropic_rank import ensure_nl_cards, rewrite_spam_cards_with_claude, select_anthropic
    from .cards import build_cards

    methods_spec = {
        "random": (select_random, False, 1, 0, False),
        "similarity": (select_similarity, False, 1, 0, False),
        "complementary": (select_complementary_greedy, False, 1, 0, False),
        "comp+rep": (select_complementary_reputation, True, 5, 80, False),
    }
    if use_anthropic:
        methods_spec["claude"] = (select_anthropic, False, 1, 0, True)

    # Fast path: whole regime already complete
    if ckpt is not None and ckpt.regime_complete(name, ks, list(methods_spec.keys())):
        print(f"  [{name}] SKIP (checkpoint complete)")
        block = ckpt.data["regimes"][name]
        return {
            "regime": name,
            "by_k": block.get("by_k", {}),
            "api": block.get("api", {"used": use_anthropic}),
        }

    results = {"regime": name, "by_k": {}, "api": {"used": use_anthropic}}
    if ckpt is not None:
        prior = ckpt.data.get("regimes", {}).get(name, {})
        results["by_k"] = dict(prior.get("by_k", {}))
        results["api"] = {**results["api"], **prior.get("api", {})}

    base = make_registry(n_agents, skill_dim, seed=seed)
    reset_honest_ads(base)
    if spam_frac > 0:
        apply_spam(base, spam_frac=spam_frac, inflate=4)
    if churn:
        apply_churn(base, leave_frac=0.15, join_frac=0.05)

    # Cards: prefer checkpointed rewritten cards
    loaded = ckpt.load_cards(name) if ckpt is not None else None
    if loaded is not None and len(loaded) == base.n:
        base.cards = loaded
        print(f"  [{name}] loaded checkpointed NL cards")
    else:
        base.cards = build_cards(base.true_skills, base.advertised)
        if use_anthropic and claude_spam_rewrite and spam_frac > 0:
            n_rewritten = rewrite_spam_cards_with_claude(base, max_agents=claude_spam_max)
            results["api"]["spam_cards_rewritten"] = n_rewritten
            print(f"  [{name}] Claude rewrote {n_rewritten} spam capability cards")
            if ckpt is not None:
                ckpt.save_cards(name, base.cards)
                ckpt.set_regime_api(name, results["api"])

    tasks = sample_tasks(n_tasks, skill_dim, seed=seed + 7)

    for k in ks:
        row = dict(results["by_k"].get(str(k), {}))
        for mname, (fn, upd, epochs, probes, task_ckpt) in methods_spec.items():
            if mname in row:
                print(f"  [{name}] k={k} method={mname} SKIP (checkpoint)")
                continue
            reg_m = make_registry(n_agents, skill_dim, seed=seed)
            reg_m.true_skills = base.true_skills.copy()
            reg_m.advertised = base.advertised.copy()
            reg_m.alive = base.alive.copy()
            reg_m.reputation = np.full(n_agents, 0.5, dtype=np.float64)
            reg_m.verified = np.zeros((n_agents, skill_dim), dtype=np.float64)
            reg_m.cards = list(base.cards) if base.cards else ensure_nl_cards(reg_m)
            print(f"  [{name}] k={k} method={mname} ...")
            try:
                stats = evaluate_method(
                    reg_m,
                    tasks,
                    k,
                    fn,
                    update_rep=upd,
                    epochs=epochs,
                    probes_per_epoch=probes,
                    rng=np.random.default_rng(seed + k),
                    ckpt=ckpt,
                    regime=name,
                    method=mname,
                    checkpoint_tasks=task_ckpt,
                )
            except Exception as e:
                print(f"  [{name}] FAILED k={k} {mname}: {e}")
                print("  Re-run the same command to resume from checkpoint.")
                raise
            row[mname] = stats
            results["by_k"][str(k)] = row
            if ckpt is not None:
                ckpt.set_method(name, k, mname, stats)
        results["by_k"][str(k)] = row
    return results


def plot_success_vs_k(all_regimes: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(all_regimes), figsize=(4.2 * len(all_regimes), 3.6), sharey=True)
    if len(all_regimes) == 1:
        axes = [axes]
    method_styles = {
        "random": ("o", "#888888"),
        "similarity": ("s", "#c0392b"),
        "complementary": ("D", "#2471a3"),
        "comp+rep": ("^", "#1e8449"),
        "claude": ("v", "#8e44ad"),
        "anthropic": ("v", "#8e44ad"),
    }
    for ax, block in zip(axes, all_regimes):
        ks = sorted(int(k) for k in block["by_k"])
        methods = list(next(iter(block["by_k"].values())).keys())
        for m in methods:
            ys = [block["by_k"][str(k)][m]["success_rate"] for k in ks]
            marker, color = method_styles.get(m, ("o", "#333"))
            ax.plot(ks, ys, marker=marker, color=color, label=m, linewidth=2)
        ax.set_title(block["regime"])
        ax.set_xlabel("team size k")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("cover success rate")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_spam_gap(honest: dict, spam: dict, k: int, out_path: Path) -> None:
    methods = list(honest["by_k"][str(k)].keys())
    x = np.arange(len(methods))
    h = [honest["by_k"][str(k)][m]["success_rate"] for m in methods]
    s = [spam["by_k"][str(k)][m]["success_rate"] for m in methods]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.35
    ax.bar(x - w / 2, h, w, label="honest ads", color="#2471a3")
    ax.bar(x + w / 2, s, w, label="capability spam", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("cover success rate")
    ax.set_title(f"Honest vs spam (k={k})")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
