"""LLM router baseline over several seeds and three shortlist conditions.

The backend follows LLM_PROVIDER (anthropic or xai). Results go to
results/<provider>_multiseed.json after every cell, and API responses are cached,
so rerunning the same command resumes where it stopped.

Usage:
    LLM_PROVIDER=openai PYTHONPATH=. python scripts/run_llm_multiseed.py --check
    LLM_PROVIDER=openai PYTHONPATH=. python scripts/run_llm_multiseed.py --dry-run
    LLM_PROVIDER=openai PYTHONPATH=. python scripts/run_llm_multiseed.py
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.anthropic_rank import (
    _model,
    build_shortlist,
    claude_text,
    provider,
    rewrite_spam_cards_with_claude,
    select_anthropic,
)
from src.cards import build_cards
from src.env import apply_churn, apply_spam, make_registry, reset_honest_ads, sample_tasks, team_covers

REGIMES = {"honest": (0.0, False), "spam": (0.4, False), "churn": (0.4, True)}


def build_regime(seed: int, regime: str, n_agents: int, spam_max: int, rewrite: bool):
    spam_frac, churn = REGIMES[regime]
    reg = make_registry(n_agents, 12, seed=seed)
    reset_honest_ads(reg)
    if spam_frac > 0:
        apply_spam(reg, spam_frac=spam_frac, inflate=4)
    if churn:
        apply_churn(reg, leave_frac=0.15, join_frac=0.05)
    reg.cards = build_cards(reg.true_skills, reg.advertised)
    if rewrite and spam_frac > 0:
        rewrite_spam_cards_with_claude(reg, max_agents=spam_max)
    return reg


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--shortlists", nargs="+", default=["cosine", "coverage", "full"])
    p.add_argument("--regimes", nargs="+", default=list(REGIMES))
    p.add_argument("--ks", type=int, nargs="+", default=[3, 5])
    p.add_argument("--n-agents", type=int, default=200)
    p.add_argument("--n-tasks", type=int, default=24)
    p.add_argument("--spam-max", type=int, default=25)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="estimate calls and input tokens, no API")
    p.add_argument("--check", action="store_true", help="one tiny uncached call to test the key and model")
    args = p.parse_args()

    print(f"provider={provider()} model={_model()}")
    if args.check:
        print("reply:", claude_text("Reply with the single word: ok", max_tokens=5, use_cache=False, max_retries=1))
        return

    OUT = Path(f"results/{provider()}_multiseed.json")
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    calls, chars = 0, 0
    for seed in args.seeds:
        for regime in args.regimes:
            reg = build_regime(seed, regime, args.n_agents, args.spam_max, rewrite=not args.dry_run)
            tasks = sample_tasks(args.n_tasks, 12, seed=seed + 7)
            for shortlist in args.shortlists:
                for k in args.ks:
                    key = f"{regime}|{shortlist}|k{k}|seed{seed}"
                    if key in done:
                        continue
                    pools = [build_shortlist(reg, tasks[t], mode=shortlist) for t in range(len(tasks))]
                    if args.dry_run:
                        calls += len(tasks)
                        chars += sum(sum(len(reg.cards[i]) + 30 for i in pool) + 600 for pool in pools)
                        continue

                    def run_task(t: int):
                        team = select_anthropic(reg, tasks[t], k, shortlist=shortlist)
                        ok = team_covers(reg.true_skills, team, tasks[t])
                        reachable = team_covers(reg.true_skills, pools[t], tasks[t])
                        return float(ok), len(team), float(reachable)

                    with ThreadPoolExecutor(max_workers=args.workers) as ex:
                        outs = list(ex.map(run_task, range(len(tasks))))
                    done[key] = {
                        "success_rate": float(np.mean([o[0] for o in outs])),
                        "avg_team_size": float(np.mean([o[1] for o in outs])),
                        "pool_cover_rate": float(np.mean([o[2] for o in outs])),
                        "pool_size": float(np.mean([len(pl) for pl in pools])),
                    }
                    OUT.write_text(json.dumps(done, indent=2))
                    print(f"{key}: success={done[key]['success_rate']:.2f} "
                          f"pool_cover={done[key]['pool_cover_rate']:.2f} pool={done[key]['pool_size']:.0f}",
                          flush=True)

    if args.dry_run:
        tokens = chars / 4
        print(f"calls={calls} input_tokens~{tokens / 1e6:.2f}M output_tokens~{calls * 30 / 1e6:.2f}M "
              "(multiply by your model's per-token price)")
        return

    print("\nregime shortlist k: mean ± std over seeds")
    for regime in args.regimes:
        for shortlist in args.shortlists:
            for k in args.ks:
                vals = [done[f"{regime}|{shortlist}|k{k}|seed{s}"]["success_rate"] for s in args.seeds]
                print(f"{regime:<7} {shortlist:<9} k={k}: {np.mean(vals):.2f} ± {np.std(vals):.2f}")


if __name__ == "__main__":
    main()
