"""Control runs for the LLM router baseline.

same_cards: the OpenAI router reads the exact spam cards Claude wrote for seed 0
            (served from the response cache), so model and card text are not confounded.
stealth:    spam cards use the same neutral template as honest cards, so the only
            spam signal is the skill list itself, not marketing tone.

Results are written after every cell; API responses are cached, so reruns resume.

Usage:
    LLM_PROVIDER=openai PYTHONPATH=. python scripts/llm_controls.py [same_cards|stealth|all]
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from scripts.run_llm_multiseed import build_regime
from src.anthropic_rank import _model, build_shortlist, provider, select_anthropic
from src.cards import honest_card
from src.env import team_covers, sample_tasks

N_AGENTS, N_TASKS, SPAM_MAX = 200, 24, 25


def run_cell(reg, tasks, k, shortlist, workers=8) -> dict:
    pools = [build_shortlist(reg, tasks[t], mode=shortlist) for t in range(len(tasks))]

    def one(t: int):
        team = select_anthropic(reg, tasks[t], k, shortlist=shortlist)
        return float(team_covers(reg.true_skills, team, tasks[t])), len(team)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(one, range(len(tasks))))
    return {
        "success_rate": float(np.mean([o[0] for o in outs])),
        "avg_team_size": float(np.mean([o[1] for o in outs])),
        "pool_cover_rate": float(np.mean([team_covers(reg.true_skills, p, tasks[t]) for t, p in enumerate(pools)])),
    }


def main() -> None:
    load_dotenv(".env")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    router = provider()
    out_path = Path(f"results/llm_controls_{router}.json")
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    print(f"router provider={router} model={_model()}")

    def save(key: str, val: dict) -> None:
        done[key] = val
        out_path.write_text(json.dumps(done, indent=2))
        print(f"{key}: success={val['success_rate']:.2f} pool_cover={val['pool_cover_rate']:.2f}", flush=True)

    if which in ("same_cards", "all"):
        for regime in ["spam", "churn"]:
            os.environ["LLM_PROVIDER"] = "anthropic"
            reg = build_regime(0, regime, N_AGENTS, SPAM_MAX, rewrite=True)
            os.environ["LLM_PROVIDER"] = router
            tasks = sample_tasks(N_TASKS, 12, seed=7)
            for k in [3, 5]:
                key = f"same_cards|{regime}|cosine|k{k}|seed0"
                if key not in done:
                    save(key, run_cell(reg, tasks, k, "cosine"))

    if which in ("stealth", "all"):
        for seed in [0, 1, 2]:
            reg = build_regime(seed, "spam", N_AGENTS, SPAM_MAX, rewrite=False)
            reg.cards = [honest_card(i, reg.advertised[i]) for i in range(reg.n)]
            tasks = sample_tasks(N_TASKS, 12, seed=seed + 7)
            for shortlist in ["cosine", "full"]:
                for k in [3, 5]:
                    key = f"stealth|spam|{shortlist}|k{k}|seed{seed}"
                    if key not in done:
                        save(key, run_cell(reg, tasks, k, shortlist))


if __name__ == "__main__":
    main()
