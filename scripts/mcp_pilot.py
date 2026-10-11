"""Pilot on real MCP registry listings (data/mcp_registry_snapshot.json).

The LLM (LLM_PROVIDER) maps each public description to the skill ontology. Those labels
act as true skills and the description itself is the honest card. Spammers get an LLM
rewrite of their own description that also claims extra skills.

Usage:
    LLM_PROVIDER=openai PYTHONPATH=. python scripts/mcp_pilot.py [--no-claude-selector]
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.anthropic_rank import _model, claude_text, provider, select_anthropic
from src.cards import SKILL_NAMES
from src.discover import (
    select_complementary_greedy,
    select_complementary_reputation,
    select_oracle_true,
    select_similarity,
)
from src.env import Registry, run_capability_probes, sample_tasks, team_covers, update_reputation

SNAPSHOT = Path("data/mcp_registry_snapshot.json")
D = len(SKILL_NAMES)


def label(desc: str) -> list[int]:
    prompt = f"""Here is the public description of an MCP server from the official registry:
\"\"\"{desc}\"\"\"

Which of these capabilities does the server directly provide? Only count capabilities the
description clearly states. Capabilities: {SKILL_NAMES}

Reply with ONLY a JSON array of capability names, e.g. ["web_search", "summarization"]. Use [] if none fit."""
    text = claude_text(prompt, max_tokens=100)
    m = re.search(r"\[[^\]]*\]", text)
    if not m:
        return []
    try:
        names = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return sorted({SKILL_NAMES.index(n) for n in names if n in SKILL_NAMES})


def spam_rewrite(desc: str, extra: list[str]) -> str:
    prompt = f"""Rewrite this MCP server listing so it sounds highly capable and also claims
these extra capabilities: {extra}. Keep it to two or three sentences, marketplace style.
Original listing: \"\"\"{desc}\"\"\"
Return only the new listing."""
    return claude_text(prompt, max_tokens=180).strip()


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--ks", type=int, nargs="+", default=[3, 5])
    p.add_argument("--n-tasks", type=int, default=24)
    p.add_argument("--spam-frac", type=float, default=0.4)
    p.add_argument("--no-claude-selector", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    out_path = Path(f"results/mcp_pilot_{provider()}.json")
    print(f"provider={provider()} model={_model()}")

    servers = json.loads(SNAPSHOT.read_text())["servers"]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        labels = list(ex.map(lambda s: label(s["description"]), servers))
    kept = [(s, l) for s, l in zip(servers, labels) if l]
    n = len(kept)
    true = np.zeros((n, D))
    for i, (_, l) in enumerate(kept):
        true[i, l] = 1.0
    stats = {
        "servers_in_snapshot": len(servers),
        "servers_with_skills": n,
        "skills_per_server_mean": float(true.sum(1).mean()),
        "skills_per_server_hist": np.bincount(true.sum(1).astype(int), minlength=D + 1).tolist(),
        "agents_per_skill": dict(zip(SKILL_NAMES, true.sum(0).astype(int).tolist())),
    }
    print(json.dumps(stats, indent=1))

    # Only keep tasks the honest pool can cover, so failures come from the selector.
    all_tasks = sample_tasks(400, D, seed=7)
    full_team = list(range(n))
    feasible = [t for t in all_tasks if team_covers(true, full_team, t)]
    tasks = np.array(feasible[: args.n_tasks])
    stats["feasible_task_frac"] = len(feasible) / len(all_tasks)

    results: dict = {}
    for spam_frac in [0.0, args.spam_frac]:
        regime = "honest" if spam_frac == 0 else "spam"
        for seed in args.seeds if spam_frac > 0 else args.seeds[:1]:
            rng = np.random.default_rng(seed)
            adv = true.copy()
            cards = [f"{s['title'] or s['name']}: {s['description']}" for s, _ in kept]
            if spam_frac > 0:
                spammers = rng.choice(n, size=int(spam_frac * n), replace=False)
                jobs = []
                for i in spammers:
                    zeros = np.where(true[i] == 0)[0]
                    add = rng.choice(zeros, size=min(4, len(zeros)), replace=False)
                    adv[i, add] = 1.0
                    jobs.append((int(i), [SKILL_NAMES[j] for j in add]))
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    rewrites = list(ex.map(lambda job: spam_rewrite(kept[job[0]][0]["description"], job[1]), jobs))
                for (i, _), text in zip(jobs, rewrites):
                    cards[i] = text

            def fresh() -> Registry:
                return Registry(
                    true_skills=true.copy(), advertised=adv.copy(), alive=np.ones(n, dtype=bool),
                    reputation=np.full(n, 0.5), verified=np.zeros((n, D)),
                    rng=np.random.default_rng(seed + 100), cards=list(cards),
                )

            for k in args.ks:
                row = {}
                for name, fn, epochs, probes in [
                    ("similarity", select_similarity, 1, 0),
                    ("complementary", select_complementary_greedy, 1, 0),
                    ("comp+rep", select_complementary_reputation, 5, 40),
                    ("oracle", select_oracle_true, 1, 0),
                ]:
                    reg = fresh()
                    order_rng = np.random.default_rng(seed + k)
                    for _ in range(epochs):
                        if probes:
                            run_capability_probes(reg, n_probes=probes)
                        succ, sizes = [], []
                        for t in order_rng.permutation(len(tasks)):
                            team = fn(reg, tasks[t], k)
                            ok = team_covers(reg.true_skills, team, tasks[t])
                            succ.append(ok)
                            sizes.append(len(team))
                            if name == "comp+rep":
                                update_reputation(reg, team, ok, req=tasks[t])
                    row[name] = {"success_rate": float(np.mean(succ)), "avg_team_size": float(np.mean(sizes))}
                if not args.no_claude_selector:
                    reg = fresh()

                    def run_task(t: int):
                        team = select_anthropic(reg, tasks[t], k, shortlist="full")
                        return float(team_covers(reg.true_skills, team, tasks[t])), len(team)

                    with ThreadPoolExecutor(max_workers=args.workers) as ex:
                        outs = list(ex.map(run_task, range(len(tasks))))
                    row["llm_full"] = {
                        "success_rate": float(np.mean([o[0] for o in outs])),
                        "avg_team_size": float(np.mean([o[1] for o in outs])),
                    }
                results[f"{regime}|k{k}|seed{seed}"] = row
                out_path.write_text(json.dumps(
                    {"provider": provider(), "model": _model(), "stats": stats, "results": results}, indent=2))
                print(regime, f"k={k}", f"seed={seed}",
                      ", ".join(f"{m}={v['success_rate']:.2f}" for m, v in row.items()), flush=True)

    out_path.write_text(json.dumps({"provider": provider(), "model": _model(), "stats": stats, "results": results}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
