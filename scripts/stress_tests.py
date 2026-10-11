"""Offline stress tests: probe budget, scale and churn, probe gaming, graded skills,
noisy feedback and collusion, and context-dependent skills (no API calls).

Usage:
    PYTHONPATH=. python scripts/stress_tests.py [budget|scale|gaming|partial|feedback|context|all]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.discover import (
    cosine_sim,
    select_complementary_greedy,
    select_complementary_reputation,
    select_complementary_reputation_guarded,
    select_oracle_true,
    select_similarity,
)
from src.env import (
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

OUT = Path("results/stress_tests")
SKILL_DIM = 12


def trust_loop(
    reg: Registry,
    tasks: np.ndarray,
    k: int,
    select: Callable,
    epochs: int,
    probes: int,
    update_rep: bool,
    revoke: bool = False,
    churn: tuple[float, float] | None = None,
    probe_fn: Callable[[Registry, int], None] | None = None,
    covers_fn: Callable[[Registry, list[int], np.ndarray], bool] | None = None,
    rng: np.random.Generator | None = None,
    report_flip: float = 0.0,
    after_epoch: Callable[[Registry], None] | None = None,
) -> dict:
    """report_flip: chance a user reports the wrong outcome. after_epoch: e.g. fake ratings."""
    rng = rng or np.random.default_rng(0)
    probe_fn = probe_fn or (lambda r, n: run_capability_probes(r, n_probes=n))
    covers_fn = covers_fn or (lambda r, team, req: team_covers(r.true_skills, team, req))
    rates, sizes, select_s = [], [], []
    failures = 0
    failures_to_95 = None
    probes_to_95 = None
    probes_used = 0
    for ep in range(epochs):
        if churn is not None and ep > 0:
            apply_churn(reg, leave_frac=churn[0], join_frac=churn[1], reset_joiners=True)
        if probes > 0:
            probe_fn(reg, probes)
            probes_used += probes
        succ = []
        for t in rng.permutation(tasks.shape[0]):
            req = tasks[t]
            t0 = time.perf_counter()
            team = select(reg, req, k)
            select_s.append(time.perf_counter() - t0)
            ok = covers_fn(reg, team, req)
            succ.append(float(ok))
            sizes.append(len(team))
            failures += int(not ok)
            if update_rep:
                reported = (not ok) if report_flip > 0 and rng.random() < report_flip else ok
                update_reputation(reg, team, reported, req=req, revoke=revoke)
        if after_epoch is not None:
            after_epoch(reg)
        rates.append(float(np.mean(succ)))
        if rates[-1] >= 0.95 and failures_to_95 is None:
            failures_to_95 = failures
            probes_to_95 = probes_used
    return {
        "epoch_rates": rates,
        "final": rates[-1],
        "first": rates[0],
        "mean_rate": float(np.mean(rates)),
        "avg_team_size": float(np.mean(sizes)),
        "probes_used": probes_used,
        "failures_total": failures,
        "failures_to_95": failures_to_95,
        "probes_to_95": probes_to_95,
        "select_ms": 1000 * float(np.mean(select_s)),
    }


def spam_registry(n: int, seed: int, spam_frac: float = 0.4) -> Registry:
    reg = make_registry(n, SKILL_DIM, seed=seed)
    reset_honest_ads(reg)
    if spam_frac > 0:
        apply_spam(reg, spam_frac=spam_frac, inflate=4)
    return reg


def summarize(runs: list[dict]) -> dict:
    out = {"runs": runs}
    for key in ["final", "first", "mean_rate", "avg_team_size", "failures_total", "select_ms"]:
        vals = [r[key] for r in runs]
        out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    reached = [r["failures_to_95"] for r in runs if r["failures_to_95"] is not None]
    out["frac_reached_95"] = len(reached) / len(runs)
    out["failures_to_95_mean"] = float(np.mean(reached)) if reached else None
    curves = np.array([r["epoch_rates"] for r in runs])
    out["curve_mean"] = curves.mean(axis=0).tolist()
    out["curve_std"] = curves.std(axis=0).tolist()
    return out


def write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}")


# ---------------------------------------------------------------- probe budget


def exp_budget(seeds=range(5), n=200, n_tasks=80, k=5, epochs=5) -> None:
    budgets = [0, 5, 10, 20, 40, 80, 160]
    payload = {"config": dict(n=n, n_tasks=n_tasks, k=k, epochs=epochs, spam=0.4), "probes_and_rep": {}, "probes_only": {}}
    for b in budgets:
        for mode, upd in [("probes_and_rep", True), ("probes_only", False)]:
            if mode == "probes_only" and b == 0:
                continue
            runs = []
            for seed in seeds:
                reg = spam_registry(n, seed)
                tasks = sample_tasks(n_tasks, SKILL_DIM, seed=seed + 7)
                runs.append(
                    trust_loop(reg, tasks, k, select_complementary_reputation, epochs, b, upd,
                               rng=np.random.default_rng(seed + k))
                )
            payload[mode][str(b)] = summarize(runs)
    write("probe_budget", payload)
    print("\nprobes/epoch | probes+rep: first final fails | probes only: final")
    for b in budgets:
        pr = payload["probes_and_rep"][str(b)]
        po = payload["probes_only"].get(str(b))
        po_s = f"{po['final']['mean']:.2f}" if po else "  -"
        print(f"{b:>5} | {pr['first']['mean']:.2f} {pr['final']['mean']:.2f}±{pr['final']['std']:.2f} "
              f"{pr['failures_total']['mean']:6.1f} | {po_s}")


# ---------------------------------------------------------------- scale + churn


def exp_scale(seeds=range(3), n_tasks=80, k=5, epochs=8, churn=(0.1, 0.1)) -> None:
    sizes = [200, 1000, 2000, 5000]
    payload = {"config": dict(n_tasks=n_tasks, k=k, epochs=epochs, churn=churn, spam=0.4), "results": {}}
    for n in sizes:
        row = {}
        policies = {
            "comp+rep fixed80": (select_complementary_reputation, True, 80),
            "comp+rep prop": (select_complementary_reputation, True, int(0.4 * n)),
            "comp+rep rep_only": (select_complementary_reputation, True, 0),
            "similarity": (select_similarity, False, 0),
            "complementary": (select_complementary_greedy, False, 0),
            "oracle": (select_oracle_true, False, 0),
        }
        for name, (fn, upd, probes) in policies.items():
            runs = []
            for seed in seeds:
                reg = spam_registry(n, seed)
                tasks = sample_tasks(n_tasks, SKILL_DIM, seed=seed + 7)
                runs.append(
                    trust_loop(reg, tasks, k, fn, epochs, probes, upd, churn=churn,
                               rng=np.random.default_rng(seed + k))
                )
            row[name] = summarize(runs)
            row[name]["probes_per_epoch"] = probes
            print(f"N={n:>5} {name:<18} mean={row[name]['mean_rate']['mean']:.2f} "
                  f"final={row[name]['final']['mean']:.2f} select={row[name]['select_ms']['mean']:.2f}ms")
        payload["results"][str(n)] = row
    write("scale_churn", payload)


# ---------------------------------------------------------------- probe gaming


def exp_gaming(seeds=range(5), n=200, n_tasks=80, k=5, epochs=8, probes=80) -> None:
    gamer_fracs = [0.0, 0.5, 1.0]
    plain, guarded = select_complementary_reputation, select_complementary_reputation_guarded
    variants = {
        "probes_only": dict(probes=probes, upd=False, revoke=False, fn=plain),
        "rep_only": dict(probes=0, upd=True, revoke=False, fn=plain),
        "probes_and_rep": dict(probes=probes, upd=True, revoke=False, fn=plain),
        "probes_rep_revoke": dict(probes=probes, upd=True, revoke=True, fn=plain),
        "probes_rep_guarded": dict(probes=probes, upd=True, revoke=False, fn=guarded),
    }
    payload = {"config": dict(n=n, n_tasks=n_tasks, k=k, epochs=epochs, probes=probes, spam=0.4), "results": {}}
    for g in gamer_fracs:
        row = {}
        for name, v in variants.items():
            runs = []
            for seed in seeds:
                reg = spam_registry(n, seed)
                spammers = np.where((reg.advertised != reg.true_skills).any(axis=1))[0]
                gamers = np.zeros(n, dtype=bool)
                pick = np.random.default_rng(seed + 99).choice(
                    spammers, size=int(round(g * len(spammers))), replace=False
                )
                gamers[pick] = True
                reg.gamers = gamers
                tasks = sample_tasks(n_tasks, SKILL_DIM, seed=seed + 7)
                runs.append(
                    trust_loop(reg, tasks, k, v["fn"], epochs, v["probes"], v["upd"],
                               revoke=v["revoke"], rng=np.random.default_rng(seed + k))
                )
            row[name] = summarize(runs)
            print(f"gamers={g:.1f} {name:<18} first={row[name]['first']['mean']:.2f} "
                  f"final={row[name]['final']['mean']:.2f}±{row[name]['final']['std']:.2f} "
                  f"fails={row[name]['failures_total']['mean']:.0f}")
        payload["results"][str(g)] = row
    write("probe_gaming", payload)


# ---------------------------------------------------------------- partial skills


def make_partial(n: int, seed: int, spam_frac: float, tau: float):
    """Graded skills in [0,1]; an agent covers a skill when its level reaches tau."""
    rng = np.random.default_rng(seed)
    levels = np.zeros((n, SKILL_DIM))
    for i in range(n):
        m = int(rng.integers(2, 6))
        idx = rng.choice(SKILL_DIM, size=m, replace=False)
        levels[i, idx] = rng.uniform(0.2, 1.0, size=m)
    adv = levels.copy()
    n_spam = int(spam_frac * n)
    if n_spam:
        for i in rng.choice(n, size=n_spam, replace=False):
            zeros = np.where(levels[i] == 0)[0]
            add = rng.choice(zeros, size=min(4, len(zeros)), replace=False)
            adv[i, add] = 0.95
            weak = (levels[i] > 0) & (levels[i] < tau)
            adv[i, weak] = 0.95
    return levels, adv


def exp_partial(seeds=range(5), n=200, n_tasks=80, k=5, epochs=5, probes=80, tau=0.6) -> None:
    payload = {"config": dict(n=n, n_tasks=n_tasks, k=k, epochs=epochs, probes=probes, tau=tau), "results": {}}
    for spam_frac in [0.0, 0.4]:
        for sigma in [0.0, 0.1, 0.2, 0.3]:
            if spam_frac == 0.0 and sigma > 0.0:
                continue
            key = f"spam={spam_frac}_noise={sigma}"
            row = {m: [] for m in ["similarity", "complementary", "comp+rep", "oracle"]}
            partial_credit = {m: [] for m in row}
            for seed in seeds:
                levels, adv = make_partial(n, seed, spam_frac, tau)
                tasks = sample_tasks(n_tasks, SKILL_DIM, seed=seed + 7)
                probe_rng = np.random.default_rng(seed + 1234)

                def build() -> Registry:
                    reg = make_registry(n, SKILL_DIM, seed=seed)
                    reg.true_skills = (levels >= tau).astype(np.float64)
                    reg.advertised = (adv >= tau).astype(np.float64)
                    reg.reputation = np.full(n, 0.5)
                    reg.verified = np.zeros((n, SKILL_DIM))
                    return reg

                def noisy_probe(reg: Registry, n_probes: int) -> None:
                    ids = reg.active_ids()
                    for _ in range(n_probes):
                        i = int(probe_rng.choice(ids))
                        claimed = np.where(reg.advertised[i] > 0)[0]
                        if len(claimed) == 0:
                            continue
                        b = int(probe_rng.choice(claimed))
                        observed = levels[i, b] + probe_rng.normal(0.0, sigma)
                        if observed >= tau:
                            reg.verified[i, b] = 1.0
                            reg.reputation[i] = min(1.0, reg.reputation[i] + 0.05)
                        else:
                            reg.reputation[i] = max(0.0, reg.reputation[i] - 0.2)

                def sim_graded(reg: Registry, req: np.ndarray, kk: int) -> list[int]:
                    sims = cosine_sim(req, adv)
                    return [int(i) for i in np.argsort(-sims)[:kk]]

                def credit(team: list[int], req: np.ndarray) -> float:
                    if not team:
                        return 0.0
                    need = req > 0
                    return float(np.mean(levels[team].max(axis=0)[need] >= tau))

                specs = {
                    "similarity": (sim_graded, False, 1, 0),
                    "complementary": (select_complementary_greedy, False, 1, 0),
                    "comp+rep": (select_complementary_reputation, True, epochs, probes),
                    "oracle": (select_oracle_true, False, 1, 0),
                }
                for m, (fn, upd, ep, pr) in specs.items():
                    reg = build()
                    res = trust_loop(reg, tasks, k, fn, ep, pr, upd, probe_fn=noisy_probe,
                                     rng=np.random.default_rng(seed + k))
                    row[m].append(res["final"])
                    partial_credit[m].append(np.mean([credit(fn(reg, tasks[t], k), tasks[t]) for t in range(n_tasks)]))
            payload["results"][key] = {
                m: {
                    "full_cover_mean": float(np.mean(row[m])),
                    "full_cover_std": float(np.std(row[m])),
                    "partial_credit_mean": float(np.mean(partial_credit[m])),
                }
                for m in row
            }
            bits = ", ".join(
                f"{m}={payload['results'][key][m]['full_cover_mean']:.2f}"
                f"({payload['results'][key][m]['partial_credit_mean']:.2f})"
                for m in row
            )
            print(f"{key:<22} {bits}")
    write("partial_skills", payload)


# ---------------------------------------------------------------- user feedback and collusion


def exp_feedback(seeds=range(5), n=200, n_tasks=80, k=5, epochs=8, probes=80, fake_ratings=3) -> None:
    """Noisy user reports, and a ring of spammers that posts fake positive ratings for itself."""
    payload = {"config": dict(n=n, n_tasks=n_tasks, k=k, epochs=epochs, probes=probes,
                              fake_ratings_per_colluder_per_epoch=fake_ratings, spam=0.4),
               "noise": {}, "collusion": {}}
    variants = {"rep_only": (0, True), "probes_and_rep": (probes, True)}

    def runs_for(setup, flip=0.0):
        out = {}
        for name, (pr, upd) in variants.items():
            runs = []
            for seed in seeds:
                reg = spam_registry(n, seed)
                after = setup(reg, seed)
                tasks = sample_tasks(n_tasks, SKILL_DIM, seed=seed + 7)
                runs.append(trust_loop(reg, tasks, k, select_complementary_reputation, epochs, pr, upd,
                                       rng=np.random.default_rng(seed + k), report_flip=flip,
                                       after_epoch=after))
            out[name] = summarize(runs)
        return out

    for flip in [0.0, 0.1, 0.2, 0.3]:
        row = runs_for(lambda reg, seed: None, flip)
        payload["noise"][str(flip)] = row
        print(f"report flip={flip:.1f} " + " | ".join(
            f"{m}: final={r['final']['mean']:.2f}±{r['final']['std']:.2f} fails={r['failures_total']['mean']:.0f}"
            for m, r in row.items()))

    for ring in [0.0, 0.5, 1.0]:
        def setup(reg, seed, ring=ring):
            spammers = np.where((reg.advertised != reg.true_skills).any(axis=1))[0]
            ring_ids = np.random.default_rng(seed + 55).choice(
                spammers, size=int(round(ring * len(spammers))), replace=False)

            def boost(r):
                for i in ring_ids:
                    for _ in range(fake_ratings):
                        update_reputation(r, [int(i)], True)
            return boost if len(ring_ids) else None

        row = runs_for(setup)
        payload["collusion"][str(ring)] = row
        print(f"colluding ring={ring:.1f} " + " | ".join(
            f"{m}: final={r['final']['mean']:.2f}±{r['final']['std']:.2f} fails={r['failures_total']['mean']:.0f}"
            for m, r in row.items()))
    write("feedback_collusion", payload)


# ---------------------------------------------------------------- context-dependent skills


def exp_context(seeds=range(5), n=200, n_tasks=80, k=5, epochs=5, probes=80, contexts=4) -> None:
    """Each true skill works only in some task contexts. Probes test a random context."""
    payload = {"config": dict(n=n, n_tasks=n_tasks, k=k, epochs=epochs, probes=probes,
                              contexts=contexts, spam=0.4), "results": {}}
    methods = ["similarity", "complementary", "comp+rep", "comp+rep per context",
               "oracle (static skills)", "oracle (context aware)"]
    for q in [1.0, 0.75, 0.5]:
        finals = {m: [] for m in methods}
        for seed in seeds:
            base = spam_registry(n, seed)
            crng = np.random.default_rng(seed + 300)
            true_ctx = base.true_skills[None] * (crng.random((contexts, n, SKILL_DIM)) < q)
            tasks = sample_tasks(n_tasks, SKILL_DIM, seed=seed + 7)
            task_ctx = np.random.default_rng(seed + 301).integers(0, contexts, size=n_tasks)

            for m in methods:
                reg = spam_registry(n, seed)
                per_ctx = m == "comp+rep per context"
                rep_ctx = np.full((contexts, n), 0.5)
                ver_ctx = np.zeros((contexts, n, SKILL_DIM))
                learn = m.startswith("comp+rep")
                order_rng = np.random.default_rng(seed + k)
                prng = np.random.default_rng(seed + 302)
                ep_count = epochs if learn else 1
                for _ in range(ep_count):
                    if learn:
                        ids = reg.active_ids()
                        for _ in range(probes):
                            i = int(prng.choice(ids))
                            claimed = np.where(reg.advertised[i] > 0)[0]
                            b = int(prng.choice(claimed))
                            c = int(prng.integers(contexts))
                            rep = rep_ctx[c] if per_ctx else reg.reputation
                            ver = ver_ctx[c] if per_ctx else reg.verified
                            if true_ctx[c, i, b] > 0:
                                ver[i, b] = 1.0
                                rep[i] = min(1.0, rep[i] + 0.05)
                            else:
                                rep[i] = max(0.0, rep[i] - 0.2)
                    succ = []
                    for t in order_rng.permutation(n_tasks):
                        req, c = tasks[t], int(task_ctx[t])
                        if per_ctx:
                            reg.reputation, reg.verified = rep_ctx[c], ver_ctx[c]
                        if m == "similarity":
                            team = select_similarity(reg, req, k)
                        elif m == "complementary":
                            team = select_complementary_greedy(reg, req, k)
                        elif m == "oracle (static skills)":
                            team = select_oracle_true(reg, req, k)
                        elif m == "oracle (context aware)":
                            saved = reg.true_skills
                            reg.true_skills = true_ctx[c]
                            team = select_oracle_true(reg, req, k)
                            reg.true_skills = saved
                        else:
                            team = select_complementary_reputation(reg, req, k)
                        ok = team_covers(true_ctx[c], team, req)
                        succ.append(float(ok))
                        if learn:
                            update_reputation(reg, team, ok, req=req)
                finals[m].append(float(np.mean(succ)))
        payload["results"][str(q)] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                      for m, v in finals.items()}
        print(f"q={q:.2f} " + ", ".join(f"{m}={np.mean(v):.2f}±{np.std(v):.2f}" for m, v in finals.items()))
    write("context_skills", payload)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    exps = {"budget": exp_budget, "scale": exp_scale, "gaming": exp_gaming, "partial": exp_partial,
            "feedback": exp_feedback, "context": exp_context}
    for name, fn in exps.items():
        if which in (name, "all"):
            print(f"\n=== {name} ===")
            fn()
