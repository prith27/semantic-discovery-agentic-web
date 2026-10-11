"""Synthetic open agent registry: true skills, advertised skills, tasks, churn, spam."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Registry:
    """N agents with true skill vectors s and advertised skills s_hat."""

    true_skills: np.ndarray  # (N, d) bool/float
    advertised: np.ndarray  # (N, d)
    alive: np.ndarray  # (N,) bool
    reputation: np.ndarray  # (N,) float in [0, 1]
    # Skills believed after successful outcomes (starts empty; grows with evidence).
    verified: np.ndarray  # (N, d)
    rng: np.random.Generator = field(repr=False)
    cards: list[str] | None = None  # natural-language marketplace listings
    # Agents that pass sandbox probes on every claimed bit but keep their true skills.
    gamers: np.ndarray | None = None


    @property
    def n(self) -> int:
        return self.true_skills.shape[0]

    @property
    def d(self) -> int:
        return self.true_skills.shape[1]

    def active_ids(self) -> np.ndarray:
        return np.where(self.alive)[0]


def make_registry(
    n_agents: int,
    skill_dim: int,
    skills_per_agent: tuple[int, int] = (1, 3),
    seed: int = 0,
) -> Registry:
    rng = np.random.default_rng(seed)
    true = np.zeros((n_agents, skill_dim), dtype=np.float64)
    for i in range(n_agents):
        k = int(rng.integers(skills_per_agent[0], skills_per_agent[1] + 1))
        idx = rng.choice(skill_dim, size=k, replace=False)
        true[i, idx] = 1.0
    return Registry(
        true_skills=true,
        advertised=true.copy(),
        alive=np.ones(n_agents, dtype=bool),
        reputation=np.full(n_agents, 0.5, dtype=np.float64),
        verified=np.zeros((n_agents, skill_dim), dtype=np.float64),
        rng=rng,
    )


def apply_spam(reg: Registry, spam_frac: float, inflate: int = 3) -> None:
    """Fraction of agents overclaim by flipping off->on bits in ads."""
    n = reg.n
    n_spam = int(spam_frac * n)
    if n_spam == 0:
        return
    spammers = reg.rng.choice(n, size=n_spam, replace=False)
    for i in spammers:
        zeros = np.where(reg.true_skills[i] == 0)[0]
        if len(zeros) == 0:
            continue
        add = reg.rng.choice(zeros, size=min(inflate, len(zeros)), replace=False)
        reg.advertised[i, add] = 1.0


def reset_honest_ads(reg: Registry) -> None:
    reg.advertised = reg.true_skills.copy()


def sample_tasks(
    n_tasks: int,
    skill_dim: int,
    need_range: tuple[int, int] = (3, 5),
    seed: int = 1,
) -> np.ndarray:
    """Return (T, d) requirement vectors (binary covers)."""
    rng = np.random.default_rng(seed)
    tasks = np.zeros((n_tasks, skill_dim), dtype=np.float64)
    for t in range(n_tasks):
        k = int(rng.integers(need_range[0], need_range[1] + 1))
        idx = rng.choice(skill_dim, size=k, replace=False)
        tasks[t, idx] = 1.0
    return tasks


def apply_churn(
    reg: Registry,
    leave_frac: float = 0.1,
    join_frac: float = 0.1,
    reset_joiners: bool = False,
) -> None:
    """Randomly deactivate some alive agents and reactivate some dead ones.

    With reset_joiners, rejoining agents arrive with no trust history.
    """
    alive = reg.active_ids()
    dead = np.where(~reg.alive)[0]
    n_leave = min(len(alive), int(leave_frac * reg.n))
    n_join = min(len(dead), int(join_frac * reg.n))
    if n_leave:
        leave = reg.rng.choice(alive, size=n_leave, replace=False)
        reg.alive[leave] = False
    if n_join:
        join = reg.rng.choice(dead, size=n_join, replace=False)
        reg.alive[join] = True
        if reset_joiners:
            reg.reputation[join] = 0.5
            reg.verified[join] = 0.0


def team_covers(true_skills: np.ndarray, team: list[int], req: np.ndarray) -> bool:
    if not team:
        return False
    cover = true_skills[team].max(axis=0)
    needed = req > 0
    return bool(np.all(cover[needed] >= req[needed]))


def update_reputation(
    reg: Registry,
    team: list[int],
    success: bool,
    req: np.ndarray | None = None,
    lr: float = 0.2,
    revoke: bool = False,
) -> None:
    """Team-level outcome update (no oracle skill labels).

    With revoke, a failed team loses its attestations on the required bits, so they
    must be re-earned through later probes.
    """
    target = 1.0 if success else 0.0
    for i in team:
        reg.reputation[i] = (1 - lr) * reg.reputation[i] + lr * target
        if not success and req is not None:
            # Extra ding if the agent claimed many required skills (responsibility heuristic).
            claimed = float(np.sum((reg.advertised[i] > 0) & (req > 0)))
            reg.reputation[i] = max(0.0, reg.reputation[i] - 0.05 * claimed)
            if revoke:
                reg.verified[i][req > 0] = 0.0


def run_capability_probes(reg: Registry, n_probes: int = 40) -> None:
    """Sandbox micro-checks: verify one advertised skill bit per probed agent.

    Models Agentic-Web attestation (unit tests / sandboxed tool calls), not oracle access
    during team selection — only probe outcomes update `verified` and reputation.
    """
    ids = reg.active_ids()
    if len(ids) == 0 or n_probes <= 0:
        return
    for _ in range(n_probes):
        i = int(reg.rng.choice(ids))
        claimed = np.where(reg.advertised[i] > 0)[0]
        if len(claimed) == 0:
            continue
        bit = int(reg.rng.choice(claimed))
        gamed = reg.gamers is not None and bool(reg.gamers[i])
        if reg.true_skills[i, bit] > 0 or gamed:
            reg.verified[i, bit] = 1.0
            reg.reputation[i] = min(1.0, reg.reputation[i] + 0.05)
        else:
            # Failed probe: do not verify; punish reputation (spam signal).
            reg.reputation[i] = max(0.0, reg.reputation[i] - 0.2)
