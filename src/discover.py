"""Discovery / team-selection methods over advertised skills."""

from __future__ import annotations

import numpy as np

from .env import Registry


def _active_mask(reg: Registry) -> np.ndarray:
    return reg.alive.copy()


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Cosine similarity of row vector a (d,) with rows of b (N, d)."""
    an = np.linalg.norm(a) + eps
    bn = np.linalg.norm(b, axis=1) + eps
    return (b @ a) / (bn * an)


def select_random(reg: Registry, req: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    ids = reg.active_ids()
    if len(ids) == 0:
        return []
    pick = rng.choice(ids, size=min(k, len(ids)), replace=False)
    return list(map(int, pick))


def select_similarity(reg: Registry, req: np.ndarray, k: int) -> list[int]:
    """Top-k by cosine similarity between task req and *advertised* skills."""
    ids = reg.active_ids()
    if len(ids) == 0:
        return []
    sims = cosine_sim(req, reg.advertised[ids])
    order = np.argsort(-sims)[:k]
    return [int(ids[i]) for i in order]


def select_complementary_greedy(reg: Registry, req: np.ndarray, k: int) -> list[int]:
    """Greedy set-cover style: pick agents that cover the most remaining required skills.

    Uses advertised skills for selection (realistic open network); evaluation uses true skills.
    """
    ids = list(map(int, reg.active_ids()))
    if not ids:
        return []
    remaining = req.copy()
    team: list[int] = []
    for _ in range(k):
        if not np.any(remaining > 0):
            break
        best_i = None
        best_gain = -1.0
        for i in ids:
            if i in team:
                continue
            gain = float(np.sum((reg.advertised[i] > 0) & (remaining > 0)))
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is None or best_gain <= 0:
            break
        team.append(best_i)
        remaining = np.maximum(remaining - reg.advertised[best_i], 0)
    # If still room and uncovered, fill by similarity as weak fallback
    if len(team) < k and np.any(remaining > 0):
        extra = select_similarity(reg, remaining, k - len(team))
        for i in extra:
            if i not in team:
                team.append(i)
    return team


def select_complementary_reputation(reg: Registry, req: np.ndarray, k: int) -> list[int]:
    """Complementary cover using trust-weighted effective skills.

    effective = verified ∪ (advertised * soft trust), so unknown ads are discounted
    until outcomes attest them. Spammers lose reputation and stop dominating cover.
    """
    ids = list(map(int, reg.active_ids()))
    if not ids:
        return []
    remaining = req.copy()
    team: list[int] = []
    for _ in range(k):
        if not np.any(remaining > 0):
            break
        best_i = None
        best_score = -1.0
        for i in ids:
            if i in team:
                continue
            trust = float(reg.reputation[i])
            # Believe verified bits fully; believe unverified ads only proportional to trust.
            unverified_ads = (reg.advertised[i] > 0) & (reg.verified[i] <= 0)
            effective = reg.verified[i].copy()
            effective[unverified_ads] = trust  # soft belief in [0,1]
            gain = float(np.sum(np.minimum(effective, remaining)))
            score = gain * (0.15 + 0.85 * trust)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None or best_score <= 0:
            break
        team.append(best_i)
        # Remove covered bits using the same effective model
        trust = float(reg.reputation[best_i])
        unverified_ads = (reg.advertised[best_i] > 0) & (reg.verified[best_i] <= 0)
        effective = reg.verified[best_i].copy()
        effective[unverified_ads] = trust
        remaining = np.maximum(remaining - effective, 0)
    return team


METHODS = {
    "random": select_random,
    "similarity": select_similarity,
    "complementary": select_complementary_greedy,
    "comp+rep": select_complementary_reputation,
}
