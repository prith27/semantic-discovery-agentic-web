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


def _greedy_cover(
    ids: np.ndarray,
    req: np.ndarray,
    k: int,
    effective: np.ndarray,
    weight: np.ndarray | None = None,
    binary_gain: bool = True,
) -> list[int]:
    """Greedy cover over rows of `effective` (aligned with `ids`).

    Ties go to the lowest agent id, matching a left-to-right scan.
    """
    remaining = req.copy()
    picked = np.zeros(len(ids), dtype=bool)
    team: list[int] = []
    for _ in range(k):
        if not np.any(remaining > 0):
            break
        if binary_gain:
            gain = ((effective > 0) & (remaining > 0)).sum(axis=1).astype(np.float64)
        else:
            gain = np.minimum(effective, remaining).sum(axis=1)
        score = gain if weight is None else gain * weight
        score = np.where(picked, -np.inf, score)
        j = int(np.argmax(score))
        if not np.isfinite(score[j]) or score[j] <= 0:
            break
        picked[j] = True
        team.append(int(ids[j]))
        remaining = np.maximum(remaining - effective[j], 0)
    return team


def select_complementary_greedy(reg: Registry, req: np.ndarray, k: int) -> list[int]:
    """Greedy set-cover style: pick agents that cover the most remaining required skills.

    Uses advertised skills for selection (realistic open network); evaluation uses true skills.
    """
    ids = reg.active_ids()
    if len(ids) == 0:
        return []
    team = _greedy_cover(ids, req, k, reg.advertised[ids])
    remaining = req.copy()
    for i in team:
        remaining = np.maximum(remaining - reg.advertised[i], 0)
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
    ids = reg.active_ids()
    if len(ids) == 0:
        return []
    trust = reg.reputation[ids]
    adv = reg.advertised[ids]
    ver = reg.verified[ids]
    # Believe verified bits fully; believe unverified ads only proportional to trust.
    effective = ver.copy()
    unverified_ads = (adv > 0) & (ver <= 0)
    effective[unverified_ads] = np.broadcast_to(trust[:, None], adv.shape)[unverified_ads]
    weight = 0.15 + 0.85 * trust
    return _greedy_cover(ids, req, k, effective, weight=weight, binary_gain=False)


def select_complementary_reputation_guarded(reg: Registry, req: np.ndarray, k: int) -> list[int]:
    """Like comp+rep, but verified bits lose weight once outcome trust falls below 0.5.

    Probe attestations then cannot outvote a record of failed teams.
    """
    ids = reg.active_ids()
    if len(ids) == 0:
        return []
    trust = reg.reputation[ids]
    adv = reg.advertised[ids]
    ver = reg.verified[ids]
    effective = ver * np.minimum(1.0, 2.0 * trust)[:, None]
    unverified_ads = (adv > 0) & (ver <= 0)
    effective[unverified_ads] = np.broadcast_to(trust[:, None], adv.shape)[unverified_ads]
    weight = 0.15 + 0.85 * trust
    return _greedy_cover(ids, req, k, effective, weight=weight, binary_gain=False)


def select_oracle_true(reg: Registry, req: np.ndarray, k: int) -> list[int]:
    """Upper bound: greedy cover using true skills (oracle knows ground truth)."""
    ids = reg.active_ids()
    if len(ids) == 0:
        return []
    return _greedy_cover(ids, req, k, reg.true_skills[ids])


METHODS = {
    "random": select_random,
    "similarity": select_similarity,
    "complementary": select_complementary_greedy,
    "comp+rep": select_complementary_reputation,
    "oracle": select_oracle_true,
}
