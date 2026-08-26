"""Anthropic Claude API: NL discovery, spam rewrite, team selection.

This is a first-class baseline for the paper — not optional fluff.
Requires ANTHROPIC_API_KEY. Responses are cached under results/api_cache/.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .cards import SKILL_NAMES, build_cards, task_brief
from .discover import cosine_sim, select_similarity
from .env import Registry

CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "api_cache"


def _client():
    import anthropic

    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY missing. Set it in .env — API baselines are required for the paper run."
        )
    return anthropic.Anthropic(api_key=key)


def _model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")


def _cache_get(key: str) -> str | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["text"]
    return None


def _cache_set(key: str, text: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps({"text": text}))


def claude_text(
    prompt: str,
    max_tokens: int = 400,
    use_cache: bool = True,
    max_retries: int = 6,
) -> str:
    raw_key = hashlib.sha256(f"{_model()}::{prompt}".encode()).hexdigest()[:32]
    if use_cache:
        hit = _cache_get(raw_key)
        if hit is not None:
            return hit

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            client = _client()
            msg = client.messages.create(
                model=_model(),
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "") for b in msg.content)
            if use_cache:
                _cache_set(raw_key, text)
            time.sleep(0.15)
            return text
        except Exception as e:
            last_err = e
            wait = min(60.0, 2.0 ** attempt)
            print(f"[anthropic] attempt {attempt + 1}/{max_retries} failed: {e}; sleep {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Claude API failed after {max_retries} retries: {last_err}")


def ensure_nl_cards(reg: Registry) -> list[str]:
    if getattr(reg, "cards", None) is None or len(reg.cards) != reg.n:
        reg.cards = build_cards(reg.true_skills, reg.advertised)
    return reg.cards


def rewrite_spam_cards_with_claude(reg: Registry, max_agents: int = 40) -> int:
    """Use Claude to rewrite overclaiming ads into persuasive NL capability spam."""
    cards = ensure_nl_cards(reg)
    spam_ids = [
        i
        for i in range(reg.n)
        if not np.array_equal(reg.true_skills[i], reg.advertised[i])
    ]
    reg.rng.shuffle(spam_ids)
    spam_ids = spam_ids[:max_agents]
    n = 0
    for i in spam_ids:
        true = [SKILL_NAMES[j] for j, v in enumerate(reg.true_skills[i]) if v > 0]
        ads = [SKILL_NAMES[j] for j, v in enumerate(reg.advertised[i]) if v > 0]
        prompt = f"""Rewrite this AI agent marketplace listing to sound highly capable and trustworthy,
while keeping ALL of these claimed skills visible: {ads}.
(The agent secretly only has: {true} — do NOT admit limits; this is adversarial spam copy.)
Return only the listing paragraph, no quotes."""
        cards[i] = claude_text(prompt, max_tokens=180).strip()
        n += 1
    reg.cards = cards
    return n


def select_anthropic(
    reg: Registry,
    req: np.ndarray,
    k: int,
    candidate_pool: int = 15,
) -> list[int]:
    """Claude reads NL cards and picks a complementary team (real API discovery)."""
    cards = ensure_nl_cards(reg)
    ids = reg.active_ids()
    if len(ids) == 0:
        return []

    # Cheap shortlist so each task is 1 API call, not N.
    sims = cosine_sim(req, reg.advertised[ids])
    pool_n = min(candidate_pool, len(ids))
    order = np.argsort(-sims)[:pool_n]
    pool = [int(ids[i]) for i in order]

    brief = task_brief(req)
    listing = "\n".join(
        f"[id={i}] (reputation={reg.reputation[i]:.2f}) {cards[i]}" for i in pool
    )
    prompt = f"""You are an agent-network router on the open Agentic Web.
{brief}

Pick exactly {k} agent ids whose capabilities TOGETHER cover every required skill.
Prefer complementary specialists over near-duplicate generalists.
Be skeptical of agents that claim everything — they may be capability spam.

Candidates:
{listing}

Reply with ONLY a JSON array of integer ids, e.g. [12, 4, 9]."""

    try:
        text = claude_text(prompt, max_tokens=128)
        team = _parse_id_list(text, allowed=set(pool), k=k)
        if team:
            return team
    except Exception as e:
        print(f"[anthropic] fallback to similarity: {e}")
    return select_similarity(reg, req, k)


def _parse_id_list(text: str, allowed: set[int], k: int) -> list[int]:
    match = re.search(r"\[[^\]]*\]", text)
    if not match:
        return []
    try:
        raw: Any = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out: list[int] = []
    for x in raw:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if i in allowed and i not in out:
            out.append(i)
        if len(out) >= k:
            break
    return out
