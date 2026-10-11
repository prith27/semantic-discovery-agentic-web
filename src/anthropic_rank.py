"""LLM API baseline: NL discovery, spam rewrite, team selection.

LLM_PROVIDER picks the backend: "anthropic" (default, ANTHROPIC_API_KEY, ANTHROPIC_MODEL),
"openai" (OPENAI_API_KEY, OPENAI_MODEL, OPENAI_REASONING), or "xai" (XAI_API_KEY, XAI_MODEL).
Responses are cached under results/api_cache/.
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
XAI_URL = "https://api.x.ai/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
# Reasoning tokens count against max_completion_tokens, so short answers need headroom.
REASONING_HEADROOM = 4000


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").strip().lower()


def _key(name: str) -> str:
    key = os.getenv(name, "").strip()
    if not key:
        raise RuntimeError(f"{name} missing. Set it in .env.")
    return key


def _model() -> str:
    if provider() == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-6-luna")
    if provider() == "xai":
        return os.getenv("XAI_MODEL", "grok-4.7")
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")


def _call(prompt: str, max_tokens: int) -> str:
    if provider() == "openai":
        import httpx

        resp = httpx.post(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {_key('OPENAI_API_KEY')}"},
            json={
                "model": _model(),
                "reasoning_effort": os.getenv("OPENAI_REASONING", "low"),
                "max_completion_tokens": max_tokens + REASONING_HEADROOM,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""
    if provider() == "xai":
        import httpx

        resp = httpx.post(
            XAI_URL,
            headers={"Authorization": f"Bearer {_key('XAI_API_KEY')}"},
            json={
                "model": _model(),
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""
    import anthropic

    client = anthropic.Anthropic(api_key=_key("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=_model(),
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(getattr(b, "text", "") for b in msg.content)


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
            text = _call(prompt, max_tokens)
            if use_cache:
                _cache_set(raw_key, text)
            time.sleep(0.15)
            return text
        except Exception as e:
            last_err = e
            wait = min(60.0, 2.0 ** attempt)
            print(f"[{provider()}] attempt {attempt + 1}/{max_retries} failed: {e}; sleep {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"{provider()} API failed after {max_retries} retries: {last_err}")


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


def build_shortlist(reg: Registry, req: np.ndarray, mode: str = "cosine", candidate_pool: int = 15) -> list[int]:
    """cosine: top cards by similarity. coverage: top cards per required skill. full: every active card."""
    ids = reg.active_ids()
    if mode == "full":
        return [int(i) for i in ids]
    sims = cosine_sim(req, reg.advertised[ids])
    order = np.argsort(-sims)
    if mode == "cosine":
        return [int(ids[i]) for i in order[: min(candidate_pool, len(ids))]]
    if mode != "coverage":
        raise ValueError(f"unknown shortlist mode {mode}")
    needed = np.where(req > 0)[0]
    quota = max(3, int(np.ceil(candidate_pool / max(1, len(needed)))))
    pool: list[int] = []
    for j in needed:
        has_j = [i for i in order if reg.advertised[ids[i], j] > 0]
        for i in has_j[:quota]:
            if int(ids[i]) not in pool:
                pool.append(int(ids[i]))
    for i in order:
        if len(pool) >= candidate_pool:
            break
        if int(ids[i]) not in pool:
            pool.append(int(ids[i]))
    return pool


def select_anthropic(
    reg: Registry,
    req: np.ndarray,
    k: int,
    candidate_pool: int = 15,
    shortlist: str = "cosine",
) -> list[int]:
    """Claude reads NL cards and picks a complementary team (real API discovery)."""
    cards = ensure_nl_cards(reg)
    if len(reg.active_ids()) == 0:
        return []

    pool = build_shortlist(reg, req, mode=shortlist, candidate_pool=candidate_pool)

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
