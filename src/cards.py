"""Skill ontology + natural-language agent / task cards for LLM discovery."""

from __future__ import annotations

import numpy as np

# Fixed ontology — LLM sees names, eval still uses bit vectors.
SKILL_NAMES = [
    "web_search",
    "code_execution",
    "data_analysis",
    "document_qa",
    "image_understanding",
    "translation",
    "email_drafting",
    "calendar_scheduling",
    "sql_querying",
    "api_integration",
    "math_reasoning",
    "summarization",
]


def skill_list(vec: np.ndarray) -> list[str]:
    return [SKILL_NAMES[j] for j, v in enumerate(vec) if j < len(SKILL_NAMES) and v > 0]


def honest_card(agent_id: int, true_skills: np.ndarray) -> str:
    skills = skill_list(true_skills)
    return (
        f"Agent {agent_id} is a specialist tool agent. "
        f"Verified capabilities: {', '.join(skills) if skills else 'none'}. "
        f"Reliable for these skills only."
    )


def spam_card(agent_id: int, true_skills: np.ndarray, advertised: np.ndarray) -> str:
    """Marketing-style overclaim card (template; Claude can rewrite later)."""
    true = set(skill_list(true_skills))
    ads = skill_list(advertised)
    fake = [s for s in ads if s not in true]
    return (
        f"Agent {agent_id} is a world-class generalist AI coworker. "
        f"Capabilities include: {', '.join(ads)}. "
        f"Especially strong at {', '.join(fake[:3]) if fake else 'everything'}. "
        f"Handles end-to-end workflows with high reliability."
    )


def task_brief(req: np.ndarray) -> str:
    needed = skill_list(req)
    return (
        "User goal: complete a multi-step workflow that requires ALL of these capabilities "
        f"working together: {', '.join(needed)}."
    )


def build_cards(
    true_skills: np.ndarray,
    advertised: np.ndarray,
) -> list[str]:
    cards = []
    for i in range(true_skills.shape[0]):
        if np.array_equal(true_skills[i], advertised[i]):
            cards.append(honest_card(i, true_skills[i]))
        else:
            cards.append(spam_card(i, true_skills[i], advertised[i]))
    return cards
