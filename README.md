# AgenticWeb-Micro

Code for: **Semantic Discovery Is Not Enough: Complementarity Gaps and Capability Spam in Open Agent Networks**  
Target: [NeurIPS 2026 Workshop on Agentic Web](https://projectnanda.org/workshops/neurips26/)

## ELI5 — what we built

Imagine a big phonebook of AI agents. Each agent lists skills (“I can do X, Y, Z”).

A user asks for a job that needs **several** skills at once. You must pick a **small team**.

Most systems pick agents whose skill list **sounds similar** to the job (embeddings / cosine).

We show:

1. **Honest world:** similarity often picks near-duplicates and **misses complementary specialists**. A simple “cover the missing skills” picker does better.
2. **Spam world:** agents **lie / overclaim** skills. Blind cover-picking collapses (it trusts the lies). Similarity also gets worse.
3. **Fix:** cheap **capability probes** (sandbox checks) + reputation → believe verified skills, discount ads → success comes back.

That is the Agentic Web point: **discovery must be network-aware (teams + trust), not just semantic search.**

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### API key (`.env`)

Offline runs need no key. For the Claude / `--paper` run, create a `.env` file in the repo root:

```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-sonnet-5
EOF
```

- Never commit `.env` (it is gitignored).
- `ANTHROPIC_API_KEY` is required for `--paper`.
- `ANTHROPIC_MODEL` defaults to `claude-sonnet-5` if omitted.

For the second model (OpenAI GPT-6 Luna), add these lines to the same `.env`:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-6-luna
OPENAI_REASONING=low
```

`LLM_PROVIDER=xai` with `XAI_API_KEY` and `XAI_MODEL` also works for Grok.

Then:

```bash
PYTHONPATH=. python scripts/run_llm_multiseed.py --check     # test key and model
PYTHONPATH=. python scripts/run_llm_multiseed.py --dry-run   # estimate tokens
PYTHONPATH=. python scripts/run_llm_multiseed.py             # 5 seeds x {cosine, coverage, full} shortlists
PYTHONPATH=. python scripts/mcp_pilot.py                     # real MCP registry pilot
```

Results go to `results/openai_multiseed.json` and `results/mcp_pilot_openai.json`. The original Claude run stays in `results/results_paper.json`.

**Offline ablation** (numpy only, no `.env` needed):

```bash
python -m src.run_all --n-agents 300 --n-tasks 80
```

**Paper run** (Claude API = first-class baseline):

```bash
python -m src.run_all --paper
```

What `--paper` does with your Anthropic key:
1. Builds **natural-language agent marketplace cards**
2. Asks **Claude to rewrite spam listings** (adversarial capability ads)
3. Asks **Claude to pick teams** from those cards (semantic discovery baseline)
4. Compares vs similarity / complementary / probe+reputation

API calls are cached in `results/api_cache/` so re-runs are cheap.

### Resume / checkpoint
If a run dies mid-API-call, just re-run the **same** command — it resumes:

- finished methods skipped
- Claude task progress resumed mid-method
- rewritten spam cards reloaded from disk
- individual Claude responses cached

```bash
python -m src.run_all --paper --n-agents 200 --api-tasks 24 --ks 3 5 --claude-spam-max 25
# wipe and start over:
python -m src.run_all --paper --fresh --n-agents 200 --api-tasks 24 --ks 3 5 --claude-spam-max 25
```

Checkpoints live in `results/checkpoints/`.

### Multi-seed stats (paper Table 1, offline)

Reproduce mean ± std over five seeds (no API):

```bash
PYTHONPATH=. python scripts/run_multiseed.py
```

Output: `results/multiseed_paper_protocol.json`

### Stress tests (paper Section 7, offline)

Probe budget, scale and churn up to N=5000, probe gaming, graded skills, noisy feedback and collusion, and context-dependent skills (no API):

```bash
PYTHONPATH=. python scripts/stress_tests.py all        # or: budget | scale | gaming | partial | feedback | context
```

Outputs go to `results/stress_tests/`.

### LLM controls (paper tone gradient)

```bash
LLM_PROVIDER=openai PYTHONPATH=. python scripts/llm_controls.py
```

Output: `results/llm_controls_openai.json`. The MCP registry snapshot used by the pilot is `data/mcp_registry_snapshot.json`.


## Layout

- `src/env.py` — registry, spam, churn, probes
- `src/discover.py` — similarity / complementary / trust-aware cover
- `src/anthropic_rank.py` — **Claude API**: NL discovery + spam card rewrite
- `src/cards.py` — natural-language skill ontology / marketplace cards
- `src/run_all.py` — full sweep + figures (`--paper` = API mode)
- `proofs/theorem_sketch.md` — propositions for the paper
- `figures/` — plots for the paper
