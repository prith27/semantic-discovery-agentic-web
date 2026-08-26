"""Run AgenticWeb-Micro — paper mode uses Anthropic Claude APIs + checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from .checkpoint import CKPT_DIR, CheckpointStore
from .metrics import plot_spam_gap, plot_success_vs_k, run_regime


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-agents", type=int, default=300)
    p.add_argument("--n-tasks", type=int, default=80)
    p.add_argument("--skill-dim", type=int, default=12)
    p.add_argument("--ks", type=int, nargs="+", default=[3, 5, 8])
    p.add_argument("--spam-frac", type=float, default=0.4)
    p.add_argument(
        "--paper",
        action="store_true",
        help="Paper run: require Anthropic API, Claude discovery + spam rewrite, fewer tasks",
    )
    p.add_argument("--with-anthropic", action="store_true")
    p.add_argument("--claude-spam-rewrite", action="store_true")
    p.add_argument("--claude-spam-max", type=int, default=30)
    p.add_argument("--api-tasks", type=int, default=24, help="tasks per regime when using Claude")
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore/delete checkpoint and start from scratch",
    )
    p.add_argument(
        "--ckpt-name",
        default="paper",
        help="Checkpoint file stem under results/checkpoints/",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    fig_dir = root / "figures"
    res_dir = root / "results"
    fig_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    use_anthropic = args.paper or args.with_anthropic or bool(
        os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    if args.paper:
        if not os.getenv("ANTHROPIC_API_KEY", "").strip():
            print("ERROR: --paper requires ANTHROPIC_API_KEY in .env", file=sys.stderr)
            sys.exit(1)
        use_anthropic = True
        args.claude_spam_rewrite = True
        n_tasks = args.api_tasks
        ks = args.ks
        print(
            f"PAPER MODE: model={os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-5')} "
            f"tasks={n_tasks} ks={ks} spam_rewrite≤{args.claude_spam_max}"
        )
    else:
        n_tasks = args.n_tasks
        ks = args.ks

    # For fingerprint, store effective n_tasks
    cfg = dict(vars(args))
    cfg["n_tasks"] = n_tasks
    cfg["api_tasks"] = n_tasks if args.paper else args.api_tasks

    if args.fresh:
        target = CKPT_DIR / f"{args.ckpt_name}.json"
        if target.exists():
            target.unlink()
            print(f"[ckpt] deleted {target}")
        for cards in CKPT_DIR.glob(f"cards_{args.ckpt_name}_*.json"):
            cards.unlink()
            print(f"[ckpt] deleted {cards.name}")

    ckpt = CheckpointStore(name=args.ckpt_name, config=cfg)

    common = dict(
        n_agents=args.n_agents,
        n_tasks=n_tasks,
        skill_dim=args.skill_dim,
        ks=ks,
        use_anthropic=use_anthropic,
        seed=args.seed,
        claude_spam_rewrite=args.claude_spam_rewrite or args.paper,
        claude_spam_max=args.claude_spam_max,
        ckpt=ckpt,
    )

    try:
        print("Running honest ...")
        honest = run_regime(name="A: honest", spam_frac=0.0, churn=False, **common)
        print("Running spam ...")
        spam = run_regime(name="B: spam", spam_frac=args.spam_frac, churn=False, **common)
        print("Running churn+spam ...")
        churn = run_regime(name="C: churn+spam", spam_frac=args.spam_frac, churn=True, **common)
    except Exception as e:
        print(f"\nSTOPPED: {e}")
        print("Checkpoint kept. Re-run the SAME command to resume:")
        print("  python -m src.run_all --paper --n-agents 200 --api-tasks 24 --ks 3 5 --claude-spam-max 25")
        print("Or wipe and restart: add --fresh")
        sys.exit(1)

    payload = {"honest": honest, "spam": spam, "churn": churn, "config": vars(args)}
    out_json = res_dir / ("results_paper.json" if args.paper else "results.json")
    out_json.write_text(json.dumps(payload, indent=2))
    # Also mirror full checkpoint export
    (CKPT_DIR / f"{args.ckpt_name}_final.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json}")

    suffix = "_paper" if args.paper else ""
    plot_success_vs_k([honest, spam, churn], fig_dir / f"success_vs_k{suffix}.png")
    mid_k = ks[len(ks) // 2]
    plot_spam_gap(honest, spam, mid_k, fig_dir / f"honest_vs_spam{suffix}.png")
    print(f"Wrote figures to {fig_dir}")

    for block in (honest, spam, churn):
        print(f"\n=== {block['regime']} ===")
        for k, row in block["by_k"].items():
            bits = ", ".join(f"{m}={row[m]['success_rate']:.2f}" for m in row)
            print(f"  k={k}: {bits}")


if __name__ == "__main__":
    main()
