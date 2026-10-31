"""Command line.

    python -m sanad run scenarios/examples/three-at-once.toml [--mode advise]
    python -m sanad demo scenarios/examples/three-at-once.toml [--pace 1.5]
    python -m sanad eval practice [--variant sanad|restart|rollback|no-canaries|...] [--max-usd 0.05]

Set SANAD_OFFLINE=1 to use only stored model replies: every published result
rebuilds that way without an API key and at no cost.
"""

import argparse
import json

from sanad import demo, evaluate, llm


def ask_person(record, decision):
    answer = input(f"\n  APPROVAL NEEDED ({decision.risk} risk): {record['action']} {record['args']}\n"
                   f"  why: {record.get('why')}\n  approve? [y/N] ")
    return answer.strip().lower().startswith("y")


def with_approval(make):
    """The same agent, but a person at the keyboard approves risky actions."""
    def build(tools):
        agent = make(tools)
        agent.approve = ask_person
        return agent
    return build


def main():
    parser = argparse.ArgumentParser(prog="sanad")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run", help="watch one scenario and narrate")
    one.add_argument("scenario")
    one.add_argument("--mode", choices=["autonomous", "advise"], default="autonomous")
    one.add_argument("--ask", action="store_true", help="ask at the keyboard before risky actions")
    one.add_argument("--max-usd", type=float, default=0.02)
    paced = sub.add_parser("demo", help="watch one scenario, paced for a person to follow")
    paced.add_argument("scenario")
    paced.add_argument("--pace", type=float, default=1.0, help="1.0 is reading speed, 0 for no pauses")
    paced.add_argument("--plain", action="store_true", help="no colour")
    paced.add_argument("--max-usd", type=float, default=0.02)
    many = sub.add_parser("eval", help="score a variant on a set of scenarios")
    many.add_argument("folder", choices=["examples", "practice", "heldout"])
    many.add_argument("--variant", default="sanad", choices=sorted(evaluate.VARIANTS) + ["restart", "rollback"])
    many.add_argument("--max-usd", type=float, default=0.05)
    many.add_argument("--label")
    many.add_argument("--only", nargs="*")
    many.add_argument("--seed-offset", type=int, default=0, help="same faults, different noise")
    args = parser.parse_args()

    if args.command == "run":
        budget = llm.Budget(args.max_usd)
        variant = "advise" if args.mode == "advise" else "sanad"
        make = evaluate.factory(variant, budget, say=print)
        if args.ask:
            make = with_approval(make)
        spec, p, agent, row = evaluate.run(args.scenario, make)
        print("\nSCORE (against the hidden answer key)")
        print(json.dumps({k: row[k] for k in ("success", "causes", "changes", "model_calls", "cost",
                                              "detect_seconds", "recover_seconds")}, indent=1, default=str))
    elif args.command == "demo":
        demo.show(args.scenario, args.pace, args.plain, args.max_usd)
    else:
        summary, _ = evaluate.evaluate(args.folder, args.variant, args.max_usd, args.label, args.only,
                                       seed_offset=args.seed_offset)
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
