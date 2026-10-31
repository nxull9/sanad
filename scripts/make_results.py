"""Build results/RESULTS.md from the run summaries in runs/.

Every number in the results page comes from a runs/<label>/summary.json
file, and every summary comes from per-scenario traces next to it, so each
figure can be followed back to the incident that produced it.

    python scripts/make_results.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COLUMNS = [
    ("Solved", lambda s: f"{s['successes']}/{s['scenarios']} ({s['success_rate']:.0%})"),
    ("Causes found", lambda s: pct(s["causes_diagnosed"])),
    ("Silent causes found", lambda s: pct(s["silent_causes_diagnosed"])),
    ("Fixable causes removed", lambda s: pct(s["fixable_causes_removed"])),
    ("Handed to people when needed", lambda s: pct(s["human_causes_escalated"])),
    ("Harmless change blamed", lambda s: s["herring_blamed"]),
    ("False alarms", lambda s: f"{s['false_alarms']}/{s['healthy_controls']}"),
    ("Changes (unneeded)", lambda s: f"{s['changes']} ({s['unneeded_changes']})"),
    ("Model calls", lambda s: s["model_calls"]),
    ("Cost", lambda s: f"${s['cost_usd']:.4f}"),
    ("Cost per solved", lambda s: f"${s['cost_per_success_usd']:.4f}" if s["cost_per_success_usd"] else "-"),
]

NAMES = {
    "sanad": "**Sanad**", "restart": "Restart everything (baseline)", "rollback": "Roll back latest change (baseline)",
    "no-canaries": "Sanad without canary questions", "trust-model": "Sanad trusting the model's \"fixed\"",
    "no-governor": "Sanad without the governor", "routine-only": "Sanad, cheap model only",
    "strong-only": "Sanad, stronger model only",
    "sanad-noise1": "Sanad, same faults, different noise (1)", "sanad-noise2": "Sanad, same faults, different noise (2)",
    "sanad-v1": "Sanad v1 (first version)", "sanad-v2": "**Sanad v2** (frozen, used for held-out)",
}
ORDER = ["restart", "rollback", "sanad", "sanad-noise1", "sanad-noise2", "routine-only", "strong-only",
         "no-canaries", "trust-model", "no-governor", "sanad-v1", "sanad-v2"]


def pct(x):
    return "-" if x is None else f"{x:.0%}"


def table(labels):
    rows = []
    for label in labels:
        path = ROOT / "runs" / label / "summary.json"
        if path.exists():
            variant = label.split("-", 1)[1]
            rows.append((NAMES.get(variant, variant), json.loads(path.read_text())))
    head = "| Agent | " + " | ".join(c for c, _ in COLUMNS) + " |"
    rule = "|---" * (len(COLUMNS) + 1) + "|"
    body = ["| " + name + " | " + " | ".join(str(f(s)) for _, f in COLUMNS) + " |" for name, s in rows]
    return "\n".join([head, rule, *body])


def main():
    runs = sorted(p.name for p in (ROOT / "runs").iterdir() if (p / "summary.json").exists())
    sections = ["# Results", "",
                "Built by `scripts/make_results.py` from `runs/*/summary.json`. Each run folder also holds one",
                "trace per scenario: the score, every model turn, every action and the hidden answer key.", ""]
    for folder in ("heldout", "practice"):
        labels = sorted((r for r in runs if r.startswith(folder + "-")),
                        key=lambda r: ORDER.index(r.split("-", 1)[1]) if r.split("-", 1)[1] in ORDER else 99)
        if labels:
            sections += [f"## {folder.replace('heldout', 'Held-out set (24 scenarios)').replace('practice', 'Practice set (21 scenarios)')}",
                         "", table(labels), ""]
    out = ROOT / "results" / "RESULTS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(sections))
    print(out.read_text())


if __name__ == "__main__":
    main()
