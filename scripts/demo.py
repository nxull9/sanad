"""Watch Sanad handle one scenario, paced for a person to follow.

    python scripts/demo.py scenarios/examples/three-at-once.toml [--pace 1.0] [--plain]

This is a viewer, not a different agent. It drives the platform and Sanad
exactly as `sanad.evaluate.run` does, tick for tick, so Sanad makes the same
decisions and the score is the same. What it adds is for the viewer only:
time to read, the faults as they are injected (Sanad never sees these), what
a normal dashboard would say, the wrong answers the canaries catch, and the
ledger after each AI turn.

With SANAD_OFFLINE=1 the AI's replies come from the stored replies of an
earlier live run of the same scenario, so every rehearsal is identical and free.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanad import evaluate, llm  # noqa: E402
from sanad.platform import scenario  # noqa: E402
from sanad.platform.canaries import BANK  # noqa: E402
from sanad.tools import Tools  # noqa: E402

FRIENDLY = {
    "gpu_thermal": "GPU fan fails: the node overheats and slows down",
    "gpu_corruption": "GPU silently corrupts its maths: wrong answers, no error",
    "bad_template": "a chat-template release breaks structured answers",
    "bad_tokenizer": "a tokenizer release garbles Arabic answers",
    "bad_quantization": "an int4 quantization release breaks maths, worst in Arabic",
    "bad_model": "a release-candidate model loses domain knowledge",
    "memory_leak": "a serving-runtime release leaks memory",
    "low_memory_limit": "a container memory limit too small for the model",
    "bad_gateway_timeout": "the gateway timeout is cut to 800 ms",
    "bad_worker_concurrency": "worker concurrency is cut from 64 to 4",
    "worker_deadlock": "the worker dispatcher deadlocks",
    "replica_crash": "a replica crashes and stays down",
    "traffic_spike": "a campaign launch sends a traffic spike",
    "index_corruption": "a vector-index segment is corrupted",
    "identity_outage": "the identity service, owned by another team, goes down",
    "rack_power": "a rack loses power",
    "benign_change": "a harmless logging change (a trap: it is not the cause)",
}


class Screen:
    def __init__(self, pace, plain):
        self.pace, self.plain = pace, plain

    def c(self, text, code):
        return text if self.plain else f"\033[{code}m{text}\033[0m"

    def out(self, text="", wait=0.0):
        print(text, flush=True)
        if wait:
            time.sleep(wait * self.pace)

    def rule(self, title=""):
        self.out(self.c(f"── {title} " + "─" * max(0, 74 - len(title)), "2"))


def canary_counts(agent, step):
    """Right answers by language in this tick's canary round, as Sanad saw them."""
    right, asked, wrong_examples = {"en": 0, "ar": 0}, {"en": 0, "ar": 0}, []
    for replica, rounds in agent.canaries.rounds.items():
        for when, results in rounds:
            if when != step:
                continue
            for a in results:
                if a["status"] not in ("right", "wrong"):
                    continue
                asked[a["lang"]] += 1
                right[a["lang"]] += a["status"] == "right"
                if a["status"] == "wrong":
                    wrong_examples.append((replica, a))
    return right, asked, wrong_examples


def question_of(answer):
    return next(c["question"] for c in BANK if c["id"] == answer["id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("--pace", type=float, default=1.0, help="1.0 is reading speed; 0 for no pauses")
    parser.add_argument("--plain", action="store_true", help="no colour")
    parser.add_argument("--max-usd", type=float, default=0.02)
    args = parser.parse_args()
    s = Screen(args.pace, args.plain)

    spec = scenario.load(args.scenario)
    p = scenario.build(spec)
    tools = Tools(p)
    events = []
    budget = llm.Budget(args.max_usd)
    agent = evaluate.factory("sanad", budget, say=events.append)(tools)
    base_p95 = sum(p.metrics["gateway"]["latency_p95_ms"][-20:]) / 20

    s.out()
    s.out(s.c("  SANAD · سند", "1;92") + s.c("   the agentic on-call engineer for AI platforms", "2"))
    s.out(s.c(f"  scenario: {spec['name']}   ·   AI: Qwen 3.7 Flash, DeepSeek V4 Flash via OpenRouter", "2"), 1.5)
    s.rule("WHAT WE WILL BREAK  (hidden from Sanad: it sees only metrics, logs, changes, canaries)")
    for f in spec.get("fault", []):
        where = f" on {', '.join(f['target'])}" if f.get("target") else ""
        s.out(f"  {p.now(f['at']).strftime('%H:%M:%S')}  {FRIENDLY.get(f['kind'], f['kind'])}{where}", 0.8)
    s.out()
    s.rule("LIVE")

    real = [f for f in spec.get("fault", []) if f["kind"] != "benign_change"]
    first = min((f["at"] for f in real), default=None)
    end = p.step_no + evaluate.HORIZON
    recovered_at, closed_at, shown_example, injected = None, None, False, 0
    while p.step_no < end:
        p.advance(evaluate.TICK)
        agent.tick()
        fixable = [t for t in p._truth if t["kind"] in evaluate.CAUSE_TYPE and evaluate.removed(p, t) is not None]
        all_gone = all(evaluate.removed(p, t) for t in fixable) and len(p._truth) == len(spec.get("fault", []))
        if all_gone and recovered_at is None and first is not None and p.step_no > first:
            recovered_at = p.step_no
        elif not all_gone:
            recovered_at = None

        if closed_at is not None and p.step_no > closed_at + 4:
            events.clear()
            continue
        for t in p._truth[injected:]:
            tag = s.c(" SILENT ", "30;43") + " " if t.get("silent") else ""
            s.out(s.c("  ⚡ injected  ", "93") + tag + FRIENDLY.get(t["kind"], t["kind"]), 0.6)
        injected = len(p._truth)

        silent_active = any(t.get("silent") for t in p._truth)
        g = p.metrics["gateway"]
        err, p95 = g["error_rate"][-1], g["latency_p95_ms"][-1]
        right, asked, wrong = canary_counts(agent, tools.step - 1)
        if not sum(asked.values()):
            right, asked, wrong = canary_counts(agent, tools.step)
        dash_ok = err < 0.02 and p95 < 1.5 * base_p95
        dash = s.c("dashboard GREEN", "32") if dash_ok else s.c("dashboard RED  ", "31")
        canary_ok = right == asked
        can = (s.c(f"canaries EN {right['en']}/{asked['en']} AR {right['ar']}/{asked['ar']}", "32" if canary_ok else "1;31"))
        s.out(f"  {p.now(p.step_no - 1).strftime('%H:%M:%S')}  errors {err:5.1%}  p95 {p95 / 1000:4.1f}s  {dash}   {can}",
              0.9)
        # Only once a silent fault is really active and many answers are wrong:
        # a healthy model also slips about once in a hundred, and that is noise.
        if silent_active and len(wrong) >= 8 and not shown_example and dash_ok:
            # The example comes from the replica with the most wrong answers, so it
            # shows the fault, not a healthy replica's occasional slip.
            worst = max({r for r, _ in wrong}, key=lambda r: sum(x == r for x, _ in wrong))
            mine = [(r, a) for r, a in wrong if r == worst]
            replica, a = next(((r, a) for r, a in mine if a["category"] == "format"), mine[0])
            s.out(s.c("           the dashboard is green, but the model is wrong:", "1;31"), 0.6)
            s.out(f"           asked {replica}: {question_of(a)[:70]}", 0.4)
            s.out(f"           it said: {a['reply'][:70]}", 1.8)
            shown_example = True

        for line in events:
            text = line.split("] ", 1)[-1]
            if "qwen" in text or "deepseek" in text:
                model, said = text.split(": ", 1)
                s.out(s.c("   AI   ", "30;45") + s.c(f" {model}: ", "35") + said, 2.4)
            elif text.startswith(("verified", "incident", "undid", "handing", "escalated", "governor", "restoring")):
                s.out(s.c("   CODE ", "30;46") + " " + text, 1.4)
            else:
                s.out(s.c("   ACT  ", "30;44") + " " + text, 1.2)
            if "recovered" in text or "handed_to_people" in text or "unresolved" in text:
                closed_at = p.step_no
        if any("qwen" in e or "deepseek" in e for e in events) and agent.incident:
            s.out(s.c("           ledger:", "2"))
            for h in agent.incident["ledger"].values():
                where = ", ".join(h.get("component") or []) or "?"
                s.out(s.c(f"             {h['id']} {h['status']:<10} {h.get('type', '')} on {where}"
                          f"  explains {len(h.get('explains', []))} symptom(s)", "2"), 0.5)
        events.clear()

    agent.finish()
    row = evaluate.score(spec, p, agent, first, recovered_at)
    s.out()
    s.rule("RESULT  (scored against the hidden answer key, not Sanad's own report)")
    for cause in row["causes"]:
        found = "found" if cause["diagnosed"] else "missed"
        done = ("handed to people" if cause["escalated"] else "removed" if cause["removed"] else "NOT removed")
        mark = s.c("✓", "32") if cause["diagnosed"] and (cause["removed"] or cause["escalated"]) else s.c("✗", "31")
        silent = s.c(" (silent)", "33") if cause["silent"] else ""
        s.out(f"  {mark} {FRIENDLY.get(cause['fault'], cause['fault'])}{silent}: {found}, {done}", 0.6)
    s.out()
    detect = f"{row['detect_seconds']} s" if row["detect_seconds"] is not None else "n/a"
    recover = f"{row['recover_seconds']} s" if row["recover_seconds"] is not None else "n/a"
    s.out(s.c(f"  AI calls {row['model_calls']}  ·  cost ${row['cost']:.5f}  ·  changes {row['changes']}"
              f"  ·  detected in {detect}  ·  recovered in {recover}", "1"))
    how = (f"paid now ${budget.spent:.5f}" if budget.spent else
           "replayed from the stored replies of the live run, $0 paid now")
    s.out(s.c(f"  AI replies: {how}", "2"))
    s.out()


if __name__ == "__main__":
    main()
