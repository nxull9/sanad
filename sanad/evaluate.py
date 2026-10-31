"""Run an agent on scenarios and score it against the hidden answer key.

The score comes from the platform's state, never from what the agent says
about itself: a bad build counts as removed only if no replica taking traffic
still runs it, a bad GPU only if no replica is left on it, and so on.

A scenario counts as a success only if all of this holds:
- healthy control: no incident was opened;
- otherwise: every cause the agent could fix is gone, every cause it could
  not fix was handed to a person, and the harmless change was not blamed.
"""

import json
import math
from pathlib import Path

from sanad import llm
from sanad.agent import Sanad
from sanad.baselines import BASELINES
from sanad.platform import catalog, scenario
from sanad.platform.platform import REPLICA_RPS, WORKER_SLOT_RPS
from sanad.tools import Tools

ROOT = Path(__file__).resolve().parent.parent
HORIZON = 170
TICK = 2

RELEASE_KEY = {"bad_model": "model_version", "bad_template": "template", "bad_tokenizer": "tokenizer",
               "bad_quantization": "quantization", "memory_leak": "server_build",
               "low_memory_limit": "memory_limit_gb"}
CAUSE_TYPE = {**{k: "bad_release" for k in RELEASE_KEY},
              "bad_gateway_timeout": "bad_config", "bad_worker_concurrency": "bad_config",
              "worker_deadlock": "deadlock", "replica_crash": "crash", "traffic_spike": "overload",
              "gpu_thermal": "gpu_overheating", "gpu_corruption": "gpu_silent_corruption",
              "index_corruption": "index_corruption", "identity_outage": "external_dependency",
              "rack_power": "power_failure"}
CONFIG_KEY = {**RELEASE_KEY, "bad_gateway_timeout": "timeout_ms", "bad_worker_concurrency": "max_concurrency"}
ANY_COMPONENT = {"traffic_spike", "worker_deadlock", "index_corruption", "identity_outage",
                 "bad_gateway_timeout", "bad_worker_concurrency"}


def removed(p, t):
    """Whether the cause behind answer-key entry `t` is gone from the platform."""
    kind, target = t["kind"], t["target"]
    serving = [r for r in p.replicas.values() if r.weight > 0]
    if kind in RELEASE_KEY:
        bad = catalog.BAD[RELEASE_KEY[kind]][0]
        return not any(r.config[RELEASE_KEY[kind]] == bad for r in serving)
    if kind == "bad_gateway_timeout":
        return p.gateway["timeout_ms"] != catalog.GATEWAY_BAD["timeout_ms"]
    if kind == "bad_worker_concurrency":
        return p.workers["max_concurrency"] != catalog.WORKERS_BAD["max_concurrency"]
    if kind == "worker_deadlock":
        return not p.workers_stuck
    if kind == "replica_crash":
        r = p.replicas.get(target[0])
        return r is None or r.up or r.weight == 0
    if kind in ("gpu_thermal", "gpu_corruption"):
        return not any(r.node == target[0] for r in serving)
    if kind == "index_corruption":
        return p.vectordb["active"] not in p.vectordb["corrupt"]
    if kind == "traffic_spike":
        if p.spike_until is None:
            return True
        demand = 120 * (1 + 0.15 * math.sin(2 * math.pi * p.step_no / 720)) * p.load_factor
        replica_cap = sum(REPLICA_RPS * (0.4 if p.nodes[r.node].fan_failed else 1.0)
                          for r in serving if r.up)
        worker_cap = p.worker_count * p.workers["max_concurrency"] * WORKER_SLOT_RPS
        return min(replica_cap, worker_cap) >= demand
    return None


def matches(h, t, p):
    """Whether hypothesis `h` names the cause in answer-key entry `t`."""
    kind = t["kind"]
    if h.get("status") == "ruled_out" or h.get("type") != CAUSE_TYPE.get(kind):
        if not (kind in CONFIG_KEY and h.get("setting") == CONFIG_KEY[kind]):
            return False
    if kind in CONFIG_KEY:
        return h.get("setting") == CONFIG_KEY[kind]
    if kind in ANY_COMPONENT:
        return True
    names = {str(c) for c in h.get("component") or []}
    targets = set(t["target"])
    if kind in ("gpu_thermal", "gpu_corruption"):
        targets |= {f"replica-{t['target'][0].split('-')[1]}"}
    if kind == "replica_crash":
        targets |= {f"gpu-{t['target'][0].split('-')[1]}"}
    if kind == "rack_power":
        targets |= {n for n, node in p.nodes.items() if node.rack == t["target"][0]}
    return bool(names & targets)


def run(path, make_agent, horizon=HORIZON, tick=TICK, seed_offset=0):
    """Watch one scenario. `seed_offset` keeps the faults and their timing
    but changes every random draw: metric noise, log noise, canary luck."""
    spec = scenario.load(path)
    spec["seed"] = spec.get("seed", 0) + seed_offset
    p = scenario.build(spec)
    tools = Tools(p)
    agent = make_agent(tools)
    real = [f for f in spec.get("fault", []) if f["kind"] != "benign_change"]
    first = min((f["at"] for f in real), default=None)
    end = p.step_no + horizon
    recovered_at = None
    while p.step_no < end:
        p.advance(tick)
        agent.tick()
        fixable = [t for t in p._truth if t["kind"] in CAUSE_TYPE and removed(p, t) is not None]
        all_gone = all(removed(p, t) for t in fixable) and len(p._truth) == len(spec.get("fault", []))
        if all_gone and recovered_at is None and first is not None and p.step_no > first:
            recovered_at = p.step_no
        elif not all_gone:
            recovered_at = None
    agent.finish()
    return spec, p, agent, score(spec, p, agent, first, recovered_at)


def score(spec, p, agent, first, recovered_at):
    report = agent.report()
    incidents = report["incidents"]
    ledger = [h for inc in incidents for h in inc.get("ledger", {}).values()]
    truth = [t for t in p._truth if t["kind"] != "benign_change"]
    actions = [a for inc in incidents for a in inc.get("actions", [])]
    ran = [a for a in actions if a.get("ran") and a["action"] not in ("escalate",)]
    causes = []
    for t in truth:
        gone = removed(p, t)
        hits = [h for h in ledger if matches(h, t, p)]
        causes.append({"fault": t["kind"], "target": t["target"], "silent": t["silent"],
                       "fixable": gone is not None, "removed": gone,
                       "diagnosed": bool(hits) if agent.has_ledger else None,
                       "escalated": any(h["status"] == "escalated" for h in hits)
                       or (not agent.has_ledger and any(inc["escalations"] for inc in incidents))})
    herring = any(t["kind"] == "benign_change" for t in p._truth)
    blamed = herring and (any(a["action"] == "rollback" and a["args"].get("key") == "log_level" for a in ran)
                          or any(h.get("setting") == "log_level" and h["status"] != "ruled_out" for h in ledger))
    unneeded = [a for a in ran if agent.has_ledger and a.get("purpose") != "restore" and not (
        a.get("cause") and any(matches(acted_cause(a), t, p) for t in truth))]
    detected = [inc["opened"] for inc in incidents]
    if not truth:
        success = not incidents
    else:
        success = (all(c["removed"] for c in causes if c["fixable"])
                   and all(c["escalated"] for c in causes if not c["fixable"]) and not blamed)
    over_escalated = sum(1 for c in causes if c["fixable"] and c["escalated"] and agent.has_ledger)
    return {
        "scenario": spec["name"], "faults": len(truth), "success": success,
        "false_alarm": not truth and bool(incidents),
        "detected": bool(truth) and bool(detected),
        "detect_seconds": (min(detected) - first) * 10 if truth and detected else None,
        "recover_seconds": (recovered_at - first) * 10 if recovered_at and first else None,
        "causes": causes, "herring_blamed": blamed, "over_escalated": over_escalated,
        "changes": len(ran), "unneeded_changes": len(unneeded),
        "blocked": sum(1 for a in actions if str(a.get("result", "")).startswith("blocked")),
        "undone": sum(1 for a in actions if a.get("undone")),
        "approvals": sum(inc.get("approvals", 0) for inc in incidents),
        "model_calls": sum(len(inc.get("turns", [])) for inc in incidents),
        "strong_calls": sum(1 for inc in incidents for tr in inc.get("turns", []) if tr["model"] != agent_routine(agent)),
        "cost": round(report.get("cost", 0.0), 6),
        "tokens_in": report.get("tokens_in", 0), "tokens_out": report.get("tokens_out", 0),
        "escalations": [e for inc in incidents for e in inc.get("escalations", [])],
    }


def acted_cause(a):
    """The cause an action was taken for. A rollback names its setting in
    its own arguments even when the model left the ledger's field blank."""
    cause = dict(a["cause"], status="acted")
    if a["action"] == "rollback" and not cause.get("setting"):
        cause["setting"] = a["args"].get("key")
    return cause


def agent_routine(agent):
    return getattr(agent, "routine", None)


VARIANTS = {
    "sanad": {},
    "no-canaries": {"use_canaries": False},
    "trust-model": {"trust_model": True},
    "no-governor": {"governor_on": False},
    "routine-only": {"routing": False},
    "strong-only": {"routing": False, "routine": "deepseek/deepseek-v4-flash"},
    "advise": {"mode": "advise"},
}


def factory(variant, budget, say=None):
    if variant in BASELINES:
        return lambda tools: BASELINES[variant](tools)

    def make(tools):
        agent = Sanad(tools, budget, say=say, **VARIANTS[variant])
        agent.has_ledger = True
        return agent
    return make


def evaluate(folder, variant="sanad", max_usd=0.05, label=None, only=None, say=None, seed_offset=0):
    budget = llm.Budget(max_usd)
    out = ROOT / "runs" / (label or f"{folder}-{variant}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    paths = sorted((ROOT / "scenarios" / folder).glob("*.toml"))
    for path in paths:
        if only and path.stem not in only:
            continue
        spent = budget.cost
        spec, p, agent, row = run(path, factory(variant, budget, say), seed_offset=seed_offset)
        row["cost"] = round(budget.cost - spent, 6)
        rows.append(row)
        trace = {"score": row, "report": agent.report(), "truth": p._truth, "changes": p.change_log}
        (out / f"{path.stem}.json").write_text(json.dumps(trace, indent=1, ensure_ascii=False, default=str))
        mark = "PASS" if row["success"] else "FAIL"
        print(f"{path.stem} {mark} faults={row['faults']} changes={row['changes']} calls={row['model_calls']} "
              f"cost=${row['cost']:.4f} recover={row['recover_seconds']}s", flush=True)
    summary = summarise(rows, budget)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary, rows


def summarise(rows, budget):
    faulty = [r for r in rows if r["faults"]]
    healthy = [r for r in rows if not r["faults"]]
    causes = [c for r in rows for c in r["causes"]]
    fixable = [c for c in causes if c["fixable"]]
    human = [c for c in causes if not c["fixable"]]
    diagnosed = [c for c in causes if c["diagnosed"] is not None]
    silent = [c for c in diagnosed if c["silent"]]
    wins = sum(r["success"] for r in rows)
    recover = [r["recover_seconds"] for r in faulty if r["recover_seconds"] is not None]
    detect = [r["detect_seconds"] for r in faulty if r["detect_seconds"] is not None]

    def share(xs, test):
        return round(sum(1 for x in xs if test(x)) / len(xs), 3) if xs else None
    return {
        "scenarios": len(rows), "success_rate": round(wins / len(rows), 3) if rows else None, "successes": wins,
        "detected": share(faulty, lambda r: r["detected"]), "false_alarms": sum(r["false_alarm"] for r in healthy),
        "healthy_controls": len(healthy),
        "causes_diagnosed": share(diagnosed, lambda c: c["diagnosed"]),
        "silent_causes_diagnosed": share(silent, lambda c: c["diagnosed"]),
        "fixable_causes_removed": share(fixable, lambda c: c["removed"]),
        "human_causes_escalated": share(human, lambda c: c["escalated"]),
        "over_escalated": sum(r["over_escalated"] for r in rows),
        "herring_blamed": sum(r["herring_blamed"] for r in rows),
        "median_detect_seconds": sorted(detect)[len(detect) // 2] if detect else None,
        "median_recover_seconds": sorted(recover)[len(recover) // 2] if recover else None,
        "changes": sum(r["changes"] for r in rows), "unneeded_changes": sum(r["unneeded_changes"] for r in rows),
        "undone": sum(r["undone"] for r in rows), "blocked": sum(r["blocked"] for r in rows),
        "approvals": sum(r["approvals"] for r in rows),
        "model_calls": sum(r["model_calls"] for r in rows), "strong_calls": sum(r["strong_calls"] for r in rows),
        "cost_usd": round(budget.cost, 5), "paid_now_usd": round(budget.spent, 5),
        "cost_per_success_usd": round(budget.cost / wins, 5) if wins else None,
        "tokens_in": budget.tokens_in, "tokens_out": budget.tokens_out,
    }
