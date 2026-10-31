"""What Sanad notices, and the evidence it hands its model.

Detection is plain code, not a language model. It runs on every tick, costs
nothing, and turns metrics and canary answers into named symptoms such as
`wrong_answers:replica-2` or `drops:queue`. The model is called only when a
symptom has no explanation yet, so a healthy platform costs nothing to watch.

Every rule compares recent data with a baseline taken while the platform was
known to be healthy. Thresholds are set so a healthy platform almost never
trips them (tests/test_observe.py checks this), because a false alarm wakes a
person at night.
"""

import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from sanad.platform.canaries import BANK

WINDOW = 3
LONG_WINDOW = 12
CANARY_ROUNDS = 3
CANARY_MIN_WRONG = 4
NOMINAL_CLOCK = 1980.0
QUESTION = {c["id"]: c for c in BANK}
LANGS = {"ar": "Arabic", "en": "English"}


@dataclass
class Symptom:
    key: str
    component: str
    text: str


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def recent(series, since, n=WINDOW):
    """The last `n` values, never reaching back before step `since`."""
    return series[max(since, len(series) - n):] or series[-1:]


def template(entry):
    """A log line with its numbers taken out, so repeats group together."""
    return entry["component"], entry["level"], re.sub(r"\d+(\.\d+)?", "#", entry["message"])


class Baseline:
    """What normal looked like, from the steps before Sanad started watching."""

    def __init__(self, tools):
        m = tools.metrics()
        self.step = tools.step
        self.gateway = {k: mean(v) for k, v in m["gateway"].items()}
        self.queue_depth = mean(m["queue"]["depth"])
        self.dispatch = mean(m["workers"]["dispatch_rps"])
        self.vectordb_latency = mean(m["vectordb"]["latency_ms"])
        replicas = [c for c in m if c.startswith("replica-")]
        self.replica = {k: mean([mean(m[r][k]) for r in replicas])
                        for k in ("latency_p95_ms", "output_tokens_avg", "memory_pct")}
        self.templates = {template(e) for e in tools.logs(last=10**6)}


class Canaries:
    """Every tick, each serving replica answers the full canary set. Asking
    replicas one by one, not through the load balancer, is what finds a bad
    build that reached only part of the fleet."""

    def __init__(self):
        self.rounds = defaultdict(list)

    def probe(self, tools):
        for name, r in tools.topology()["replicas"].items():
            if r["up"]:
                self.rounds[name].append((tools.step, tools.canaries(replica=name)))

    def judge(self, replica, since):
        """Wrong answers by language and category over the last rounds."""
        rounds = [res for step, res in self.rounds[replica] if step >= since][-CANARY_ROUNDS:]
        wrong, asked, example = Counter(), Counter(), {}
        for results in rounds:
            for a in results:
                if a["status"] in ("right", "wrong"):
                    group = (a["lang"], a["category"])
                    asked[group] += 1
                    if a["status"] == "wrong":
                        wrong[group] += 1
                        example.setdefault(group, a)
        return wrong, asked, example


def detect(tools, base, canaries, since=0):
    """The symptoms visible now, judged only on data from step `since` on."""
    m, topo = tools.metrics(), tools.topology()
    found = []

    def add(key, component, text):
        found.append(Symptom(key, component, text))

    g = m["gateway"]
    err = mean(recent(g["error_rate"], since))
    if err > max(0.02, 4 * base.gateway["error_rate"]):
        add("errors:gateway", "gateway", f"error rate {err:.1%} (normal {base.gateway['error_rate']:.2%})")
    lat = mean(recent(g["latency_p95_ms"], since))
    if lat > 1.5 * base.gateway["latency_p95_ms"]:
        add("slow:gateway", "gateway", f"p95 latency {lat:.0f} ms (normal {base.gateway['latency_p95_ms']:.0f} ms)")
    depth = mean(recent(m["queue"]["depth"], since))
    if depth > max(500, 5 * base.queue_depth):
        add("backlog:queue", "queue", f"{depth:.0f} requests waiting (normal {base.queue_depth:.0f})")
    dropped = recent(m["queue"]["dropped"], since)
    if sum(dropped) > 0:
        add("drops:queue", "queue", f"queue full: dropped {sum(dropped):.0f} requests in the last {len(dropped)} steps")
    dispatch = mean(recent(m["workers"]["dispatch_rps"], since))
    if dispatch < 0.5 * base.dispatch:
        add("stalled:workers", "workers", f"workers dispatching {dispatch:.0f} req/s (normal {base.dispatch:.0f})")
    vlat = mean(recent(m["vectordb"]["latency_ms"], since))
    if vlat > 3 * base.vectordb_latency:
        add("slow:vectordb", "vectordb", f"lookup latency {vlat:.0f} ms (normal {base.vectordb_latency:.0f} ms)")

    for name, r in topo["replicas"].items():
        s = m.get(name)
        if not s:
            continue
        if max(recent(s["up"], since)) == 0:
            add(f"down:{name}", name, f"{name} down for the last {len(recent(s['up'], since))} steps")
        restarts = recent(s["restarts"], since, LONG_WINDOW)
        if restarts[-1] > restarts[0]:
            add(f"restarting:{name}", name,
                f"{name} restarted {restarts[-1] - restarts[0]:.0f} times in the last {len(restarts)} steps")
        mem = recent(s["memory_pct"], since, 6)
        tail = mem[max((i + 1 for i, v in enumerate(mem) if v == 0), default=0):]
        slope = (tail[-1] - tail[0]) / (len(tail) - 1) if len(tail) >= 3 else 0.0
        if slope > 0.8 or (tail and tail[-1] > 85):
            add(f"memory:{name}", name, f"{name} memory at {tail[-1]:.0f}% and rising {slope:.1f} points per step "
                                        f"(normal steady near {base.replica['memory_pct']:.0f}%)")
        if not r["up"]:
            continue
        rlat = mean([v for v in recent(s["latency_p95_ms"], since) if v > 0])
        if rlat > 1.5 * base.replica["latency_p95_ms"]:
            add(f"slow:{name}", name, f"{name} p95 latency {rlat:.0f} ms (fleet normal {base.replica['latency_p95_ms']:.0f} ms)")
        tokens = mean([v for v in recent(s["output_tokens_avg"], since) if v > 0])
        if tokens > 1.5 * base.replica["output_tokens_avg"]:
            add(f"long_answers:{name}", name, f"{name} answers average {tokens:.0f} tokens "
                                              f"(fleet normal {base.replica['output_tokens_avg']:.0f})")
        wrong, asked, example = canaries.judge(name, since)
        bad = sorted(g for g in wrong if wrong[g] >= CANARY_MIN_WRONG)
        if bad:
            add(f"wrong_answers:{name}", name, wrong_answers_text(name, bad, wrong, asked, example))

    for name, n in topo["nodes"].items():
        if n["drained"]:
            continue
        s = m[name]
        temp, clock = mean(recent(s["temp_c"], since)), mean(recent(s["sm_clock_mhz"], since))
        if temp == 0 and clock == 0:
            add(f"unreachable:{name}", name, f"{name} ({n['rack']}) sends no telemetry: temperature and clock read 0")
            continue
        if n["serving"] and (temp > 85 or clock < 0.8 * NOMINAL_CLOCK):
            add(f"hot:{name}", name, f"{name} at {temp:.0f} C, clock {clock:.0f} MHz (normal under 80 C, {NOMINAL_CLOCK:.0f} MHz)")
        ecc = recent(s["ecc_errors_total"], since, LONG_WINDOW)
        if ecc[-1] > ecc[0]:
            add(f"ecc:{name}", name, f"{name} logged {ecc[-1] - ecc[0]:.0f} new ECC errors in the last {len(ecc)} steps")
    return {s.key: s for s in found}


def wrong_answers_text(name, bad, wrong, asked, example):
    parts = [f"{LANGS[lang]} {cat} {wrong[(lang, cat)]}/{asked[(lang, cat)]}" for lang, cat in bad]
    fine = [lang for lang in LANGS if not any(g[0] == lang for g in bad)]
    text = f"{name} answers canary questions wrongly: " + ", ".join(parts)
    if fine:
        fine_wrong = sum(wrong[g] for g in asked if g[0] in fine)
        fine_asked = sum(asked[g] for g in asked if g[0] in fine)
        text += "; " + " and ".join(LANGS[lang] for lang in fine) + f" fine ({fine_wrong}/{fine_asked} wrong)"
    a = example[bad[0]]
    q = QUESTION[a["id"]]
    text += f'. Example: asked "{q["question"][:70]}", expected "{q["answer"]}", got "{a["reply"][:70]}"'
    return text


def evidence(tools, base, incident_start):
    """The facts an engineer would pull up first, kept short: the model reads
    every token of this, and pays for it."""
    topo = tools.topology()
    lines = ["PLATFORM", "path: " + " -> ".join(topo["path"])]
    configs = {n: r["config"] for n, r in topo["replicas"].items()}
    keys = list(next(iter(configs.values())))
    differ = [k for k in keys if len({c[k] for c in configs.values()}) > 1]
    for name, r in topo["replicas"].items():
        cfg = ", ".join(f"{k}={r['config'][k]}" + ("*" if k in differ else "") for k in keys)
        lines.append(f"{name}: node {r['node']}, {'up' if r['up'] else 'DOWN'}, traffic weight {r['weight']}, {cfg}")
    if differ:
        lines.append("(* = replicas differ on this setting)")
    spares = [n for n, v in topo["nodes"].items() if not v["serving"] and not v["drained"]]
    drained = [n for n, v in topo["nodes"].items() if v["drained"]]
    lines.append(f"nodes: {', '.join(f'{n} {v['rack']}' for n, v in topo['nodes'].items())}; "
                 f"spare: {', '.join(spares) or 'none'}" + (f"; drained: {', '.join(drained)}" if drained else ""))
    lines.append(f"gateway: {topo['gateway']}; workers: {topo['workers']}; vectordb: {topo['vectordb']}")
    rps = mean(recent(tools.metrics('gateway')['gateway']['rps'], 0))
    lines.append(f"incoming traffic: {rps:.0f} req/s, {rps / base.gateway['rps']:.2f}x normal")

    lines += ["", "CHANGE LOG (newest last)"]
    for c in tools.changes():
        who = "you (Sanad)" if c["author"] == "sanad" else c["author"]
        lines.append(f"step {c['step']} {c['time'][11:]}: {c['component']} {c['key']} {c['old']} -> {c['new']} on "
                     f"{', '.join(c['scope'])} by {who} ({c['ticket']}: {c['note']})")
    if not tools.changes():
        lines.append("(none)")

    lines += ["", f"LOG LINES NOT SEEN BEFORE THE INCIDENT (since step {incident_start}; count, first step, example)"]
    groups = {}
    for e in tools.logs(last=10**6):
        if e["step"] < incident_start or e["component"] == "sanad":
            continue
        t = template(e)
        if t in base.templates:
            continue
        g = groups.setdefault(t, {"count": 0, "first": e["step"], "example": e})
        g["count"] += 1
    for t, g in sorted(groups.items(), key=lambda kv: kv[1]["first"])[:15]:
        e = g["example"]
        lines.append(f"{g['count']}x from step {g['first']} {e['level']} {e['component']}: {e['message']}")
    if not groups:
        lines.append("(none)")
    return "\n".join(lines)
