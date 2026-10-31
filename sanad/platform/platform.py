"""A small AI serving platform, simulated in steps of ten seconds.

Requests arrive at the gateway, which checks each one with an identity service
owned by another team. They wait in a queue until a worker dispatches them to
one of the model replicas. Each replica runs on its own GPU node and looks
things up in a vector database. There are spare GPU nodes for scaling out and
for moving a replica off a bad node.

Everything the platform reports comes from its state: metrics, logs, a change
log, and the answers its model gives to canary questions. Faults and fixes
change that state. So a fix works because it removes a cause, never because a
script says this fix goes with that fault.

The agent sees the platform only through the methods under "Tools". What was
injected, and why, stays in `_truth`.
"""

import copy
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sanad.platform import canaries, catalog

STEP_SECONDS = 10
START = datetime(2026, 10, 30, 9, 0, 0)

REPLICA_RPS = 45.0
WORKER_SLOT_RPS = 0.5
QUEUE_LIMIT = 5000
GEN_SECONDS = 1.2
RESTART_STEPS = 2
MOVE_STEPS = 3
REBUILD_STEPS = 20


@dataclass
class Node:
    name: str
    rack: str
    fan_failed: bool = False
    corrupting: bool = False
    powered: bool = True
    drained: bool = False
    ecc_total: int = 0


@dataclass
class Replica:
    name: str
    node: str
    config: dict
    history: dict = field(default_factory=dict)
    up: bool = True
    down_steps: int = 0
    memory_pct: float = 55.0
    restarts: int = 0
    weight: float = 1.0
    crashed: bool = False


class Platform:
    def __init__(self, seed=0):
        self.seed = seed
        self.step_no = 0
        self.nodes = {f"gpu-{i}": Node(f"gpu-{i}", "rack-a" if i <= 3 else "rack-b") for i in range(1, 7)}
        self.replicas = {}
        for i in range(1, 5):
            history = {k: list(v) for k, v in catalog.HISTORY.items()}
            self.replicas[f"replica-{i}"] = Replica(f"replica-{i}", f"gpu-{i}", dict(catalog.GOOD), history)
        self.gateway = dict(catalog.GATEWAY_GOOD)
        self.gateway_history = {k: [] for k in self.gateway}
        self.workers = dict(catalog.WORKERS_GOOD)
        self.workers_history = {"max_concurrency": []}
        self.worker_count = 8
        self.workers_stuck = False
        self.workers_down_steps = 0
        self.vectordb = {"active": "primary", "corrupt": set(), "rebuild_steps": 0}
        self.identity_up = True
        self.load_factor = 1.0
        self.spike_until = None
        self.queue_depth = 0.0
        self.metrics = {}
        self.logs = []
        self.change_log = []
        self._truth = []
        self._scheduled = []
        self.actions_taken = []


    def now(self, step=None):
        return START + timedelta(seconds=STEP_SECONDS * (self.step_no if step is None else step))

    def rng(self, *parts):
        """Randomness that depends only on the seed, the step and what is
        being drawn, never on what the agent did before. So the same scenario
        gives the same noise whatever the agent tries."""
        return random.Random(":".join(str(p) for p in (self.seed, self.step_no, *parts)))

    def schedule(self, step, fault):
        self._scheduled.append((step, fault))

    def advance(self, steps=1):
        for _ in range(steps):
            for _, fault in [s for s in self._scheduled if s[0] == self.step_no]:
                fault.inject(self)
            self._tick()
            self.step_no += 1


    def defects(self, replica):
        """Defects on a replica's answer path: its own builds, its node, and
        the vector index it reads."""
        found = [catalog.DEFECT_OF[(k, v)] for k, v in replica.config.items() if (k, v) in catalog.DEFECT_OF]
        if self.nodes[replica.node].corrupting:
            found.append("silent_data_corruption")
        if self.vectordb["active"] in self.vectordb["corrupt"] and not self.vectordb["rebuild_steps"]:
            found.append("index_corruption")
        return found

    def _replica_tick(self, r):
        defects = self.defects(r)
        node = self.nodes[r.node]
        if r.down_steps:
            r.down_steps -= 1
            r.up = r.down_steps == 0 and not r.crashed
            if r.up:
                r.memory_pct = 55.0
        elif not r.crashed and "memory_limit_too_low" in defects:
            r.restarts += 1
            r.up, r.down_steps = False, RESTART_STEPS
            self._log("ERROR", r.name, f"OOMKilled while loading weights: limit {r.config['memory_limit_gb']}GB, need 68GB")
        if "memory_leak" in defects and r.up:
            r.memory_pct += 1.6 + self.rng(r.name, "leak").random() * 0.4
            if r.memory_pct >= 100:
                r.restarts += 1
                r.up, r.down_steps = False, RESTART_STEPS
                self._log("ERROR", r.name, "CUDA out of memory. Tried to allocate 2.00 GiB; process exited with code 137")
            elif r.memory_pct > 85:
                self._log("WARN", r.name, f"host memory at {r.memory_pct:.0f}% of limit")
        if not node.powered and r.up:
            r.up = False
            r.crashed = True
            self._log("ERROR", r.name, f"lost connection to {node.name}")

    def _tick(self):
        s = self.step_no
        for r in self.replicas.values():
            self._replica_tick(r)
        if self.workers_down_steps:
            self.workers_down_steps -= 1
        if self.vectordb["rebuild_steps"]:
            self.vectordb["rebuild_steps"] -= 1
            if not self.vectordb["rebuild_steps"]:
                self.vectordb["corrupt"].discard(self.vectordb["active"])
                self._log("INFO", "vectordb", "index rebuild complete, segment checksums verified")
        if self.spike_until is not None and s >= self.spike_until:
            self.load_factor, self.spike_until = 1.0, None

        demand = 120 * (1 + 0.15 * math.sin(2 * math.pi * s / 720)) * self.load_factor
        demand *= 1 + self.rng("demand").gauss(0, 0.03)

        auth_fail = 0.0 if self.identity_up else 0.62
        admitted = demand * (1 - auth_fail)

        serving = [r for r in self.replicas.values() if r.up and r.weight > 0]
        speed = {r.name: (0.4 if self.nodes[r.node].fan_failed else 1.0) for r in serving}
        replica_cap = sum(REPLICA_RPS * speed[r.name] for r in serving)
        stuck = self.workers_stuck or self.workers_down_steps
        worker_cap = 0.0 if stuck else self.worker_count * self.workers["max_concurrency"] * WORKER_SLOT_RPS
        service = min(replica_cap, worker_cap)

        self.queue_depth = max(0.0, self.queue_depth + (admitted - service) * STEP_SECONDS)
        dropped = max(0.0, self.queue_depth - QUEUE_LIMIT)
        self.queue_depth -= dropped
        wait = self.queue_depth / service if service > 0 else 60.0

        total_w = sum(r.weight for r in serving) or 1.0
        gen_of = {r.name: GEN_SECONDS / speed[r.name] for r in serving}
        gen = sum(gen_of[r.name] * r.weight for r in serving) / total_w if serving else 0.0
        tail = max((gen_of[r.name] for r in serving if r.weight / total_w >= 0.05), default=gen)
        retrieval = 0.3 if self.vectordb["rebuild_steps"] else 0.04

        timeout = self.gateway["timeout_ms"] / 1000
        if not serving:
            timed_out = 1.0
        elif wait >= timeout:
            timed_out = 1.0
        else:
            timed_out = math.exp(-(timeout - wait) / max(gen, 0.01))
        served = min(admitted, service) if service else 0.0
        crash_errors = sum(1 for r in self.replicas.values() if r.down_steps == RESTART_STEPS) * REPLICA_RPS * 0.5
        failed = demand * auth_fail + dropped / STEP_SECONDS + served * timed_out + crash_errors
        error_rate = min(1.0, failed / demand) if demand else 0.0
        p95 = (wait + 3 * tail + retrieval) * 1000

        self._metric("gateway", "rps", demand, 0.01)
        self._metric("gateway", "error_rate", error_rate, 0.0, floor=0.0005)
        self._metric("gateway", "latency_p95_ms", p95 if serving else self.gateway["timeout_ms"], 0.04)
        self._metric("queue", "depth", self.queue_depth, 0.02)
        self._metric("queue", "dropped", dropped, 0.0)
        self._metric("workers", "dispatch_rps", 0.0 if stuck else service, 0.02)
        for r in self.replicas.values():
            share = (r.weight / total_w) if r in serving else 0.0
            self._metric(r.name, "up", 1.0 if r.up else 0.0, 0.0)
            self._metric(r.name, "rps", served * share, 0.03)
            r_p95 = (wait + 3 * gen_of[r.name] + retrieval) * 1000 if r in serving else 0.0
            self._metric(r.name, "latency_p95_ms", r_p95, 0.04)
            self._metric(r.name, "memory_pct", r.memory_pct if r.up else 0.0, 0.01)
            self._metric(r.name, "restarts", r.restarts, 0.0)
            tokens = 180 * (2.4 if "tokenizer_mismatch" in self.defects(r) else 1.0)
            self._metric(r.name, "output_tokens_avg", tokens if r.up else 0.0, 0.05)
        for n in self.nodes.values():
            busy = next((r for r in self.replicas.values() if r.node == n.name and r.up), None)
            if not n.powered:
                util, temp, clock = 0.0, 0.0, 0.0
            else:
                util = 0.0 if busy is None else min(99.0, 100 * (served * (busy.weight / total_w)) / REPLICA_RPS + 20)
                temp = (91.0 if n.fan_failed else 62.0 + util * 0.12) if busy else 38.0
                clock = 1980.0 * (0.45 if n.fan_failed else 1.0)
            if n.corrupting and busy is not None:
                n.ecc_total += 1 if self.rng(n.name, "ecc").random() < 0.35 else 0
            self._metric(n.name, "util_pct", util, 0.03)
            self._metric(n.name, "temp_c", temp, 0.01)
            self._metric(n.name, "sm_clock_mhz", clock, 0.005)
            self._metric(n.name, "ecc_errors_total", n.ecc_total, 0.0)
        self._metric("vectordb", "latency_ms", retrieval * 1000, 0.05)
        self._logs_for_tick(error_rate, wait, timed_out)

    def _metric(self, component, name, value, jitter, floor=None):
        noisy = value * (1 + self.rng(component, name).gauss(0, jitter)) if jitter else value
        if floor is not None:
            noisy = max(noisy, floor * (1 + self.rng(component, name, "floor").random()))
        self.metrics.setdefault(component, {}).setdefault(name, []).append(round(max(noisy, 0.0), 4))

    def _log(self, level, component, message):
        self.logs.append({"step": self.step_no, "time": self.now().isoformat(), "level": level,
                          "component": component, "message": message})

    def _logs_for_tick(self, error_rate, wait, timed_out):
        rng = self.rng("logs")
        self._log("INFO", "gateway", f"POST /v1/chat 200 ({int(1000 * (wait + GEN_SECONDS))}ms)")
        if not self.identity_up and rng.random() < 0.8:
            self._log("ERROR", "gateway", "503 from identity-service: connection refused (owner: identity team)")
        if timed_out > 0.05 and self.identity_up:
            self._log("ERROR", "gateway", f"upstream timeout after {self.gateway['timeout_ms']}ms")
        if self.queue_depth > 1000:
            self._log("WARN", "queue", f"backlog {int(self.queue_depth)} requests, oldest waiting {wait:.0f}s")
        if self.workers_stuck and rng.random() < 0.6:
            self._log("WARN", "workers", "dispatcher: 0 tasks dispatched in last 10s; waiting on dispatch_mutex")
        for n in self.nodes.values():
            if n.fan_failed and n.powered and rng.random() < 0.5:
                self._log("WARN", n.name, "GPU temperature 91C, clocks throttled (HW slowdown: thermal)")
            if n.corrupting and n.powered and rng.random() < 0.25:
                self._log("WARN", n.name, "Xid 94: contained ECC error, application may continue")
            if not n.powered and rng.random() < 0.5:
                self._log("ERROR", n.name, f"node unreachable; BMC reports PSU fault on {n.rack}")
        if self.vectordb["active"] in self.vectordb["corrupt"] and rng.random() < 0.12:
            self._log("WARN", "vectordb", "segment 7 checksum mismatch; serving cached segment")
        if rng.random() < 0.15:
            self._log("WARN", "gateway", f"slow request {3 + rng.random() * 3:.1f}s for tenant t-{rng.randint(100, 999)}")
        if rng.random() < 0.1:
            self._log("INFO", "autoscaler", "no scaling action: within target utilisation")


    def record_change(self, component, key, old, new, scope, author, ticket, note):
        self.change_log.append({"step": self.step_no, "time": self.now().isoformat(), "component": component,
                                "key": key, "old": old, "new": new, "scope": scope, "author": author,
                                "ticket": ticket, "note": note})

    def deploy(self, key, value, scope, author="ci-bot", ticket="REL-0", note=""):
        """Change a replica setting on some replicas, as a release would."""
        old = None
        for name in scope:
            r = self.replicas[name]
            old = r.config[key]
            if old != value:
                r.history.setdefault(key, []).append(old)
                r.config[key] = value
                r.memory_pct = 55.0
        self.record_change("model-server", key, old, value, list(scope), author, ticket, note)

    def set_gateway(self, key, value, author="ops", ticket="OPS-0", note=""):
        old = self.gateway[key]
        self.gateway_history[key].append(old)
        self.gateway[key] = value
        self.record_change("gateway", key, old, value, ["gateway"], author, ticket, note)

    def set_workers(self, key, value, author="ops", ticket="OPS-0", note=""):
        old = self.workers[key]
        self.workers_history[key].append(old)
        self.workers[key] = value
        self.record_change("workers", key, old, value, ["workers"], author, ticket, note)


    def topology(self):
        return {
            "path": ["gateway", "identity-service (external, owned by identity team)", "queue", "workers",
                     "replicas", "vectordb"],
            "replicas": {r.name: {"node": r.node, "up": r.up, "weight": r.weight, "config": dict(r.config)}
                         for r in self.replicas.values()},
            "nodes": {n.name: {"rack": n.rack, "drained": n.drained,
                               "serving": next((r.name for r in self.replicas.values() if r.node == n.name), None)}
                      for n in self.nodes.values()},
            "gateway": dict(self.gateway), "workers": dict(self.workers, count=self.worker_count),
            "vectordb": {"active": self.vectordb["active"], "rebuilding": bool(self.vectordb["rebuild_steps"])},
            "step": self.step_no, "time": self.now().isoformat(),
        }

    def get_metrics(self, component=None, last=30):
        chosen = self.metrics if component is None else {component: self.metrics.get(component, {})}
        return {c: {m: v[-last:] for m, v in series.items()} for c, series in chosen.items()}

    def get_logs(self, component=None, level=None, contains=None, last=200):
        out = [e for e in self.logs if (component is None or e["component"] == component)
               and (level is None or e["level"] == level)
               and (contains is None or contains.lower() in e["message"].lower())]
        return copy.deepcopy(out[-last:])

    def get_changes(self, since_step=0):
        return [dict(c) for c in self.change_log if c["step"] >= since_step]

    def ask(self, canary, replica=None):
        """Send one canary question through the platform. With no replica
        named, it is routed like real traffic."""
        rng = self.rng("canary", canary["id"], replica or "routed", len(self.actions_taken))
        if not self.identity_up and rng.random() < 0.62:
            return {"id": canary["id"], "status": "error", "error": "503 identity-service", "replica": None}
        serving = [r for r in self.replicas.values() if r.up and (replica is None and r.weight > 0 or r.name == replica)]
        if not serving:
            return {"id": canary["id"], "status": "error", "error": "no healthy replica", "replica": replica}
        if replica is None:
            pick = rng.random() * sum(r.weight for r in serving)
            for r in serving:
                pick -= r.weight
                if pick <= 0:
                    break
        else:
            r = serving[0]
        reply = canaries.answer(canary, self.defects(r), rng)
        ok = canaries.grade(canary, reply)
        return {"id": canary["id"], "lang": canary["lang"], "category": canary["category"],
                "status": "right" if ok else "wrong", "replica": r.name, "reply": reply}

    def run_canaries(self, lang=None, category=None, replica=None):
        return [self.ask(c, replica) for c in canaries.select(lang, category)]


    def act(self, action, **args):
        handler = getattr(self, f"_do_{action}", None)
        if handler is None:
            return {"ok": False, "message": f"unknown action {action!r}"}
        result = handler(**args)
        self.actions_taken.append({"step": self.step_no, "action": action, "args": args, "ok": result["ok"]})
        if result["ok"]:
            self._log("INFO", "sanad", f"action {action} {args}: {result['message']}")
        return result

    def _do_restart(self, target):
        if target in self.replicas:
            r = self.replicas[target]
            r.up, r.down_steps, r.crashed = False, RESTART_STEPS, False
            r.restarts += 1
            return {"ok": True, "message": f"{target} restarting", "undo": None}
        if target == "workers":
            self.workers_stuck = False
            self.workers_down_steps = RESTART_STEPS
            return {"ok": True, "message": "worker pool restarting", "undo": None}
        return {"ok": False, "message": f"cannot restart {target!r}"}

    def _do_rollback(self, key, replicas=None):
        if key in self.gateway:
            if not self.gateway_history[key]:
                return {"ok": False, "message": f"no earlier gateway {key} to roll back to"}
            value = self.gateway_history[key].pop()
            old = self.gateway[key]
            self.gateway[key] = value
            self.record_change("gateway", key, old, value, ["gateway"], "sanad", "INC", "rollback")
            return {"ok": True, "message": f"gateway {key} {old} -> {value}", "undo": {"action": "set_gateway", "key": key, "value": old}}
        if key in self.workers:
            if not self.workers_history[key]:
                return {"ok": False, "message": f"no earlier workers {key} to roll back to"}
            value = self.workers_history[key].pop()
            old = self.workers[key]
            self.workers[key] = value
            self.record_change("workers", key, old, value, ["workers"], "sanad", "INC", "rollback")
            return {"ok": True, "message": f"workers {key} {old} -> {value}", "undo": {"action": "set_workers", "key": key, "value": old}}
        scope = replicas or list(self.replicas)
        undo = {}
        for name in scope:
            r = self.replicas.get(name)
            if r is None or key not in r.config:
                return {"ok": False, "message": f"no {key} on {name}"}
            if not r.history.get(key):
                continue
            undo[name] = r.config[key]
            r.config[key] = r.history[key].pop()
            r.memory_pct = 55.0
        if not undo:
            return {"ok": False, "message": f"nothing earlier to roll {key} back to on {scope}"}
        self.record_change("model-server", key, next(iter(undo.values())), self.replicas[next(iter(undo))].config[key],
                           list(undo), "sanad", "INC", "rollback")
        return {"ok": True, "message": f"rolled back {key} on {sorted(undo)}",
                "undo": {"action": "set_replica", "key": key, "values": undo}}

    def _do_set_replica(self, key, values):
        for name, value in values.items():
            r = self.replicas[name]
            r.history.setdefault(key, []).append(r.config[key])
            r.config[key] = value
            r.memory_pct = 55.0
        return {"ok": True, "message": f"set {key} on {sorted(values)}", "undo": None}

    def _do_set_gateway(self, key, value):
        self.set_gateway(key, value, author="sanad", ticket="INC", note="undo")
        return {"ok": True, "message": f"gateway {key} = {value}", "undo": None}

    def _do_set_workers(self, key, value):
        self.set_workers(key, value, author="sanad", ticket="INC", note="undo")
        return {"ok": True, "message": f"workers {key} = {value}", "undo": None}

    def _do_set_weight(self, replica, weight):
        r = self.replicas.get(replica)
        if r is None or not 0 <= weight <= 1:
            return {"ok": False, "message": f"bad weight change {replica}={weight}"}
        old, r.weight = r.weight, weight
        return {"ok": True, "message": f"{replica} traffic weight {old} -> {weight}",
                "undo": {"action": "set_weight", "replica": replica, "weight": old}}

    def _spare(self):
        used = {r.node for r in self.replicas.values()}
        return next((n for n in self.nodes.values() if n.name not in used and n.powered and not n.drained), None)

    def _do_drain(self, node):
        n = self.nodes.get(node)
        r = next((r for r in self.replicas.values() if r.node == node), None)
        if n is None or r is None:
            return {"ok": False, "message": f"nothing running on {node!r}"}
        spare = self._spare()
        if spare is None:
            return {"ok": False, "message": "no spare GPU node to move to"}
        n.drained = True
        r.node = spare.name
        r.up, r.down_steps, r.crashed = False, MOVE_STEPS, False
        return {"ok": True, "message": f"{r.name} moving {node} -> {spare.name}; {node} drained",
                "undo": {"action": "undrain", "node": node, "replica": r.name}}

    def _do_undrain(self, node, replica):
        n, r = self.nodes[node], self.replicas[replica]
        n.drained = False
        r.node = node
        r.up, r.down_steps = False, MOVE_STEPS
        return {"ok": True, "message": f"{replica} moved back to {node}", "undo": None}

    def _do_scale_out(self, count=1):
        added = []
        for _ in range(count):
            spare = self._spare()
            if spare is None:
                break
            template = next(iter(self.replicas.values()))
            name = f"replica-{len(self.replicas) + 1}"
            self.replicas[name] = Replica(name, spare.name, dict(template.config),
                                          {k: list(v) for k, v in template.history.items()},
                                          up=False, down_steps=MOVE_STEPS)
            added.append(name)
        if not added:
            return {"ok": False, "message": "no spare GPU node to scale onto"}
        return {"ok": True, "message": f"added {added}", "undo": {"action": "scale_in", "replicas": added}}

    def _do_scale_in(self, replicas):
        for name in replicas:
            self.replicas.pop(name, None)
        return {"ok": True, "message": f"removed {replicas}", "undo": None}

    def _do_rebuild_index(self):
        if self.vectordb["rebuild_steps"]:
            return {"ok": False, "message": "index rebuild already running"}
        self.vectordb["rebuild_steps"] = REBUILD_STEPS
        return {"ok": True, "message": f"rebuilding {self.vectordb['active']} index (about {REBUILD_STEPS * STEP_SECONDS}s)", "undo": None}

    def _do_failover_vectordb(self):
        old = self.vectordb["active"]
        self.vectordb["active"] = "standby" if old == "primary" else "primary"
        return {"ok": True, "message": f"vectordb {old} -> {self.vectordb['active']}",
                "undo": {"action": "failover_vectordb"}}
