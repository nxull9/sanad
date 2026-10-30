"""Faults: the ways the platform breaks, and the hidden answer key for each.

A fault changes the platform's state when it fires, the way the real thing
would: a bad release is a deploy that shows in the change log, a dying fan is
not. Each one also writes its answer key to `platform._truth`: what the cause
is, what kind of fix removes it, and whether any action available to an agent
can remove it at all. Some cannot: an outage at a service another team owns,
or a power fault on a rack, must be handed to a person.

A "benign change" fires too. It is a real, harmless change near the start of
an incident, because the most recent change is not always the cause.
"""

from dataclasses import dataclass, field

from sanad.platform import catalog

SILENT = {"bad_model", "bad_template", "bad_quantization", "bad_tokenizer", "gpu_corruption", "index_corruption"}


@dataclass
class Fault:
    kind: str
    at: int
    target: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    name: str = ""

    def inject(self, platform):
        KINDS[self.kind](self, platform)

    def truth(self, platform, cause, fix, fixable=True, human=None):
        platform._truth.append({"fault": self.name or self.kind, "kind": self.kind, "step": self.at,
                                "target": list(self.target), "cause": cause, "fix": fix,
                                "fixable": fixable, "needs_human": human, "silent": self.kind in SILENT})


def _bad_release(key, label):
    def inject(fault, p):
        value, _ = catalog.BAD[key]
        scope = fault.target or list(p.replicas)
        p.deploy(key, value, scope, author=fault.params.get("author", "ci-bot"),
                 ticket=fault.params.get("ticket", "REL-2291"), note=fault.params.get("note", label))
        fault.truth(p, f"{key} {value} deployed to {', '.join(scope)}", f"rollback:{key}")
    return inject


def bad_gateway_timeout(fault, p):
    p.set_gateway("timeout_ms", catalog.GATEWAY_BAD["timeout_ms"], author="jsmith", ticket="OPS-118",
                  note="tighten timeout to protect upstream")
    fault.target = ["gateway"]
    fault.truth(p, "gateway timeout_ms cut to 800", "rollback:timeout_ms")


def bad_worker_concurrency(fault, p):
    p.set_workers("max_concurrency", catalog.WORKERS_BAD["max_concurrency"], author="ops-bot", ticket="OPS-131",
                  note="reduce concurrency during db migration")
    fault.target = ["workers"]
    fault.truth(p, "workers max_concurrency cut to 4", "rollback:max_concurrency")


def worker_deadlock(fault, p):
    p.workers_stuck = True
    fault.target = ["workers"]
    fault.truth(p, "worker dispatcher deadlocked on dispatch_mutex", "restart:workers")


def replica_crash(fault, p):
    r = p.replicas[fault.target[0]]
    r.up, r.crashed = False, True
    r.restarts += 1
    p._log("ERROR", r.name, "segfault in attention kernel; process exited with code 139")
    fault.truth(p, f"{r.name} crashed and did not come back", f"restart:{r.name}")


def traffic_spike(fault, p):
    p.load_factor = fault.params.get("factor", 1.7)
    p.spike_until = fault.at + fault.params.get("duration", 400)
    fault.target = ["gateway"]
    fault.truth(p, f"demand rose {p.load_factor}x (campaign launch)", "scale_out")


def gpu_thermal(fault, p):
    p.nodes[fault.target[0]].fan_failed = True
    fault.truth(p, f"{fault.target[0]} fan failure: thermal throttling", f"drain:{fault.target[0]}",
                human="replace the fan on the node")


def gpu_corruption(fault, p):
    p.nodes[fault.target[0]].corrupting = True
    fault.truth(p, f"{fault.target[0]} silent data corruption", f"drain:{fault.target[0]}",
                human="pull the GPU for vendor diagnostics (RMA)")


def index_corruption(fault, p):
    p.vectordb["corrupt"].add(p.vectordb["active"])
    fault.target = ["vectordb"]
    fault.truth(p, f"vectordb {p.vectordb['active']} index segment corrupted", "rebuild_index|failover_vectordb")


def identity_outage(fault, p):
    p.identity_up = False
    fault.target = ["identity-service"]
    fault.truth(p, "identity-service (owned by the identity team) is down", "escalate", fixable=False,
                human="page the identity team")


def rack_power(fault, p):
    rack = fault.params.get("rack", "rack-b")
    for n in p.nodes.values():
        if n.rack == rack:
            n.powered = False
    fault.target = [rack]
    fault.truth(p, f"power supply fault on {rack}", "escalate", fixable=False,
                human="data-centre facilities team")


def benign_change(fault, p):
    p.set_gateway("log_level", "debug", author="mfahad", ticket="OPS-140", note="more logging for audit")
    fault.target = ["gateway"]
    p._truth.append({"fault": fault.name or "benign_change", "kind": "benign_change", "step": fault.at,
                     "target": ["gateway"], "cause": None, "fix": None, "fixable": None,
                     "needs_human": None, "silent": False, "red_herring": True})


KINDS = {
    "bad_model": _bad_release("model_version", "ALLaM 2.4 release candidate"),
    "bad_template": _bad_release("template", "chat template refresh"),
    "bad_quantization": _bad_release("quantization", "int4 to halve GPU memory"),
    "bad_tokenizer": _bad_release("tokenizer", "tokenizer upgrade for new vocab"),
    "memory_leak": _bad_release("server_build", "serving runtime upgrade"),
    "low_memory_limit": _bad_release("memory_limit_gb", "right-size container memory"),
    "bad_gateway_timeout": bad_gateway_timeout,
    "bad_worker_concurrency": bad_worker_concurrency,
    "worker_deadlock": worker_deadlock,
    "replica_crash": replica_crash,
    "traffic_spike": traffic_spike,
    "gpu_thermal": gpu_thermal,
    "gpu_corruption": gpu_corruption,
    "index_corruption": index_corruption,
    "identity_outage": identity_outage,
    "rack_power": rack_power,
    "benign_change": benign_change,
}
