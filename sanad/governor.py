"""The safety governor: code that decides whether a proposed action may run.

The language model proposes and the governor decides. A model can be talked
into things and can loop; these rules cannot. They are what keeps Sanad from
the failure the challenge names, "an AI loop that retries forever instead of
asking for help": a cause gets two attempts, an action cannot be repeated
without limit, an incident has a fixed allowance of turns and changes, and
when any of these runs out Sanad stops and hands the incident to a person.

Two modes:
- advise: every change waits for a person's approval;
- autonomous: low- and medium-risk changes run at once, high-risk ones wait.
"""

from dataclasses import dataclass

REPLICA_KEYS = {"model_version", "template", "quantization", "tokenizer", "server_build", "memory_limit_gb"}
GATEWAY_KEYS = {"timeout_ms", "log_level"}
WORKER_KEYS = {"max_concurrency"}

RISK = {
    "set_weight": "low",
    "rollback": "low",
    "failover_vectordb": "low",
    "scale_out": "low",
    "restart": "medium",
    "drain": "medium",
    "rebuild_index": "high",
}
RESTART_WORKERS_RISK = "high"

SETTLE = {"restart": 3, "rollback": 2, "drain": 4, "set_weight": 1, "scale_out": 4,
          "failover_vectordb": 1, "rebuild_index": 21}

MAX_TURNS = 12
MAX_CHANGES = 10
MAX_ATTEMPTS = 2
MAX_REPEATS = 2
MAX_ACTIONS_PER_TURN = 3


@dataclass
class Decision:
    allowed: bool
    needs_approval: bool = False
    risk: str = "none"
    reason: str = ""


def check(action, args, hypothesis, topology, incident, mode="autonomous", enforce_limits=True):
    """Whether `action` may run now, and whether a person must approve it."""
    if action in ("inspect", "wait"):
        return Decision(True, risk="none")
    if action == "escalate":
        return Decision(True, risk="none")
    if action not in RISK:
        return Decision(False, reason=f"unknown action {action!r}")
    problem = _bad_args(action, args, topology)
    if problem:
        return Decision(False, reason=problem)

    if enforce_limits:
        if incident["changes"] >= MAX_CHANGES:
            return Decision(False, reason=f"change allowance used up ({MAX_CHANGES}); escalate instead")
        same = sum(1 for a in incident["actions"] if a["action"] == action and a["args"] == args and a["ran"])
        if same >= MAX_REPEATS:
            return Decision(False, reason=f"{action} {args} already ran {same} times; it is not working, rethink or escalate")
        hyp = incident["ledger"].get(hypothesis)
        if hyp is not None and hyp.get("attempts", 0) >= MAX_ATTEMPTS:
            return Decision(False, reason=f"{hyp['id']} has had {MAX_ATTEMPTS} fixes that did not work; escalate it")

    risk = RESTART_WORKERS_RISK if action == "restart" and args.get("target") == "workers" else RISK[action]
    if _thins_fleet(action, args, topology):
        risk = "high"
    needs = mode == "advise" or risk == "high"
    return Decision(True, needs_approval=needs, risk=risk)


def _bad_args(action, args, topo):
    replicas, nodes = topo["replicas"], topo["nodes"]
    if action == "rollback":
        key = args.get("key")
        if key in GATEWAY_KEYS | WORKER_KEYS:
            return None
        if key not in REPLICA_KEYS:
            return f"no setting called {key!r}"
        unknown = [r for r in args.get("replicas") or [] if r not in replicas]
        return f"no such replicas {unknown}" if unknown else None
    if action == "restart":
        target = args.get("target")
        return None if target in replicas or target == "workers" else f"cannot restart {target!r}"
    if action == "drain":
        node = args.get("node")
        if node not in nodes or not nodes[node]["serving"]:
            return f"{node!r} is not a node with a replica on it"
        if not any(not v["serving"] and not v["drained"] for v in nodes.values()):
            return "no spare node to move the replica to"
        return None
    if action == "set_weight":
        weight = args.get("weight")
        if args.get("replica") not in replicas or not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
            return f"bad set_weight arguments {args}"
        return None
    if action == "scale_out":
        return None if args.get("count", 1) in (1, 2) else "scale_out count must be 1 or 2"
    return None


def _thins_fleet(action, args, topo):
    """True if the action would leave fewer than two replicas taking traffic
    while two or more do now: that needs a person, even in autonomous mode."""
    serving = {n for n, r in topo["replicas"].items() if r["up"] and r["weight"] > 0}
    if len(serving) < 2:
        return False
    if action == "restart" and args.get("target") in serving:
        return len(serving) - 1 < 2
    if action == "set_weight" and args.get("weight") == 0 and args.get("replica") in serving:
        return len(serving) - 1 < 2
    if action == "drain":
        moving = next((n for n, r in topo["replicas"].items() if r["node"] == args.get("node")), None)
        return moving in serving and len(serving) - 1 < 2
    return False
