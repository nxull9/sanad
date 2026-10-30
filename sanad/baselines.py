"""Two simple agents to beat. Neither uses a language model, so both are free.

They see the same symptoms Sanad sees, canaries included, so any gap between
them and Sanad comes from reasoning, not from better eyesight.

- restart: what a tired engineer does at 3 a.m. Restart every replica and
  the workers, wait, try again; after three rounds, page someone.
- rollback: blame the most recent change. Roll back the newest change made
  by someone else, wait, take the next one; after three, page someone.
"""

from sanad import observe

ROUNDS = 3
WAIT = 8


class Baseline:
    has_ledger = False
    routine = None

    def __init__(self, tools):
        self.tools = tools
        self.base = observe.Baseline(tools)
        self.canaries = observe.Canaries()
        self.since = tools.step
        self.wait_until = tools.step
        self.incidents = []
        self.incident = None
        self.previous = set()

    def tick(self):
        t = self.tools
        self.canaries.probe(t)
        if t.step < self.wait_until:
            return
        symptoms = observe.detect(t, self.base, self.canaries, self.since)
        inc = self.incident
        if inc is None:
            persistent = set(symptoms) & self.previous
            self.previous = set(symptoms)
            if not persistent:
                return
            inc = self.incident = {"opened": t.step, "actions": [], "escalations": [], "rounds": 0, "approvals": 0}
            self.incidents.append(inc)
        if not symptoms:
            self.incident, self.previous = None, set()
            return
        if inc["escalations"]:
            return
        if inc["rounds"] >= ROUNDS:
            inc["escalations"].append({"step": t.step, "team": "on-call engineer", "summary": "still broken",
                                       "hypothesis": None, "automatic": True})
            return
        inc["rounds"] += 1
        for action, args in self.plan():
            result = t.act(action, **args)
            inc["actions"].append({"step": t.step, "action": action, "args": args, "ran": result["ok"],
                                   "result": result["message"]})
        self.since = t.step + 3
        self.wait_until = self.since + WAIT

    def finish(self):
        pass

    def report(self):
        return {"incidents": self.incidents, "cost": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0}


class RestartEverything(Baseline):
    def plan(self):
        replicas = list(self.tools.topology()["replicas"])
        return [("restart", {"target": r}) for r in replicas] + [("restart", {"target": "workers"})]


class RollbackLatest(Baseline):
    def plan(self):
        done = {(a["args"].get("key")) for inc in self.incidents for a in inc["actions"]}
        theirs = [c for c in self.tools.changes() if c["author"] != "sanad" and c["key"] not in done]
        if not theirs:
            return []
        c = theirs[-1]
        args = {"key": c["key"]}
        if c["component"] == "model-server":
            args["replicas"] = c["scope"]
        return [("rollback", args)]


BASELINES = {"restart": RestartEverything, "rollback": RollbackLatest}
