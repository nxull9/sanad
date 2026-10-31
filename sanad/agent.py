"""Sanad: watch, explain every symptom, fix safely, prove the fix.

The loop, once per tick:

1. Ask every serving replica the canary questions and run the detection rules
   (plain code, free). With no symptoms, that is all: no model call.
2. When symptoms persist, open an incident. A model reads the symptoms and
   the evidence and keeps a ledger of hypotheses, one per cause. Every open
   symptom must sit under some hypothesis; any that does not is shown back to
   the model as unexplained, so a second or third cause is not forgotten
   after the first is found.
3. The governor checks each proposed action. Allowed ones run.
4. After the platform settles, code (not the model) checks which of the
   hypothesis's symptoms cleared. A fix that cleared nothing is undone, and
   the hypothesis loses an attempt.
5. The incident closes only when monitoring shows no symptoms for several
   checks in a row. Causes nobody here can fix are handed to people with a
   summary; so is everything left when an allowance runs out.

The model sees one message per turn: the symptoms, the ledger and the
evidence, rebuilt from scratch. It never sees a growing chat history, so a
long incident costs about as much per turn as a short one.
"""

import json

from sanad import governor, llm, observe

ROUTINE = "qwen/qwen3.7-flash"
STRONG = "deepseek/deepseek-v4-flash"
VERIFY_STEPS = 6
QUIET_CHECKS = 2
CODE_STATUSES = {"testing", "fixed", "partly_fixed", "not_fixed", "confirmed", "escalated"}

SYSTEM = """You are Sanad, the on-call agent for an AI serving platform that answers users in Arabic and English.
Requests flow: gateway -> identity-service (owned by the identity team, not you) -> queue -> workers -> model replicas (one per GPU node) -> vector database.

Each turn you get the open symptoms (found by monitoring code), the evidence, your ledger of hypotheses, and what your last actions did. Monitoring code, not you, decides whether a fix worked.

How to work:
- Several causes can be active at once. Keep one hypothesis per cause. Every open symptom must appear in `explains` of a hypothesis that is not ruled_out.
- Back each hypothesis with concrete evidence: a metric, a log line, a change log entry, a canary answer.
- The most recent change is not always the cause. A change can only explain symptoms on what it touched, starting after it was made.
- Canary questions are asked of each replica separately. If only replicas with a certain setting answer wrongly, suspect that setting. If every replica fails the same kind of question, suspect what they share, such as the vector database. If one replica fails while its settings match healthy ones, suspect its GPU node.
- Restarting does not undo a bad deploy; rolling back does. Roll back only the replicas that carry the bad value.
- Overload: when incoming traffic is well above normal and every replica is slow or the queue backs up, add capacity with scale_out. set_weight only moves load between replicas; it adds none.
- A replica that crashed (segfault, non-zero exit) while its node shows no hardware trouble (no ECC errors, normal temperature) needs a restart. Drain a node only when the node itself shows hardware trouble.
- A symptom that follows from a cause already in your ledger (for example, lost capacity after hardware went down) belongs in that hypothesis's explains, not in a new cause.
- Change only what the evidence points at. Leave healthy components alone.
- Some causes are not yours to fix: another team's service, hardware that needs hands, power. Mitigate what you safely can, then escalate with a clear summary.
- If a fix did not clear its symptoms, revise the hypothesis rather than repeat the fix.

Actions (one to three per turn):
- rollback {"key": setting, "replicas": [...]} returns a setting to its previous value. Replica settings: model_version, template, quantization, tokenizer, server_build, memory_limit_gb (give "replicas"). Gateway: timeout_ms, log_level. Workers: max_concurrency (no "replicas").
- restart {"target": "replica-N" or "workers"}
- drain {"node": "gpu-N"} moves that node's replica to a spare node and takes the node out of service.
- set_weight {"replica": "replica-N", "weight": 0 to 1} sends less or no traffic to a replica. Undoable.
- scale_out {"count": 1 or 2} adds replicas on spare nodes.
- failover_vectordb {} switches to the standby vector database.
- rebuild_index {} rebuilds the active vector index (about 200 s).
- escalate {"team": "...", "summary": "..."} hands a cause to people.
- inspect {"what": "logs" or "metrics" or "canaries", "component": "...", "contains": "..."} reads more and changes nothing.
- wait {} changes nothing.

Reply with JSON only:
{"ledger": [{"id": "H1", "type": TYPE, "component": ["..."], "setting": "setting key or null", "explains": ["symptom keys"], "evidence": ["short facts"], "status": "suspected" or "ruled_out", "confidence": 0.0 to 1.0}],
 "actions": [{"action": "...", "args": {}, "hypothesis": "H1", "purpose": "fix" or "experiment" or "investigate" or "escalate", "why": "short reason"}],
 "summary": "one sentence"}
TYPE is one of: bad_release, bad_config, crash, deadlock, overload, gpu_overheating, gpu_silent_corruption, index_corruption, external_dependency, power_failure, unknown."""

TRUST_MODEL = """
Add "resolved": true to your reply when you believe the incident is over."""


class Sanad:
    def __init__(self, tools, budget, mode="autonomous", routine=ROUTINE, strong=STRONG, routing=True,
                 use_canaries=True, trust_model=False, governor_on=True, approve=None, say=None):
        self.tools = tools
        self.budget = budget
        self.mode = mode
        self.routine, self.strong, self.routing = routine, strong, routing
        self.use_canaries = use_canaries
        self.trust_model = trust_model
        self.governor_on = governor_on
        self.approve = approve or (lambda action, decision: True)
        self.say = say or (lambda text: None)
        self.base = observe.Baseline(tools)
        self.canaries = observe.Canaries()
        self.since = tools.step
        self.quiet_until = tools.step
        self.incident = None
        self.incidents = []
        self.previous = set()
        self.ignored = set()
        self.pending = []
        self.parked = {}


    def tick(self):
        t = self.tools
        if self.use_canaries:
            self.canaries.probe(t)
        if t.step < self.quiet_until:
            return
        symptoms = observe.detect(t, self.base, self.canaries, self.since)
        symptoms = {k: s for k, s in symptoms.items() if k not in self.ignored}
        if self.pending:
            self._verify(symptoms)
            if t.step < self.quiet_until:
                return
        inc = self.incident
        if inc is None:
            persistent = set(symptoms) & self.previous
            self.previous = set(symptoms)
            if not persistent:
                return
            inc = self._open(symptoms)
        if not symptoms:
            inc["quiet"] += 1
            if inc["quiet"] >= QUIET_CHECKS and self._restore(inc):
                return
            if inc["quiet"] >= QUIET_CHECKS:
                self._close(inc, "recovered", "monitoring shows no symptoms; recovery verified by code")
            return
        inc["quiet"] = 0
        handed = {k for h in inc["ledger"].values() if h["status"] == "escalated" for k in h["explains"]}
        active = {k: s for k, s in symptoms.items() if k not in handed}
        if not active or inc["exhausted"]:
            return
        self._turns(inc, symptoms, active)

    def finish(self):
        """End of the watch: whatever is still open is reported as it stands."""
        if self.incident is not None:
            inc = self.incident
            open_ = observe.detect(self.tools, self.base, self.canaries, self.since)
            status = "handed_to_people" if inc["escalations"] else "unresolved"
            self._close(inc, status, f"watch ended with {len(open_)} symptoms open")


    def _open(self, symptoms):
        inc = {"id": len(self.incidents) + 1, "opened": self.tools.step, "first_symptoms": sorted(symptoms),
               "ledger": {}, "actions": [], "turns": [], "notes": [], "escalations": [], "changes": 0,
               "approvals": 0, "quiet": 0, "exhausted": False, "status": "open", "closed": None}
        self.incident = inc
        self.incidents.append(inc)
        self.say(f"[step {self.tools.step}] incident {inc['id']} opened: {', '.join(sorted(symptoms))}")
        return inc

    def _close(self, inc, status, why):
        inc["status"], inc["closed"], inc["close_reason"] = status, self.tools.step, why
        self.incident = None
        self.previous = set()
        self.say(f"[step {self.tools.step}] incident {inc['id']} {status}: {why}")

    def _give_up(self, inc, symptoms, why):
        """Out of allowance: stop acting and hand everything open to a person."""
        inc["exhausted"] = True
        open_ = sorted(symptoms)
        summary = f"Sanad stopped ({why}). Still open: {', '.join(open_)}. Ledger: " + "; ".join(
            f"{h['id']} {h['type']} {h.get('setting') or ''} on {h['component']} [{h['status']}]"
            for h in inc["ledger"].values())
        inc["escalations"].append({"step": self.tools.step, "team": "on-call engineer", "summary": summary,
                                   "hypothesis": None, "automatic": True})
        self.say(f"[step {self.tools.step}] handing to on-call engineer: {why}")


    def _turns(self, inc, symptoms, active):
        extra = ""
        for _ in range(3):
            if inc["exhausted"]:
                return
            if len(inc["turns"]) >= governor.MAX_TURNS and self.governor_on:
                self._give_up(inc, symptoms, f"used all {governor.MAX_TURNS} model turns")
                return
            model, reasoning = self._route(inc, symptoms)
            prompt = self._prompt(inc, symptoms, active, extra)
            messages = [{"role": "system", "content": SYSTEM + (TRUST_MODEL if self.trust_model else "")},
                        {"role": "user", "content": prompt}]
            try:
                reply, stored = llm.chat(model, messages, self.budget, reasoning=reasoning)
            except llm.BudgetExceeded as error:
                self._give_up(inc, symptoms, f"spending cap reached: {error}")
                return
            inc["turns"].append({"step": self.tools.step, "model": model, "stored": stored,
                                 "symptoms": sorted(symptoms), "reply": reply})
            if "_unparsed" in reply:
                inc["notes"].append(f"step {self.tools.step}: reply was not valid JSON")
                return
            renamed = self._merge(inc, reply.get("ledger") or [])
            if reply.get("summary"):
                self.say(f"[step {self.tools.step}] {model.split('/')[-1]}: {reply['summary']}")
            actions = [dict(a, hypothesis=renamed.get(a.get("hypothesis"), a.get("hypothesis")))
                       for a in (reply.get("actions") or []) if isinstance(a, dict)][:governor.MAX_ACTIONS_PER_TURN]
            reads = [a for a in actions if a.get("action") == "inspect"]
            changes = [a for a in actions if a.get("action") not in ("inspect", "wait")]
            if changes:
                self._run(inc, changes, symptoms)
            if self.trust_model and reply.get("resolved"):
                self.ignored |= set(symptoms)
                self.pending = []
                self._close(inc, "recovered", "the model said the incident was over")
                return
            if changes:
                return
            if not reads:
                self.quiet_until = self.tools.step + 4
                return
            extra = self._inspect(reads)

    def _route(self, inc, symptoms):
        """The cheap model for routine turns; the stronger one when the
        incident is clearly hard: many symptoms, a fix that failed, a long
        incident, or a reply that could not be read."""
        if not self.routing:
            return self.routine, None
        failed = any(h["status"] == "not_fixed" for h in inc["ledger"].values())
        unreadable = bool(inc["turns"]) and "_unparsed" in inc["turns"][-1]["reply"]
        kinds = {k.split(":")[0] for k in symptoms}
        if len(kinds) >= 3 or failed or len(inc["turns"]) >= 4 or unreadable:
            return self.strong, None
        return self.routine, None

    def _merge(self, inc, proposed):
        """Take the model's ledger, but keep what only code may say: whether
        a fix worked, and how many attempts a cause has had.

        The model may refine a guess it has not acted on. A hypothesis that
        code has acted on is kept: if the model reuses its id for a different
        cause, the new cause gets a new id, and a cause verified as fixed or
        handed to people cannot be ruled out after the fact. Returns the ids
        it renamed, so actions in the same reply follow them."""
        renamed = {}
        proposed = [h for h in proposed if isinstance(h, dict) and h.get("id")]
        taken = set(inc["ledger"]) | {h["id"] for h in proposed}
        for h in proposed:
            old = inc["ledger"].get(h["id"], {})
            acted_on = old.get("status") in CODE_STATUSES
            if acted_on and (old["type"], old.get("setting")) != (h.get("type", "unknown"), h.get("setting")):
                n = len(taken) + 1
                while f"H{n}" in taken:
                    n += 1
                taken.add(f"H{n}")
                renamed[h["id"]], h, old, acted_on = f"H{n}", dict(h, id=f"H{n}"), {}, False
            comp = h.get("component") or []
            explains = [clean_key(k) for k in h.get("explains") or []] or old.get("explains", [])
            new = {"id": h["id"], "type": h.get("type", "unknown"),
                   "component": comp if isinstance(comp, list) else [comp],
                   "setting": h.get("setting"), "explains": explains,
                   "evidence": list(h.get("evidence") or [])[:6] or old.get("evidence", []),
                   "confidence": h.get("confidence"),
                   "attempts": old.get("attempts", 0), "history": old.get("history", [])}
            wanted = h.get("status", "suspected")
            settled = old.get("status") in {"fixed", "partly_fixed", "confirmed", "escalated", "testing"}
            if settled or (acted_on and wanted != "ruled_out"):
                new["status"] = old["status"]
            else:
                new["status"] = wanted if wanted in ("suspected", "ruled_out") else "suspected"
            inc["ledger"][h["id"]] = new
        return renamed


    def _run(self, inc, actions, symptoms):
        settle = 0
        for a in actions:
            name, args, hyp_id = a.get("action"), dict(a.get("args") or {}), a.get("hypothesis")
            record = {"step": self.tools.step, "action": name, "args": args, "hypothesis": hyp_id,
                      "purpose": a.get("purpose"), "why": a.get("why"), "ran": False}
            inc["actions"].append(record)
            decision = governor.check(name, args, hyp_id, self.tools.topology(), inc, self.mode, self.governor_on)
            record["risk"] = decision.risk
            if not decision.allowed:
                record["result"] = f"blocked: {decision.reason}"
                self.say(f"[step {self.tools.step}] governor blocked {name} {args}: {decision.reason}")
                continue
            if decision.needs_approval:
                inc["approvals"] += 1
                if not self.approve(record, decision):
                    record["result"] = "a person declined it"
                    continue
                record["approved"] = True
            hyp = inc["ledger"].get(hyp_id)
            if hyp:
                record["cause"] = {k: hyp.get(k) for k in ("type", "setting", "component")}
            if name == "escalate":
                self._escalate(inc, record, hyp, args)
                continue
            if name == "set_weight":
                before_weight = self.tools.topology()["replicas"][args["replica"]]["weight"]
            result = self.tools.act(name, **args)
            if name == "set_weight" and result["ok"]:
                self.parked.setdefault(args["replica"], before_weight)
            record["ran"], record["result"] = result["ok"], result["message"]
            self.say(f"[step {self.tools.step}] {name} {args}: {result['message']}")
            if not result["ok"]:
                continue
            inc["changes"] += 1
            settle = max(settle, governor.SETTLE.get(name, 2))
            before = sorted(set(hyp["explains"]) & set(symptoms)) if hyp else sorted(symptoms)
            if hyp:
                hyp["status"] = "testing"
            self.pending.append({"record": record, "hypothesis": hyp_id, "before": before,
                                 "undo": result.get("undo"), "purpose": a.get("purpose")})
        if settle:
            self.since = self.tools.step + settle
            self.quiet_until = self.since + VERIFY_STEPS

    def _restore(self, inc):
        """Before calling an incident over, give back any traffic Sanad took
        from a replica, then watch again. A weight left at zero after the
        fix is capacity quietly lost, and the next traffic peak finds it."""
        topo = self.tools.topology()["replicas"]
        back = {r: w for r, w in self.parked.items() if r in topo and topo[r]["weight"] != w}
        self.parked = {}
        if not back:
            return False
        for replica, weight in back.items():
            result = self.tools.act("set_weight", replica=replica, weight=weight)
            inc["actions"].append({"step": self.tools.step, "action": "set_weight",
                                   "args": {"replica": replica, "weight": weight}, "hypothesis": None,
                                   "purpose": "restore", "why": "give back traffic taken during the incident",
                                   "ran": result["ok"], "result": result["message"], "risk": "low"})
            self.say(f"[step {self.tools.step}] restoring {replica} traffic: {result['message']}")
        inc["quiet"] = 0
        self.since = self.tools.step + governor.SETTLE["set_weight"]
        self.quiet_until = self.since + VERIFY_STEPS
        return True

    def _escalate(self, inc, record, hyp, args):
        if hyp and hyp["status"] == "escalated":
            record["result"] = f"{hyp['id']} was already handed to people"
            return
        record["ran"], record["result"] = True, f"handed to {args.get('team', 'on-call')}"
        if hyp:
            hyp["status"] = "escalated"
        inc["escalations"].append({"step": self.tools.step, "team": args.get("team", "on-call engineer"),
                                   "summary": args.get("summary", ""), "hypothesis": hyp_id_of(hyp),
                                   "automatic": False})
        self.say(f"[step {self.tools.step}] escalated to {args.get('team')}: {args.get('summary', '')[:160]}")

    def _verify(self, symptoms):
        """Judge each change by what it did to the symptoms it was meant to
        clear. This is code, so a fix that did nothing cannot be called a
        success however confident the model was."""
        inc = self.incident
        reverted = False
        for p in self.pending:
            before, now = set(p["before"]), set(symptoms)
            cleared, left = sorted(before - now), sorted(before & now)
            p["record"]["verified"] = {"step": self.tools.step, "cleared": cleared, "still_open": left}
            hyp = inc["ledger"].get(p["hypothesis"]) if inc else None
            if cleared and not left:
                outcome = "confirmed" if p["purpose"] == "experiment" else "fixed"
            elif cleared:
                outcome = "partly_fixed"
            else:
                outcome = "not_fixed"
            if hyp:
                hyp["status"] = outcome
                hyp["history"].append(f"step {p['record']['step']} {p['record']['action']} -> {outcome}")
                if outcome == "not_fixed":
                    hyp["attempts"] = hyp.get("attempts", 0) + 1
            self.say(f"[step {self.tools.step}] verified {p['record']['action']}: {outcome}"
                     + (f", cleared {', '.join(cleared)}" if cleared else "") + (f", still {', '.join(left)}" if left else ""))
            if outcome == "not_fixed" and p["undo"] and self.governor_on:
                undo = dict(p["undo"])
                result = self.tools.act(undo.pop("action"), **undo)
                p["record"]["undone"] = result["message"]
                reverted = True
                self.say(f"[step {self.tools.step}] undid it: {result['message']}")
        self.pending = []
        if reverted:
            self.since = self.tools.step + 2
            self.quiet_until = self.since + 2

    def _inspect(self, reads):
        out = []
        for a in reads:
            args = a.get("args") or {}
            what, comp = args.get("what"), args.get("component")
            if what == "logs":
                lines = self.tools.logs(component=comp, contains=args.get("contains"), last=12)
                out.append(f"logs {comp or ''} {args.get('contains') or ''}:")
                out += [f"  step {e['step']} {e['level']} {e['component']}: {e['message']}" for e in lines] or ["  (none)"]
            elif what == "metrics":
                series = self.tools.metrics(comp, last=8)
                out.append(f"metrics {comp} (last 8 steps):")
                out += [f"  {c}.{m}: {[round(v, 3) for v in vals]}" for c, ms in series.items() for m, vals in ms.items()]
            elif what == "canaries" and self.canaries.rounds.get(comp):
                step, results = self.canaries.rounds[comp][-1]
                out.append(f"canaries on {comp} at step {step}:")
                out += [f"  {r['id']} {r['status']}: {str(r.get('reply', r.get('error')))[:80]}" for r in results]
            else:
                out.append(f"inspect {args}: nothing to show")
        return "\n".join(out)


    def _prompt(self, inc, symptoms, active, extra):
        t = self.tools
        explained = {k for h in inc["ledger"].values() if h["status"] != "ruled_out" for k in h["explains"]}
        lines = [f"STEP {t.step}. Incident opened at step {inc['opened']}. Each step is 10 seconds.", "",
                 "OPEN SYMPTOMS"]
        for k, s in sorted(symptoms.items()):
            tag = "(handed to people) " if k not in active else ("" if k in explained else "(UNEXPLAINED) ")
            lines.append(f"- {tag}{k}: {s.text}")
        lines += ["", "YOUR LEDGER"]
        for h in inc["ledger"].values():
            lines.append(f"- {h['id']} {h['type']} setting={h.get('setting')} on {h['component']} status={h['status']} "
                         f"explains={h['explains']} evidence={h['evidence']}"
                         + (f" history={h['history']}" if h["history"] else ""))
        if not inc["ledger"]:
            lines.append("(empty)")
        done = [a for a in inc["actions"] if a["step"] >= (inc["turns"][-2]["step"] if len(inc["turns"]) > 1 else 0)]
        if done:
            lines += ["", "YOUR RECENT ACTIONS"]
            for a in done:
                line = f"- step {a['step']} {a['action']} {json.dumps(a['args'])} for {a['hypothesis']}: {a.get('result')}"
                if a.get("verified"):
                    v = a["verified"]
                    line += f"; checked at step {v['step']}: cleared {v['cleared'] or 'nothing'}, still open {v['still_open'] or 'nothing'}"
                if a.get("undone"):
                    line += f"; undone because it cleared nothing ({a['undone']})"
                lines.append(line)
        if extra:
            lines += ["", "WHAT YOU INSPECTED", extra]
        lines += ["", observe.evidence(t, self.base, inc["opened"] - 6), "",
                  f"ALLOWANCE LEFT: {governor.MAX_TURNS - len(inc['turns'])} turns, "
                  f"{governor.MAX_CHANGES - inc['changes']} changes. Mode: {self.mode}."]
        return "\n".join(lines)


    def report(self):
        return {"incidents": self.incidents, "cost": self.budget.cost, "calls": self.budget.calls,
                "tokens_in": self.budget.tokens_in, "tokens_out": self.budget.tokens_out}


def hyp_id_of(hyp):
    return hyp["id"] if hyp else None


def clean_key(key):
    """A symptom key as monitoring wrote it, without any note the model added."""
    return str(key).replace("[", " ").split()[0].rstrip(":,") if str(key).strip() else ""
