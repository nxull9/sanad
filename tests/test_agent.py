"""Detection, the governor and the agent's rules, with a scripted model in
place of a real one: these run offline and cost nothing."""

import pytest

from sanad import agent as agent_module
from sanad import governor, llm, observe
from sanad.agent import Sanad, clean_key
from sanad.platform import Platform
from sanad.platform.faults import Fault
from sanad.tools import Tools


def platform(*faults, seed=3):
    p = Platform(seed=seed)
    for kind, at, target in faults:
        p.schedule(at, Fault(kind, at, list(target)))
    p.advance(30)
    return p


def watch(p, sanad, steps):
    for _ in range(steps // 2):
        p.advance(2)
        sanad.tick()


@pytest.fixture
def scripted(monkeypatch):
    """A model that replies with the next reply in a list."""
    replies, calls = [], []

    def chat(model, messages, budget, **_):
        calls.append(messages[-1]["content"])
        return (replies.pop(0) if replies else {"ledger": [], "actions": [{"action": "wait"}]}), True
    monkeypatch.setattr(agent_module.llm, "chat", chat)
    return replies, calls


def test_a_healthy_platform_raises_no_symptom_for_a_long_watch():
    for seed in range(5):
        p = platform(seed=seed)
        t = Tools(p)
        base, can = observe.Baseline(t), observe.Canaries()
        for _ in range(100):
            p.advance(2)
            can.probe(t)
            assert observe.detect(t, base, can, since=30) == {}, (seed, p.step_no)


def test_silent_faults_become_wrong_answer_symptoms_on_the_right_replicas():
    p = platform(("bad_tokenizer", 31, ["replica-3"]))
    t = Tools(p)
    base, can = observe.Baseline(t), observe.Canaries()
    for _ in range(6):
        p.advance(2)
        can.probe(t)
    found = observe.detect(t, base, can, since=31)
    assert "wrong_answers:replica-3" in found and "Arabic" in found["wrong_answers:replica-3"].text
    assert not any(k.startswith("wrong_answers:") and k != "wrong_answers:replica-3" for k in found)


def test_watching_a_healthy_platform_never_calls_the_model(scripted):
    _, calls = scripted
    p = platform()
    sanad = Sanad(Tools(p), llm.Budget(0.0))
    watch(p, sanad, 120)
    assert calls == [] and sanad.incidents == []


def test_a_fix_is_judged_by_code_and_a_useless_one_is_undone(scripted):
    replies, _ = scripted
    p = platform(("bad_tokenizer", 31, ["replica-1", "replica-2"]))
    replies += [
        {"ledger": [{"id": "H1", "type": "bad_release", "setting": "template", "component": ["replica-1"],
                     "explains": ["wrong_answers:replica-1", "wrong_answers:replica-2"]}],
         "actions": [{"action": "rollback", "args": {"key": "template", "replicas": ["replica-1", "replica-2"]},
                      "hypothesis": "H1", "purpose": "fix"}]},
        {"ledger": [{"id": "H2", "type": "bad_release", "setting": "tokenizer", "component": ["replica-1"],
                     "explains": ["wrong_answers:replica-1", "wrong_answers:replica-2"]}],
         "actions": [{"action": "rollback", "args": {"key": "tokenizer", "replicas": ["replica-1", "replica-2"]},
                      "hypothesis": "H2", "purpose": "fix"}]},
    ]
    sanad = Sanad(Tools(p), llm.Budget(0.0))
    watch(p, sanad, 80)
    inc = sanad.incidents[0]
    assert inc["ledger"]["H1"]["status"] == "not_fixed" and inc["ledger"]["H1"]["attempts"] == 1
    assert p.replicas["replica-1"].config["template"] == "chat-tpl-12"
    assert inc["ledger"]["H2"]["status"] == "fixed"
    assert inc["status"] == "recovered"


def test_a_cause_handed_to_people_stops_further_turns(scripted):
    replies, calls = scripted
    p = platform(("identity_outage", 31, []))
    replies.append({"ledger": [{"id": "H1", "type": "external_dependency", "component": ["identity-service"],
                                "explains": ["errors:gateway [UNEXPLAINED]"]}],
                    "actions": [{"action": "escalate", "args": {"team": "identity team", "summary": "503s"},
                                 "hypothesis": "H1", "purpose": "escalate"}]})
    sanad = Sanad(Tools(p), llm.Budget(0.0))
    watch(p, sanad, 100)
    assert len(calls) == 1
    assert sanad.incidents[0]["ledger"]["H1"]["status"] == "escalated"


def test_a_verified_cause_cannot_be_overwritten_or_ruled_out(scripted):
    p = platform()
    sanad = Sanad(Tools(p), llm.Budget(0.0))
    inc = sanad._open({})
    inc["ledger"]["H1"] = {"id": "H1", "type": "gpu_overheating", "setting": None, "component": ["gpu-2"],
                           "explains": ["hot:gpu-2"], "evidence": [], "status": "fixed", "attempts": 0, "history": []}
    renamed = sanad._merge(inc, [{"id": "H1", "type": "gpu_overheating", "status": "ruled_out", "explains": []},
                                 {"id": "H1", "type": "bad_release", "setting": "template", "explains": ["x"]}])
    assert inc["ledger"]["H1"]["status"] == "fixed" and inc["ledger"]["H1"]["explains"] == ["hot:gpu-2"]
    assert renamed == {"H1": "H2"} and inc["ledger"]["H2"]["setting"] == "template"


def test_the_governor_stops_repeats_and_spent_causes_and_guards_the_fleet():
    p = platform()
    topo = Tools(p).topology()
    inc = {"changes": 0, "actions": [], "ledger": {"H1": {"id": "H1", "attempts": 2}}}
    assert not governor.check("rollback", {"key": "tokenizer"}, "H1", topo, inc).allowed
    twice = [{"action": "restart", "args": {"target": "replica-1"}, "ran": True}] * 2
    assert not governor.check("restart", {"target": "replica-1"}, None, topo, dict(inc, actions=twice)).allowed
    assert not governor.check("restart", {"target": "gpu-9"}, None, topo, inc).allowed
    assert governor.check("restart", {"target": "workers"}, None, topo, inc).needs_approval
    assert not governor.check("set_weight", {"replica": "replica-1", "weight": 0}, None, topo, inc).needs_approval
    assert governor.check("set_weight", {"replica": "replica-1", "weight": 0}, None, topo, inc, "advise").needs_approval
    for r in ("replica-2", "replica-3"):
        p.replicas[r].weight = 0
    assert governor.check("restart", {"target": "replica-1"}, None, Tools(p).topology(), inc).needs_approval


def test_symptom_keys_are_cleaned_of_notes_the_model_adds():
    assert clean_key("errors:gateway [UNEXPLAINED]") == "errors:gateway"
    assert clean_key("wrong_answers:replica-2: Arabic") == "wrong_answers:replica-2"
