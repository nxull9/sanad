"""The simulated platform: faults show the symptoms they should, and only a
fix that removes the cause brings it back."""

from pathlib import Path

from sanad.platform import Platform, canaries, scenario
from sanad.platform.faults import Fault


def run(kind, target=(), at=30, until=60, seed=1, **params):
    p = Platform(seed=seed)
    p.schedule(at, Fault(kind, at, list(target), params))
    p.advance(until)
    return p


def right(p, **which):
    found = p.run_canaries(**which)
    return sum(c["status"] == "right" for c in found), len(found)


def last(p, component, metric):
    return p.metrics[component][metric][-1]


def test_the_same_seed_gives_the_same_platform():
    a, b = run("gpu_thermal", ["gpu-2"]), run("gpu_thermal", ["gpu-2"])
    assert a.metrics == b.metrics and a.logs == b.logs
    assert run("gpu_thermal", ["gpu-2"], seed=2).metrics != a.metrics


def test_a_healthy_platform_rarely_misses_a_canary_and_has_few_errors():
    # A healthy model still slips about 1 time in 100: an agent must not
    # call a single miss an incident.
    missed = 0
    for seed in range(10):
        p = Platform(seed=seed)
        p.advance(60)
        missed += 24 - right(p)[0]
        assert max(p.metrics["gateway"]["error_rate"]) < 0.01
    assert 0 < missed <= 8


def test_silent_faults_leave_errors_and_latency_normal_but_canaries_catch_them():
    healthy = Platform(seed=1)
    healthy.advance(60)
    for kind in ["bad_model", "bad_template", "bad_quantization", "bad_tokenizer", "index_corruption"]:
        p = run(kind)
        assert max(p.metrics["gateway"]["error_rate"][30:]) < 0.01, kind
        assert last(p, "gateway", "latency_p95_ms") < 1.2 * last(healthy, "gateway", "latency_p95_ms"), kind
        assert right(p)[0] <= 18, kind


def test_a_tokenizer_mismatch_garbles_arabic_and_spares_english():
    en = ar = 0
    for seed in range(10):
        p = run("bad_tokenizer", seed=seed)
        en += right(p, lang="en")[0]
        ar += right(p, lang="ar")[0]
    assert en >= 105 and ar <= 50          # out of 120 each
    assert any("Ø" in c["reply"] for c in p.run_canaries(lang="ar") if c["status"] == "wrong")


def test_bad_quantization_hurts_arabic_more_than_english():
    en = ar = 0
    for seed in range(10):
        p = run("bad_quantization", seed=seed)
        en += right(p, lang="en")[0]
        ar += right(p, lang="ar")[0]
    assert ar < en


def test_a_bad_release_is_in_the_change_log_and_a_dying_fan_is_not():
    assert [c["key"] for c in run("bad_model").get_changes()] == ["model_version"]
    assert run("gpu_thermal", ["gpu-2"]).get_changes() == []


def test_restarting_does_not_fix_a_bad_release_but_rolling_it_back_does():
    p = run("bad_model")
    for r in list(p.replicas):
        p.act("restart", target=r)
    p.advance(5)
    assert right(p)[0] <= 18
    assert p.act("rollback", key="model_version")["ok"]
    p.advance(5)
    assert right(p) == (24, 24)


def test_a_partial_rollout_can_be_found_by_asking_each_replica():
    p = run("bad_tokenizer", ["replica-2"])
    by_replica = {r: right(p, lang="ar", replica=r)[0] for r in p.replicas}
    assert by_replica["replica-2"] <= 7
    assert all(n >= 11 for r, n in by_replica.items() if r != "replica-2")


def test_draining_a_throttled_gpu_brings_latency_back():
    p = run("gpu_thermal", ["gpu-2"])
    slow = last(p, "gateway", "latency_p95_ms")
    result = p.act("drain", node="gpu-2")
    assert result["ok"] and result["undo"]["action"] == "undrain"
    p.advance(6)
    assert last(p, "gateway", "latency_p95_ms") < slow / 2


def test_moving_traffic_off_a_suspect_replica_is_an_experiment_that_can_be_undone():
    p = run("gpu_corruption", ["gpu-3"])
    shift = p.act("set_weight", replica="replica-3", weight=0.0)
    p.advance(1)
    assert right(p) == (24, 24)          # routed traffic now avoids the bad GPU
    p.act(**shift["undo"])
    p.advance(1)
    assert p.replicas["replica-3"].weight == 1.0


def test_a_deadlock_needs_a_restart_and_a_spike_needs_more_replicas():
    p = run("worker_deadlock")
    assert last(p, "gateway", "error_rate") > 0.5
    p.act("restart", target="workers")
    p.advance(40)
    assert last(p, "gateway", "error_rate") < 0.01

    p = run("traffic_spike", factor=1.7)
    assert last(p, "queue", "depth") > 1000
    assert p.act("scale_out", count=2)["ok"]
    p.advance(60)
    assert last(p, "queue", "depth") < 100


def test_nothing_the_agent_can_do_brings_back_an_outage_at_another_teams_service():
    p = run("identity_outage")
    for r in list(p.replicas):
        p.act("restart", target=r)
    p.act("restart", target="workers")
    p.act("scale_out", count=2)
    p.advance(20)
    assert last(p, "gateway", "error_rate") > 0.5
    assert p._truth[0]["fixable"] is False


def test_index_corruption_is_fixed_by_a_rebuild_after_it_finishes():
    p = run("index_corruption")
    assert right(p, category="retrieval")[0] <= 2
    p.act("rebuild_index")
    p.advance(25)
    assert right(p, category="retrieval") == (6, 6)


def test_canaries_are_graded_by_meaning_not_wording():
    capital = next(c for c in canaries.BANK if c["id"] == "fact-capital-ar")
    leap = next(c for c in canaries.BANK if c["id"] == "fact-leap-ar")
    city = next(c for c in canaries.BANK if c["id"] == "format-city-en")
    assert canaries.grade(capital, "العاصمة هي الرِّياض")          # diacritics do not matter
    assert canaries.grade(leap, "366 يوماً") and canaries.grade(leap, "٣٦٦")
    assert not canaries.grade(leap, "3660")
    assert canaries.grade(city, '{"city": "riyadh"}') and not canaries.grade(city, "Riyadh")
    assert not canaries.grade(capital, canaries.garble(capital["reply"]))


def test_every_example_scenario_loads_and_fires_its_faults():
    for path in sorted(Path("scenarios/examples").glob("*.toml")):
        spec = scenario.load(path)
        p = scenario.build(spec)
        p.advance(20)
        assert len(p._truth) == len(spec["fault"]), path.name
