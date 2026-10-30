"""The scenario sets: the held-out set is the one that was committed, and it
keeps the promises that make a held-out score mean something."""

import hashlib
import subprocess
import sys
from pathlib import Path

from sanad.platform import scenario

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_scenarios as make  # noqa: E402


def load(folder):
    return [scenario.load(p) for p in sorted((ROOT / "scenarios" / folder).glob("*.toml"))]


def kinds(spec):
    return {f["kind"] for f in spec.get("fault", []) if f["kind"] != "benign_change"}


def test_the_committed_scenarios_are_exactly_what_the_script_makes():
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_scenarios.py"), "--check"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout


def test_no_held_out_scenario_has_changed_since_it_was_locked():
    manifest = (ROOT / "scenarios" / "heldout" / "MANIFEST.sha256").read_text().split("\n")
    locked = dict(reversed(line.split("  ")) for line in manifest if line)
    found = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (ROOT / "scenarios" / "heldout").glob("*.toml")}
    assert found == locked


def test_held_out_fault_kinds_never_appear_in_practice():
    seen = set().union(*(kinds(s) for s in load("practice")))
    assert not seen & make.HELD_OUT_KINDS
    assert make.HELD_OUT_KINDS <= set().union(*(kinds(s) for s in load("heldout")))


def test_no_two_faults_that_meet_in_held_out_ever_meet_in_practice():
    practice = set().union(*(make.pairs(kinds(s)) for s in load("practice")))
    heldout = set().union(*(make.pairs(kinds(s)) for s in load("heldout")))
    assert not practice & heldout


def test_seeds_are_never_shared():
    assert not {s["seed"] for s in load("practice")} & {s["seed"] for s in load("heldout")}


def test_both_sets_have_healthy_controls_and_held_out_goes_up_to_four_faults():
    for folder in ("practice", "heldout"):
        assert any(not kinds(s) for s in load(folder)), folder
    assert max(len(kinds(s)) for s in load("heldout")) == 4


def test_every_fault_in_every_scenario_fires():
    for folder in ("examples", "practice", "heldout"):
        for spec in load(folder):
            p = scenario.build(spec)
            last = max((f["at"] for f in spec.get("fault", [])), default=0)
            p.advance(max(0, last - p.step_no + 1))
            assert len(p._truth) == len(spec.get("fault", [])), (folder, spec["name"])
