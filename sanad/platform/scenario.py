"""Scenarios: a seed, a quiet warm-up, and the faults to inject, from a TOML file.

    name = "arabic-goes-quiet"
    seed = 7
    warmup = 30

    [[fault]]
    kind = "bad_tokenizer"
    at = 40
    target = ["replica-1", "replica-2"]
"""

import tomllib
from pathlib import Path

from sanad.platform.faults import KINDS, Fault
from sanad.platform.platform import Platform


def load(path):
    data = tomllib.loads(Path(path).read_text())
    data.setdefault("name", Path(path).stem)
    return data


def build(scenario):
    """A platform warmed up to just before the first fault, with every fault
    scheduled. Warm-up gives the agent a normal baseline to compare against."""
    platform = Platform(seed=scenario.get("seed", 0))
    for i, spec in enumerate(scenario.get("fault", [])):
        if spec["kind"] not in KINDS:
            raise ValueError(f"unknown fault kind {spec['kind']!r} in {scenario.get('name')}")
        platform.schedule(spec["at"], Fault(spec["kind"], spec["at"], list(spec.get("target", [])),
                                            dict(spec.get("params", {})), spec.get("name", f"f{i + 1}")))
    platform.advance(scenario.get("warmup", 30))
    return platform
