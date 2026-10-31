"""Write the practice and held-out scenario sets.

The files this writes are committed, and the held-out set is committed before
any agent code exists, so git history shows Sanad was never tuned on it.
Running this again writes the same files byte for byte; `--check` confirms
the committed files are exactly what this script makes.

How the held-out set differs from practice, so a held-out score means something:

- seeds: practice uses 100 and up, held-out 1000 and up;
- fault kinds: three kinds (HELD_OUT_KINDS) never appear in practice;
- combinations: no two faults that share a held-out scenario ever share a
  practice one;
- difficulty: held-out has more faults at once (up to four), more faults that
  arrive late, after an agent may think it is done, and stronger traffic spikes;
- both sets have healthy controls, where the right answer is "nothing is wrong".

    python -m scripts.make_scenarios          # write scenarios/practice and scenarios/heldout
    python -m scripts.make_scenarios --check  # fail if the committed files differ
"""

import hashlib
import itertools
import random
import sys
from pathlib import Path

from sanad.platform import scenario
from sanad.platform.faults import KINDS, SILENT

ROOT = Path(__file__).resolve().parent.parent

REPLICAS = [f"replica-{i}" for i in range(1, 5)]
NODES = [f"gpu-{i}" for i in range(1, 5)]
RELEASES = {"bad_model", "bad_template", "bad_tokenizer", "bad_quantization", "memory_leak", "low_memory_limit"}
NODE_FAULTS = {"gpu_thermal", "gpu_corruption"}
HUMAN_ONLY = {"identity_outage", "rack_power"}
FAULT_KINDS = sorted(set(KINDS) - {"benign_change"})
HELD_OUT_KINDS = {"bad_quantization", "low_memory_limit", "index_corruption"}
SEEN_KINDS = [k for k in FAULT_KINDS if k not in HELD_OUT_KINDS]
SPARES = 2
WARMUP = 30


def compatible(kinds):
    """Combinations whose answer key is unambiguous and whose fixes fit the
    spare capacity: two spare nodes, shared by draining and scaling out."""
    kinds = set(kinds)
    if "rack_power" in kinds and kinds & (NODE_FAULTS | {"replica_crash", "traffic_spike"}):
        return False
    if "traffic_spike" in kinds and kinds & NODE_FAULTS:
        return False
    return len(kinds & NODE_FAULTS) <= SPARES


def pairs(kinds):
    return {frozenset(p) for p in itertools.combinations(sorted(kinds), 2)}


def fault_spec(kind, rng, used_nodes, spike_factors):
    spec = {"kind": kind}
    if kind in RELEASES and rng.random() < 0.5:
        spec["target"] = sorted(rng.sample(REPLICAS, rng.randint(1, 3)))
    elif kind in NODE_FAULTS:
        node = rng.choice([n for n in NODES if n not in used_nodes])
        used_nodes.add(node)
        spec["target"] = [node]
    elif kind == "replica_crash":
        spec["target"] = [rng.choice([r for r, n in zip(REPLICAS, NODES) if n not in used_nodes])]
    elif kind == "rack_power":
        spec["params"] = {"rack": rng.choice(["rack-a", "rack-b"])}
    elif kind == "traffic_spike":
        spec["params"] = {"factor": rng.choice(spike_factors)}
    return spec


def make(name, seed, kinds, rng, herring=False, late=False, spike_factors=(1.7,)):
    """One scenario. Node faults pick their nodes first, so a crash is never
    put on a node that is also failing."""
    kinds = sorted(kinds, key=lambda k: (k not in NODE_FAULTS, k))
    used = set()
    faults = [fault_spec(k, rng, used, spike_factors) for k in kinds]
    rng.shuffle(faults)
    at = WARMUP + rng.randint(4, 8)
    for i, f in enumerate(faults):
        if i:
            at += rng.randint(30, 50) if late and i == len(faults) - 1 else rng.randint(1, 8)
        f["at"] = at
    if herring:
        first = faults[0]["at"] if faults else WARMUP + rng.randint(4, 8)
        faults.insert(0, {"kind": "benign_change", "at": first - rng.randint(1, 3)})
    return {"name": name, "seed": seed, "faults": faults, "late": late}


def pick(rng, pool, size, banned_pairs, must=None):
    """A compatible combination of `size` kinds from `pool` that shares no
    pair with `banned_pairs`, including `must` if given."""
    for _ in range(10000):
        combo = set(rng.sample(sorted(pool), size))
        if must is not None and not combo & must:
            continue
        if compatible(combo) and not pairs(combo) & banned_pairs:
            return combo
    raise RuntimeError(f"no combination of {size} from {sorted(pool)}")


def practice_set():
    rng = random.Random(1)
    seeds = itertools.count(100)
    out = [make("p01", next(seeds), [], rng), make("p02", next(seeds), [], rng, herring=True)]
    for kind in SEEN_KINDS:
        out.append(make(f"p{len(out) + 1:02d}", next(seeds), [kind], rng, herring=rng.random() < 0.4,
                        spike_factors=(1.5, 1.6, 1.7)))
    for size, late in [(2, False)] * 4 + [(3, False), (2, True)]:
        combo = pick(rng, SEEN_KINDS, size, set())
        out.append(make(f"p{len(out) + 1:02d}", next(seeds), combo, rng, herring=rng.random() < 0.5, late=late,
                        spike_factors=(1.5, 1.6, 1.7)))
    return out


def heldout_set(practice):
    rng = random.Random(2)
    seeds = itertools.count(1000)
    banned = set().union(*(pairs(f["kind"] for f in s["faults"] if f["kind"] != "benign_change") for s in practice))
    factors = (1.6, 1.7, 1.8, 1.9)
    fixable = set(FAULT_KINDS) - HUMAN_ONLY
    plan = [((), False, False), ((), False, False), ((), True, False), ((), True, False)]
    plan += [((k,), rng.random() < 0.5, False) for k in sorted(HELD_OUT_KINDS)]
    plan += [((k,), rng.random() < 0.5, False) for k in rng.sample(SEEN_KINDS, 3)]
    combos = []
    for human in sorted(HUMAN_ONLY):
        combos.append((pick(rng, fixable | {human}, 2, banned, must={human}), False))
    for i in range(6):
        combos.append((pick(rng, FAULT_KINDS, 2, banned, must=HELD_OUT_KINDS if i < 3 else None), i >= 4))
    for i in range(4):
        combos.append((pick(rng, FAULT_KINDS, 3, banned, must=HELD_OUT_KINDS if i < 2 else None), i == 3))
    for _ in range(2):
        combos.append((pick(rng, FAULT_KINDS, 4, banned), False))
    plan += [(tuple(sorted(c)), rng.random() < 0.5, late) for c, late in combos]
    out = []
    for kinds, herring, late in plan:
        if not compatible(kinds) or pairs(kinds) & banned:
            raise RuntimeError(f"held-out combination {kinds} breaks the split rules")
        out.append(make(f"h{len(out) + 1:02d}", next(seeds), kinds, rng, herring, late, factors))
    return out


def describe(f):
    where = f" on {', '.join(f['target'])}" if f.get("target") else ""
    extra = "".join(f" {k}={v}" for k, v in f.get("params", {}).items())
    tags = []
    if f["kind"] in SILENT:
        tags.append("silent")
    if f["kind"] in HUMAN_ONLY:
        tags.append("needs a person")
    if f["kind"] == "benign_change":
        tags.append("red herring")
    return f"step {f['at']}: {f['kind']}{where}{extra}" + (f" ({', '.join(tags)})" if tags else "")


def toml(s):
    real = [f for f in s["faults"] if f["kind"] != "benign_change"]
    lines = [f"# {len(real)} fault{'s' if len(real) != 1 else ''}" + (", the last arriving late" if s["late"] else "")
             + ("" if real else ": a healthy control, the right answer is that nothing needs fixing")]
    lines += [f"#   {describe(f)}" for f in s["faults"]]
    lines += ["", f'name = "{s["name"]}"', f"seed = {s['seed']}", f"warmup = {WARMUP}"]
    for i, f in enumerate(s["faults"], 1):
        lines += ["", "[[fault]]", f'name = "f{i}"', f'kind = "{f["kind"]}"', f"at = {f['at']}"]
        if f.get("target"):
            lines.append("target = [" + ", ".join(f'"{t}"' for t in f["target"]) + "]")
        if f.get("params"):
            lines.append("[fault.params]")
            lines += [f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}" for k, v in f["params"].items()]
    return "\n".join(lines) + "\n"


def validate(path):
    """Every fault in the file fires, and writes its answer key."""
    spec = scenario.load(path)
    p = scenario.build(spec)
    last = max((f["at"] for f in spec.get("fault", [])), default=WARMUP)
    p.advance(max(0, last - p.step_no + 1))
    assert len(p._truth) == len(spec.get("fault", [])), path


def files():
    practice = practice_set()
    heldout = heldout_set(practice)
    out = {}
    for folder, chosen in [("practice", practice), ("heldout", heldout)]:
        for s in chosen:
            out[ROOT / "scenarios" / folder / f"{s['name']}.toml"] = toml(s)
    manifest = "".join(f"{hashlib.sha256(text.encode()).hexdigest()}  {path.name}\n"
                       for path, text in out.items() if path.parent.name == "heldout")
    out[ROOT / "scenarios" / "heldout" / "MANIFEST.sha256"] = manifest
    return out


def main():
    wanted = files()
    if "--check" in sys.argv:
        stale = [p for p, text in wanted.items() if not p.exists() or p.read_text() != text]
        extra = [p for folder in ("practice", "heldout") for p in (ROOT / "scenarios" / folder).glob("*")
                 if p not in wanted]
        for p in stale + extra:
            print(f"differs: {p.relative_to(ROOT)}")
        sys.exit(1 if stale or extra else 0)
    for path, text in wanted.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for path in wanted:
        if path.suffix == ".toml":
            validate(path)
    print(f"wrote {sum(p.parent.name == 'practice' for p in wanted)} practice and "
          f"{sum(p.suffix == '.toml' and p.parent.name == 'heldout' for p in wanted)} held-out scenarios")


if __name__ == "__main__":
    main()
