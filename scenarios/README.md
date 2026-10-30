# Scenarios

A scenario is a seed, a quiet warm-up and a list of faults to inject into the
simulated platform. Sanad never reads these files. It sees the platform only
through its tools, the way an on-call engineer would. The comment at the top
of each file is the answer key, for people.

| Folder | What it is for |
|---|---|
| `examples/` | Three hand-written scenarios used to explain the project. |
| `practice/` | 21 scenarios for building and tuning Sanad. |
| `heldout/` | 24 scenarios used only to report results. These were committed before any agent code existed. |

`practice/` and `heldout/` are written by
[`scripts/make_scenarios.py`](../scripts/make_scenarios.py), which produces
the same files on every run. To confirm the committed files are exactly what
it makes, run:

```bash
python scripts/make_scenarios.py --check
```

## Why the held-out score means something

The held-out set is harder than practice and different from it in ways
tuning cannot reach:

- **New fault kinds.** Quantization damage, a memory limit set too low and a
  corrupted vector index never appear in practice. Sanad has to diagnose
  them from evidence, not from having seen them before.
- **New combinations.** No two faults that share a held-out scenario ever
  share a practice one.
- **More at once.** Held-out scenarios have up to four simultaneous faults,
  against three in practice.
- **Faults that arrive late.** In some scenarios a fault lands 30 to 50 steps
  after the others, when an agent may already believe it has finished.
- **Stronger traffic spikes**, up to 1.9 times normal load.
- **Separate seeds**: practice starts at 100 and held-out at 1000, so even the
  noise in the metrics and logs is different.

Both sets include **healthy controls**, where nothing is wrong and the right
answer is to change nothing. Some controls contain a harmless config change
to tempt a false alarm. Many faulty scenarios also contain a harmless change
just before the real fault, because the latest change is not always the
cause.

Some faults cannot be fixed by any action Sanad has: an outage at the
identity service, which another team owns, and a power fault on a rack. In
those scenarios the right move is to hand the problem to a person, not to
keep retrying.

## Locked

[`heldout/MANIFEST.sha256`](heldout/MANIFEST.sha256) holds the hash of
every held-out file. `tests/test_scenarios.py` fails if any of them changes,
and git history shows they were committed before the agent was written.
