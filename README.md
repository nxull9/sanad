# Sanad · سند

**An agent that keeps an AI serving platform healthy.** It notices when the
platform breaks, including when the model keeps answering with no errors
but gives wrong answers. It finds every cause, not just the first, fixes
what it can safely fix, proves the platform has recovered, and hands an
engineer a clear summary when a cause is not its to fix.

HUMAIN × Misk hackathon, **Challenge 1: Agentic Recovery of a Failing System**.

*Sanad* (سند) means support. It is also the root of *isnad*, the chain of
narrators that authenticates a report. Every conclusion Sanad reaches
carries its chain of evidence: the metric, the log line, the change, the
canary answer.

---

## The problem

When an AI platform breaks, the worst failures don't look like failures.

- **Wrong answers raise no alarm.** In four years of Microsoft's
  generative-AI incidents, 14.5% were invalid output, and 38% of incidents
  were found by people rather than monitors
  ([arXiv 2504.08865](https://arxiv.org/abs/2504.08865)).
- **Arabic breaks first.** Compressing a model to save memory hurts
  non-Latin scripts more than English
  ([arXiv 2407.03211](https://arxiv.org/abs/2407.03211)). An English-only
  health check can say "all fine" while Arabic users get nonsense.
- **Several causes arrive at once, and agents stop at the first.** On IBM's
  ITBench, agents built on state-of-the-art models resolve 11.4% of SRE
  scenarios ([ICML 2025](https://proceedings.mlr.press/v267/jha25a.html)).

## How Sanad works

```
every 20 s ── canary questions to each replica ─┐
          └── metrics, logs, change log ────────┴─► detection rules (code, free)
                                                         │ symptoms, nothing unexplained? ─► keep watching
                                                         ▼
                              model keeps a LEDGER: one hypothesis per cause,
                              every open symptom must sit under one
                                                         │ proposed actions
                                                         ▼
                              GOVERNOR (code): allowed? risky? retried too often?
                                                         │
                                                         ▼
                              act ─► wait ─► CODE checks which symptoms cleared
                                            cleared nothing? undo it, one attempt spent
                                                         │
                     no symptoms for 2 checks ─► recovered     not ours / allowance spent ─► hand to people
```

| Part | The trap it handles | Proof it earns its place |
|---|---|---|
| Canary questions in Arabic and English, asked of each replica | Wrong answers with normal errors and latency; a bad build on part of the fleet | Ablation: without canaries |
| Ledger: every symptom must be explained | Finding one cause and stopping | Multi-fault results |
| Governor in code | Retrying forever; risky moves without a person | Ablation: without the governor |
| Recovery checked by code | An agent saying "fixed" when nothing is | Ablation: trusting the model |
| Cheap model by default, stronger one when hard | Paying frontier prices for routine incidents | Cheap-only and strong-only runs |

Detection is plain code, so **a healthy platform costs nothing to watch**.
The model is called only while a symptom has no explanation, and it gets
one fresh message per turn, never a growing chat history.

## Results

On the **held-out set**, 24 scenarios Sanad never saw while it was built:

| Agent | Solved | Silent causes found | Harmless change blamed | False alarms | Cost (all 24) |
|---|---|---|---|---|---|
| Restart everything (baseline) | 4/24 (17%) | n/a | 0 | 0/4 | $0 |
| Roll back the latest change (baseline) | 9/24 (38%) | n/a | 4 | 0/4 | $0 |
| **Sanad** | **19/24 (79%)** | **88%** | **0** | **0/4** | **$0.024** |
| Sanad, same faults, different noise | 18/24, 18/24 | 94%, 94% | 1, 1 | 0/4, 0/4 | $0.037, $0.032 |

Sanad solves twice as many scenarios as the better baseline, at about
**a tenth of a US cent per incident**. Rerunning the same faults with
different random noise moves the score by one scenario. Full tables, with
every variant and the practice set, are in [results/RESULTS.md](results/RESULTS.md).
Every trace is in `runs/`.

### What each part proved (ablations on the held-out set)

| Remove this | Solved | What changed |
|---|---|---|
| Nothing (Sanad) | 19/24 | |
| **Canary questions** | **12/24** | Silent causes found fall from 88% to **29%**. Without known-answer questions, a model giving wrong answers looks healthy. This is the largest effect we measured. |
| **Code-checked recovery** (the model decides when it's over) | 17/24 | The model declared victory while causes were still active. Fixable causes removed fall from 84% to 74%. |
| **The governor** (no limits, no undo) | 20/24 | The solve rate didn't drop, but model calls rose from 84 to 144 and cost more than doubled. It solved h16 *because* nothing undid a slow-acting fix (see failures). The governor's value is bounded cost and bounded risk, not more solves. |
| **Routing** (cheap model only) | 20/24 | As good, for **$0.014**. |
| **Routing** (stronger model only) | 19/24 | The same, for $0.041. |

**The honest finding on models:** once code handles detection, the
ledger and verification, the cheapest model was enough. The stronger
model added cost, not accuracy. Structure did more than model size. On
this evidence the default should be the cheap model alone. We report it
rather than retune, because the agent was frozen before the held-out
run. Differences of one or two scenarios are within the noise shown above.

### Where Sanad failed, and why (all 5 held-out failures)

| Scenario | What went wrong | What it teaches |
|---|---|---|
| h11 | Sanad escalated the identity outage, and the traffic spike's symptoms were filed under that handed-off cause, so the spike was never treated | Escalating one cause must not absorb symptoms of another |
| h14, h16 | A correct scale-out was undone because the backlog had not drained within the check window. In h16 a deadlock struck one step after the fix | Code checks need more patience for slow-acting fixes, judged by trend rather than by whether the symptom is gone |
| h14 | Late in the run, 1.9× traffic exceeds what the worker pool can ever dispatch, so no available action fixes it: a flaw in our scenario generator, left in place because the held-out set is locked | Check every generated scenario for a reachable fix |
| h20 | Scale-out copied a replica that carried the bad tokenizer, spreading the fault to new capacity | New capacity must clone a known-good configuration |
| h24 | Four silent faults at once: all wrong answers were filed under one cause, and two causes were never separated | Split "wrong answers on replica-2" by kind of question, so the ledger can tell causes apart |

### From v1 to v2 (practice set only)

v1 solved 19/21. Reading its traces led to general rules, not
scenario-specific hints: meet overload with capacity; restart a crash
that shows no hardware signs instead of draining its node; give back
borrowed traffic before closing. Code now also stops a verified cause
from being overwritten. v2 solved 21/21 and cost less than a third as
much ($0.019 → $0.0055). The v1 summary was scored before two scoring
fixes; the v2 summary was rescored offline from stored replies.

## Run it

Python 3.13, standard library only.

```bash
python -m sanad run scenarios/examples/three-at-once.toml
```

That narrates one incident live. To watch it paced for a person, with the
injected faults, the canary answers and the ledger on screen:

```bash
SANAD_OFFLINE=1 python -m sanad demo scenarios/examples/three-at-once.toml --pace 1.5
```

The plain run above narrates one incident live. With `--mode advise`, every change waits
for a person; add `--ask` to approve risky actions at the keyboard. A
model call needs `OPENROUTER_API_KEY` in `.env`.

**Rebuild the published numbers without a key and at no cost.** Every
model reply is stored in `replays/`:

```bash
SANAD_OFFLINE=1 scripts/run_heldout.sh
```

Nine of the ten held-out runs rebuild exactly: both baselines, every
ablation and both noise reruns. The main Sanad run (19/24) rebuilds exactly
through h11 and then stops. A network outage interrupted that run, and before
it was restarted one line was changed: inspecting canaries for a replica with
no answers yet no longer crashes. For the other variants that change touches
nothing, but in h12 (a rack power failure, replica-4 down) it changes what
Sanad reads, so the replies stored for h12 onwards match neither version.
Rebuilding h12 to h24 takes live model calls, about half a US cent.

```bash
python -m pytest
```

## How it was evaluated

- **45 synthetic scenarios** from a simulated serving platform. See
  [docs/DATA.md](docs/DATA.md) for how the data is generated, where each
  fault comes from in published incident reports, and what the simulator
  does not model.
- **A held-out set locked before any agent code existed** (commit
  `db317d0`, with a SHA-256 manifest and a test that fails on any change).
  It has three fault kinds never seen in practice, no fault pair shared
  with practice, up to four faults at once, late faults and healthy
  controls.
- **The agent was frozen before the held-out run** (commit `3528cb9`). All
  tuning used the practice set only; `runs/practice-sanad-v1` and `-v2`
  show what changed and why.
- **Scored against the hidden answer key from the platform's state**, never
  from the agent's own report. A scenario counts as solved only if every
  fixable cause is gone, every cause that needed a person was handed over,
  and the harmless change was not blamed. On a healthy control, it counts
  as solved only if nothing was touched.
- **Baselines, ablations and robustness:** two free baselines (restart
  everything; roll back the latest change), four ablations, a cheap-only
  and a strong-only run, and the held-out set rerun with different noise.

## Repository

| Path | What |
|---|---|
| `sanad/agent.py` | The loop, the ledger, verification, routing |
| `sanad/observe.py` | Detection rules, canaries, the evidence the model reads |
| `sanad/governor.py` | What may run, what needs a person, allowances |
| `sanad/llm.py` | OpenRouter calls, stored replies, the spending cap |
| `sanad/tools.py` | The only interface between Sanad and a platform |
| `sanad/platform/` | The simulated platform and its faults |
| `sanad/evaluate.py`, `sanad/baselines.py` | Scoring and the baselines |
| `scenarios/` | Examples, practice, held-out |
| `runs/`, `results/` | Every trace and the tables built from them |

## Limits

- A simulator can only produce failures we thought of. A real platform
  would plug in behind `sanad/tools.py`, but that has not been done.
- Canaries are free here. On a real platform they cost inference, and the
  right sampling rate is unmeasured.
- Results come from one run per scenario per variant at temperature 0.
  The noise reruns show how much they move, but the sets are small (24
  held-out scenarios), so a difference of one or two scenarios is within
  noise.
