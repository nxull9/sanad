# How the data is generated

Every number in Sanad's results comes from a simulated AI serving platform.
The organisers allow synthetic data and asked teams to explain how it was
made. This page explains what is simulated, where each fault comes from in
real incident reports, how scenarios are generated and split, and what the
simulator leaves out.

## Why a simulator

- **An answer key.** Diagnosis can only be scored if the true causes are
  known. A real outage doesn't say what caused it; the simulator writes
  down every cause it injects, where Sanad cannot see it.
- **Repeatability.** The same seed always produces the same incident,
  down to the noise in the metrics, so every result can be rerun.
- **Breaking things on purpose.** We have no GPUs, and nobody lets a team
  break a production model 45 times in a weekend.

## What is simulated

[`sanad/platform/platform.py`](../sanad/platform/platform.py) simulates a
small serving stack in steps of 10 seconds:

```
gateway -> identity-service (another team's) -> queue -> workers -> 4 model replicas on 6 GPU nodes (2 racks) -> vector database
```

Every metric, log line, change-log entry and canary answer is computed
from the platform's state: queue depth from arrivals minus service, latency
from queueing plus generation time, GPU temperature from load and fan
health. No fault writes a symptom directly. A fault changes the state, and
the symptoms follow from it. That means **a fix works only if it removes
the cause**: restarting a replica that runs a bad build doesn't help,
because the bad build is still there afterwards.

Every scenario also includes background noise that real systems have:
slow requests for random tenants, autoscaler messages, metric jitter, and
the daily rise and fall of traffic.

## Where each fault comes from

| Fault | What it does in the simulator | Grounded in |
|---|---|---|
| Bad model release, bad chat template, bad tokenizer | Answers turn wrong with no error. A template change breaks structured output; a tokenizer mismatch garbles Arabic and mostly spares English | Microsoft's study of four years of its generative-AI incidents: 14.5% were invalid output, and 38% were reported by people rather than monitors ([arXiv 2504.08865](https://arxiv.org/abs/2504.08865)) |
| Bad quantization | Maths answers break, and Arabic suffers more than English | Cohere, *How does quantization affect multilingual LLMs?* ([arXiv 2407.03211](https://arxiv.org/abs/2407.03211)) |
| GPU silent data corruption | Wrong answers of every kind on one node, plus occasional ECC warnings | Meta's Llama 3 run: 419 unexpected interruptions in 54 days, about 78% hardware, with faulty GPUs (148) and HBM3 memory (72) the largest groups ([DatacenterDynamics](https://datacenterdynamics.com/en/news/meta-report-details-hundreds-of-gpu-and-hbm3-related-interruptions-to-llama-3-training-run)) |
| GPU overheating | Clocks throttle and that replica's latency triples | Same source (hardware is the largest failure class) |
| Memory leak in a server build, memory limit set too low | Memory climbs until the process is killed, or the process dies while loading weights | Inference-engine failures are about 60% of serious LLM-serving incidents ([arXiv 2511.07424](https://arxiv.org/abs/2511.07424)) |
| Gateway timeout cut, worker concurrency cut | Requests time out; the queue backs up and drops work | Configuration changes as a leading trigger of incidents (Microsoft study above) |
| Worker deadlock | Dispatch stops while workers wait on a lock: "a workflow that hangs forever" | Challenge brief, item 4 |
| Traffic spike | Demand rises 1.5–1.9×; the queue fills and drops requests: "a queue that loses messages under load" | Challenge brief, item 2 |
| Corrupted vector index | Retrieval answers turn wrong on every replica at once: "a quiet bug that turns good records bad" | Challenge brief, item 3 |
| Identity service outage, rack power fault | Nothing Sanad can do removes them; the right move is to hand them to a person | Challenge brief, item 5: "a repair agent that never escalates" |
| A harmless change | A logging change made just before the real fault, to tempt the agent into blaming the latest change | A known trap in incident diagnosis |

The rates at which each fault spoils each kind of answer are listed in
[`sanad/platform/catalog.py`](../sanad/platform/catalog.py). They are
judgement calls informed by the sources above, not measurements.

## Canary questions

[`sanad/data/canaries.json`](../sanad/data/canaries.json) holds 24
questions with known answers, 3 for each pair of language (Arabic, English)
and category (fact, maths, structured format, retrieval). Answers are
graded by meaning, not wording. Arabic is normalised first: diacritics are
dropped, alef forms unified and Arabic-Indic digits read as digits. A
healthy model still gets a canary wrong about 1 time in 100, so detection
requires 4 wrong answers out of 9 in one language and category before it
raises a symptom.

## Scenarios, and how the held-out set is kept honest

A scenario is a seed, a 30-step warm-up and a list of faults with their
timing and targets. [`scripts/make_scenarios.py`](../scripts/make_scenarios.py)
writes all of them and produces the same files on every run.

| | Practice (21) | Held-out (24) |
|---|---|---|
| Used for | building and tuning Sanad | reporting results only |
| Seeds | 100 and up | 1000 and up |
| Fault kinds | 13 | the same 13 plus 3 never seen in practice: quantization damage, memory limit too low, corrupted index |
| Faults at once | up to 3 | up to 4 |
| Combinations | | no pair of faults that meets in held-out ever meets in practice |
| Late faults | 1 scenario | 3 scenarios |
| Healthy controls | 2 | 4 |

The held-out set was committed before any agent code existed (commit
`db317d0`). A SHA-256 manifest and a test fail if any held-out file
changes.

## What the simulator does not model

Stated plainly, so nobody has to guess:

- **It can only produce failures we thought of.** A real platform fails in
  ways no catalogue lists. The held-out fault kinds test diagnosis of
  causes Sanad has not seen, but they are still faults we wrote.
- **Failure rates are judgement calls**, informed by published incident
  studies but not fitted to them.
- **Logs are cleaner than real ones.** Real logs are larger and noisier,
  and their formats vary more.
- **One fault per component.** A replica can carry several defects at once,
  but two faults never fight over the same setting.
- **Canaries are not free in real life.** Here they cost nothing. On a real
  platform, 24 questions per replica every 20 seconds would be an
  inference bill, and the right sampling rate is a trade-off we have not
  measured.
- **Time is simplified.** Fixes take effect in fixed steps; real rollouts
  are slower and less even.

Sanad talks to the platform only through [`sanad/tools.py`](../sanad/tools.py):
topology, metrics, logs, change log, canaries and actions. A real platform
(Prometheus, a log store, a deploy system and a serving endpoint) could sit
behind the same interface. That is the path from this prototype to a real
deployment.
