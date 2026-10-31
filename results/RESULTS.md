# Results

Built by `python -m scripts.make_results` from `runs/*/summary.json`. Each run folder also holds one
trace per scenario: the score, every model turn, every action and the hidden answer key.

## Held-out set (24 scenarios)

| Agent | Solved | Causes found | Silent causes found | Fixable causes removed | Handed to people when needed | Harmless change blamed | False alarms | Changes (unneeded) | Model calls | Cost | Cost per solved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Restart everything (baseline) | 4/24 (17%) | - | - | 3% | 100% | 0 | 0/4 | 300 (0) | 0 | $0.0000 | - |
| Roll back latest change (baseline) | 9/24 (38%) | - | - | 47% | 100% | 4 | 0/4 | 22 (0) | 0 | $0.0000 | - |
| **Sanad** | 19/24 (79%) | 86% | 88% | 84% | 100% | 0 | 0/4 | 49 (8) | 84 | $0.0238 | $0.0013 |
| Sanad, same faults, different noise (1) | 18/24 (75%) | 95% | 94% | 84% | 100% | 1 | 0/4 | 56 (9) | 121 | $0.0372 | $0.0021 |
| Sanad, same faults, different noise (2) | 18/24 (75%) | 90% | 94% | 87% | 100% | 1 | 0/4 | 55 (6) | 99 | $0.0320 | $0.0018 |
| Sanad, cheap model only | 20/24 (83%) | 86% | 82% | 90% | 100% | 0 | 0/4 | 49 (10) | 87 | $0.0143 | $0.0007 |
| Sanad, stronger model only | 19/24 (79%) | 83% | 82% | 84% | 100% | 0 | 0/4 | 49 (6) | 106 | $0.0408 | $0.0022 |
| Sanad without canary questions | 12/24 (50%) | 57% | 29% | 55% | 100% | 0 | 0/4 | 40 (6) | 81 | $0.0210 | $0.0018 |
| Sanad trusting the model's "fixed" | 17/24 (71%) | 81% | 94% | 74% | 100% | 0 | 0/4 | 49 (5) | 110 | $0.0377 | $0.0022 |
| Sanad without the governor | 20/24 (83%) | 88% | 94% | 87% | 100% | 0 | 0/4 | 53 (9) | 144 | $0.0544 | $0.0027 |

## Practice set (21 scenarios)

| Agent | Solved | Causes found | Silent causes found | Fixable causes removed | Handed to people when needed | Harmless change blamed | False alarms | Changes (unneeded) | Model calls | Cost | Cost per solved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Restart everything (baseline) | 6/21 (29%) | - | - | 25% | 100% | 0 | 0/2 | 265 (0) | 0 | $0.0000 | - |
| Roll back latest change (baseline) | 9/21 (43%) | - | - | 50% | 100% | 5 | 0/2 | 17 (0) | 0 | $0.0000 | - |
| Sanad v1 (first version) | 19/21 (90%) | 88% | 100% | 92% | 100% | 0 | 0/2 | 44 (8) | 80 | $0.0191 | $0.0010 |
| **Sanad v2** (frozen, used for held-out) | 21/21 (100%) | 100% | 100% | 100% | 100% | 0 | 0/2 | 30 (4) | 41 | $0.0055 | $0.0003 |
