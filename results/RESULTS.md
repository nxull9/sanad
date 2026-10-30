# Results

Built by `scripts/make_results.py` from `runs/*/summary.json`. Each run folder also holds one
trace per scenario: the score, every model turn, every action and the hidden answer key.

## Practice set (21 scenarios)

| Agent | Solved | Causes found | Silent causes found | Fixable causes removed | Handed to people when needed | Harmless change blamed | False alarms | Changes (unneeded) | Model calls | Cost | Cost per solved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Restart everything (baseline) | 6/21 (29%) | - | - | 25% | 100% | 0 | 0/2 | 265 (0) | 0 | $0.0000 | - |
| Roll back latest change (baseline) | 9/21 (43%) | - | - | 50% | 100% | 5 | 0/2 | 17 (0) | 0 | $0.0000 | - |
| sanad-v1 | 19/21 (90%) | 88% | 100% | 92% | 100% | 0 | 0/2 | 44 (8) | 80 | $0.0191 | $0.0010 |
