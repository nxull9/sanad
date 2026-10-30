#!/bin/sh
# Every held-out run reported in results/RESULTS.md, in the order they were made.
# With SANAD_OFFLINE=1 this replays stored model replies: no key, no cost.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
for v in restart rollback; do
  $PY -m sanad eval heldout --variant $v --label heldout-$v
done
for v in sanad no-canaries trust-model no-governor routine-only strong-only; do
  $PY -m sanad eval heldout --variant $v --label heldout-$v --max-usd 0.04
done
for k in 1 2; do
  $PY -m sanad eval heldout --variant sanad --label heldout-sanad-noise$k --seed-offset $k --max-usd 0.04
done
$PY scripts/make_results.py
