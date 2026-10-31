#!/bin/sh
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
for v in restart rollback; do
  $PY -m sanad eval heldout --variant $v --label heldout-$v
done
for v in no-canaries trust-model no-governor routine-only strong-only; do
  $PY -m sanad eval heldout --variant $v --label heldout-$v --max-usd 0.04
done
for k in 1 2; do
  $PY -m sanad eval heldout --variant sanad --label heldout-sanad-noise$k --seed-offset $k --max-usd 0.04
done
$PY -m sanad eval heldout --variant sanad --label heldout-sanad --max-usd 0.04 || echo "heldout-sanad: replay stops at h12, see README"
$PY -m scripts.make_results
