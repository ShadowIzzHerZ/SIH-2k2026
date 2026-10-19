#!/bin/bash
# after queue23: average every seed of the new best recipe (incl. the ones trained by the drift-experiments thread)
cd "$(dirname "$0")/../.."
while pgrep -f "queue(1[0-9]|2[0-3]).sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2; C=experiments/drift/checkpoints
args=""; n=0
for f in $C/hist10_speedaware_lx_chain6.pt $C/hist10_speedaware_lx_chain6_s*.pt; do
  b=$(basename $f .pt); args="$args ens=$b"; n=$((n+1))
done
python3 -m experiments.drift.run_ensembles seedens speed hist=100 $args > $L/lxchain6ens$n.log 2>&1
echo Q24_DONE >> $L/queue24.done
