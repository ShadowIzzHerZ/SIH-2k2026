#!/bin/bash
# after queue13: average the 3 best-val and 3 last-epoch context+speed models together
cd "$(dirname "$0")/../.."
while pgrep -f "queue1[023].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2
python3 -m experiments.drift.run_ensembles seedens speed hist=100 ens=hist10_speedaware ens=hist10_speedaware_s7 ens=hist10_speedaware_s3 ens=hist10_speedaware_lx ens=hist10_speedaware_lx_s7 ens=hist10_speedaware_lx_s3 > $L/histspeedens6.log 2>&1
echo Q14_DONE >> $L/queue14.done
