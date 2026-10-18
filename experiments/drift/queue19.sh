#!/bin/bash
# after queue18: average the 3 picked models with the 3 weight-averaged ones
cd "$(dirname "$0")/../.."
while pgrep -f "queue1[0-8].sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2
python3 -m experiments.drift.run_ensembles seedens speed hist=100 ens=hist10_speedaware ens=hist10_speedaware_s7 ens=hist10_speedaware_s3 ens=hist10_speedaware_swa ens=hist10_speedaware_swa_s7 ens=hist10_speedaware_swa_s3 > $L/swaens6.log 2>&1
echo Q19_DONE >> $L/queue19.done
