#!/bin/bash
# after queue25: average the 3 chain6 models with the 3 30 s-weighted ones
cd "$(dirname "$0")/../.."
while pgrep -f "queue(1[0-9]|2[0-5]).sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2
python3 -m experiments.drift.run_ensembles seedens speed hist=100 ens=hist10_speedaware_lx_chain6 ens=hist10_speedaware_lx_chain6_s7 ens=hist10_speedaware_lx_chain6_s3 ens=hist10_speedaware_lx_chain6_ew ens=hist10_speedaware_lx_chain6_ew_s7 ens=hist10_speedaware_lx_chain6_ew_s3 > $L/lxchain6mixens6.log 2>&1
echo Q26_DONE >> $L/queue26.done
