#!/bin/bash
# after queue27: 5-model average of the 30 s-weighted variant (2 seeds trained by drift-experiments)
cd "$(dirname "$0")/../.."
while pgrep -f "queue(1[0-9]|2[0-7]).sh" >/dev/null; do sleep 30; done
L=experiments/drift/logs2
python3 -m experiments.drift.run_ensembles seedens speed hist=100 ens=hist10_speedaware_lx_chain6_ew ens=hist10_speedaware_lx_chain6_ew_s7 ens=hist10_speedaware_lx_chain6_ew_s3 ens=hist10_speedaware_lx_chain6_ew_s11 ens=hist10_speedaware_lx_chain6_ew_s13 > $L/lxchain6ewens5.log 2>&1
echo Q28_DONE >> $L/queue28.done
