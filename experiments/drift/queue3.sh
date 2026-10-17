#!/bin/bash
# follow-ups once queue2 is done: longer-context seed 3, its seed ensemble, 30 s context
cd "$(dirname "$0")/../.."
while pgrep -f queue2.sh >/dev/null; do sleep 30; done
L=experiments/drift/logs2; R=experiments/drift/results
[ -f $R/hist10_s3.json ] || python3 -m experiments.drift.run_exp2 hist10 --seed 3 > $L/hist10_s3.log 2>&1
[ -f "$R/seedens_hist10+hist10_s7+hist10_s3.json" ] || python3 -m experiments.drift.run_ensembles seedens hist=100 ens=hist10 ens=hist10_s7 ens=hist10_s3 > $L/histens.log 2>&1
for s in 42 7; do
  tag=hist25; [ $s != 42 ] && tag=hist25_s$s
  [ -f $R/$tag.json ] || python3 -m experiments.drift.run_exp2 hist25 --seed $s > $L/$tag.log 2>&1
done
echo Q3_DONE >> $L/queue3.done
