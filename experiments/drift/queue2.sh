#!/bin/bash
# round-2 queue, one run at a time (4 cores)
cd "$(dirname "$0")/../.."
[ -f experiments/drift/results/rotyaw_augbias.json ] || python3 -m experiments.drift.run_ensembles rot_yaw seedens > experiments/drift/logs2/ensembles.log 2>&1
for e in aligned speedhead hist10 endweight aligned_speedaware; do
  for s in 42 7; do
    tag=$e; [ $s != 42 ] && tag=${e}_s$s
    [ -f experiments/drift/results/$tag.json ] && continue
    python3 -m experiments.drift.run_exp2 $e --seed $s > experiments/drift/logs2/$tag.log 2>&1
  done
done
echo QUEUE_DONE >> experiments/drift/logs2/queue.done
