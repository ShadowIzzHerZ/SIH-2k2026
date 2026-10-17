"""Train + score one round-2 experiment (ideas2.py), same as run_exp.py.

    python -m experiments.drift.run_exp2 speedhead --seed 7
"""
import argparse
import time

from experiments.drift import aligned, ideas, ideas2, lib

aligned.use_aligned_chains()

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--stage1", type=int, default=6)
ap.add_argument("--stage2", type=int, default=6)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

spec = {**ideas.EXPERIMENTS, **ideas2.EXPERIMENTS}[args.name]
t0 = time.time()
splits = spec.get("splits", lambda: lib.build_splits(spec.get("calib", "orig")))()
model = spec.get("model", lib.new_model)()
kw = dict(spec.get("train_kw", {}))
for k in ("loss_fn", "chain_loss_fn", "chain_fn", "train_ds"):
    if k in spec:
        kw[k] = spec[k](model, splits)
if "fwd" in spec:
    kw["fwd"] = spec["fwd"]
name = args.name if args.seed == 42 else f"{args.name}_s{args.seed}"
model, hist = lib.train(model, splits, name, stage1_epochs=args.stage1, stage2_epochs=args.stage2,
                        seed=args.seed, log=lambda s: print(s, flush=True), **kw)
res = lib.evaluate(model, splits, chain_fn=kw.get("chain_fn", lib.chained_drift_x), fwd=kw.get("fwd", lib.forward_pass))
rec = lib.save_result(name, spec["desc"], res, hist)
print(f"RESULT {name}: {lib.fmt_headline(rec['headline'])}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
