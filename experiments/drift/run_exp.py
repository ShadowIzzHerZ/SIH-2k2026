"""Train + score one named experiment on the fixed drive-level split.

    python -m experiments.drift.run_exp baseline
    python -m experiments.drift.run_exp calibfix
"""
import argparse
import time

from experiments.drift import ideas, lib

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--stage1", type=int, default=6)
ap.add_argument("--stage2", type=int, default=6)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

from experiments.drift import combos
if args.name in combos.COMBOS:  # checkpoint combination, no training
    combos.run(args.name)
    raise SystemExit
spec = ideas.EXPERIMENTS[args.name]
t0 = time.time()
splits = spec.get("splits", lambda: lib.build_splits(spec.get("calib", "orig")))()
model = spec.get("model", lib.new_model)()
kw = dict(spec.get("train_kw", {}))
for k in ("loss_fn", "chain_loss_fn", "chain_fn", "train_ds"):  # factories taking (model, splits)
    if k in spec:
        kw[k] = spec[k](model, splits)
if "fwd" in spec:
    kw["fwd"] = spec["fwd"]
name = args.name if args.seed == 42 else f"{args.name}_s{args.seed}"
model, hist = lib.train(model, splits, name, stage1_epochs=args.stage1, stage2_epochs=args.stage2,
                        seed=args.seed, **kw)
res = lib.evaluate(model, splits, chain_fn=kw.get("chain_fn", lib.chained_drift), fwd=kw.get("fwd", lib.forward_pass))
rec = lib.save_result(name, spec["desc"], res, hist)
print(f"RESULT {name}: {lib.fmt_headline(rec['headline'])}  ({(time.time() - t0) / 60:.0f} min)")
