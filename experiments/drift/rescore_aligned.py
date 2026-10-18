"""Rescore finished runs/ensembles with the off-by-one-free metric from the
drift-search-2 branch (experiments/drift/aligned.py, copied unchanged), so the
two searches' numbers can be compared. Saves results/A_<name>.json.

    python -m experiments.drift.rescore_aligned ens_lastepoch8 duo_chain6_last8_7 ...
"""
import sys

from experiments.drift import aligned, combos, ideas, lib


def model_for(name):
    if name in combos.COMBOS:
        mode, members, desc, *sw = combos.COMBOS[name]
        if mode not in ("ens", "duo"):
            raise SystemExit(f"{name}: only ens/duo combos are supported")
        if isinstance(members[0], list):
            members = [combos.Mix(members[0])] + members[1:]
        if mode == "duo" and isinstance(members[1], list):
            members = [members[0], combos.Mix(members[1])]
        return combos.Mix(members, mode, *sw).eval(), desc
    return combos.load(name).eval(), ideas.EXPERIMENTS[name.rsplit("_s", 1)[0] if name not in ideas.EXPERIMENTS else name]["desc"]


def main():
    sp = aligned.build_splits("fwdfix")
    for name in sys.argv[1:]:
        m, desc = model_for(name)
        res = aligned.evaluate(m, sp, set_v0=True)
        rec = lib.save_result(f"A_{name}", f"aligned metric: {desc}", res)
        print(f"RESULT A_{name}: {lib.fmt_headline(rec['headline'])}", flush=True)


if __name__ == "__main__":
    main()
