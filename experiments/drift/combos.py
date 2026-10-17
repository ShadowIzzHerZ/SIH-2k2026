"""Combinations of already-trained checkpoints, no training.

ens:  average the per-sample corrections of several models (seeds disagree on
      noise, agree on signal, so averaging should cut the random part).
duo:  pick a model by the current speed estimate (fast one >= switch m/s).
Members are checkpoint names; speed-aware ones get the current speed set.
Run through run_exp like any experiment: python -m experiments.drift.run_exp ens_moredata
"""
import torch

from experiments.drift import ideas, lib


def load(name):
    spec = ideas.EXPERIMENTS[name.split("_s")[0] if name not in ideas.EXPERIMENTS else name]
    m = spec.get("model", lib.new_model)()
    m.load_state_dict(torch.load(lib.ROOT / "checkpoints" / f"{name}.pt"))
    return m.eval()


class Mix(torch.nn.Module):
    def __init__(self, members, mode="ens", switch=6.0):
        super().__init__()
        self.members = torch.nn.ModuleList(load(n) if isinstance(n, str) else n for n in members)
        self.mode, self.switch = mode, switch
        self.cur_v0 = torch.zeros(1)

    def forward(self, x):
        outs = []
        for m in self.members:
            if hasattr(m, "cur_v0"):
                m.cur_v0 = self.cur_v0
            outs.append(m(x))
        if self.mode == "ens":
            return torch.stack(outs).mean(0)
        fast = self.cur_v0.reshape(-1, 1, 1) >= self.switch
        return torch.where(fast, outs[0], outs[1])


COMBOS = {
    "ens_moredata": ("ens", ["moredata", "moredata_s7"], "average of moredata seeds 42 and 7"),
    "ens_moredata3": ("ens", ["moredata", "moredata_s7", "moredata_s3"], "average of moredata seeds 42, 7 and 3"),
    "ens_round3": ("ens", ["moredata", "moredata_s7", "richaug", "richaug_s7"],
                   "average of moredata and richaug, both seeds"),
    "duo_moredata_slowspec": ("duo", ["moredata", "slowspec"], "moredata above 6 m/s, slow specialist below"),
    "duo_ens_slowspec": ("duo", [None, "slowspec"], "moredata seed-average above 6 m/s, slow specialist below"),
}


def run(name):
    mode, members, desc = COMBOS[name]
    if members[0] is None:
        members = [Mix(["moredata", "moredata_s7"])] + members[1:]
    m = Mix(members, mode).eval()
    sp = lib.build_splits("fwdfix")  # only val/test are used here
    res = lib.evaluate(m, sp, chain_fn=lib.chained_drift_x, fwd=ideas.speedaware_forward)
    rec = lib.save_result(name, desc, res)
    print(f"RESULT {name}: {lib.fmt_headline(rec['headline'])}", flush=True)
