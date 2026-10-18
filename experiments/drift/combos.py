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
        if self.mode == "med":
            return torch.stack(outs).median(0).values
        fast = self.cur_v0.reshape(-1, 1, 1) >= self.switch
        return torch.where(fast, outs[0], outs[1])


COMBOS = {
    "ens_moredata": ("ens", ["moredata", "moredata_s7"], "average of moredata seeds 42 and 7"),
    "ens_moredata3": ("ens", ["moredata", "moredata_s7", "moredata_s3"], "average of moredata seeds 42, 7 and 3"),
    "duo_ens_slowplain8": ("duo", [None, "slowspec_plain"], "moredata seed-average above 8 m/s, plain slow specialist below", 8.0),
    "duo_ens_slowspec7": ("duo", [None, "slowspec"], "moredata seed-average above 7 m/s, slow specialist below", 7.0),
    "ens_all5": ("ens", ["moredata", "moredata_s7", "moredata_s3", "richaug", "richaug_s7"],
                 "average of moredata (3 seeds) and richaug (2 seeds)"),
    "ens_moredata4": ("ens", ["moredata", "moredata_s7", "moredata_s3", "moredata_s11"], "average of 4 moredata seeds"),
    "duo_ens3_slowspec7": ("duo", [["moredata", "moredata_s7", "moredata_s3"], "slowspec"],
                           "3-seed moredata average above 7 m/s, slow specialist below", 7.0),
    "duo_all5_slowspec7": ("duo", [["moredata", "moredata_s7", "moredata_s3", "richaug", "richaug_s7"], "slowspec"],
                           "5-model average above 7 m/s, slow specialist below", 7.0),
    "ens_moredata6": ("ens", ["moredata", "moredata_s7", "moredata_s3", "moredata_s11", "moredata_s5", "moredata_s13"],
                      "average of all 6 moredata seeds (no seed picked by test results)"),
    "duo_ens6_slowspec7": ("duo", [["moredata", "moredata_s7", "moredata_s3", "moredata_s11", "moredata_s5", "moredata_s13"],
                                   "slowspec"], "6-seed moredata average above 7 m/s, slow specialist below", 7.0),
    "ens_speedaware3": ("ens", ["speedaware", "speedaware_s7", "speedaware_s3"],
                        "average of 3 speedaware seeds (no extra drives): fair control for the moredata ensembles"),
    "ens_all11": ("ens", ["moredata", "moredata_s7", "moredata_s3", "moredata_s11", "moredata_s5", "moredata_s13",
                          "speedaware", "speedaware_s7", "speedaware_s3", "speedaware_s5", "speedaware_s13"],
                  "average of every speedaware-family seed: 6 moredata + 5 speedaware"),
    "ens_lastepoch4": ("ens", ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3"], "average of 4 last-epoch moredata seeds"),
    "ens_lastepoch6": ("ens", ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13"], "average of 6 last-epoch moredata seeds"),
    "duo_last4_slowspec7": ("duo", [["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3"], "slowspec"],
                            "4-seed last-epoch average above 7 m/s, slow specialist below", 7.0),
    "duo_last6_slowspec7": ("duo", [["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13"], "slowspec"], "6-seed last-epoch average above 7 m/s, slow specialist below", 7.0),
    "ens_lastnoextra3": ("ens", ["lastepoch_noextra", "lastepoch_noextra_s7", "lastepoch_noextra_s3"],
                         "average of 3 last-epoch speedaware seeds without the recovered drives"),
    "ens_lastepoch8": ("ens", ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13",
                               "lastepoch_s17", "lastepoch_s21"], "average of 8 last-epoch moredata seeds"),
    "ens_last9mix": ("ens", ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13"] + ["lastepoch_noextra", "lastepoch_noextra_s7", "lastepoch_noextra_s3"], "average of 6 last-epoch moredata + 3 last-epoch no-extra seeds"),
    "duo_last6_noextra7": ("duo", [["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13"], ["lastepoch_noextra", "lastepoch_noextra_s7", "lastepoch_noextra_s3"]], "6-seed last-epoch moredata average above 7 m/s, 3-seed no-extra average below", 7.0),
    "ens_chain6_4": ("ens", ["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3"],
                     "average of 4 last-epoch 30 s-chain seeds"),
    "ens_last6_chain6_4": ("ens", ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13",
                                   "lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3"],
                           "average of 6 last-epoch (15 s chains) + 4 last-epoch 30 s-chain seeds"),
    "ens_lastepoch10": ("ens", ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21", "lastepoch_s25", "lastepoch_s29"], "average of 10 last-epoch moredata seeds"),
    "duo_last10_noextra7": ("duo", [["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21", "lastepoch_s25", "lastepoch_s29"], ["lastepoch_noextra", "lastepoch_noextra_s7", "lastepoch_noextra_s3"]], "10-seed last-epoch average above 7 m/s, 3-seed no-extra average below", 7.0),
    "duo_chain6_noextra7": ("duo", [["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3"], ["lastepoch_noextra", "lastepoch_noextra_s7", "lastepoch_noextra_s3"]], "4-seed 30 s-chain average above 7 m/s, 3-seed no-extra average below", 7.0),
    "duo_chain6_last8_7": ("duo", [["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3"], ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21"]], "4-seed 30 s-chain average above 7 m/s, 8-seed last-epoch average below", 7.0),
    "duo_chain6_last8_6": ("duo", [["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3"], ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21"]], "4-seed 30 s-chain average above 6 m/s, 8-seed last-epoch average below", 6.0),
    "duo_chain6_last8_8": ("duo", [["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3"], ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21"]], "4-seed 30 s-chain average above 8 m/s, 8-seed last-epoch average below", 8.0),
    "duo_chain6_last8_9": ("duo", [["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3"], ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21"]], "4-seed 30 s-chain average above 9 m/s, 8-seed last-epoch average below", 9.0),
    "ens_lastvariants16": ("ens", ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21", "lastepoch_speedw", "lastepoch_speedw_s7", "lastepoch_lowlr2", "lastepoch_lowlr2_s7", "lastepoch_hilr2", "lastepoch_hilr2_s7", "lastepoch_s1heavy", "lastepoch_s2heavy"], "average of 8 last-epoch seeds and the 8 last-epoch recipe variants (speed weight, chained rate, stage budget)"),
    "ens_drop4_4": ("ens", ["lastepoch_drop4", "lastepoch_drop4_s7", "lastepoch_drop4_s3", "lastepoch_drop4_s11"], "average of 4 last-epoch dropout-0.4 seeds"),
    "ens_last8_drop4_4": ("ens", ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21", "lastepoch_drop4", "lastepoch_drop4_s7", "lastepoch_drop4_s3", "lastepoch_drop4_s11"], "average of 8 last-epoch seeds and 4 dropout-0.4 seeds"),
    "duo_chain6_lastdrop12_7": ("duo", [["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3"], ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21", "lastepoch_drop4", "lastepoch_drop4_s7", "lastepoch_drop4_s3", "lastepoch_drop4_s11"]], "4-seed 30 s-chain average above 7 m/s, 8 last-epoch + 4 dropout-0.4 seeds below", 7.0),
    "ens_drop4_8": ("ens", ["lastepoch_drop4", "lastepoch_drop4_s7", "lastepoch_drop4_s3", "lastepoch_drop4_s11", "lastepoch_drop4_s5", "lastepoch_drop4_s13", "lastepoch_drop4_s17", "lastepoch_drop4_s21"], "average of 8 last-epoch dropout-0.4 seeds"),
    "duo_chain6_drop8_7": ("duo", [["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3"], ["lastepoch_drop4", "lastepoch_drop4_s7", "lastepoch_drop4_s3", "lastepoch_drop4_s11", "lastepoch_drop4_s5", "lastepoch_drop4_s13", "lastepoch_drop4_s17", "lastepoch_drop4_s21"]], "4-seed 30 s-chain average above 7 m/s, 8-seed dropout-0.4 average below", 7.0),
    "ens_chain6_8": ("ens", ["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3", "lastepoch_chain6_s5", "lastepoch_chain6_s13", "lastepoch_chain6_s17", "lastepoch_chain6_s21"], "average of 8 last-epoch 30 s-chain seeds"),
    "duo_chain6x8_last8_7": ("duo", [["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3", "lastepoch_chain6_s5", "lastepoch_chain6_s13", "lastepoch_chain6_s17", "lastepoch_chain6_s21"], ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21"]], "8-seed 30 s-chain average above 7 m/s, 8-seed last-epoch average below", 7.0),
    "med_lastepoch8": ("med", ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21"], "per-element median of 8 last-epoch seeds (robust to one bad seed)"),
    "med_chain6_8": ("med", ["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3", "lastepoch_chain6_s5", "lastepoch_chain6_s13", "lastepoch_chain6_s17", "lastepoch_chain6_s21"], "per-element median of 8 30 s-chain seeds"),
    "dmed_chain6x8_last8_7": ("dmed", [["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3", "lastepoch_chain6_s5", "lastepoch_chain6_s13", "lastepoch_chain6_s17", "lastepoch_chain6_s21"], ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21"]], "median of 8 30 s-chain seeds above 7 m/s, median of 8 last-epoch seeds below", 7.0),
    "dslowmed_chain6_last8_7": ("dslowmed", [["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3"], ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21"]], "4-seed 30 s-chain mean above 7 m/s, median of 8 last-epoch seeds below", 7.0),
    "ens_chain6_6": ("ens", ["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3",
                             "lastepoch_chain6_s5", "lastepoch_chain6_s13"], "average of 6 last-epoch 30 s-chain seeds"),
    "duo_chain6x6_last8_7": ("duo", [["lastepoch_chain6", "lastepoch_chain6_s7", "lastepoch_chain6_s11", "lastepoch_chain6_s3", "lastepoch_chain6_s5", "lastepoch_chain6_s13"], ["lastepoch", "lastepoch_s7", "lastepoch_s11", "lastepoch_s3", "lastepoch_s5", "lastepoch_s13", "lastepoch_s17", "lastepoch_s21"]], "6-seed 30 s-chain average above 7 m/s, 8-seed last-epoch average below", 7.0),
    "ens_round3": ("ens", ["moredata", "moredata_s7", "richaug", "richaug_s7"],
                   "average of moredata and richaug, both seeds"),
    "duo_moredata_slowspec": ("duo", ["moredata", "slowspec"], "moredata above 6 m/s, slow specialist below"),
    "duo_ens_slowspec": ("duo", [None, "slowspec"], "moredata seed-average above 6 m/s, slow specialist below"),
    "duo_ens_slowspec8": ("duo", [None, "slowspec"], "moredata seed-average above 8 m/s, slow specialist below", 8.0),
    "duo_ens_slowspec10": ("duo", [None, "slowspec"], "moredata seed-average above 10 m/s, slow specialist below", 10.0),
}


def run(name):
    mode, members, desc, *sw = COMBOS[name]
    if members[0] is None:
        members = [Mix(["moredata", "moredata_s7"])] + members[1:]
    # dmed: speed-switched pair of per-element medians; dslowmed: mean above the switch, median below
    inner = {"dmed": ("med", "med"), "dslowmed": ("ens", "med")}.get(mode, ("ens", "ens"))
    if mode in ("dmed", "dslowmed"):
        mode = "duo"
    if isinstance(members[0], list):
        members = [Mix(members[0], inner[0])] + members[1:]
    if mode == "duo" and isinstance(members[1], list):
        members = [members[0], Mix(members[1], inner[1])]
    m = Mix(members, mode, *sw).eval()
    sp = lib.build_splits("fwdfix")  # only val/test are used here
    res = lib.evaluate(m, sp, chain_fn=lib.chained_drift_x, fwd=ideas.speedaware_forward)
    rec = lib.save_result(name, desc, res)
    print(f"RESULT {name}: {lib.fmt_headline(rec['headline'])}", flush=True)
