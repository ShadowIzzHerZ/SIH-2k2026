"""Scores without training, all with the aligned (off-by-one fixed) metric:
round-1 checkpoints re-scored, rotation ensembles (mounting-error copies)
and seed ensembles.

    python -m experiments.drift.run_ensembles rescore rot_yaw seedens
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import torch
from experiments.drift import aligned, ideas, ideas2, lib


def load(name, cls=lib.new_model):
    m = cls()
    m.load_state_dict(torch.load(lib.ROOT / "checkpoints" / f"{name}.pt"))
    return m.eval()


class Duo(torch.nn.Module):
    def __init__(self, fast, slow, switch=6.0):
        super().__init__()
        self.fast, self.slow, self.switch = fast, slow, switch
        self.cur_v0 = torch.zeros(1)

    def forward(self, x):
        self.fast.cur_v0 = self.cur_v0
        use_fast = (self.cur_v0.reshape(-1, 1, 1) >= self.switch)
        return torch.where(use_fast, self.fast(x), self.slow(x))


which = sys.argv[1:]
hist = [int(a.split("=", 1)[1]) for a in which if a.startswith("hist=")]
sp = ideas2.hist_splits(hist[0])() if hist else aligned.build_splits()
net = (lambda: ideas2.HistNet()) if hist else lib.new_model
runs = []  # (name, desc, model, needs current speed)
if "rescore" in which:
    runs.append(("A_physics", "physics only, fixed calibration (aligned metric)", lib.ZeroNet(), False))
    for n in ("augbias", "augbias_s7", "augbias_long"):
        runs.append((f"A_{n}", f"round-1 {n} checkpoint, aligned metric", load(n), False))
    for n in ("speedaware", "speedaware_s7"):
        runs.append((f"A_duo_{n}", f"round-1 duo ({n} >= 6 m/s, augbias_long below), aligned metric",
                     Duo(load(n, ideas.SpeedAwareNet), load("augbias_long")), True))
for base in [a.split("=", 1)[1] for a in which if a.startswith("base=")] or ["augbias", "augbias_s7"]:
    if "rot_yaw" in which:
        runs.append((f"rotyaw_{base}", f"{base}, test-time mean over yaw mounting offsets -3,0,+3 deg",
                     ideas2.RotEnsemble(load(base), yaw_deg=(-3, 0, 3)), False))
    if "rot_pitch" in which:
        runs.append((f"rotpitch_{base}", f"{base}, test-time mean over yaw +-3 x pitch +-1 deg (9 copies)",
                     ideas2.RotEnsemble(load(base), yaw_deg=(-3, 0, 3), pitch_deg=(-1, 0, 1)), False))
if "seedens" in which:
    names = [a.split("=", 1)[1] for a in which if a.startswith("ens=")] or ["augbias", "augbias_s7", "augbias_long"]
    runs.append(("seedens_" + "+".join(names), "mean correction of " + " + ".join(names),
                 ideas2.MeanEnsemble([load(n, net) for n in names]), False))
for name, desc, m, v0 in runs:
    res = aligned.evaluate(m.eval(), sp, set_v0=v0)
    rec = lib.save_result(name, desc, res)
    print(f"RESULT {name}: {lib.fmt_headline(rec['headline'])}", flush=True)
