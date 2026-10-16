"""Speed-switched pair, no training: the speed-aware model when the current
speed estimate is >= 6 m/s, the slow-traffic-best augbias_long model below."""
import warnings
warnings.filterwarnings("ignore")
import torch
from experiments.drift import ideas, lib


class Duo(torch.nn.Module):
    def __init__(self, fast, slow, switch=6.0):
        super().__init__()
        self.fast, self.slow, self.switch = fast, slow, switch
        self.cur_v0 = torch.zeros(1)

    def forward(self, x):
        self.fast.cur_v0 = self.cur_v0
        use_fast = (self.cur_v0.reshape(-1, 1, 1) >= self.switch)
        return torch.where(use_fast, self.fast(x), self.slow(x))


sp = lib.build_splits("fwdfix")
for fast_name in ("speedaware", "speedaware_s7"):
    fast = ideas.SpeedAwareNet(); fast.load_state_dict(torch.load(lib.ROOT / "checkpoints" / f"{fast_name}.pt")); fast.eval()
    slow = lib.new_model(); slow.load_state_dict(torch.load(lib.ROOT / "checkpoints" / "augbias_long.pt")); slow.eval()
    m = Duo(fast, slow).eval()
    res = lib.evaluate(m, sp, chain_fn=lib.chained_drift_x, fwd=ideas.speedaware_forward)
    rec = lib.save_result(f"duo_{fast_name}", f"{fast_name} above 6 m/s, augbias_long below (switch on current speed estimate)", res)
    print(f"RESULT duo_{fast_name}: {lib.fmt_headline(rec['headline'])}", flush=True)
