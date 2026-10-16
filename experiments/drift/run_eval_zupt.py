"""Stop detection (ZUPT) added at inference to an already-trained checkpoint."""
import sys, warnings
warnings.filterwarnings("ignore")
import torch
torch.set_num_threads(1)
from experiments.drift import ideas, lib
name = sys.argv[1]
sp = lib.build_splits("fwdfix")
m = lib.new_model(); m.load_state_dict(torch.load(lib.ROOT / "checkpoints" / f"{name}.pt")); m.eval()
res = lib.evaluate(m, sp, chain_fn=ideas.zupt_chain_fn())
rec = lib.save_result(f"{name}+zupt", f"{name} checkpoint + stop detection at inference", res)
print(f"RESULT {name}+zupt: {lib.fmt_headline(rec['headline'])}")
