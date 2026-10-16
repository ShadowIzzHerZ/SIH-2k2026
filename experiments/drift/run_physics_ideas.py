"""No-training check of the integration-side ideas on physics only (fixed calibration)."""
import warnings
warnings.filterwarnings("ignore")
from experiments.drift import ideas, lib

sp = lib.build_splits("fwdfix")
import sys
rows = {
    "phys_bias_both": lambda: ideas.bias_chain_fn(sp),
    "phys_bias_gyro": lambda: ideas.bias_chain_fn(sp, accel=False),
    "phys_bias_accel": lambda: ideas.bias_chain_fn(sp, gyro=False),
    "phys_bias_gyro_120s": lambda: ideas.bias_chain_fn(sp, accel=False, lookback_s=120, min_s=60),
    "phys_zupt": lambda: ideas.zupt_chain_fn(),
}
rows = {k: v() for k, v in rows.items() if len(sys.argv) < 2 or k in sys.argv[1:]}
for name, fn in rows.items():
    res = lib.evaluate(lib.ZeroNet(), sp, chain_fn=fn)
    rec = lib.save_result(name, f"physics only, fixed calibration, {name}", res)
    print(f"{name}: {lib.fmt_headline(rec['headline'])}", flush=True)
