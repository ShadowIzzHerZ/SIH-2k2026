"""No-training reference rows: physics only (orig vs fixed calibration) and the
existing best_v3chain.pt (trained on the old leaky split, so its IO-VNBD
numbers here are optimistic for drives it saw: reference only)."""
import time

import numpy as np
import torch

from experiments.drift import lib

for calib in ("orig", "fwdfix"):
    t = time.time()
    sp = lib.build_splits(calib)
    print(f"[{calib}] windows: " + ", ".join(f"{k} {len(v)}" for k, v in sp.items() if not k.startswith("_")),
          f"({time.time() - t:.0f}s)")
    if calib == "fwdfix":
        info = sp["_calib_info"]
        r = [v["r_long"] for v in info.values() if v]
        print(f"forward-axis fit: median r(longitudinal accel vs GPS dv/dt) = {np.nanmedian(r):.2f}")
    t = time.time()
    res = lib.evaluate(lib.ZeroNet(), sp)
    rec = lib.save_result(f"ref_physics_{calib}", f"physics only, {calib} calibration", res)
    print(f"physics_{calib}: {lib.fmt_headline(rec['headline'])} ({time.time() - t:.0f}s)")
    m = lib.new_model()
    m.load_state_dict(torch.load("checkpoints/best_v3chain.pt", map_location="cpu"))
    res = lib.evaluate(m.eval(), sp)
    rec = lib.save_result(f"ref_v3chain_{calib}", f"existing best_v3chain.pt (old leaky split), {calib} calibration", res)
    print(f"v3chain_{calib}: {lib.fmt_headline(rec['headline'])}")
