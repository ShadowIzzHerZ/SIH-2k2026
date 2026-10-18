"""Recover IO-VNBD drives that prep.py skipped, for TRAINING ONLY.

prep.py uses the main pipeline's yaw-axis thresholds (fit R >= 0.6 and >= 400
moving 10 s spans). 19 of the 29 skipped drives miss only those cut-offs.
Here they are retried with R >= 0.45 and >= 100 spans. A recovered drive is
kept only if its session group is already in train or is a new group; a drive
whose group sits in val/test is dropped, so val and test stay exactly as they
were and every result stays comparable.

    python -m experiments.drift.prep_extra   ->  cache/extra_sequences.pkl
"""
import json
import pickle
from pathlib import Path

from experiments.drift.prep import CACHE, HZ, IOVNBD, ROOT, group_of
from src.data.io_vnbd_loader import load_sequence, recover_yaw_axis
from src.data.windowing import resample_uniform

MIN_R, MIN_SPANS = 0.45, 100


def main():
    split = json.load(open(ROOT / "split.json"))
    skipped, groups = split["skipped"], split["group_split"]
    paths = {p.stem.lower(): p for p in sorted(IOVNBD.rglob("S-*.csv"), key=lambda p: "Uncategorised" not in str(p))[::-1]}
    extra, log = {}, {}
    for key in sorted(skipped):
        stem = key.split(":", 1)[1]
        g = groups.get(group_of(key), "train")
        if g != "train":
            log[key] = f"group in {g}, not used"
            continue
        try:
            seq = resample_uniform(load_sequence(paths[stem.lower()], None), HZ)
            seq = recover_yaw_axis(seq, min_r=MIN_R, min_moving_windows=MIN_SPANS)
            extra[key] = seq
            log[key] = f"recovered, {seq.n / HZ / 60:.1f} min"
        except Exception as e:
            log[key] = "still skipped: " + str(e).split(": ", 1)[-1]
    pickle.dump(extra, open(CACHE / "extra_sequences.pkl", "wb"))
    json.dump(log, open(ROOT / "extra_drives.json", "w"), indent=1)
    for k, v in log.items():
        print(k, v)
    print(f"{len(extra)} extra training drives, {sum(s.n for s in extra.values()) / HZ / 60:.0f} min")


if __name__ == "__main__":
    main()
