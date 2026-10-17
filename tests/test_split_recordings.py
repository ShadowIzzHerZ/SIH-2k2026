"""
IO-VNBD split by recording: every copy of one drive lands in the same split
(synthetic file lists and CSVs, no data needed).
"""
from __future__ import annotations

import numpy as np

from src.data.windowing import load_dataset_splits, recording_key, split_recordings


def _paths():
    """Each drive in a Categorised and an Uncategorised folder, like IO-VNBD,
    with the case of the file name not always matching."""
    out = []
    for i in range(30):
        out.append(f"root/Categorised IOVNB Dataset/D{i % 4}/S-Vw{i}.csv")
        out.append(f"root/Uncategorised IOVNB Dataset/S-Dataset/S-VW{i}.csv" if i % 3 == 0
                   else f"root/Uncategorised IOVNB Dataset/S-Dataset/S-Vw{i}.csv")
    return out


def test_copies_of_a_drive_share_a_split():
    parts = split_recordings(_paths(), train_split=0.7, val_split=0.15, seed=0)
    keys = {s: {recording_key(p) for p in ps} for s, ps in parts.items()}
    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]
    assert sorted(p for ps in parts.values() for p in ps) == sorted(_paths())
    for ps in parts.values():
        assert len(ps) == 2 * len({recording_key(p) for p in ps})


def test_fractions_apply_to_recordings_and_seed_is_stable():
    a = split_recordings(_paths(), 0.7, 0.15, seed=0)
    b = split_recordings(list(reversed(_paths())), 0.7, 0.15, seed=0)
    assert a == b
    n = {s: len({recording_key(p) for p in ps}) for s, ps in a.items()}
    assert n == {"train": 21, "val": 4, "test": 5}


def test_load_dataset_splits_never_puts_one_drive_in_two_splits(tmp_path, monkeypatch):
    for p in _paths():
        f = tmp_path / "IO-VNBD" / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x\n")
    seen = []

    def fake_load(path, column_map):
        seen.append(path)
        raise ValueError("synthetic")

    monkeypatch.setattr("src.data.windowing.load_sequence", fake_load)
    load_dataset_splits(str(tmp_path), "IO-VNBD", {}, 10.0, 50, 10, 0.7, 0.15, file_prefix="S-")
    assert len(seen) == 60
    # load order is train, val, test: each recording's copies must be adjacent
    # within one split, i.e. one recording never shows up in two splits
    order = [recording_key(str(p)) for p in seen]
    first = {k: order.index(k) for k in set(order)}
    last = {k: len(order) - 1 - order[::-1].index(k) for k in set(order)}
    cut = [21 * 2, (21 + 4) * 2]
    for k in first:
        assert np.searchsorted(cut, first[k], side="right") == np.searchsorted(cut, last[k], side="right"), k
