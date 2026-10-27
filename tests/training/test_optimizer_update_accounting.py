"""Optimizer-update accounting replays Ultralytics 8.4.174 trainer logic (accumulate ramp in warmup, then every 4 batches)."""
import numpy as np
def updates(nb, epochs, nbs=64, bs=16, warm=2, upto=None):
    nw = round(min(warm, max(epochs - 1, 0)) * nb); last = -1; acc = round(nbs / bs); n = 0
    for ni in range(nb * (upto or epochs)):
        if ni < nw: acc = max(1, int(np.interp(ni, [0, nw], [1, nbs / bs]).round()))
        if ni - last >= acc: n += 1; last = ni
    return n
def test_p2_budgets_not_equal():
    v2, v3 = updates(511, 50), updates(530, 48)
    assert (v2, v3) == (6629, 6610) and v2 != v3
    assert 511 * 50 == 25550 and 530 * 48 == 25440          # batches, not optimizer updates
def test_evaluated_epoch1_checkpoints():
    assert (updates(511, 50, upto=1), updates(530, 48, upto=1)) == (341, 353)
def test_no_accumulation_after_warmup():
    assert updates(100, 10) < 100 * 10 / 3
