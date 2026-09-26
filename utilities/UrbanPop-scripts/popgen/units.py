"""Iteration order over independent units, switchable for the order-independence tests.

Loops whose iterations must not interact -- block groups within a PUMA, industries, regions within
a scale, school levels, teacher types, work and class groups -- iterate through units.each().
Production runs visit units in their natural order; test_stages.py re-runs every stage with the
units reversed and shuffled and requires bitwise-identical output, which is what lets the C++ port
run the same loops in parallel. Loops that are sequential by design (TRS draws without replacement, the
university and teacher neighbour passes over overlapping county neighbourhoods, IPF iterations and
repair sweeps) do not use it.
"""

import numpy as np

MODE = "forward"          # "forward", "reverse" or "shuffle"
_rng = np.random.default_rng(12345)


def each(items):
    """items (a sequence or 1-D array) in the current test order."""
    if MODE == "forward":
        return items
    idx = np.arange(len(items))
    idx = idx[::-1] if MODE == "reverse" else _rng.permutation(idx)
    if isinstance(items, np.ndarray):
        return items[idx]
    items = list(items)
    return [items[i] for i in idx]
