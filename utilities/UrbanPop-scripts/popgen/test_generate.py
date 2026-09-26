"""Order-independence and determinism of bundle -> placements -> persons.

Run from utilities/UrbanPop-scripts:  python -m popgen.test_generate BUNDLE [PUMA ...]

A population must be a pure function of (bundle, seed, rep): the same PUMAs generated in a
different order, or alone, must give identical placements, and a different seed must not.
"""

import sys

import numpy as np

from generate_exaepi import generate
from generate_exaepi import placement_digest as digest
from popgen import bundle


def by_puma(b, placements, pumas):
    """Split placements by PUMA using the bundle's block-group ranges."""
    names = bundle.strings(b, "solve.puma")
    off, geo = b["solve.bg_offset"], b["solve.bg_geoid"]
    out = {}
    for p in pumas:
        i = names.index(p)
        m = np.isin(placements[0], geo[off[i]:off[i + 1]])
        out[p] = tuple(a[m] for a in placements)
    return out


def main():
    path = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--digest":
        # Print one population's digest, for comparing separate processes.
        b = bundle.read(path)
        print(digest(generate(b, 5, 0, sys.argv[3:], verbose=False)[0]))
        return 0
    pumas = sys.argv[2:] or ["3500804", "3500200", "3500806"]
    b = bundle.read(path)
    fwd, _ = generate(b, 5, 0, pumas, verbose=False)
    names = bundle.strings(b, "solve.puma")
    rev_order = sorted(pumas, key=lambda p: -names.index(p))
    rev, _ = generate(b, 5, 0, rev_order, verbose=False)
    assert digest(fwd) == digest(rev), "PUMA order changed the population"
    alone = {p: generate(b, 5, 0, [p], verbose=False)[0] for p in pumas}
    split = by_puma(b, fwd, pumas)
    for p in pumas:
        assert digest(alone[p]) == digest(split[p]), f"PUMA {p} alone differs from in a group"
    other, _ = generate(b, 6, 0, pumas, verbose=False)
    assert digest(other) != digest(fwd), "a different seed gave the same population"
    rep1, _ = generate(b, 5, 1, pumas, verbose=False)
    assert digest(rep1) != digest(fwd), "a different replicate gave the same population"
    print(f"popgen generate: {len(pumas)} PUMAs order-independent (digest {digest(fwd)}), "
          f"alone == grouped, seed and replicate each change the result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
