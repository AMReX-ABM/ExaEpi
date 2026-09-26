"""Tests for popgen.kr64. Run from utilities/UrbanPop-scripts:  python -m popgen.test_kr64

The known-answer vectors pin the definition across languages (utilities/tests/kr64_kat.cpp checks
the C++ side against the same files); these check the Python implementation against itself --
vectorised calls must equal scalar calls exactly -- and that the draws behave like uniform random
numbers.
"""

import os
import sys

import numpy as np

from . import kr64, stages

HERE = os.path.dirname(__file__)


def check_vectors():
    n = 0
    with open(os.path.join(HERE, "kr64_vectors.txt")) as fh:
        for line in fh:
            f = [int(x) for x in line.split()]
            m = f[0]
            words, rest = f[1:1 + m], f[1 + m:]
            k = kr64.key(*words)
            assert int(k) == rest[0], line
            assert [int(kr64.draw(k, j)) for j in range(3)] == rest[1:4], line
            n += 1
    return n


def check_vectorised():
    """Array keys must reproduce the scalar computation element for element."""
    geo = np.array([350010001001, 350010001002, -1, 0, 2**62], dtype=np.int64)
    h = np.array([0, 5, 7, 1, 3], dtype=np.int64)
    kv = kr64.key(42, 0, stages.CHILDCARE, geo, h, 2)
    for i in range(len(geo)):
        ks = kr64.key(42, 0, stages.CHILDCARE, int(geo[i]), int(h[i]), 2)
        assert kv[i] == ks
        assert kr64.draw(kv, 1)[i] == kr64.draw(ks, 1)
        assert kr64.index(kr64.draw(kv, 0), 13)[i] == kr64.index(kr64.draw(ks, 0), 13)
    # Broadcasting a scalar word against an array word.
    assert np.array_equal(kr64.key(1, np.arange(4)), [kr64.key(1, i) for i in range(4)])


def check_uniformity(n=1_000_000):
    k = kr64.key(7, 0, stages.TRS, np.arange(n, dtype=np.int64))
    u = kr64.u01(kr64.draw(k, 0))
    assert 0.0 <= u.min() and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 5 * np.sqrt(1 / 12 / n), u.mean()
    # Chi-square on 64 bins, and on index() over a non-power-of-two range.
    for vals, bins in ((np.floor(u * 64).astype(int), 64), (kr64.index(kr64.draw(k, 1), 37), 37)):
        obs = np.bincount(vals, minlength=bins)
        exp = n / bins
        chi2 = ((obs - exp) ** 2 / exp).sum()
        assert chi2 < bins + 6 * np.sqrt(2 * bins), (bins, chi2)
    # Adjacent keys and adjacent counters must be uncorrelated.
    v = kr64.u01(kr64.draw(k, 2))
    assert abs(np.corrcoef(u, v)[0, 1]) < 5 / np.sqrt(n)
    assert abs(np.corrcoef(u[:-1], u[1:])[0, 1]) < 5 / np.sqrt(n)


def check_cdfs(n=400_000):
    w = np.array([0, 3, 0, 1, 6], dtype=np.int64)
    xs = kr64.draw(kr64.key(3, 0, np.arange(n)), 0)
    picks = np.array([kr64.int_cdf(w, x) for x in xs[:20000]])
    assert not np.isin(picks, [0, 2]).any(), "zero-weight entry chosen"
    share = np.bincount(picks, minlength=5) / len(picks)
    assert np.allclose(share, w / w.sum(), atol=0.015), share
    wf = np.array([0.5, 0.0, 1.5, 2.0])
    picks = np.array([kr64.float_cdf(wf, x) for x in xs[:20000]])
    assert not (picks == 1).any()
    assert np.allclose(np.bincount(picks, minlength=4) / len(picks), wf / wf.sum(), atol=0.015)


def check_shuffle():
    ids = np.arange(1000)
    k = kr64.key(11, 0, stages.STU_SCHOOL_PERM, ids)
    p = kr64.shuffle_order(k, ids)
    assert sorted(p) == list(ids)
    # Order-independence: shuffling the input rows gives the same permutation of identities.
    perm = np.random.default_rng(0).permutation(1000)
    p2 = kr64.shuffle_order(k[perm], ids[perm])
    assert np.array_equal(ids[p], ids[perm][p2])


def main():
    n = check_vectors()
    check_vectorised()
    check_uniformity()
    check_cdfs()
    check_shuffle()
    print(f"popgen.kr64: {n} known-answer vectors, vectorised == scalar, uniformity, CDFs and "
          f"shuffle order-independence all pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
