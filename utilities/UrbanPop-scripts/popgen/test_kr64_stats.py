"""Statistical sanity battery for KR64 on the structured keys population generation feeds it.

Run from utilities/UrbanPop-scripts:  python -m popgen.test_kr64_stats [--n N]

KR64 is a keyed (counter-based) generator: every draw is a hash of (seed, rep, stage, ids...),
which is what makes a population independent of rank count, thread count and iteration order,
and identical between Python and C++ (the known-answer vectors in test_kr64.py and
utilities/tests/kr64_kat.cpp tie the two bit for bit, so testing Python tests both). Its mixing
function is SplitMix64's finalizer, applied twice per key word. This is not BigCrush; it targets
the one plausible failure of a hash used this way -- inputs with structure (consecutive ids,
geoid-shaped numbers, neighbouring seeds) not mixing into independent-looking outputs:

    families    the key shapes the stages use: persons (bg, h, p) with geoid-shaped block
                groups, consecutive seeds, consecutive reps, every stage id, and consecutive draw
                counters under one key
    per family  balance of each of the 64 output bits; chi-square of u01 on 256 bins; serial
                chi-square of neighbouring keys' (u01, u01) on 16 x 16 bins; lag-1 correlation;
                chi-square of index(x, n) for n = 2, 3, 7, 100
    avalanche   flipping any one of the 64 bits of an id word, or of the seed, flips each output
                bit with probability 1/2 (strict avalanche criterion, 64 x 64 cells each)
    collisions  no two persons of a 2-million-person population share a key

Each test's p-value is Bonferroni-corrected over its sub-tests (bits, bins, cells); the battery
fails if any corrected p-value is below 1e-4. With ~25 family tests one p near 1e-3 turns up by
chance now and then (at 2^20 draws, consecutive reps' index() test gave 0.00095); a real defect
grows with n, so rerun with --n 2^22 or 2^24, which draw the same keys and more (there that test
gave 1 and 0.81). Runs in seconds at the default n.
"""

import argparse
import sys

import numpy as np
from scipy import stats as sps

from . import kr64, stages

P_FAIL = 1e-4
SEED, REP = 12345, 0


def geoids(nbg):
    """Block-group geoids shaped like real ones: state, odd county, tract, bg 1-4."""
    i = np.arange(nbg, dtype=np.int64)
    bg = 1 + i % 4
    tract = 100 + 100 * ((i // 4) % 300)
    county = 1 + 2 * (i // 1200)
    return 35 * 10**10 + county * 10**7 + tract * 10 + bg


def families(n):
    """name -> uint64 keys whose natural order is the family's adjacency."""
    nbg = max(1, n // 64)                      # ~16 households x 4 persons per block group
    g = np.repeat(geoids(nbg), 64)[:n]
    h = np.tile(np.repeat(np.arange(16), 4), nbg)[:n]
    p = np.tile(np.arange(4), n // 4 + 1)[:n]
    ids = np.arange(n, dtype=np.int64)
    stage_words = np.array([stages.word(s) for s in stages.STAGES], dtype=np.int64)
    fixed = (350010001001, 3, 1)
    return {
        "persons (bg,h,p)": kr64.key(SEED, REP, stages.WORK_ASSIGN, g, h, p),
        "consecutive seeds": kr64.key(ids, REP, stages.CHILDCARE, *fixed),
        "consecutive reps": kr64.key(SEED, ids, stages.CHILDCARE, *fixed),
        "all stages x ids": kr64.key(SEED, REP, np.repeat(stage_words, n // len(stage_words) + 1)[:n],
                                     np.tile(ids[: n // len(stage_words) + 1], len(stage_words))[:n]),
        "draw counters j": kr64.draw(kr64.key(SEED, REP, stages.IPF_TRS_ADD, 17, 350010001001), ids),
    }


def bonferroni(p, m):
    return min(1.0, float(p) * m)


def chi2_p(obs, exp):
    obs = np.asarray(obs, dtype=np.float64).ravel()
    exp = np.broadcast_to(np.asarray(exp, dtype=np.float64), obs.shape)
    return sps.chi2.sf(((obs - exp) ** 2 / exp).sum(), obs.size - 1)


def family_tests(x):
    """Corrected p-values of the per-family tests on 64-bit outputs x (in adjacency order)."""
    n = x.size
    u = kr64.u01(x)
    out = {}
    ones = np.array([((x >> np.uint64(b)) & np.uint64(1)).sum() for b in range(64)], dtype=np.float64)
    z = (ones - n / 2) / np.sqrt(n / 4)
    out["bit balance (64)"] = bonferroni(2 * sps.norm.sf(np.abs(z).max()), 64)
    out["u01 chi2 (256)"] = chi2_p(np.bincount(np.floor(u * 256).astype(int), minlength=256), n / 256)
    a, b = np.floor(u[:-1] * 16).astype(int), np.floor(u[1:] * 16).astype(int)
    out["serial chi2 (16x16)"] = chi2_p(np.bincount(a * 16 + b, minlength=256), (n - 1) / 256)
    r = np.corrcoef(u[:-1], u[1:])[0, 1]
    out["lag-1 correlation"] = 2 * sps.norm.sf(abs(r) * np.sqrt(n - 1))
    ps = [chi2_p(np.bincount(kr64.index(x, m), minlength=m), n / m) for m in (2, 3, 7, 100)]
    out["index() chi2 (n=2,3,7,100)"] = bonferroni(min(ps), 4)
    return out


def avalanche(n, word):
    """Strict avalanche over 64 x 64 (input bit, output bit) cells, flipping one bit of `word`."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 2**62, size=n, dtype=np.int64)

    def keyed(w):
        return kr64.draw(kr64.key(w, REP, stages.WORK_ASSIGN, 350010001001, 3, 1), 0) if word == "seed" \
            else kr64.draw(kr64.key(SEED, REP, stages.WORK_ASSIGN, w, 3, 1), 0)

    x0 = keyed(base)
    worst = 0.0
    for i in range(64):
        flipped = (base.view(np.uint64) ^ np.uint64(1 << i)).view(np.int64)
        d = x0 ^ keyed(flipped)
        cnt = np.array([((d >> np.uint64(b)) & np.uint64(1)).sum() for b in range(64)], dtype=np.float64)
        worst = max(worst, np.abs((cnt - n / 2) / np.sqrt(n / 4)).max())
    return bonferroni(2 * sps.norm.sf(worst), 64 * 64), worst


def collisions(npersons=2_000_000):
    nbg = npersons // 64
    g = np.repeat(geoids(nbg), 64)
    h = np.tile(np.repeat(np.arange(16), 4), nbg)
    p = np.tile(np.arange(4), 16 * nbg)
    k = kr64.key(SEED, REP, stages.WORK_ASSIGN, g, h, p)
    return k.size - np.unique(k).size, k.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 20, help="draws per family")
    args = ap.parse_args()
    ok = True
    print(f"{args.n} draws per family; corrected p-values, fail below {P_FAIL:g}")
    for name, k in families(args.n).items():
        x = kr64.draw(k, 0) if name != "draw counters j" else k
        res = family_tests(x)
        bad = [t for t, p in res.items() if p < P_FAIL]
        ok &= not bad
        print(f"  {name:20s} " + "  ".join(f"{t} {p:.3g}" for t, p in res.items()) + ("  FAIL" if bad else ""))
    for word in ("id", "seed"):
        p, worst = avalanche(20000, word)
        ok &= p >= P_FAIL
        print(f"  avalanche on {word:4s}      64x64 cells, worst |z| {worst:.2f}, corrected p {p:.3g}"
              + ("  FAIL" if p < P_FAIL else ""))
    dup, total = collisions()
    ok &= dup == 0
    print(f"  collisions           {dup} duplicate keys among {total} persons" + ("  FAIL" if dup else ""))
    print("popgen.test_kr64_stats: " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
