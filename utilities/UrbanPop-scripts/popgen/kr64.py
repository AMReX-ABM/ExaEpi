"""KR64: the keyed, counter-based random number generator for in-process population generation.

Every random decision in population generation is a pure function of a key tuple of global, stable
identifiers -- (seed, replicate, stage, geoid, household, person, ...) -- never a position in a
shared stream. That is what makes a generated population independent of processing order, thread
layout and MPI rank count, and what lets the C++ port (src/KeyedRNG.H) reproduce this reference
bit for bit. The definition below is normative; both implementations are checked against the
known-answer vectors in kr64_vectors.json.

All arithmetic is modulo 2^64. Key words are int64 values reinterpreted as uint64 (two's
complement), so negative identifiers such as -1 are valid.

    mix64(z):     z = (z ^ (z >> 30)) * M1;  z = (z ^ (z >> 27)) * M2;  return z ^ (z >> 31)
    absorb(h, w): mix64(h ^ mix64(w + G))
    key(w1..wn):  fold absorb over the words, starting from h = IV
    draw(K, j):   absorb(K, j)     -- the j-th 64-bit output for key K

mix64 is the SplitMix64 finalizer (Stafford's Mix13). absorb mixes the word before combining it:
a bare mix64(h ^ w) would make (a, b) and (a', b ^ D) collide for a fixed D, and Weyl-offset streams
overlap whenever two keys differ by a small multiple of G.

Derived draws, all exact in IEEE double and identical in C++:

    u01(x)          (x >> 11) * 2^-53, in [0, 1)
    index(x, n)     min(n - 1, floor(u01(x) * n)),  1 <= n < 2^53
    int_cdf(w, x)   integer weights: r = index(x, sum w); smallest i with cumsum(w)[i] > r.
                    Zero-weight entries are never chosen.
    float_cdf(w, x) float weights: F = sequential running sum; t = u01(x) * F[-1];
                    smallest i with F[i] > t, clamped to len(w) - 1
    shuffle         sort items by (draw(key(..., item), 0), item identity)

Every function accepts numpy arrays for the key words and broadcasts, so a whole population's draws
are one vectorised call.
"""

import numpy as np

G = np.uint64(0x9E3779B97F4A7C15)
M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)
IV = np.uint64(0x243F6A8885A308D3)
_S30, _S27, _S31, _S11 = (np.uint64(s) for s in (30, 27, 31, 11))
_TWO_M53 = 2.0 ** -53


def _u64(w):
    """int64 (or uint64) words as uint64, two's complement for negatives."""
    a = np.asarray(w)
    if a.dtype == np.uint64:
        return a
    return a.astype(np.int64).view(np.uint64) if a.ndim else np.int64(a).view(np.uint64)


def mix64(z):
    with np.errstate(over="ignore"):
        z = (z ^ (z >> _S30)) * M1
        z = (z ^ (z >> _S27)) * M2
        return z ^ (z >> _S31)


def absorb(h, w):
    with np.errstate(over="ignore"):
        return mix64(h ^ mix64(_u64(w) + G))


def key(*words):
    """Fold key words (scalars or broadcastable arrays) into a uint64 key."""
    h = IV
    for w in words:
        h = absorb(h, w)
    return h


def draw(k, j=0):
    """The j-th 64-bit output for key k."""
    return absorb(k, j)


def u01(x):
    return (x >> _S11).astype(np.float64) * _TWO_M53


def index(x, n):
    """Uniform integer in [0, n) from a 64-bit draw; n may be an array."""
    n = np.asarray(n)
    return np.minimum(n - 1, np.floor(u01(x) * n.astype(np.float64)).astype(np.int64))


def int_cdf(weights, x):
    """Index chosen with probability proportional to non-negative integer weights."""
    c = np.cumsum(np.asarray(weights, dtype=np.int64))
    if c[-1] <= 0:
        raise ValueError("int_cdf needs a positive total weight")
    return np.searchsorted(c, index(x, c[-1]), side="right")


def float_cdf(weights, x):
    """Index chosen with probability proportional to float weights (sequential running sum)."""
    f = np.cumsum(np.asarray(weights, dtype=np.float64))
    t = u01(x) * f[-1]
    return np.minimum(np.searchsorted(f, t, side="right"), len(f) - 1)


def shuffle_order(k, identity):
    """Permutation sorting items by (draw(k), identity); k and identity are per-item arrays."""
    return np.lexsort((np.asarray(identity), draw(k, 0)))
