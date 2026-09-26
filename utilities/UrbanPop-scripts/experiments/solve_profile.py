#!/usr/bin/env python
"""Where does a ~40 s P-MEDM solve spend its time, and how fast could it be?

Two questions, answered on the same PUMA:

1. Breakdown of pymedm's own run: loading the cached ACS inputs, building the problem (the Kronecker
   products that assemble the sparse design matrix), JIT compilation, the L-BFGS iterations, and
   recovering the allocation. Iteration count and cost per objective evaluation separate "each step
   is slow" from "it takes many steps".

2. Whether the problem has a better shape than pymedm gives it. pymedm materialises the design matrix
   as kron(C^T, A) -- one sparse row per (characteristic, zone) over all (donor, block group)
   cells. But the exponent for a cell (d, g) is just

       sum_k C[d,k] * (lam0[k] + lam1[k, tract(g)] + lam2[k, g])  =  (C @ L)[d, g]

   with L a small dense (K x G) matrix built from the three levels of prices. So one objective
   evaluation is two dense matrix products of size (donors x K) @ (K x block groups), an
   exponential and a normalisation. That is GEMM-shaped work, which is exactly what GPUs and BLAS
   are fastest at. This implements that formulation, checks it against pymedm's own objective and
   gradient to rounding error, and times both.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspaces/ExaEpi/utilities/UrbanPop-scripts")
from build_precompute import EXAEPI_MINIMAL, repair_controlled_se  # noqa: E402


def timed(label, fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    dt = time.perf_counter() - t0
    print(f"  {label:44s}{dt:8.2f} s")
    return out, dt


def main():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from livelike import acs, config
    from pymedm import PMEDM

    fips = sys.argv[1] if len(sys.argv) > 1 else "3500804"
    key = os.environ.get("CENSUS_API_KEY") or None
    print(f"PUMA {fips}\n\npymedm, stage by stage:")

    pup, t_load = timed("acs.puma (cached ACS + PUMS)", acs.puma, fips,
                        constraints_selection=EXAEPI_MINIMAL,
                        constraints_theme_order=config.up_constraints_theme_order,
                        year=2019, target_zone="bg", cache=True,
                        cache_folder="./llcache_minimal", censusapikey=key)
    s1, _ = repair_controlled_se(pup.se_g1, np.asarray(pup.est_g1, float), "t")
    s2, _ = repair_controlled_se(pup.se_g2, np.asarray(pup.est_g2, float), "b")
    s1 = pd.DataFrame(s1, index=pup.se_g1.index, columns=pup.se_g1.columns)
    s2 = pd.DataFrame(s2, index=pup.se_g2.index, columns=pup.se_g2.columns)

    pmd, t_build = timed("PMEDM() -- build kron design matrix", PMEDM, pup.year,
                         pup.est_ind.index, pup.wt, pup.est_ind, pup.est_g1, pup.est_g2,
                         s1, s2, n_reps=0, random_state=1, keep_solver=True)
    X = pmd.X
    print(f"    design matrix {X.shape[0]:,} cells x {X.shape[1]:,} targets, "
          f"{X.nse:,} stored nonzeros ({X.nse * 12 / 1e6:.0f} MB)")

    _, t_solve = timed("solve() -- JIT + L-BFGS + allocation", pmd.solve)
    st = pmd.res.state
    iters = int(st.iter_num)
    print(f"    L-BFGS iterations: {iters}, final gradient norm {float(st.error):.2e}")

    # Per-evaluation cost of pymedm's objective + gradient, after compilation.
    kw = dict(q=pmd.q, X=pmd.X, Y_vec=pmd.Y_vec, sV=pmd.sV)
    vg = jax.jit(jax.value_and_grad(lambda l: pmd.f(l, **kw)))
    lam = pmd.lam
    t0 = time.perf_counter(); v, g = vg(lam); jax.block_until_ready(g)
    t_jit = time.perf_counter() - t0
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        v, g = vg(lam)
    jax.block_until_ready(g)
    t_eval = (time.perf_counter() - t0) / n
    print(f"    one objective+gradient: {1e3 * t_eval:.1f} ms  (first call incl. compile "
          f"{t_jit:.2f} s)")
    print(f"    => iterations account for roughly {iters * t_eval:.1f} s of the solve "
          f"(L-BFGS also spends extra evaluations in its line search)")

    # ---- dense GEMM formulation ------------------------------------------------------------
    C = jnp.asarray(np.asarray(pup.est_ind, float))            # donors x K
    nd, K = C.shape
    G = pup.est_g2.shape[0]
    A1 = jnp.asarray(pmd.A1.astype(float))                     # tracts x G
    T = A1.shape[0]
    L0 = len(lam) - K * (T + G)
    assert L0 in (0, K), f"unexpected PUMA-level block size {L0}"
    wt = jnp.asarray(np.asarray(pup.wt, float))
    logq = jnp.log(wt / wt.sum() / G)                          # prior: weight spread evenly over G
    Y = pmd.Y_vec
    V = jnp.asarray(np.asarray(pmd.V_vec))

    def unpack(l):
        o = 0
        l0 = l[:L0] if L0 else jnp.zeros(K); o += L0
        l1 = l[o:o + K * T].reshape(K, T); o += K * T
        l2 = l[o:o + K * G].reshape(K, G)
        return l0, l1, l2

    def f_dense(l):
        l0, l1, l2 = unpack(l)
        Leff = l0[:, None] + l1 @ A1 + l2                      # K x G, one GEMM
        E = logq[:, None] - C @ Leff                           # donors x G, one GEMM
        return Y @ l + jax.scipy.special.logsumexp(E) + 0.5 * l @ (V * l)

    vg_d = jax.jit(jax.value_and_grad(f_dense))
    t0 = time.perf_counter(); vd, gd = vg_d(lam); jax.block_until_ready(gd)
    t_jit_d = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n):
        vd, gd = vg_d(lam)
    jax.block_until_ready(gd)
    t_eval_d = (time.perf_counter() - t0) / n

    print("\ndense-GEMM formulation of the same objective:")
    print(f"  value  pymedm {float(v):.12f}  dense {float(vd):.12f}")
    print(f"  gradient max abs difference {float(jnp.abs(g - gd).max()):.2e} "
          f"(gradient scale {float(jnp.abs(g).max()):.2e})")
    print(f"  one objective+gradient: {1e3 * t_eval_d:.2f} ms  (compile {t_jit_d:.2f} s)")
    print(f"  speed-up per evaluation: {t_eval / t_eval_d:.1f}x")

    flops = 2 * 2 * nd * K * G + 2 * 2 * K * T * G             # fwd+bwd, the two GEMMs
    print(f"\n  arithmetic per evaluation: {flops / 1e6:.0f} MFLOP, "
          f"exp over {nd * G:,} cells")
    print(f"  at this CPU's achieved rate: {flops / t_eval_d / 1e9:.1f} GFLOP/s")
    print(f"  {iters} iterations x ~2 evaluations at this cost: "
          f"{2 * iters * t_eval_d:.2f} s per solve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
