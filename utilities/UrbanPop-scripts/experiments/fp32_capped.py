#!/usr/bin/env python
"""Is 32-bit good enough for a capped re-solve?

gpu_solve.py found that float32 can never reach the gradient tolerance (it stalls at ~5e-3), so a
converged solve needs float64 -- which consumer GPUs run at 1/64 of their float32 rate. But
resolve_spread.py showed convergence is not needed: a 1,000-iteration cap lands within 0.3-1.5% of
households of each draw's own converged answer and leaves every spread, fit and composition
measure unchanged. The question is only whether float32 reaches the same place in the same cap.

Three precisions, each at fixed iteration caps, on the recommended perturbation (nested targets +
replicate-weight prior, both_nested in resolve_spread.py):

    f64     everything in float64 (the reference path)
    f32     everything in float32, including the L-BFGS state
    mixed   the two dense matrix products -- nearly all the arithmetic -- in float32, then the
            exponent matrix cast to float64 for logsumexp, the objective, the gradient
            accumulation into the prices and the L-BFGS state
    hybrid  the objective value in float64, the gradient in float32. The value is what the line
            search compares; in float32 the late-solve decrease per step falls below rounding and
            the line search keeps retrying (measured ~28 evaluations per iteration against ~2).
            The gradient only sets the search direction, where float32 error is far below the
            gradient's own size at a 1,000-iteration cap. With the zoom (strong Wolfe) line
            search, and as hybrid_bt with backtracking (Armijo)
    inc32   lbfgs_incremental.py: a hand-written fixed-iteration L-BFGS that keeps the exponent
            matrix in float64 and updates it along the search direction, so line-search trials
            need no matrix product; the two products per iteration are float32
    inc64   the same algorithm entirely in float64, to separate algorithm from precision

Scored against the float64 converged solve of the same draw: households placed differently, and
the ensemble's spread ratio and composition (composition.py), which are what matter downstream.

Run from data/UrbanPop/experiments with JAX_PLATFORMS=cuda,cpu.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jaxopt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composition  # noqa: E402
from gpu_solve import f, load  # noqa: E402
from lbfgs_incremental import make_incremental_solver  # noqa: E402
from resolve_spread import logq_of, nested_Y, prior_rep, replicate_weights  # noqa: E402


def f_mixed(l, C32, logq, A1, Y, V):
    """gpu_solve.f with the two large products in float32 and everything else in float64."""
    K = C32.shape[1]
    T, G = A1.shape
    Leff = l[:K, None] + l[K:K + K * T].reshape(K, T) @ A1 + l[K + K * T:].reshape(K, G)
    E = logq - (C32 @ Leff.astype(jnp.float32)).astype(jnp.float64)
    return Y @ l + jax.scipy.special.logsumexp(E) + 0.5 * l @ (V * l)


def f_hybrid(l, C64, C32, logq, A1, Y, V):
    """(float64 value, float32 gradient) for jaxopt's value_and_grad=True."""
    g = jax.grad(f)(l.astype(jnp.float32), C32, logq.astype(jnp.float32),
                    A1.astype(jnp.float32), Y.astype(jnp.float32), V.astype(jnp.float32))
    return f(l, C64, logq, A1, Y, V), g.astype(jnp.float64)


def allocation(l, C, logq, A1, N):
    K = C.shape[1]
    T, G = A1.shape
    Leff = l[:K, None] + l[K:K + K * T].reshape(K, T) @ A1 + l[K + K * T:].reshape(K, G)
    E = logq - C @ Leff
    return jnp.exp(E - jax.scipy.special.logsumexp(E)) * N


def make_solver(fun, maxiter, value_and_grad=False, linesearch="zoom"):
    solver = jaxopt.LBFGS(fun=fun, tol=1e-6, maxiter=maxiter, jit=True,
                          value_and_grad=value_and_grad, linesearch=linesearch)

    @jax.jit
    def go(init, *a):
        r = solver.run(init, *a)
        return r.params, r.state.iter_num, r.state.error

    return go


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pumas", nargs="+", default=["3500804", "3500300", "3500100"])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--caps", type=int, nargs="+", default=[1000, 2000])
    ap.add_argument("--precisions", nargs="+",
                    default=["f64", "f32", "mixed", "hybrid", "hybrid_bt", "inc32", "inc64"])
    ap.add_argument("--moe_csv", default="/workspaces/ExaEpi/data/UrbanPop/acs_moe_35.csv")
    args = ap.parse_args()

    key = os.environ.get("CENSUS_API_KEY") or None
    dev = jax.devices()[0]
    print(f"device {dev.device_kind}, {args.draws} draws (nested targets + replicate-weight "
          f"prior)\n" + composition.header() + "\n", flush=True)
    moe = pd.read_csv(args.moe_csv, dtype={"geoid": str})
    moe = moe[moe.acs > 0].set_index("geoid")
    conv = make_solver(f, 50000)
    kinds = {"f64": (f, False, "zoom"), "f32": (f, False, "zoom"),
             "mixed": (f_mixed, False, "zoom"), "hybrid": (f_hybrid, True, "zoom"),
             "hybrid_bt": (f_hybrid, True, "backtracking")}
    solvers = {(prec, c): make_solver(kinds[prec][0], c, *kinds[prec][1:])
               for prec in args.precisions if prec in kinds for c in args.caps}

    for fips in args.pumas:
        p = load(fips, key, "./llcache_minimal")
        p["repwt"] = replicate_weights(fips, p["pup"].est_ind.index, p["wt"], key,
                                       "./repwt_cache")
        G = p["A1"].shape[1]
        acs_cv = (moe["se"] / moe["acs"]).reindex(p["pup"].est_g2.index.astype(str)).to_numpy()
        rng = np.random.default_rng(7)
        draws = [(prior_rep(p, rng), nested_Y(p, rng)) for _ in range(args.draws)]
        put = lambda a, dt: jax.device_put(jnp.asarray(a, dt), dev)
        C64, A64, V64 = (put(p[k], jnp.float64) for k in ("C", "A1", "V"))

        def run(prec, fn, w, Y):
            lq = logq_of(w, G)
            if prec.startswith("inc"):
                a = [C64, put(lq, jnp.float64), A64, put(Y, jnp.float64), V64]
                init = jnp.zeros(len(Y), jnp.float64)
            elif prec == "f32":
                a = [put(x, jnp.float32) for x in (p["C"], lq, p["A1"], Y, p["V"])]
                init = jnp.zeros(len(Y), jnp.float32)
            elif prec.startswith("hybrid"):
                a = [C64, put(p["C"], jnp.float32), put(lq, jnp.float64), A64,
                     put(Y, jnp.float64), V64]
                init = jnp.zeros(len(Y), jnp.float64)
            else:
                C = put(p["C"], jnp.float32) if prec == "mixed" else C64
                a = [C, put(lq, jnp.float64), A64, put(Y, jnp.float64), V64]
                init = jnp.zeros(len(Y), jnp.float64)
            t0 = time.perf_counter()
            lam, it, err = fn(init, *a)
            jax.block_until_ready(lam)
            t = time.perf_counter() - t0
            al = np.asarray(allocation(jnp.asarray(lam, jnp.float64), C64, put(lq, jnp.float64),
                                       A64, p["N"]))
            return al, int(it), float(err), t

        print(f"PUMA {fips}: {p['C'].shape[0]} donors x {G} block groups", flush=True)
        K, T = p["C"].shape[1], p["A1"].shape[0]
        inc = {(prec, c): make_incremental_solver(
                   c, K, T, G, jnp.float32 if prec == "inc32" else jnp.float64)
               for prec in args.precisions if prec.startswith("inc") for c in args.caps}
        run("f64", conv, *draws[0])                                   # compile
        ref = [run("f64", conv, w, Y) for w, Y in draws]
        ref_als = [r[0] for r in ref]

        def report(label, res):
            als = [r[0] for r in res]
            moved = [0.5 * np.abs(a - b).sum() / b.sum() for a, b in zip(als, ref_als)]
            pops = np.array([a.T @ p["C"][:, 0] for a in als])
            m = pops.mean(0)
            cv = pops.std(0, ddof=1) / np.where(m > 0, m, np.nan)
            ok = np.isfinite(cv) & np.isfinite(acs_cv)
            finite = all(np.isfinite(a).all() for a in als)
            print(f"  {label:18s} {np.mean([r[3] for r in res]):6.2f} s/draw   iters "
                  f"{min(r[1] for r in res)}-{max(r[1] for r in res)}   final grad "
                  f"{max(r[2] for r in res):.1e}   vs f64 converged: mean "
                  f"{100 * np.mean(moved):.2f}% max {100 * np.max(moved):.2f}%   ratio "
                  f"{np.median(cv[ok]) / np.median(acs_cv[ok]):.3f}"
                  + ("" if finite else "   NON-FINITE"), flush=True)
            print(f"  {'':18s} " + composition.line(composition.measure(
                als, p["C"], p["cols"], p["A1"], p["est1"], p["est2"], p["se1"], p["se2"])),
                flush=True)

        report("f64 converged", ref)
        for (prec, c), fn in {**solvers, **inc}.items():
            run(prec, fn, *draws[0])                                  # compile
            report(f"{prec} cap {c}", [run(prec, fn, w, Y) for w, Y in draws])
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
