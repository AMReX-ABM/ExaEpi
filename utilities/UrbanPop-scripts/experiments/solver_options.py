#!/usr/bin/env python
"""Two open settings of the float32 incremental solver (lbfgs_incremental.py), before porting.

A. Line search. "vector" evaluates all 12 candidate steps every iteration -- 12 element-wise passes
   over the (donors x block groups) matrix, a large share of the memory traffic. "first" tries the
   full step alone and evaluates the rest only when it is rejected. Reported: which step each
   iteration accepted, time, and distance from the float64 converged solution.

B. Stopping rule, replacing the fixed cap. Every `check_every` iterations, measure the share of
   households the allocation moved since the previous check; stop below a tolerance or at the
   maximum. Validated on all 18 New Mexico PUMAs against float64 converged solves (jaxopt, gradient
   1e-6) of the same draws, alongside the fixed 2,000-iteration cap.

Draws are the recommended perturbation (nested targets + replicate-weight prior). Distance from
converged = share of households placed differently, the measure used throughout.

Run from data/UrbanPop/experiments with JAX_PLATFORMS=cuda,cpu.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp32_capped import allocation, make_solver  # noqa: E402
from gpu_solve import PUMAS, f, load  # noqa: E402
from lbfgs_incremental import make_incremental_solver  # noqa: E402
from resolve_spread import logq_of, nested_Y, prior_rep, replicate_weights  # noqa: E402


def setup(fips, key, n_draws):
    p = load(fips, key, "./llcache_minimal")
    p["repwt"] = replicate_weights(fips, p["pup"].est_ind.index, p["wt"], key, "./repwt_cache")
    G = p["A1"].shape[1]
    rng = np.random.default_rng(7)
    p["draws"] = [(jnp.asarray(logq_of(w, G)), jnp.asarray(y))
                  for w, y in (( prior_rep(p, rng), nested_Y(p, rng)) for _ in range(n_draws))]
    p["dev"] = {k: jnp.asarray(p[k]) for k in ("C", "A1", "V")}
    return p


def reference(p, conv):
    C, A1, V = p["dev"]["C"], p["dev"]["A1"], p["dev"]["V"]
    z = jnp.zeros(len(p["Y"]))
    out = []
    for lq, y in p["draws"]:
        lam = conv(z, C, lq, A1, y, V)[0]
        out.append(np.asarray(allocation(lam, C, lq, A1, p["N"])))
    return out


def run(p, solver, ref):
    C, A1, V = p["dev"]["C"], p["dev"]["A1"], p["dev"]["V"]
    z = jnp.zeros(len(p["Y"]))
    jax.block_until_ready(solver(z, C, *p["draws"][0][:1], A1, p["draws"][0][1], V))  # compile
    moved, iters, secs, hist = [], [], [], 0
    for (lq, y), r in zip(p["draws"], ref):
        t0 = time.perf_counter()
        out = solver(z, C, lq, A1, y, V)
        jax.block_until_ready(out)
        secs.append(time.perf_counter() - t0)
        al = np.asarray(allocation(out[0], C, lq, A1, p["N"]))
        moved.append(0.5 * np.abs(al - r).sum() / r.sum())
        iters.append(int(out[1]))
        hist = hist + np.asarray(out[3])
    return dict(moved=np.array(moved), iters=np.array(iters), secs=np.array(secs),
                hist=hist, hh=float(ref[0].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--ls_pumas", nargs="+", default=["3500804", "3500300", "3500100"])
    ap.add_argument("--tols", type=float, nargs="+", default=[1e-3, 5e-4, 2e-4])
    ap.add_argument("--check_every", type=int, default=250)
    ap.add_argument("--max_iter", type=int, default=6000)
    ap.add_argument("--skip_a", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("CENSUS_API_KEY") or None
    print(f"device {jax.devices()[0].device_kind}; {args.draws} draws per PUMA; float32 "
          f"incremental L-BFGS, full-precision matmuls\n", flush=True)
    conv = make_solver(f, 50000)
    problems = {}

    def get(fips):
        if fips not in problems:
            p = setup(fips, key, args.draws)
            p["ref"] = reference(p, conv)
            problems[fips] = p
        return problems[fips]

    def solver(p, **kw):
        K, (T, G) = p["C"].shape[1], p["A1"].shape
        return make_incremental_solver(K=K, T=T, G=G, diagnostics=True, **kw)

    if not args.skip_a:
        print("A. line search: 'vector' (all 12 candidates) vs 'first' (full step, rest only if "
              "rejected)", flush=True)
        print(f"  {'PUMA':9s}{'mode':8s}{'iters':>7s}{'s/draw':>8s}{'vs converged':>14s}"
              f"   accepted step 1 / 1/2 / 1/4 / smaller / none", flush=True)
        for fips in args.ls_pumas:
            p = get(fips)
            for it in (1000, 2000):
                for mode in ("vector", "first"):
                    r = run(p, solver(p, maxiter=it, linesearch=mode), p["ref"])
                    h = r["hist"] / r["hist"].sum()
                    print(f"  {fips:9s}{mode:8s}{it:7d}{r['secs'].mean():8.3f}"
                          f"{100 * r['moved'].mean():13.2f}%   "
                          f"{100 * h[0]:.1f}% / {100 * h[1]:.1f}% / {100 * h[2]:.1f}% / "
                          f"{100 * h[3:-1].sum():.1f}% / {100 * h[-1]:.1f}%", flush=True)
        print(flush=True)

    print(f"B. stopping rule: check every {args.check_every} iterations, stop when the allocation "
          f"moved less than the tolerance since the last check, max {args.max_iter}; all 18 PUMAs",
          flush=True)
    configs = [("fixed 2000", dict(maxiter=2000))] + [
        (f"tol {100 * t:g}%", dict(maxiter=args.max_iter, check_every=args.check_every,
                                   tol_moved=t)) for t in args.tols]
    results = {c: [] for c, _ in configs}
    for fips in PUMAS:
        p = get(fips)
        line = f"  {fips:9s}"
        for name, kw in configs:
            r = run(p, solver(p, linesearch="first", **kw), p["ref"])
            results[name].append(r)
            line += (f" | {name}: {r['iters'].mean():5.0f} it {r['secs'].mean():5.2f} s "
                     f"{100 * r['moved'].mean():5.2f}%")
        print(line, flush=True)

    print(f"\n  {'config':12s}{'NM realization s':>18s}{'iters min-max':>16s}"
          f"{'statewide misplaced':>21s}{'worst PUMA mean':>17s}{'worst draw':>12s}", flush=True)
    for name, _ in configs:
        rs = results[name]
        tot_s = sum(r["secs"].mean() for r in rs)
        hh = np.array([r["hh"] for r in rs])
        mv = np.array([r["moved"].mean() for r in rs])
        print(f"  {name:12s}{tot_s:17.2f}s{min(r['iters'].min() for r in rs):>9d}-"
              f"{max(r['iters'].max() for r in rs):<6d}{100 * (mv * hh).sum() / hh.sum():19.3f}%"
              f"{100 * mv.max():16.2f}%{100 * max(r['moved'].max() for r in rs):11.2f}%",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
