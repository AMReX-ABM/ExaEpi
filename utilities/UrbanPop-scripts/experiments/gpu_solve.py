#!/usr/bin/env python
"""How fast is the P-MEDM solve on the GPU, alone and batched?

solve_profile.py showed the P-MEDM objective can be written as two dense matrix products per
evaluation instead of pymedm's 5-million-nonzero sparse Kronecker product: identical to 1e-14, and
17x faster per evaluation on CPU. That estimate predicted a GPU would gain little on one PUMA -- the
matrices are small and L-BFGS is sequential -- but a great deal when many independent problems
(PUMAs, bootstrap replicates) are batched into each step. This measures it.

Three tests, all New Mexico, 123-constraint minimal set:

    A  one PUMA (3500804): CPU vs GPU, float64 vs float32, stopped at 500 iterations as pymedm is,
       checked against pymedm's own allocation; then run to actual convergence (pymedm's gradient
       tolerance, 1e-6), which pymedm never reaches.
    B  all 18 PUMAs in one batched solve.
    C  18 PUMAs x 20 bootstrap replicates = 360 problems, each with its ACS targets perturbed by
       their standard errors. The perturbation here is only a stand-in for timing; it is not the
       validated scheme.

Batching pads every PUMA to the largest shape. Padded households and block groups get a prior of
exp(-1e30), so they receive no mass and contribute nothing; padded tract rows touch no block group.
Their prices start at zero and stay there, since their gradient is zero at zero.

Run from data/UrbanPop/experiments with JAX_PLATFORMS unset or "cuda,cpu", so both are visible.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jaxopt  # noqa: E402

sys.path.insert(0, "/workspaces/ExaEpi/utilities/UrbanPop-scripts")
from build_precompute import EXAEPI_MINIMAL, repair_controlled_se  # noqa: E402

PUMAS = [f"35{p:05d}" for p in (100, 1001, 1002, 1100, 1200, 200, 300, 400, 500, 600, 700,
                                801, 802, 803, 804, 805, 806, 900)]
NEG = -1e30


def load(fips, key, cache, selection=None):
    from livelike import acs, config
    from pymedm import PMEDM

    pup = acs.puma(fips, constraints_selection=selection or EXAEPI_MINIMAL,
                   constraints_theme_order=config.up_constraints_theme_order,
                   year=2019, target_zone="bg", cache=True, cache_folder=cache,
                   censusapikey=key)
    s1, _ = repair_controlled_se(pup.se_g1, np.asarray(pup.est_g1, float), "t")
    s2, _ = repair_controlled_se(pup.se_g2, np.asarray(pup.est_g2, float), "b")
    s1d = pd.DataFrame(s1, index=pup.se_g1.index, columns=pup.se_g1.columns)
    s2d = pd.DataFrame(s2, index=pup.se_g2.index, columns=pup.se_g2.columns)
    pmd = PMEDM(pup.year, pup.est_ind.index, pup.wt, pup.est_ind, pup.est_g1, pup.est_g2,
                s1d, s2d, n_reps=0, random_state=1)
    C = np.asarray(pup.est_ind, float)
    K = C.shape[1]
    A1 = pmd.A1.astype(float)
    Y = np.asarray(pmd.Y_vec, float)
    assert len(Y) == K * (1 + A1.shape[0] + A1.shape[1]), "expected PUMA-level targets"
    return dict(
        fips=fips, pmd=pmd, pup=pup, cols=list(pup.est_ind.columns),
        C=C, A1=A1, Y=Y, V=np.asarray(pmd.V_vec, float),
        wt=np.asarray(pup.wt, float), N=float(pmd.N),
        est1=np.asarray(pup.est_g1, float), est2=np.asarray(pup.est_g2, float),
        se1=s1, se2=s2,
    )


def perturbed_Y(p, rng):
    """ACS targets redrawn from their standard errors, in pymedm's layout and scaling.

    Tract and block-group estimates are drawn independently and clipped at zero; the PUMA-level
    targets are then rebuilt as block-group sums, which is how pymedm derives them.
    """
    y1 = np.maximum(p["est1"] + p["se1"] * rng.standard_normal(p["est1"].shape), 0)
    y2 = np.maximum(p["est2"] + p["se2"] * rng.standard_normal(p["est2"].shape), 0)
    y0 = y2.sum(axis=0)
    return np.concatenate([y0, y1.flatten("F"), y2.flatten("F")]) / p["N"]


def pad(problems, Ys):
    """Stack problems of different shapes into padded batch arrays."""
    K = problems[0]["C"].shape[1]
    nd = max(p["C"].shape[0] for p in problems)
    T = max(p["A1"].shape[0] for p in problems)
    G = max(p["A1"].shape[1] for p in problems)
    B = len(Ys)
    C = np.zeros((B, nd, K)); logq = np.full((B, nd, G), NEG); A1 = np.zeros((B, T, G))
    Yb = np.zeros((B, K * (1 + T + G))); Vb = np.ones_like(Yb)
    for b, (p, y) in enumerate(zip(problems, Ys)):
        n, t, g = p["C"].shape[0], p["A1"].shape[0], p["A1"].shape[1]
        C[b, :n] = p["C"]
        logq[b, :n, :g] = np.log(p["wt"] / p["wt"].sum() / g)[:, None]
        A1[b, :t, :g] = p["A1"]
        for src, dst in ((y, Yb), (p["V"], Vb)):
            o = K + K * t
            dst[b, :K] = src[:K]
            dst[b, K:K + K * T] = np.pad(src[K:o].reshape(K, t), ((0, 0), (0, T - t))).ravel()
            dst[b, K + K * T:] = np.pad(src[o:].reshape(K, g), ((0, 0), (0, G - g))).ravel()
    return C, logq, A1, Yb, Vb


def f(l, C, logq, A1, Y, V):
    """P-MEDM dual objective as two dense matrix products (see solve_profile.py)."""
    K = C.shape[1]
    T, G = A1.shape
    Leff = l[:K, None] + l[K:K + K * T].reshape(K, T) @ A1 + l[K + K * T:].reshape(K, G)
    E = logq - C @ Leff
    return Y @ l + jax.scipy.special.logsumexp(E) + 0.5 * l @ (V * l)


def allocation(l, C, logq, A1, N):
    K = C.shape[1]
    T, G = A1.shape
    Leff = l[:K, None] + l[K:K + K * T].reshape(K, T) @ A1 + l[K + K * T:].reshape(K, G)
    E = logq - C @ Leff
    return jnp.exp(E - jax.scipy.special.logsumexp(E)) * N


def make_solver(maxiter, tol):
    solver = jaxopt.LBFGS(fun=f, tol=tol, maxiter=maxiter, jit=True)

    def one(C, logq, A1, Y, V):
        r = solver.run(jnp.zeros_like(Y), C, logq, A1, Y, V)
        return r.params, r.state.iter_num, r.state.error, f(r.params, C, logq, A1, Y, V)

    return jax.jit(jax.vmap(one))


def run(batch, device, dtype, maxiter, tol, chunk):
    """Solve a padded batch in chunks on one device. Returns results and timings."""
    run_fn = make_solver(maxiter, tol)
    arrs = [a.astype(dtype) for a in batch]
    B = arrs[0].shape[0]
    chunks = [(i, min(i + chunk, B)) for i in range(0, B, chunk)]
    put = lambda s, e: [jax.device_put(a[s:e], device) for a in arrs]

    # Compile on the first chunk's shape; a short last chunk compiles once more, inside the timing.
    t0 = time.perf_counter()
    jax.block_until_ready(run_fn(*put(*chunks[0])))
    t_compile = time.perf_counter() - t0

    out = []
    t0 = time.perf_counter()
    for s, e in chunks:
        r = run_fn(*put(s, e))
        jax.block_until_ready(r)
        out.append([np.asarray(x) for x in r])
    t_run = time.perf_counter() - t0
    params, iters, err, obj = (np.concatenate([o[i] for o in out]) for i in range(4))
    return dict(params=params, iters=iters, err=err, obj=obj,
                t_compile=t_compile, t_run=t_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./llcache_minimal")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--skip_cpu_bootstrap", action="store_true")
    ap.add_argument("--out", default="./gpu_solve.json")
    args = ap.parse_args()

    key = os.environ.get("CENSUS_API_KEY") or None
    gpu = jax.devices("cuda")[0]
    cpu = jax.devices("cpu")[0]
    print(f"GPU {gpu.device_kind}   CPU {os.cpu_count()} threads\n")

    t0 = time.perf_counter()
    problems = [load(f, key, args.cache) for f in PUMAS]
    print(f"loaded 18 PUMAs in {time.perf_counter() - t0:.1f} s")
    shapes = [(p["C"].shape[0], p["A1"].shape[0], p["A1"].shape[1]) for p in problems]
    print(f"  donors {min(s[0] for s in shapes)}-{max(s[0] for s in shapes)}, "
          f"tracts {min(s[1] for s in shapes)}-{max(s[1] for s in shapes)}, "
          f"block groups {min(s[2] for s in shapes)}-{max(s[2] for s in shapes)}\n")
    results = {}

    # ---- A: one PUMA ---------------------------------------------------------------------
    pA = problems[PUMAS.index("3500804")]
    t0 = time.perf_counter()
    pA["pmd"].solve()
    t_pymedm = time.perf_counter() - t0
    ref = np.asarray(pA["pmd"].almat)
    print(f"A. PUMA 3500804, pymedm on CPU (sparse): {t_pymedm:.2f} s incl. compile")

    single = pad([pA], [pA["Y"]])
    nd, G = pA["C"].shape[0], pA["A1"].shape[1]
    print(f"   {'config':26s}{'solve':>9s}{'compile':>9s}{'iters':>7s}{'grad norm':>11s}"
          f"{'objective':>16s}{'vs pymedm':>11s}")
    for name, dev, dt, mi in [("CPU f64, 500 iter", cpu, np.float64, 500),
                              ("GPU f64, 500 iter", gpu, np.float64, 500),
                              ("GPU f32, 500 iter", gpu, np.float32, 500),
                              ("CPU f64, to tol 1e-6", cpu, np.float64, 50000),
                              ("GPU f64, to tol 1e-6", gpu, np.float64, 50000),
                              ("GPU f32, to tol 1e-6", gpu, np.float32, 50000)]:
        r = run(single, dev, dt, mi, 1e-6, 1)
        al = np.asarray(allocation(jnp.asarray(r["params"][0], jnp.float64),
                                   *[jnp.asarray(a[0], jnp.float64) for a in single[:3]],
                                   pA["N"]))[:nd, :G]
        moved = 0.5 * np.abs(al - ref).sum()
        results[f"A {name}"] = dict(t=r["t_run"], compile=r["t_compile"],
                                    iters=int(r["iters"][0]), err=float(r["err"][0]),
                                    obj=float(r["obj"][0]), moved=float(moved))
        print(f"   {name:26s}{r['t_run']:8.2f}s{r['t_compile']:8.2f}s{int(r['iters'][0]):7d}"
              f"{float(r['err'][0]):11.2e}{float(r['obj'][0]):16.10f}{moved:9.0f} hh")
    print("   ('vs pymedm' = households placed differently from pymedm's own 500-iteration "
          "allocation, of 51,825)\n")

    # ---- B: all 18 PUMAs batched -----------------------------------------------------------
    allb = pad(problems, [p["Y"] for p in problems])
    print("B. all 18 PUMAs in one batch, 500 iterations")
    for name, dev, dt, ch in [("CPU f64", cpu, np.float64, 18),
                              ("GPU f64", gpu, np.float64, 18),
                              ("GPU f32", gpu, np.float32, 18)]:
        r = run(allb, dev, dt, 500, 1e-6, ch)
        results[f"B {name}"] = dict(t=r["t_run"], compile=r["t_compile"])
        print(f"   {name:10s} {r['t_run']:7.2f} s   (compile {r['t_compile']:.1f} s)")

    # ---- C: bootstrap batch -------------------------------------------------------------------
    rng = np.random.default_rng(1)
    probs, Ys = [], []
    for p in problems:
        for _ in range(args.reps):
            probs.append(p)
            Ys.append(perturbed_Y(p, rng))
    bootb = pad(probs, Ys)
    print(f"\nC. {len(Ys)} bootstrap problems (18 PUMAs x {args.reps}), 500 iterations")
    configs = [("GPU f32", gpu, np.float32, 120), ("GPU f64", gpu, np.float64, 40)]
    if not args.skip_cpu_bootstrap:
        configs.insert(0, ("CPU f64", cpu, np.float64, 60))
    for name, dev, dt, ch in configs:
        r = run(bootb, dev, dt, 500, 1e-6, ch)
        results[f"C {name}"] = dict(t=r["t_run"], compile=r["t_compile"], chunk=ch)
        print(f"   {name:10s} {r['t_run']:7.2f} s   ({r['t_run'] / len(Ys) * 1e3:.0f} ms per "
              f"problem; compile {r['t_compile']:.1f} s; chunks of {ch})")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
