#!/usr/bin/env python
"""Why was batching the P-MEDM solve slower than solving PUMAs one at a time?

gpu_solve.py found the opposite of what was expected. One PUMA (3500804) solves in 0.31 s on the
GPU in float64, so all 18 New Mexico PUMAs one after another should take roughly 6 s -- but solving
them as one vmapped batch took 36 s, and on CPU 72 s against ~15 s sequential. Two suspects:

    padding       every PUMA is padded to the largest shape (5,855 households x 113 block groups
                  x 50 tracts): 3.7x the cells of 3500804, 8.5x those of the smallest PUMA
    line search   under vmap, jaxopt's zoom line search is an inner while_loop, so at every step
                  the whole batch waits for whichever problem needs the most line-search
                  evaluations

Tests, each at 500 iterations (pymedm's effective cap) and to convergence (gradient norm 1e-6):

    same18        18 copies of PUMA 3500804, vmapped -- no padding, so any slowdown against 18x
                  the single solve is the batched line search
    sequential    each PUMA solved alone at its own shape (one compile per shape, reported apart)
    sum18         one L-BFGS over all 18 problems' stacked prices, minimising the SUM of their
                  objectives. The problems are independent, so the minimum is the same, but there
                  is one line search for the whole batch instead of 18 interleaved ones
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jaxopt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_solve import PUMAS, f, load, make_solver, pad  # noqa: E402


def timed(fn, *a):
    t0 = time.perf_counter()
    out = fn(*a)
    jax.block_until_ready(out)
    return out, time.perf_counter() - t0


def vmapped(batch, dev, maxiter, tol):
    fn = make_solver(maxiter, tol)
    args = [jax.device_put(a, dev) for a in batch]
    _, tc = timed(fn, *args)
    out, t = timed(fn, *args)
    return out, t, tc


def summed(batch, dev, maxiter, tol):
    """One L-BFGS over the stacked prices of every problem, minimising the sum of objectives."""
    fsum = lambda L, C, logq, A1, Y, V: jnp.sum(jax.vmap(f)(L, C, logq, A1, Y, V))
    solver = jaxopt.LBFGS(fun=fsum, tol=tol, maxiter=maxiter, jit=True)

    @jax.jit
    def go(C, logq, A1, Y, V):
        r = solver.run(jnp.zeros_like(Y), C, logq, A1, Y, V)
        per = jax.vmap(jax.grad(f))(r.params, C, logq, A1, Y, V)
        return (r.params, r.state.iter_num, jnp.linalg.norm(per, axis=1),
                jax.vmap(f)(r.params, C, logq, A1, Y, V))

    args = [jax.device_put(a, dev) for a in batch]
    _, tc = timed(go, *args)
    out, t = timed(go, *args)
    return out, t, tc


def main():
    key = os.environ.get("CENSUS_API_KEY") or None
    gpu, cpu = jax.devices("cuda")[0], jax.devices("cpu")[0]
    problems = [load(fp, key, "./llcache_minimal") for fp in PUMAS]
    pA = problems[PUMAS.index("3500804")]
    single = pad([pA], [pA["Y"]])
    same18 = [np.repeat(a, 18, axis=0) for a in single]
    allb = pad(problems, [p["Y"] for p in problems])
    cells = sum(p["C"].shape[0] * p["A1"].shape[1] for p in problems)
    print(f"18 PUMAs: {cells:,} real cells; padded batch {allb[0].shape[0]} x "
          f"{allb[0].shape[1]} x {allb[1].shape[2]} = {allb[1].size:,} cells "
          f"({allb[1].size / cells:.1f}x)\n")

    conv = {}
    for label, maxiter in (("500 iterations", 500), ("to gradient 1e-6", 50000)):
        print(f"=== {label}")
        for dname, dev in (("GPU f64", gpu), ("CPU f64", cpu)):
            _, t1, _ = vmapped(single, dev, maxiter, 1e-6)
            _, ts, _ = vmapped(same18, dev, maxiter, 1e-6)
            print(f"  {dname}  one PUMA {t1:6.2f} s | 18 identical, vmapped {ts:7.2f} s "
                  f"({ts / t1:.1f}x the single; 18x would mean no batching gain)")

            tseq, tcomp, iters, objs = 0.0, 0.0, [], []
            for p in problems:
                b = pad([p], [p["Y"]])
                out, t, tc = vmapped(b, dev, maxiter, 1e-6)
                tseq += t; tcomp += tc
                iters.append(int(out[1][0])); objs.append(float(out[3][0]))
            print(f"  {dname}  18 PUMAs sequential, own shapes: {tseq:7.2f} s "
                  f"(+{tcomp:.0f} s compiling 18 shapes)  iterations {min(iters)}-{max(iters)}")

            # Capped at 30,000 when running to convergence: a single PUMA needs ~5,200, and an
            # uncapped run that fails to converge would otherwise take most of an hour.
            out, tsum, tc = summed(allb, dev, min(maxiter, 30000), 1e-6)
            gn = np.asarray(out[2])
            print(f"  {dname}  18 PUMAs, one summed L-BFGS:       {tsum:7.2f} s "
                  f"(compile {tc:.0f} s)  iterations {int(out[1])}, worst per-PUMA gradient "
                  f"{gn.max():.1e}")
            if maxiter > 500:
                conv[dname] = (np.array(objs), np.asarray(out[3]))
        print()

    for dname, (seq, summ) in conv.items():
        print(f"converged objectives, sequential vs summed ({dname}): max difference "
              f"{np.abs(seq - summ).max():.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
