#!/usr/bin/env python
"""All 18 New Mexico PUMAs solved one after another, to convergence, on GPU and CPU.

gpu_batch.py established that solving PUMAs sequentially at their own shapes beats every batched
arrangement tried (vmapped, padded, or one summed L-BFGS). This measures what that costs when each
solve is run to pymedm's own gradient tolerance (1e-6) -- which pymedm itself never reaches,
because it inherits jaxopt's 500-iteration cap -- and how much that cap moves the answer
statewide, as households placed differently from the converged allocation.

JIT compilation (one per PUMA shape) is reported separately and cached on disk in ./jax_cache; it
is an artifact of JAX and would not exist in a compiled port.
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
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_solve import PUMAS, allocation, load, make_solver, pad  # noqa: E402


def solve(b, dev, maxiter):
    fn = make_solver(maxiter, 1e-6)
    args = [jax.device_put(a, dev) for a in b]
    t0 = time.perf_counter(); jax.block_until_ready(fn(*args)); tc = time.perf_counter() - t0
    t0 = time.perf_counter(); out = fn(*args); jax.block_until_ready(out)
    return out, time.perf_counter() - t0, tc


def main():
    key = os.environ.get("CENSUS_API_KEY") or None
    devs = {"GPU f64": jax.devices("cuda")[0], "CPU f64": jax.devices("cpu")[0]}
    problems = [load(fp, key, "./llcache_minimal") for fp in PUMAS]

    res = {d: [] for d in devs}
    moved_total, hh_total = 0.0, 0.0
    print(f"{'PUMA':9s}{'cells':>9s}" + "".join(f"{d + ' iters':>15s}{d + ' s':>11s}" for d in devs)
          + f"{'obj diff':>10s}{'cap moves':>11s}")
    for p in problems:
        b = pad([p], [p["Y"]])
        nd, G = p["C"].shape[0], p["A1"].shape[1]
        row = []
        for d, dev in devs.items():
            out, t, tc = solve(b, dev, 50000)
            res[d].append(dict(t=t, tc=tc, iters=int(out[1][0]), err=float(out[2][0]),
                               obj=float(out[3][0]), params=np.asarray(out[0][0])))
            row.append(res[d][-1])
        # Households the 500-iteration cap places differently from the converged answer.
        capped, _, _ = solve(b, devs["GPU f64"], 500)
        a64 = [jnp.asarray(x[0], jnp.float64) for x in b[:3]]
        conv_al = np.asarray(allocation(jnp.asarray(row[0]["params"]), *a64, p["N"]))[:nd, :G]
        cap_al = np.asarray(allocation(jnp.asarray(capped[0][0]), *a64, p["N"]))[:nd, :G]
        mv = 0.5 * np.abs(conv_al - cap_al).sum()
        moved_total += mv; hh_total += conv_al.sum()
        print(f"{p['fips']:9s}{nd * G:9,d}"
              + "".join(f"{r['iters']:15,d}{r['t']:11.2f}" for r in row)
              + f"{abs(row[0]['obj'] - row[1]['obj']):10.1e}{100 * mv / conv_al.sum():10.2f}%",
              flush=True)

    print()
    for d in devs:
        r = res[d]
        print(f"{d}: {sum(x['t'] for x in r):7.1f} s to solve all 18 to convergence "
              f"(+{sum(x['tc'] for x in r):.0f} s JIT compile); iterations "
              f"{min(x['iters'] for x in r):,}-{max(x['iters'] for x in r):,}; "
              f"worst final gradient {max(x['err'] for x in r):.1e}")
    g, c = (sum(x["t"] for x in res[d]) for d in devs)
    print(f"GPU speed-up over CPU: {c / g:.1f}x")
    print(f"500-iteration cap places {100 * moved_total / hh_total:.2f}% of NM households "
          f"({moved_total:,.0f} of {hh_total:,.0f}) differently from the converged allocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
