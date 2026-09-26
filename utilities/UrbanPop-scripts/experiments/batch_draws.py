#!/usr/bin/env python
"""Does batching draws of one PUMA into a single solve fill the GPU?

gpu_batch.py found that vmapping jaxopt's L-BFGS over many problems gained at most ~1.5x: its zoom
line search is a data-dependent loop, so the whole batch waits for the slowest problem at every
step. lbfgs_incremental.py has no data-dependent loop -- a fixed iteration count, and all candidate
line-search steps evaluated at once -- so draws can run in lockstep. At one draw its kernels are
small (a 2,226 x 80 GEMM output is a few dozen thread blocks on a ~36-SM GPU) and most of the GPU
idles even while nvidia-smi reports 95% "utilization", which counts time with any kernel running.

For each PUMA, B draws of the recommended perturbation (nested targets + replicate-weight prior)
are solved in one vmapped call, sharing the donor matrix, at B = 1, 2, 4, ..., and compared against
the same draws solved one at a time: wall time per draw, speed-up, and the largest difference in
allocation between batched and unbatched solves (they run the same arithmetic, so this should be
at rounding level). Power draw is sampled from nvidia-smi during each batch as a rough measure of
how much of the chip is working.

Run from data/UrbanPop/experiments with JAX_PLATFORMS=cuda,cpu.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_solve import load  # noqa: E402
from lbfgs_incremental import make_incremental_solver  # noqa: E402
from resolve_spread import logq_of, nested_Y, prior_rep, replicate_weights  # noqa: E402


class PowerSampler:
    """Mean nvidia-smi power draw and SM clock while a block of work runs."""

    def __enter__(self):
        self.samples, self.stop = [], False
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu", "--format=csv,noheader,nounits",
             "-lms", "100"], stdout=subprocess.PIPE, text=True)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()
        return self

    def _read(self):
        for ln in self.proc.stdout:
            try:
                self.samples.append([float(x) for x in ln.split(",")])
            except ValueError:
                pass

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()
        s = np.array(self.samples) if self.samples else np.full((1, 2), np.nan)
        self.watts, self.util = float(s[:, 0].mean()), float(s[:, 1].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pumas", nargs="+", default=["3500804", "3500300", "3500100"])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    key = os.environ.get("CENSUS_API_KEY") or None
    dev = jax.devices()[0]
    print(f"device {dev.device_kind}; float32 incremental L-BFGS, {args.iters} iterations\n",
          flush=True)
    idle = PowerSampler()
    with idle:
        time.sleep(2)
    print(f"idle power {idle.watts:.1f} W\n", flush=True)

    for fips in args.pumas:
        p = load(fips, key, "./llcache_minimal")
        p["repwt"] = replicate_weights(fips, p["pup"].est_ind.index, p["wt"], key,
                                       "./repwt_cache")
        K, (T, G) = p["C"].shape[1], p["A1"].shape
        rng = np.random.default_rng(7)
        nmax = max(args.batches)
        draws = [(prior_rep(p, rng), nested_Y(p, rng)) for _ in range(nmax)]
        LQ = np.stack([logq_of(w, G) for w, _ in draws])
        YY = np.stack([y for _, y in draws])
        C, A1, V = (jax.device_put(jnp.asarray(p[k]), dev) for k in ("C", "A1", "V"))
        single = make_incremental_solver(args.iters, K, T, G)
        batched = jax.jit(jax.vmap(single, in_axes=(0, None, 0, None, 0, None)))

        def alloc(l, lq):
            v0, v1, v2 = l[:K], l[K:K + K * T].reshape(K, T), l[K + K * T:].reshape(K, G)
            E = lq - C @ (v0[:, None] + v1 @ A1 + v2)
            return jnp.exp(E - jax.scipy.special.logsumexp(E)) * p["N"]

        # Reference: the first 8 draws one at a time (timed), for the batched-vs-unbatched check.
        n_ref = min(8, nmax)
        z = jnp.zeros(len(p["Y"]))
        jax.block_until_ready(single(z, C, jnp.asarray(LQ[0]), A1, jnp.asarray(YY[0]), V))
        t0 = time.perf_counter()
        ref = [single(z, C, jnp.asarray(LQ[i]), A1, jnp.asarray(YY[i]), V)[0]
               for i in range(n_ref)]
        jax.block_until_ready(ref)
        t_one = (time.perf_counter() - t0) / n_ref
        ref_al = [np.asarray(alloc(l, jnp.asarray(LQ[i]))) for i, l in enumerate(ref)]
        print(f"PUMA {fips}: {p['C'].shape[0]} donors x {G} block groups, dual {len(p['Y'])}; "
              f"one draw alone {t_one:.3f} s", flush=True)
        print(f"  {'B':>4s}{'batch s':>10s}{'s/draw':>9s}{'speed-up':>10s}{'power W':>9s}"
              f"{'max diff vs unbatched':>24s}", flush=True)

        for B in args.batches:
            args_b = (jnp.zeros((B, len(p["Y"]))), C, jnp.asarray(LQ[:B]), A1,
                      jnp.asarray(YY[:B]), V)
            try:
                jax.block_until_ready(batched(*args_b))                 # compile
                with PowerSampler() as ps:
                    t0 = time.perf_counter()
                    out = batched(*args_b)
                    jax.block_until_ready(out)
                    t = time.perf_counter() - t0
            except Exception as e:                                      # e.g. out of memory
                print(f"  {B:4d}  failed: {type(e).__name__}: {str(e).splitlines()[0][:80]}",
                      flush=True)
                break
            diff = max(0.5 * np.abs(np.asarray(alloc(out[0][i], jnp.asarray(LQ[i]))) - ref_al[i]
                                    ).sum() / ref_al[i].sum() for i in range(min(B, n_ref)))
            print(f"  {B:4d}{t:10.3f}{t / B:9.3f}{t_one / (t / B):9.1f}x{ps.watts:9.1f}"
                  f"{100 * diff:22.4f}%", flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
