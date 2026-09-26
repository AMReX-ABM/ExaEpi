#!/usr/bin/env python
"""Does solving different PUMAs concurrently fill the GPU better, without breaking invariance?

batch_draws.py showed that vmapping draws of one PUMA fills the GPU by ~8 draws (power 30 -> ~88
W) but gains at most 1.4x, and that a batched solve rounds differently from a solo one, so a capped
result depends on the batch it ran in. Solving DIFFERENT PUMAs concurrently, each alone at its own
shape, cannot change any solve's arithmetic, so it should be invariant; the question is throughput.

In JAX all threads of one process share a single GPU compute stream, so concurrency means separate
processes. Without MPS the driver time-slices between their contexts; under MPS (NVIDIA
Multi-Process Service) their kernels genuinely run side by side.

Each PUMA is first solved alone (its solo time and a hash of every solution); then the first N
PUMAs are solved concurrently, N = 2, 4, 8. Speed-up = sum of their solo times / concurrent wall
time. Each worker loads and compiles before a shared start signal, so only solving is timed.

Run from data/UrbanPop/experiments with JAX_PLATFORMS=cuda. `--worker` is internal.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PUMAS8 = ["3500804", "3500300", "3500100", "3501100", "3500500", "3500600", "3501200", "3500900"]


def worker(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import numpy as np
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    sys.path.insert(0, HERE)
    from gpu_solve import load
    from lbfgs_incremental import make_incremental_solver
    from resolve_spread import logq_of, nested_Y, prior_rep, replicate_weights

    key = os.environ.get("CENSUS_API_KEY") or None
    p = load(args.worker, key, "./llcache_minimal")
    p["repwt"] = replicate_weights(args.worker, p["pup"].est_ind.index, p["wt"], key,
                                   "./repwt_cache")
    K, (T, G) = p["C"].shape[1], p["A1"].shape
    rng = np.random.default_rng(7)
    draws = [(prior_rep(p, rng), nested_Y(p, rng)) for _ in range(args.draws)]
    solve = make_incremental_solver(args.iters, K, T, G)
    C, A1, V = (jnp.asarray(p[k]) for k in ("C", "A1", "V"))
    z = jnp.zeros(len(p["Y"]))
    ins = [(jnp.asarray(logq_of(w, G)), jnp.asarray(y)) for w, y in draws]
    jax.block_until_ready(solve(z, C, ins[0][0], A1, ins[0][1], V))       # compile
    open(args.ready, "w").close()
    while not os.path.exists(args.go):
        time.sleep(0.01)
    t0 = time.perf_counter()
    out = [solve(z, C, lq, A1, y, V)[0] for lq, y in ins]
    jax.block_until_ready(out)
    t = time.perf_counter() - t0
    hashes = [hashlib.sha256(np.asarray(l).tobytes()).hexdigest()[:16] for l in out]
    with open(args.out, "w") as fh:
        json.dump(dict(puma=args.worker, t=t, hashes=hashes), fh)


def run_group(pumas, draws, iters, tmp):
    """Launch one worker per PUMA, start them together, return per-PUMA results and wall time."""
    os.makedirs(tmp, exist_ok=True)
    go = os.path.join(tmp, "go")
    if os.path.exists(go):
        os.remove(go)
    procs, files = [], []
    for fp in pumas:
        ready, out = os.path.join(tmp, f"ready_{fp}"), os.path.join(tmp, f"out_{fp}.json")
        for f in (ready, out):
            if os.path.exists(f):
                os.remove(f)
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--worker", fp, "--draws", str(draws), "--iters",
             str(iters), "--ready", ready, "--go", go, "--out", out],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        files.append((ready, out))
    while not all(os.path.exists(r) for r, _ in files):
        if any(pr.poll() not in (None, 0) for pr in procs):
            raise RuntimeError("a worker failed before starting")
        time.sleep(0.05)
    t0 = time.perf_counter()
    open(go, "w").close()
    for pr in procs:
        pr.wait()
    wall = time.perf_counter() - t0
    res = {}
    for fp, (_, out) in zip(pumas, files):
        with open(out) as fh:
            res[fp] = json.load(fh)
    return res, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--ready")
    ap.add_argument("--go")
    ap.add_argument("--out")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--groups", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--label", default="")
    ap.add_argument("--tmp", default="./concurrent_tmp")
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    print(f"{args.label}{args.draws} draws per PUMA, {args.iters} iterations, float32 "
          f"incremental L-BFGS\n", flush=True)
    solo = {}
    for fp in PUMAS8[:max(args.groups)]:
        r, _ = run_group([fp], args.draws, args.iters, args.tmp)
        solo[fp] = r[fp]
        print(f"  solo {fp}: {r[fp]['t']:.2f} s", flush=True)
    print(f"\n  {'N':>3s}{'sum of solo s':>15s}{'concurrent s':>14s}{'speed-up':>10s}"
          f"{'identical to solo':>20s}", flush=True)
    for n in args.groups:
        group = PUMAS8[:n]
        res, wall = run_group(group, args.draws, args.iters, args.tmp)
        same = sum(res[fp]["hashes"] == solo[fp]["hashes"] for fp in group)
        ssum = sum(solo[fp]["t"] for fp in group)
        print(f"  {n:3d}{ssum:15.2f}{wall:14.2f}{ssum / wall:9.2f}x{same:>14d} of {n}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
