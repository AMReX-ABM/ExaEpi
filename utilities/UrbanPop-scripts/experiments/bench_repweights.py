#!/usr/bin/env python
"""Compare the ACS replicate-weight path against the Hessian path for generating replicates.

Measured so far on PUMA 3500804 (dual 31,886):

    solve alone                      1.9 s,  8.14 GB
    solve + Hessian replicates     331   s, 42-46 GB   <- independent of n_reps
    container memory cap                    48    GB

The Hessian path leaves under 2 GB of headroom on the median PUMA and should not fit the four
largest NM PUMAs at all, and its memory is structural -- n_reps=2 cost the same as n_reps=20, so
batching does not help.

The alternative is to re-solve once per ACS replicate weight. Each solve is cheap, so N replicates
should cost roughly N x 5.7 s at the solve's 8 GB, which would fit comfortably, restore
cross-PUMA parallelism, and rest on the Census's own published replicate weights rather than a
quadratic approximation at the converged dual.

This measures whether that holds, and whether the resulting spread matches the Hessian path's
per-block-group population CV of 0.1425 against an ACS CV of 0.1604.
"""

import argparse
import os
import resource
import time

import numpy as np
import pandas as pd


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


class Phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        print(f"--- {self.name} ...", flush=True)
        return self

    def __exit__(self, *exc):
        self.dt = time.time() - self.t0
        print(f"--- {self.name}: {self.dt:.1f} s   peak RSS {rss_gb():.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fips", required=True)
    ap.add_argument("--nreps", type=int, default=8)
    ap.add_argument("--moe_csv", default="/workspaces/ExaEpi/data/UrbanPop/acs_moe_35.csv")
    ap.add_argument("--cache_folder", default="./livelike_acs_cache")
    args = ap.parse_args()

    from livelike import config, multi
    from pymedm import PMEDM

    key = os.environ.get("CENSUS_API_KEY") or None
    print(f"PUMA {args.fips}   nreps {args.nreps}   key: {'set' if key else 'NOT set'}")

    with Phase(f"make_replicate_pumas nreps={args.nreps}") as p_build:
        pumas = multi.make_replicate_pumas(
            args.fips,
            constraints_selection=config.up_expanded_constraints_selection,
            constraints_theme_order=config.up_constraints_theme_order,
            year=2019, target_zone="bg", nreps=args.nreps,
            censusapikey=key, cache_folder=args.cache_folder,
        )
    print(f"    got {len(pumas)} replicate puma objects: {list(pumas)[:5]}{' ...' if len(pumas) > 5 else ''}")

    almats, solve_times = [], []
    for i, (name, pup) in enumerate(pumas.items()):
        t0 = time.time()
        pmd = PMEDM(
            pup.year, pup.est_ind.index, pup.wt,
            pup.est_ind, pup.est_g1, pup.est_g2, pup.se_g1, pup.se_g2,
        )
        pmd.solve()
        dt = time.time() - t0
        solve_times.append(dt)
        almats.append(np.asarray(pmd.almat))
        print(f"    [{i + 1}/{len(pumas)}] {name}: build+solve {dt:.1f} s   peak RSS {rss_gb():.2f} GB",
              flush=True)
        del pmd

    ref = list(pumas.values())[0]
    hh_people = ref.sporder.groupby(level=0).size().reindex(ref.est_ind.index).fillna(1).to_numpy()
    total_pop = float(ref.est_g2.iloc[:, 0].sum()) if ref.est_g2.shape[1] else 1.0

    # Replicate weight sets can differ in donor count, so align on the reference's donors.
    pops = []
    for m, (_, pup) in zip(almats, pumas.items()):
        w = pup.sporder.groupby(level=0).size().reindex(pup.est_ind.index).fillna(1).to_numpy()
        pops.append(m.T @ w * total_pop)
    n_bg = min(len(p) for p in pops)
    pops = np.vstack([p[:n_bg] for p in pops])

    mean = pops.mean(axis=0)
    cv = pd.Series(pops.std(axis=0, ddof=1) / np.where(mean > 0, mean, np.nan),
                   index=ref.est_g2.index[:n_bg]).dropna()

    print()
    print("=== replicate-weight per-block-group population spread ===")
    print(f"  block groups {len(cv)}   replicates {len(almats)}")
    print(f"  CV  median {cv.median():.4f}  mean {cv.mean():.4f}  p90 {cv.quantile(.9):.4f}  max {cv.max():.4f}")

    if os.path.exists(args.moe_csv):
        moe = pd.read_csv(args.moe_csv, dtype={"geoid": str})
        moe = moe[moe.acs > 0].set_index("geoid")
        acs_cv = (moe["se"] / moe["acs"]).reindex(cv.index.astype(str)).dropna()
        if len(acs_cv):
            print(f"  ACS CV, same block groups: median {acs_cv.median():.4f}")
            print(f"  ratio: {cv.median() / acs_cv.median():.3f}"
                  f"   (Hessian path 0.889 at n=20, TRS 0.028)")

    tot = sum(solve_times)
    print()
    print("=== summary ===")
    print(f"  make_replicate_pumas {p_build.dt:.1f} s | {len(almats)} solves {tot:.1f} s "
          f"({np.mean(solve_times):.1f} s each) | peak RSS {rss_gb():.2f} GB")
    print(f"  Hessian path, same PUMA: 331 s, 42-46 GB peak, cap 48 GB")
    if rss_gb() < 20:
        n_par = int(46 / max(rss_gb(), 1))
        print(f"  => fits with headroom; ~{n_par} PUMAs could run concurrently under the 48 GB cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
