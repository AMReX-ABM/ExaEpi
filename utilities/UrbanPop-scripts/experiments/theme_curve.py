#!/usr/bin/env python
"""Which constraint theme buys the most replicate spread per unit of memory?

The bundle's ability to vary residential placement between runs is set by the replicate spread,
and that is what the constraint reduction cost: measured on PUMA 3500804, the 123-constraint
minimal set gives a per-block-group population CV of 0.0494 (ratio 0.308 against an ACS sampling
CV of 0.1604), while up_expanded's 298 constraints give 0.1425 (ratio 0.889). Fit is not the issue
-- minimal fits the variables ExaEpi reads slightly better -- but the P-MEDM Hessian is
(dual x dual), so buying spread by going back to up_expanded costs ~42 GB against a 48 GB cap and
makes a national build sequential.

The question this answers is whether the 0.308 -> 0.889 gap is spread evenly across the dropped
themes or concentrated in one or two. If it is concentrated, most of the uncertainty is available
for a fraction of the memory.

Each theme is added to the minimal set ALONE, so the numbers are marginal contributions rather
than a path-dependent cumulative sequence. A separate cumulative pass can then follow the best
ordering.

Run one variant per process: peak RSS is a high-water mark that never decreases, so several
variants in one process would all report the largest one's footprint.
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspaces/ExaEpi/utilities/UrbanPop-scripts")
from build_precompute import EXAEPI_MINIMAL, repair_controlled_se  # noqa: E402


def variants():
    """minimal, each single addition, and up_expanded as the ceiling."""
    from livelike import config

    base = {k: (list(v) if isinstance(v, list) else v) for k, v in EXAEPI_MINIMAL.items()}

    def plus(**kw):
        d = {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}
        for k, v in kw.items():
            if isinstance(v, list) and isinstance(d.get(k), list):
                d[k] = d[k] + [x for x in v if x not in d[k]]
            else:
                d[k] = v
        return d

    out = {
        "minimal": base,
        "+economic": plus(economic=["hhinc", "ipr"]),
        "+housing": plus(housing=["units", "year_built"]),
        "+hsplat": plus(social=["hsplat"]),
        "+hhtype": plus(demographic=["hhtype"]),
        "+worker_all": plus(worker=True),
        "+student_all": plus(student=True),
        "+mobility_all": plus(mobility=True),
        "up_expanded": config.up_expanded_constraints_selection,
    }

    # Cumulative path, cheapest addition first, so the dual grows as slowly as possible. Single
    # additions each moved the spread by almost nothing while all of them together moved it by
    # +0.576, which is not additive in variance either -- so the question is whether the jump
    # arrives gradually along this path (a usable midpoint exists) or all at once near the end
    # (it does not).
    steps = [
        ("hsplat", dict(social=["hsplat"])),
        ("hhtype", dict(demographic=["hhtype"])),
        ("housing", dict(housing=["units", "year_built"])),
        ("mobility_all", dict(mobility=True)),
        ("economic", dict(economic=["hhinc", "ipr"])),
        ("student_all", dict(student=True)),
        ("worker_all", dict(worker=True)),
    ]
    acc = {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}
    for i, (label, kw) in enumerate(steps, start=1):
        for k, v in kw.items():
            if isinstance(v, list) and isinstance(acc.get(k), list):
                acc[k] = acc[k] + [x for x in v if x not in acc[k]]
            else:
                acc[k] = v
        out[f"cum{i}_{label}"] = {k: (list(v) if isinstance(v, list) else v)
                                  for k, v in acc.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--fips", default="3500804")
    ap.add_argument("--n_reps", type=int, default=20)
    ap.add_argument("--cache_root", default="./theme_cache")
    ap.add_argument("--moe_csv", default="/workspaces/ExaEpi/data/UrbanPop/acs_moe_35.csv")
    ap.add_argument("--out", default="./theme_curve.jsonl")
    args = ap.parse_args()

    from livelike import acs, config
    from pymedm import PMEDM

    sel = variants()[args.variant]
    key = os.environ.get("CENSUS_API_KEY") or None

    # livelike's ACS cache key ignores constraints_selection, so a shared folder would hand one
    # variant another's columns and fail with a length mismatch. One folder per variant.
    cache = os.path.join(args.cache_root, args.variant)
    os.makedirs(cache, exist_ok=True)

    t0 = time.time()
    pup = acs.puma(
        args.fips, constraints_selection=sel,
        constraints_theme_order=config.up_constraints_theme_order,
        year=2019, target_zone="bg", cache=True,
        cache_folder=cache, censusapikey=key,
    )
    # Controlled ACS estimates carry no margin of error, and one NaN makes the whole objective NaN.
    se1, n1 = repair_controlled_se(pup.se_g1, np.asarray(pup.est_g1, dtype=float), "tract")
    se2, n2 = repair_controlled_se(pup.se_g2, np.asarray(pup.est_g2, dtype=float), "bg")
    se1 = pd.DataFrame(se1, index=pup.se_g1.index, columns=pup.se_g1.columns)
    se2 = pd.DataFrame(se2, index=pup.se_g2.index, columns=pup.se_g2.columns)

    n_con = pup.est_ind.shape[1]
    dual = n_con * (pup.est_g2.shape[0] + pup.est_g1.shape[0] + 1)

    pmd = PMEDM(pup.year, pup.est_ind.index, pup.wt,
                pup.est_ind, pup.est_g1, pup.est_g2, se1, se2,
                n_reps=args.n_reps, random_state=1)
    pmd.solve()
    dt = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    reps = np.asarray(pmd.almat_reps)
    if not np.isfinite(reps).all():
        print(f"{args.variant}: NON-FINITE replicates -- solve failed")
        return 1

    hh = pup.sporder.groupby(level=0).size().reindex(pup.est_ind.index).fillna(0).to_numpy()
    pops = np.vstack([reps[i].T @ hh for i in range(reps.shape[0])])
    mean = pops.mean(axis=0)
    cv = pd.Series(pops.std(axis=0, ddof=1) / np.where(mean > 0, mean, np.nan),
                   index=pup.est_g2.index).dropna()

    moe = pd.read_csv(args.moe_csv, dtype={"geoid": str})
    moe = moe[moe.acs > 0].set_index("geoid")
    acs_cv = (moe["se"] / moe["acs"]).reindex(cv.index.astype(str)).dropna()

    rec = {
        "variant": args.variant,
        "constraints": int(n_con),
        "dual": int(dual),
        "cv_median": float(cv.median()),
        "acs_cv_median": float(acs_cv.median()),
        "ratio": float(cv.median() / acs_cv.median()),
        "seconds": round(dt, 1),
        "peak_rss_gb": round(rss, 2),
        "se_repaired": int(n1 + n2),
    }
    print(json.dumps(rec))
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
