#!/usr/bin/env -S python -u
"""Generate a fresh ExaEpi population from a version-2 precompute bundle (reference generator).

This is the Python reference ("oracle") for ExaEpi's in-process population generation: the same
bundle, the same stages and the same keyed random draws the C++ port implements, so every C++
stage can be checked against it. It is also usable on its own -- its output is a `.bin` that
today's ExaEpi runs unchanged -- which gives per-run fresh populations before any C++ exists.

Stages (popgen/):

    perturb     targets redrawn within their ACS standard errors (block groups first, tracts
                rebuilt from them) and the PUMS prior weights by a Bayesian bootstrap
    solve       P-MEDM re-solve per PUMA (float32 incremental L-BFGS, popgen/solver.py)
    place       livelike-style synthesis: household reweighting, TRS by household type x size,
                donor draws per (block group, type); expansion to persons (bg, h, p)
    S1-S2       childcare and the worker/student split (persons.py)
    S3          worker destinations: CBP-sized establishment slots, IPF fill (workers.py)
    S4-S5       student and teacher school assignment (students.py, teachers.py)
    S6-S10      dense ids, home and work groups, school classes, day neighbourhoods (groups.py)

Every draw is keyed on (seed, rep, stage, global identifiers) through KR64, so the result depends
only on the bundle, the seed and the replicate number -- not on processing order.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from popgen import (bundle, cbp, groups, perturb, persons, placement, solver,  # noqa: E402
                    students, teachers, workers)
from popgen.problem import PumaProblem, puma_count  # noqa: E402


def generate(b, seed, rep=0, pumas=None, verbose=True):
    """Placements for every (or the named) PUMA: (bg_geoid, donor_global, count) plus stats."""
    names = bundle.strings(b, "solve.puma")
    cols = bundle.strings(b, "solve.constraints")
    parts, stats = [], []
    for p in range(puma_count(b)):
        if pumas and names[p] not in pumas:
            continue
        t0 = time.perf_counter()
        prob = PumaProblem(b, p)
        Y, _ = perturb.perturbed_targets(prob, seed, rep)
        logq = perturb.log_prior(perturb.perturbed_prior(prob, seed, rep), prob.G)
        al, iters, gnorm = solver.solve(prob, Y, logq)
        bg, row, ct = placement.place_puma(b, prob, al, cols, seed, rep)
        parts.append((bg, prob.donor_index[row].astype(np.int64), ct))
        stats.append(dict(puma=names[p], donors=prob.D, bgs=prob.G, iters=iters,
                          grad=gnorm, households=int(ct.sum()), seconds=time.perf_counter() - t0))
        if verbose:
            s = stats[-1]
            print(f"  PUMA {s['puma']}: {s['donors']} donors x {s['bgs']} bgs, {s['iters']} "
                  f"iterations, {s['households']} households, {s['seconds']:.2f} s", flush=True)
    placements = tuple(np.concatenate([x[i] for x in parts]) for i in range(3))
    return placements, stats


def assign(b, pers, seed, rep=0, verbose=True):
    """Stages S1-S10 on expanded persons; returns the agent columns the .bin carries."""
    times = {}

    def lap(name, t0):
        times[name] = time.perf_counter() - t0
        return time.perf_counter()

    t = time.perf_counter()
    P = persons.build(b, pers, seed, rep)
    t = lap("S1-S2", t)
    tables = cbp.SizeTables(b)
    work, wst = workers.allocate(b, P, tables, seed, rep)
    t = lap("S3 workers", t)
    school, sst = students.allocate(b, P, work, seed, rep)
    t = lap("S4 students", t)
    tst = teachers.allocate(b, P, work, school, seed, rep)
    t = lap("S5 teachers", t)
    sid = groups.school_ids(b, P, work, school)
    nb, hhc = groups.home_groups(P, seed, rep)
    wg, wgrp = groups.work_groups(P, work, sid, tables, seed, rep)
    scls, scg = groups.school_groups(b, P, work, school, sid, seed, rep)
    wnb = groups.day_neighborhoods(b, P, work, school, sid, wg, scls, scg, seed, rep)
    t = lap("S6-S10 groups", t)
    if verbose:
        print("  " + ", ".join(f"{k} {v:.1f} s" for k, v in times.items()))
        print(f"  workers {wst['workers']}, fallback {wst['fallback']}; students unplaced by level "
              + ", ".join(f"{k} {u}" for k, (n, u) in sst.items())
              + "; teachers placed " + ", ".join(f"{k} {g}/{r}" for k, (r, g) in tst.items()))
    cols = {
        "id": P["id"], "home_geoid": P["bg"], "work_geoid": work,
        "school_class_group": scg, "work_group": wgrp, "naics": P["naics"],
        "household_id": P["h"], "school_id": sid, "nborhood": nb, "work_nborhood": wnb,
        "workgroup": wg, "hh_cluster": hhc, "school_class": scls, "age": P["age"],
        "sex": P["sex"], "race": P["race"], "travel": P["travel"], "veh_occ": P["veh_occ"],
        "grade": P["grade"],
    }
    return cols


# Column types of the .bin record (UrbanPopAgentStruct.H), in on-disk order.
BIN_TYPES = [("id", "Int64"), ("home_geoid", "Int64"), ("work_geoid", "Int64"),
             ("school_class_group", "Int32"), ("work_group", "Int32"), ("naics", "Int16"),
             ("household_id", "Int16"), ("school_id", "Int16"), ("nborhood", "Int16"),
             ("work_nborhood", "Int16"), ("workgroup", "Int16"), ("hh_cluster", "Int16"),
             ("school_class", "Int16"), ("age", "Int8"), ("sex", "Int8"), ("race", "Int8"),
             ("travel", "Int8"), ("veh_occ", "Int8"), ("grade", "Int8")]


def write_bin(cols, out_prefix):
    """Write the population as <out_prefix>.bin with upop_to_exaepi's own writer."""
    import polars as pl

    import upop_to_exaepi as U

    for name, _ in BIN_TYPES:
        if name in ("id", "home_geoid", "work_geoid"):
            continue
        lim = {"Int32": 2**31, "Int16": 2**15, "Int8": 2**7}[dict(BIN_TYPES)[name]]
        v = np.asarray(cols[name])
        if len(v) and (v.max() >= lim or v.min() < -lim):
            raise SystemExit(f"{name} does not fit the .bin's {dict(BIN_TYPES)[name]} field")
    df = pl.DataFrame({name: pl.Series(name, np.asarray(cols[name]), dtype=getattr(pl, t))
                       for name, t in BIN_TYPES})
    U.sanity_checks(df)
    U.print_agents(df, out_prefix)


def compare(pers, feathers):
    """Totals, per-block-group population correlation and mean age against delivered feathers."""
    import polars as pl

    ref = pl.concat([pl.read_ipc(p, columns=["geoid", "pr_age"]) for p in feathers])
    rc = ref.group_by("geoid").agg(pl.len().alias("n"))
    ref_n = dict(zip(rc["geoid"].cast(pl.Int64).to_list(), rc["n"].to_list()))
    g, n = np.unique(pers["bg"], return_counts=True)
    keys = sorted(set(ref_n) | set(g.tolist()))
    gen_n = dict(zip(g.tolist(), n.tolist()))
    a = np.array([gen_n.get(k, 0) for k in keys])
    r = np.array([ref_n.get(k, 0) for k in keys])
    return dict(persons=int(a.sum()), delivered=int(r.sum()),
                corr=float(np.corrcoef(a, r)[0, 1]), ref_age=float(ref["pr_age"].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--pumas", nargs="*", default=None, help="restrict to these PUMAs")
    ap.add_argument("--compare", default=None, help="glob of delivered feathers to score against")
    ap.add_argument("--out", default=None, help="write the person arrays to this .npz")
    ap.add_argument("--bin", default=None,
                    help="run stages S1-S10 and write the population to <BIN>.bin for ExaEpi")
    args = ap.parse_args()

    t0 = time.perf_counter()
    b = bundle.read(args.bundle)
    print(f"bundle {args.bundle}: {puma_count(b)} PUMAs, {len(b['bg.geoid'])} block groups")
    placements, stats = generate(b, args.seed, args.rep, args.pumas)
    pers = placement.expand(b, placements)
    age = b["donors.age"][pers["src"]]
    print(f"\n{len(pers['bg'])} persons in {pers['n_households']} households, "
          f"{len(np.unique(pers['bg']))} block groups, mean age {age.mean():.2f}; "
          f"solve+place {sum(s['seconds'] for s in stats):.1f} s, total "
          f"{time.perf_counter() - t0:.1f} s")
    if args.compare:
        c = compare(pers, sorted(glob.glob(args.compare)))
        print(f"vs delivered: {c['persons']} vs {c['delivered']} persons "
              f"({100 * c['persons'] / c['delivered'] - 100:+.2f}%), per-block-group correlation "
              f"{c['corr']:.4f}, delivered mean age {c['ref_age']:.2f}")
    if args.out:
        np.savez_compressed(args.out, **{k: v for k, v in pers.items() if k != "n_households"})
        print(f"wrote {args.out}")
    if args.bin:
        cols = assign(b, pers, args.seed, args.rep)
        write_bin(cols, args.bin)
        print(f"total {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
