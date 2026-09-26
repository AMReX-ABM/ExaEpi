#!/usr/bin/env python
"""What does the greedy fill actually buy, and what would a max-entropy solve give up?

Step 2 of the plan asks whether a P-MEDM O-D solve can replace `alloc_workers`' destination-first
greedy fill, scored against its measured 0.881 LODES flow correlation. But correlation alone is
the wrong scoreboard: the trivial allocator -- send each worker to a destination drawn in
proportion to its own home's real LODES flow -- matches LODES almost by construction, and would
"win" on that metric while producing nothing ExaEpi needs.

The greedy fill deliberately spends LODES fidelity to buy industry concentration at destinations:
it sizes each destination from real inbound flow, splits it into discrete establishment slots,
draws each slot's NAICS from the local commute-shed, and draws each slot's size from the real CBP
establishment-size distribution. That concentration is what makes work groups epidemiologically
meaningful -- a workplace is a mixing unit, and a destination diluted across 251 NAICS codes has
no workplaces in it at all.

So this measures BOTH axes on the same population:

    fidelity     Pearson correlation of generated (home, dest) pair counts vs LODES, exactly as
                 check_flows_correlation computes it (outer join, missing pairs as 0).
    structure    how concentrated each destination's industry mix is, and how the resulting
                 (dest, NAICS) cell sizes -- which become ExaEpi's work groups -- are distributed.

Two allocators here establish the endpoints of the tradeoff:

    greedy       the shipped result, read back from workers_nt_dt.intermediate.csv
    flowprop     per-worker multinomial on the home's own LODES row -- the same fallback
                 alloc_workers uses for leftovers, applied to everyone

A max-entropy solve subject to LODES margins lands at or near `flowprop`, because the entropy term
is maximised by exactly the dispersion `flowprop` produces. That is the claim this script is here
to test before any solver gets written.
"""

import argparse
import os
import sys

import numpy as np
import polars as pl


def load_lodes(path: str) -> pl.DataFrame:
    """Block-group-level LODES flows, matching get_lodes_groups in upop_to_exaepi.py."""
    df = pl.read_csv(path, columns=["w_geocode", "h_geocode", "S000"])
    df = df.with_columns(
        pl.col("w_geocode").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("w_geocode"),
        pl.col("h_geocode").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("h_geocode"),
        pl.col("S000").cast(pl.Int32),
    )
    return df.group_by(["w_geocode", "h_geocode"]).agg(pl.col("S000").sum().alias("count"))


def flow_correlation(home: np.ndarray, work: np.ndarray, lodes: pl.DataFrame) -> tuple:
    """check_flows_correlation, reimplemented on arrays.

    Outer join on (home, dest): a pair the allocator invented that LODES never saw counts as
    LODES 0, and a real LODES pair nobody was sent to counts as generated 0. Both directions
    penalise, which is what makes this a fidelity measure rather than a coverage one.
    """
    gen = (
        pl.DataFrame({"h": home, "w": work})
        .group_by(["h", "w"])
        .agg(pl.len().alias("total"))
        .with_columns((pl.col("h") + "-" + pl.col("w")).alias("key"))
    )
    ref = lodes.with_columns((pl.col("h_geocode") + "-" + pl.col("w_geocode")).alias("key"))
    merged = (
        gen.select(["key", "total"])
        .join(ref.select(["key", "count"]), on="key", how="full", coalesce=True)
        .with_columns(pl.col("total").fill_null(0), pl.col("count").fill_null(0))
    )
    corr = merged.select(pl.corr("total", "count")).item()
    return corr, len(merged), int(merged["total"].sum())


def structure_metrics(work: np.ndarray, naics: np.ndarray) -> dict:
    """How concentrated is each destination's industry mix, and how big are the resulting cells?

    effective NAICS per destination is 1 / sum(p^2) over that destination's NAICS shares -- the
    inverse Simpson index. A destination whose workers are all in one industry scores 1; one
    diluted evenly across k industries scores k. ExaEpi builds work groups inside (dest, NAICS)
    cells, so this is directly the number of distinct workplaces-in-kind a destination supports.
    """
    valid = naics >= 0
    df = pl.DataFrame({"w": work[valid], "n": naics[valid]})
    cell = df.group_by(["w", "n"]).agg(pl.len().alias("cnt"))

    dest_tot = cell.group_by("w").agg(pl.col("cnt").sum().alias("tot"))
    share = cell.join(dest_tot, on="w").with_columns(
        (pl.col("cnt") / pl.col("tot")).pow(2).alias("p2")
    )
    eff = (
        share.group_by("w")
        .agg(pl.col("p2").sum().alias("hhi"), pl.col("tot").first().alias("tot"))
        .with_columns((1.0 / pl.col("hhi")).alias("eff_naics"))
    )
    # Weight by destination size: a tiny destination is trivially concentrated and should not
    # dominate the summary.
    w = eff["tot"].to_numpy().astype(float)
    e = eff["eff_naics"].to_numpy()
    sizes = cell["cnt"].to_numpy()
    return {
        "n_dest": len(eff),
        "n_cells": len(cell),
        "eff_naics_median": float(np.median(e)),
        "eff_naics_wmean": float((e * w).sum() / w.sum()),
        "cell_size_median": float(np.median(sizes)),
        "cell_size_mean": float(sizes.mean()),
        "cell_size_p90": float(np.quantile(sizes, 0.9)),
        "cell_size_max": int(sizes.max()),
        "frac_in_cells_ge_20": float(sizes[sizes >= 20].sum() / sizes.sum()),
    }


def alloc_flowprop(home: np.ndarray, lodes: pl.DataFrame, rng) -> np.ndarray:
    """Per-worker multinomial on the home's own LODES row -- the no-structure endpoint.

    Vectorised by home: draw all of a home's workers' destinations in one call rather than one
    per worker, which matters at 800k workers.
    """
    dests = sorted(lodes["w_geocode"].unique().to_list())
    dest_id = {g: i for i, g in enumerate(dests)}
    dest_arr = np.array(dests, dtype=object)

    by_home = {}
    for (hg,), grp in lodes.group_by("h_geocode"):
        c = grp["count"].to_numpy().astype(float)
        by_home[hg] = (
            np.array([dest_id[g] for g in grp["w_geocode"].to_list()]),
            c / c.sum(),
        )

    out = np.empty(len(home), dtype=object)
    order = np.argsort(home, kind="stable")
    hs = home[order]
    bounds = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = order[a:b]
        hd = by_home.get(hs[a])
        if hd is None:
            out[rows] = hs[a]  # no LODES row for this origin: work at home, as alloc_workers does
            continue
        idx, p = hd
        out[rows] = dest_arr[idx[rng.choice(len(idx), size=b - a, p=p)]]
    return out


def report(name: str, home, work, naics, lodes):
    corr, npairs, ngen = flow_correlation(home, work, lodes)
    s = structure_metrics(work, naics)
    print(f"\n=== {name} ===")
    print(f"  LODES flow correlation   {corr:.3f}   ({npairs} pairs in union, {ngen} workers placed)")
    print(f"  destinations used        {s['n_dest']}")
    print(f"  (dest, NAICS) cells      {s['n_cells']}")
    print(f"  effective NAICS/dest     median {s['eff_naics_median']:.2f}   "
          f"size-weighted mean {s['eff_naics_wmean']:.2f}")
    print(f"  cell size                median {s['cell_size_median']:.1f}  mean "
          f"{s['cell_size_mean']:.1f}  p90 {s['cell_size_p90']:.0f}  max {s['cell_size_max']}")
    print(f"  workers in cells >= 20   {s['frac_in_cells_ge_20']:.3f}")
    return corr, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="/workspaces/ExaEpi/data/UrbanPop/workers_nt_dt.intermediate.csv")
    ap.add_argument("--lodes", default="/workspaces/ExaEpi/data/LODES7/nm_od_main_JT00_2019.csv.gz")
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()

    for p in (args.workers, args.lodes):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    print(f"loading {args.workers}")
    w = pl.read_csv(
        args.workers,
        columns=["home_geoid", "naics", "work_geoid"],
        schema_overrides={"home_geoid": pl.Utf8, "work_geoid": pl.Utf8, "naics": pl.Int32},
    )
    print(f"loading {args.lodes}")
    lodes = load_lodes(args.lodes)

    # alloc_workers restricts LODES to pairs whose BOTH ends host synthetic workers, so any
    # baseline has to see the same flow matrix or the comparison is not like for like.
    wg = set(w["home_geoid"].unique().to_list())
    lodes = lodes.filter(pl.col("w_geocode").is_in(wg) & pl.col("h_geocode").is_in(wg))
    print(f"  {len(w)} workers, {len(lodes)} LODES pairs after restricting to populated geoids")

    home = w["home_geoid"].to_numpy()
    naics = w["naics"].to_numpy()

    c_greedy, s_greedy = report("greedy (shipped alloc_workers)", home, w["work_geoid"].to_numpy(),
                                naics, lodes)

    rng = np.random.default_rng(args.seed)
    c_flow, s_flow = report("flowprop (per-worker LODES multinomial)",
                            home, alloc_flowprop(home, lodes, rng), naics, lodes)

    print("\n" + "=" * 78)
    print("the tradeoff, stated plainly:")
    print(f"  flowprop beats greedy on LODES correlation by {c_flow - c_greedy:+.3f} "
          f"({c_flow:.3f} vs {c_greedy:.3f})")
    print(f"  greedy beats flowprop on concentration: effective NAICS per destination "
          f"{s_greedy['eff_naics_wmean']:.2f} vs {s_flow['eff_naics_wmean']:.2f}")
    print(f"  and on workplace size: {s_greedy['frac_in_cells_ge_20']:.3f} vs "
          f"{s_flow['frac_in_cells_ge_20']:.3f} of workers sit in a (dest, NAICS) cell of 20+")
    print()
    print("A max-entropy solve maximises dispersion subject to its margins, so it lands on the")
    print("flowprop side of this. If the gap in structure is large, P-MEDM cannot replace the")
    print("greedy fill outright -- only the flow layer under it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
