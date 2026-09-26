#!/usr/bin/env python
"""Replace only the greedy fill LOOP with per-NAICS IPF, keeping the establishment-slot draw.

flow_baselines.py settled the framing question: a max-entropy solve subject to LODES margins
lands on the dispersed side of the tradeoff (cells of median size 2, 54% of workers in a cell of
20+), while the greedy fill produces cells of median size 24 and 96% in cells of 20+. Entropy
maximisation cannot manufacture industry concentration, because concentration is precisely what
maximising entropy destroys. So P-MEDM cannot replace `alloc_workers` outright.

But the plan never wanted P-MEDM for fidelity -- it wanted it for determinism. The concern was
that the fill is "stateful, order-dependent, and sequential", the hardest step to port to AMReX.
Reading the five steps separately, only the last one is:

    1. dest_total[d] from real LODES inbound flow            deterministic, precomputable
    2. local_naics_weight = flow @ home_naics                deterministic given the population
    3. slot NAICS + slot size draws per destination          stochastic, INDEPENDENT per destination
    4. rescale to statewide NAICS totals, largest remainder  one reduction + one sort
    5. fill (dest, NAICS) pairs in descending demand order   stateful, sequential  <-- the problem

Steps 1-4 produce `demand_int[d, n]`, and THAT is where the concentration lives: it is generated
by the establishment-slot draw against real CBP size distributions, not by the fill. Step 3 is
embarrassingly parallel -- each destination draws its own slots from its own commute-shed -- so it
ports to Philox keyed on (seed, dest_idx) with no ordering at all.

Step 5 is then just a transportation problem, and it separates by NAICS:

    for each NAICS n:   find x[h, d] >= 0  with
        sum_d x[h, d] = W[h, n]           every worker living at h in industry n is placed
        sum_h x[h, d] = demand_int[d, n]  every workplace slot is filled
        x supported only where LODES flow[h, d] > 0, and proportional to it otherwise

which is the classic doubly-constrained gravity model, solved by IPF. IPF is order-independent,
converges to a unique fixed point, and parallelises as two sparse segment-sums per iteration.

So the hypothesis under test: IPF hits the SAME demand margins as the greedy fill -- preserving
concentration exactly -- while matching LODES flows at least as well, because it distributes
proportionally to real flow instead of racing pairs in demand order.

The targets here are the greedy fill's own REALIZED (dest, NAICS) cell counts. That deliberately
hands greedy the home-field advantage: the margins are known achievable because greedy achieved
them, so any shortfall IPF shows is a real IPF weakness and not an infeasible ask.
"""

import argparse
import os
import sys

import numpy as np
import polars as pl
import scipy.sparse as sp


def load_lodes(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, columns=["w_geocode", "h_geocode", "S000"])
    df = df.with_columns(
        pl.col("w_geocode").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("w_geocode"),
        pl.col("h_geocode").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("h_geocode"),
        pl.col("S000").cast(pl.Int32),
    )
    return df.group_by(["w_geocode", "h_geocode"]).agg(pl.col("S000").sum().alias("count"))


def ipf_sparse(P: sp.csr_matrix, row_t: np.ndarray, col_t: np.ndarray, n_iter=60, tol=1e-9):
    """Doubly-constrained IPF on a sparse prior, as row/column multipliers.

    Works on P's stored values directly rather than rebuilding a matrix each sweep: the sparsity
    pattern never changes, only the values, so a whole sweep is two segment-sums and two gathers.
    That is exactly the shape an AMReX port would take.

    Returns (values, row_residual, col_residual). Residuals are what reveal infeasibility from
    structural zeros -- a home whose entire commute shed wants none of its industry cannot place
    its workers no matter how many sweeps run.
    """
    rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
    cols = P.indices
    v0 = P.data.astype(np.float64)
    v = v0.copy()

    for _ in range(n_iter):
        rs = np.bincount(rows, weights=v, minlength=P.shape[0])
        rf = np.divide(row_t, rs, out=np.zeros_like(row_t), where=rs > 0)
        v = v * rf[rows]

        cs = np.bincount(cols, weights=v, minlength=P.shape[1])
        cf = np.divide(col_t, cs, out=np.zeros_like(col_t), where=cs > 0)
        v = v * cf[cols]

        rs = np.bincount(rows, weights=v, minlength=P.shape[0])
        if np.abs(rs - row_t).sum() < tol * max(1.0, row_t.sum()):
            break

    rs = np.bincount(rows, weights=v, minlength=P.shape[0])
    cs = np.bincount(cols, weights=v, minlength=P.shape[1])
    return v, np.abs(rs - row_t).sum(), np.abs(cs - col_t).sum()


def trs(vals: np.ndarray, target: int, rng) -> np.ndarray:
    """Truncate, replicate, sample: whole parts kept, remainder drawn proportional to fractions."""
    whole = np.floor(vals).astype(np.int64)
    short = target - int(whole.sum())
    if short > 0:
        frac = vals - whole
        s = frac.sum()
        if s > 0:
            pick = rng.choice(len(vals), size=short, replace=True, p=frac / s)
            whole += np.bincount(pick, minlength=len(vals))
    elif short < 0:
        nz = np.flatnonzero(whole > 0)
        if len(nz):
            pick = rng.choice(nz, size=min(-short, len(nz)), replace=False)
            whole[pick] -= 1
    return whole


def flow_correlation(pairs: pl.DataFrame, lodes: pl.DataFrame):
    """pairs: columns h, w, total."""
    gen = pairs.with_columns((pl.col("h") + "-" + pl.col("w")).alias("key"))
    ref = lodes.with_columns((pl.col("h_geocode") + "-" + pl.col("w_geocode")).alias("key"))
    m = (
        gen.select(["key", "total"])
        .join(ref.select(["key", "count"]), on="key", how="full", coalesce=True)
        .with_columns(pl.col("total").fill_null(0), pl.col("count").fill_null(0))
    )
    return m.select(pl.corr("total", "count")).item(), len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="/workspaces/ExaEpi/data/UrbanPop/workers_nt_dt.intermediate.csv")
    ap.add_argument("--lodes", default="/workspaces/ExaEpi/data/LODES7/nm_od_main_JT00_2019.csv.gz")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()

    for p in (args.workers, args.lodes):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    rng = np.random.default_rng(args.seed)
    w = pl.read_csv(
        args.workers,
        columns=["home_geoid", "naics", "work_geoid"],
        schema_overrides={"home_geoid": pl.Utf8, "work_geoid": pl.Utf8, "naics": pl.Int32},
    ).filter(pl.col("naics") >= 0)
    lodes = load_lodes(args.lodes)
    wg = set(w["home_geoid"].unique().to_list())
    lodes = lodes.filter(pl.col("w_geocode").is_in(wg) & pl.col("h_geocode").is_in(wg))

    homes = sorted(wg)
    hid = {g: i for i, g in enumerate(homes)}
    dests = sorted(lodes["w_geocode"].unique().to_list())
    did = {g: i for i, g in enumerate(dests)}
    nH, nD = len(homes), len(dests)
    print(f"{len(w)} workers with NAICS, {nH} homes, {nD} destinations, {len(lodes)} LODES pairs")

    li = lodes.with_columns(
        pl.col("h_geocode").replace_strict(hid).alias("hi"),
        pl.col("w_geocode").replace_strict(did).alias("di"),
    )
    flow = sp.csr_matrix(
        (li["count"].to_numpy().astype(float), (li["hi"].to_numpy(), li["di"].to_numpy())),
        shape=(nH, nD),
    )

    # Supply: workers by (home, NAICS). Demand: the greedy fill's own realized (dest, NAICS) cells.
    sup = (
        w.group_by(["home_geoid", "naics"]).agg(pl.len().alias("c"))
        .with_columns(pl.col("home_geoid").replace_strict(hid).alias("hi"))
    )
    dem = (
        w.group_by(["work_geoid", "naics"]).agg(pl.len().alias("c"))
        .with_columns(pl.col("work_geoid").replace_strict(did).alias("di"))
    )
    naics_codes = sorted(set(sup["naics"].to_list()))
    sup_by_n = {k[0]: g for k, g in sup.group_by("naics")}
    dem_by_n = {k[0]: g for k, g in dem.group_by("naics")}

    out_h, out_d, out_c = [], [], []
    cell_h, cell_d, cell_n, cell_c = [], [], [], []
    unplaced = 0
    worst = []

    for n in naics_codes:
        s = np.zeros(nH)
        sg = sup_by_n[n]
        s[sg["hi"].to_numpy()] = sg["c"].to_numpy().astype(float)
        t = np.zeros(nD)
        if n in dem_by_n:
            dg = dem_by_n[n]
            t[dg["di"].to_numpy()] = dg["c"].to_numpy().astype(float)
        if s.sum() == 0 or t.sum() == 0:
            continue

        ri = np.flatnonzero(s > 0)
        ci = np.flatnonzero(t > 0)
        P = flow[ri][:, ci].tocsr()
        if P.nnz == 0:
            unplaced += int(s.sum())
            continue

        # A home with no flow to any destination wanting this industry is structurally unplaceable;
        # drop it from the system and count it rather than letting IPF diverge trying.
        rowok = np.diff(P.indptr) > 0
        colok = np.diff(P.tocsc().indptr) > 0
        lost = int(s[ri][~rowok].sum())
        unplaced += lost
        ri2, ci2 = ri[rowok], ci[colok]
        P = flow[ri2][:, ci2].tocsr()
        if P.nnz == 0:
            unplaced += int(s[ri2].sum())
            continue

        rt = s[ri2].copy()
        ct = t[ci2].copy()
        ct = ct * (rt.sum() / ct.sum())  # margins must agree for IPF to converge

        v, rres, cres = ipf_sparse(P, rt, ct, n_iter=args.iters)
        worst.append((n, rres / max(1.0, rt.sum())))

        cnt = trs(v, int(rt.sum()), rng)
        keep = cnt > 0
        if not keep.any():
            continue
        rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))[keep]
        cols = P.indices[keep]
        c = cnt[keep]
        out_h.append(ri2[rows]); out_d.append(ci2[cols]); out_c.append(c)
        cell = np.bincount(cols, weights=c, minlength=P.shape[1])
        nzc = np.flatnonzero(cell > 0)
        cell_h.append(nzc); cell_d.append(ci2[nzc]); cell_c.append(cell[nzc])
        cell_n.append(np.full(len(nzc), n))

    H = np.concatenate(out_h); D = np.concatenate(out_d); C = np.concatenate(out_c)
    pairs = (
        pl.DataFrame({"hi": H, "di": D, "total": C})
        .group_by(["hi", "di"]).agg(pl.col("total").sum())
        .with_columns(
            pl.col("hi").map_elements(lambda i: homes[i], return_dtype=pl.Utf8).alias("h"),
            pl.col("di").map_elements(lambda i: dests[i], return_dtype=pl.Utf8).alias("w"),
        )
    )
    corr, npairs = flow_correlation(pairs.select(["h", "w", "total"]), lodes)

    sizes = np.concatenate(cell_c)
    placed = int(C.sum())
    res = np.array([r for _, r in worst])

    print(f"\n=== IPF fill (targets = greedy's own realized cells) ===")
    print(f"  LODES flow correlation   {corr:.3f}   ({npairs} pairs in union)")
    print(f"  workers placed           {placed} of {len(w)}  "
          f"({100.0 * (len(w) - placed) / len(w):.2f}% unplaced)")
    print(f"  structurally unplaceable {unplaced}  ({100.0 * unplaced / len(w):.2f}%)")
    print(f"  (dest, NAICS) cells      {len(sizes)}")
    print(f"  cell size                median {np.median(sizes):.1f}  mean {sizes.mean():.1f}  "
          f"p90 {np.quantile(sizes, 0.9):.0f}  max {int(sizes.max())}")
    print(f"  workers in cells >= 20   {sizes[sizes >= 20].sum() / sizes.sum():.3f}")
    print(f"  IPF row residual         median {np.median(res):.2e}  max {res.max():.2e}  "
          f"({len(res)} NAICS solved, {args.iters} sweeps)")
    print()
    print("  reference, from flow_baselines.py:")
    print("    greedy    corr 0.861   cells 31469   median size 24.0   in cells>=20  0.962")
    print("    flowprop  corr 0.927   cells 119424  median size  2.0   in cells>=20  0.544")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
