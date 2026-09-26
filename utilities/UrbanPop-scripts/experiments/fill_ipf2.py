#!/usr/bin/env python
"""How should the IPF solution be integerized, and does it hold the demand margins?

fill_ipf.py showed per-NAICS IPF beats the greedy fill on LODES correlation (0.922 vs 0.861) while
matching its cell-size distribution and placing every worker. But it filled 27,189 (dest, NAICS)
cells against 31,469 targets -- 14% of the demanded workplaces came out empty, and mean cell size
rose from 26.6 to 30.8. That is the integerization, not the solve: a single global TRS over each
NAICS's whole value vector matches row and column margins only in expectation, so small cells get
rounded away and their workers pile into large ones.

Two things are being conflated and need separating, because they have different requirements:

    row margins (supply)   MUST be exact. Every synthetic worker is a real person who works
                           exactly one place; a row residual means a worker vanished or was cloned.
    column margins (demand) SHOULD be close. These are the establishment slots from the CBP draw,
                           and they carry the industry concentration -- the whole point of keeping
                           steps 1-4. Drift here erodes workplace structure.

Three integerizations, scored on exactly that:

    global    one TRS over the whole per-NAICS value vector          (what fill_ipf.py did)
    byrow     TRS within each home's row, so row sums are exact by construction
    bycol     TRS within each destination column, so column sums are exact by construction,
              then a repair sweep to restore exact row sums

`bycol` is the interesting one: it protects the margins that carry the structure, and the repair
moves whole workers between destinations inside a home's own commute shed, so it cannot invent a
pair LODES never saw.
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


def ipf_sparse(P, row_t, col_t, n_iter=60, tol=1e-9):
    rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
    cols = P.indices
    v = P.data.astype(np.float64).copy()
    for _ in range(n_iter):
        rs = np.bincount(rows, weights=v, minlength=P.shape[0])
        rf = np.divide(row_t, rs, out=np.zeros_like(row_t), where=rs > 0)
        v *= rf[rows]
        cs = np.bincount(cols, weights=v, minlength=P.shape[1])
        cf = np.divide(col_t, cs, out=np.zeros_like(col_t), where=cs > 0)
        v *= cf[cols]
        rs = np.bincount(rows, weights=v, minlength=P.shape[0])
        if np.abs(rs - row_t).sum() < tol * max(1.0, row_t.sum()):
            break
    return v, rows, cols


def trs_group(vals, group, n_groups, targets, rng):
    """TRS applied independently within each group, hitting each group's integer target exactly.

    Vectorised across groups: whole parts first, then each group's shortfall drawn from its own
    fractional parts. The loop runs over groups that are actually short, which is far fewer than
    the group count once most land on integers.
    """
    whole = np.floor(vals).astype(np.int64)
    frac = vals - whole
    have = np.bincount(group, weights=whole, minlength=n_groups).astype(np.int64)
    short = targets - have

    order = np.argsort(group, kind="stable")
    g = group[order]
    bounds = np.flatnonzero(np.r_[True, g[1:] != g[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        gi = g[a]
        s = int(short[gi])
        if s == 0:
            continue
        idx = order[a:b]
        if s > 0:
            f = frac[idx]
            tot = f.sum()
            p = f / tot if tot > 0 else np.full(len(idx), 1.0 / len(idx))
            whole[idx] += np.bincount(rng.choice(len(idx), size=s, p=p), minlength=len(idx))
        else:
            nz = idx[whole[idx] > 0]
            if len(nz):
                pick = rng.choice(len(nz), size=min(-s, len(nz)), replace=False)
                whole[nz[pick]] -= 1
    return whole


def repair_rows(cnt, rows, cols, row_t, rng, sweeps=12, key=None, col_key=None):
    """Restore exact row sums by moving whole workers between cells within the same row.

    A row over its supply gives workers back; a row under it takes them. Movement is confined to
    the row's own existing cells, which are exactly the destinations its LODES commute shed
    reaches -- so this can never create a flow that does not exist in the data.

    `key`, if given, maps a row to its own generator, so a row's repair draws from a stream keyed
    on that row rather than on how many rows were repaired before it. Sharing one stream across
    rows makes the result depend on the order rows are visited, which is exactly the rank
    dependence this whole design exists to avoid.
    """
    if key is not None:
        rng_for = key
    else:
        def rng_for(_ri, _sweep):
            return rng
    # Sort each row's cells by GLOBAL column id, not by their position in whatever layout the
    # caller happens to be using, so a row's candidate list is the same list in the same order
    # under any permutation of the matrix.
    gcol = cols if col_key is None else col_key[cols]
    order = np.lexsort((gcol, rows))
    r = rows[order]
    bounds = np.flatnonzero(np.r_[True, r[1:] != r[:-1], True])
    row_slices = {r[a]: order[a:b] for a, b in zip(bounds[:-1], bounds[1:])}

    for sweep in range(sweeps):
        rs = np.bincount(rows, weights=cnt, minlength=len(row_t)).astype(np.int64)
        delta = row_t.astype(np.int64) - rs
        bad = np.flatnonzero(delta != 0)
        if len(bad) == 0:
            break
        for ri in bad:
            idx = row_slices.get(ri)
            if idx is None or len(idx) == 0:
                continue
            rng = rng_for(int(ri), sweep)
            d = int(delta[ri])
            if d > 0:
                w = cnt[idx].astype(float) + 0.5
                pick = rng.choice(len(idx), size=d, p=w / w.sum())
                cnt[idx] += np.bincount(pick, minlength=len(idx))
            else:
                nz = idx[cnt[idx] > 0]
                if len(nz) == 0:
                    continue
                take = min(-d, int(cnt[nz].sum()))
                w = cnt[nz].astype(float)
                pick = rng.choice(len(nz), size=take, replace=True, p=w / w.sum())
                for j in pick:
                    if cnt[nz[j]] > 0:
                        cnt[nz[j]] -= 1
    return cnt


def flow_correlation(pairs, lodes):
    gen = pairs.with_columns((pl.col("h") + "-" + pl.col("w")).alias("key"))
    ref = lodes.with_columns((pl.col("h_geocode") + "-" + pl.col("w_geocode")).alias("key"))
    m = (
        gen.select(["key", "total"])
        .join(ref.select(["key", "count"]), on="key", how="full", coalesce=True)
        .with_columns(pl.col("total").fill_null(0), pl.col("count").fill_null(0))
    )
    return m.select(pl.corr("total", "count")).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="/workspaces/ExaEpi/data/UrbanPop/workers_nt_dt.intermediate.csv")
    ap.add_argument("--lodes", default="/workspaces/ExaEpi/data/LODES7/nm_od_main_JT00_2019.csv.gz")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--modes", default="global,byrow,bycol")
    args = ap.parse_args()

    for p in (args.workers, args.lodes):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    w = pl.read_csv(
        args.workers, columns=["home_geoid", "naics", "work_geoid"],
        schema_overrides={"home_geoid": pl.Utf8, "work_geoid": pl.Utf8, "naics": pl.Int32},
    ).filter(pl.col("naics") >= 0)
    lodes = load_lodes(args.lodes)
    wg = set(w["home_geoid"].unique().to_list())
    lodes = lodes.filter(pl.col("w_geocode").is_in(wg) & pl.col("h_geocode").is_in(wg))

    homes = sorted(wg); hid = {g: i for i, g in enumerate(homes)}
    dests = sorted(lodes["w_geocode"].unique().to_list()); did = {g: i for i, g in enumerate(dests)}
    nH, nD = len(homes), len(dests)

    li = lodes.with_columns(
        pl.col("h_geocode").replace_strict(hid).alias("hi"),
        pl.col("w_geocode").replace_strict(did).alias("di"),
    )
    flow = sp.csr_matrix(
        (li["count"].to_numpy().astype(float), (li["hi"].to_numpy(), li["di"].to_numpy())),
        shape=(nH, nD),
    )

    sup = (w.group_by(["home_geoid", "naics"]).agg(pl.len().alias("c"))
           .with_columns(pl.col("home_geoid").replace_strict(hid).alias("hi")))
    dem = (w.group_by(["work_geoid", "naics"]).agg(pl.len().alias("c"))
           .with_columns(pl.col("work_geoid").replace_strict(did).alias("di")))
    sup_by_n = {k[0]: g for k, g in sup.group_by("naics")}
    dem_by_n = {k[0]: g for k, g in dem.group_by("naics")}
    naics_codes = sorted(sup_by_n)

    # Solve once; the integerizations are then compared on the SAME continuous solution, so any
    # difference between them is the rounding and nothing else.
    solved = []
    n_target_cells = 0
    for n in naics_codes:
        if n not in dem_by_n:
            continue
        s = np.zeros(nH); sg = sup_by_n[n]
        s[sg["hi"].to_numpy()] = sg["c"].to_numpy().astype(float)
        t = np.zeros(nD); dg = dem_by_n[n]
        t[dg["di"].to_numpy()] = dg["c"].to_numpy().astype(float)
        ri, ci = np.flatnonzero(s > 0), np.flatnonzero(t > 0)
        if len(ri) == 0 or len(ci) == 0:
            continue
        P = flow[ri][:, ci].tocsr()
        if P.nnz == 0:
            continue
        rowok = np.diff(P.indptr) > 0
        colok = np.diff(P.tocsc().indptr) > 0
        ri, ci = ri[rowok], ci[colok]
        P = flow[ri][:, ci].tocsr()
        if P.nnz == 0:
            continue
        rt, ct = s[ri].copy(), t[ci].copy()
        n_target_cells += len(ci)
        ct *= rt.sum() / ct.sum()
        v, rows, cols = ipf_sparse(P, rt, ct)
        solved.append((n, v, rows, cols, ri, ci, rt, ct))

    print(f"solved {len(solved)} NAICS, {n_target_cells} target (dest, NAICS) cells\n")
    print(f"{'mode':8s}{'corr':>8s}{'cells':>9s}{'med':>6s}{'mean':>7s}{'>=20':>8s}"
          f"{'row err':>10s}{'col err':>10s}")

    for mode in args.modes.split(","):
        rng = np.random.default_rng(args.seed)
        H, D, C, cellc = [], [], [], []
        row_err = col_err = 0
        for n, v, rows, cols, ri, ci, rt, ct in solved:
            ct_i = np.round(ct).astype(np.int64)
            if mode == "global":
                whole = np.floor(v).astype(np.int64)
                frac = v - whole
                short = int(rt.sum()) - int(whole.sum())
                if short > 0 and frac.sum() > 0:
                    whole += np.bincount(
                        rng.choice(len(v), size=short, p=frac / frac.sum()), minlength=len(v))
                cnt = whole
            elif mode == "byrow":
                cnt = trs_group(v, rows, len(rt), rt.astype(np.int64), rng)
            else:
                cnt = trs_group(v, cols, len(ct), ct_i, rng)
                cnt = repair_rows(cnt, rows, cols, rt, rng)

            row_err += int(np.abs(np.bincount(rows, weights=cnt, minlength=len(rt)) - rt).sum())
            col_err += int(np.abs(np.bincount(cols, weights=cnt, minlength=len(ct)) - ct_i).sum())
            keep = cnt > 0
            H.append(ri[rows[keep]]); D.append(ci[cols[keep]]); C.append(cnt[keep])
            cc = np.bincount(cols[keep], weights=cnt[keep], minlength=len(ct))
            cellc.append(cc[cc > 0])

        H, D, C = np.concatenate(H), np.concatenate(D), np.concatenate(C)
        sizes = np.concatenate(cellc)
        pairs = (pl.DataFrame({"hi": H, "di": D, "total": C})
                 .group_by(["hi", "di"]).agg(pl.col("total").sum())
                 .with_columns(
                     pl.col("hi").map_elements(lambda i: homes[i], return_dtype=pl.Utf8).alias("h"),
                     pl.col("di").map_elements(lambda i: dests[i], return_dtype=pl.Utf8).alias("w")))
        corr = flow_correlation(pairs.select(["h", "w", "total"]), lodes)
        print(f"{mode:8s}{corr:8.3f}{len(sizes):9d}{np.median(sizes):6.0f}{sizes.mean():7.1f}"
              f"{sizes[sizes >= 20].sum() / sizes.sum():8.3f}{row_err:10d}{col_err:10d}")

    print(f"\n{'greedy':8s}{0.861:8.3f}{31469:9d}{24:6d}{26.6:7.1f}{0.962:8.3f}"
          f"{0:10d}{'n/a':>10s}   <- targets are its own cells, so col err is 0 by definition")
    print(f"{'flowprop':8s}{0.927:8.3f}{119424:9d}{2:6d}{7.0:7.1f}{0.544:8.3f}")
    print("\nrow err is workers lost or cloned -- must be 0. col err is drift away from the")
    print("establishment-slot demand that carries the industry concentration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
