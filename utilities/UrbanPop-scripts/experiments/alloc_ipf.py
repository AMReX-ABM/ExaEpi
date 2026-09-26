#!/usr/bin/env python
"""Greedy fill vs IPF fill on IDENTICAL demand, generated from the real CBP tables.

fill_ipf2.py used the greedy fill's own realized (dest, NAICS) cells as IPF's demand targets. That
was deliberately conservative -- the margins were known achievable because greedy achieved them --
but circular about where the concentration came from. This closes that loop.

Steps 1-4 of `alloc_workers` are run here verbatim (reusing upop_to_exaepi's own loaders, so the
CBP establishment-size distributions and workgroup targets are the shipped ones), producing
`demand_int[d, n]` ONCE. Both allocators then consume that same array:

    greedy   step 5 as shipped: visit (dest, NAICS) pairs in descending demand, pulling from
             shrinking per-(home, NAICS) pools, with per-worker home-flow sampling for leftovers
    ipf      per-NAICS IPF onto the same margins, column-wise TRS, row-repair sweep

So this is step 5 against step 5, with concentration supplied independently by CBP. The scoreboard
is three-sided:

    fidelity     LODES flow correlation, as check_flows_correlation computes it
    structure    how closely realized cell sizes track the DEMANDED cell sizes -- the CBP-derived
                 establishment slots that are the whole reason steps 1-4 exist
    coverage     what fraction of workers each allocator fails to place from demand and has to
                 fall back on (greedy's own comments record 1.69-3.96% for CA)
"""

import argparse
import os
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import upop_to_exaepi as U  # noqa: E402

from fill_ipf2 import ipf_sparse, repair_rows  # noqa: E402

WG_FILE = "/workspaces/ExaEpi/data/UrbanPop/workgroup_sizes_us.txt"
EST_FILE = "/workspaces/ExaEpi/data/UrbanPop/establishment_sizes_us.txt"


def load_lodes(path):
    df = pl.read_csv(path, columns=["w_geocode", "h_geocode", "S000"])
    df = df.with_columns(
        pl.col("w_geocode").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("w_geocode"),
        pl.col("h_geocode").cast(pl.Utf8).str.zfill(15).str.slice(0, 12).alias("h_geocode"),
        pl.col("S000").cast(pl.Int32),
    )
    return df.group_by(["w_geocode", "h_geocode"]).agg(pl.col("S000").sum().alias("count"))


def build_demand(flow, home_naics, dest_geoids, home_geoid_arr, naics_arr, n_naics, seed):
    """Steps 1-4 of alloc_workers, transcribed so both allocators get the same targets.

    Kept structurally identical to the shipped code (including the per-destination Python loops)
    rather than rewritten, so any difference downstream is the fill and not a re-derivation of
    the demand.
    """
    np.random.seed(seed)
    naics_categs = list(U.categ_types["pr_naics"].categories)
    wg_targets = U.load_workgroup_targets(WG_FILE)
    est_dists = U.load_establishment_size_dists(EST_FILE)

    def target_for(state_fips, ni):
        if not (0 <= ni < n_naics):
            return U.DEFAULT_WORKGROUP_TARGET
        return wg_targets.get((state_fips, naics_categs[ni]), U.DEFAULT_WORKGROUP_TARGET)

    n_dest = flow.shape[0]
    dest_total = np.asarray(flow.sum(axis=1)).ravel()
    local_naics_weight = np.asarray(flow @ home_naics)

    state_fips_arr = np.array([int(g[:2]) for g in home_geoid_arr])
    avg_target = np.mean([target_for(s, n) for s, n in zip(state_fips_arr, naics_arr)])
    n_slots = np.where(dest_total > 0, np.maximum(1, np.round(dest_total / avg_target)), 0).astype(np.int64)

    slot_naics_count = np.zeros((n_dest, n_naics), dtype=np.int64)
    for d in range(n_dest):
        if n_slots[d] <= 0:
            continue
        w = local_naics_weight[d]
        tw = w.sum()
        if tw <= 0:
            continue
        slot_naics_count[d] = np.bincount(
            np.random.choice(n_naics, size=n_slots[d], p=w / tw), minlength=n_naics)

    dest_state = np.array([int(g[:2]) for g in dest_geoids])
    tgt_by_state = {s: np.array([target_for(s, ni) for ni in range(n_naics)])
                    for s in np.unique(dest_state)}
    tgt_matrix = np.stack([tgt_by_state[s] for s in dest_state])

    implied = np.zeros((n_dest, n_naics))
    slot_sizes = []
    for d in range(n_dest):
        for ni in np.flatnonzero(slot_naics_count[d]):
            k = slot_naics_count[d, ni]
            dist = est_dists.get((int(dest_state[d]), naics_categs[ni]))
            if dist is None:
                implied[d, ni] = k * tgt_matrix[d, ni]
                slot_sizes.append(np.full(k, tgt_matrix[d, ni]))
            else:
                sizes, probs = dist
                drawn = np.random.choice(sizes, size=k, p=probs)
                implied[d, ni] = drawn.sum()
                slot_sizes.append(drawn)

    true_tot = np.bincount(naics_arr[naics_arr >= 0], minlength=n_naics)
    imp_tot = implied.sum(axis=0)
    scale = np.divide(true_tot, imp_tot, out=np.zeros(n_naics), where=imp_tot > 0)
    implied = implied * scale[np.newaxis, :]

    demand = np.zeros((n_dest, n_naics), dtype=np.int64)
    for ni in range(n_naics):
        tt = int(true_tot[ni])
        if tt == 0:
            continue
        col = implied[:, ni]
        fl = np.floor(col).astype(np.int64)
        rem = tt - int(fl.sum())
        if rem > 0:
            fl[np.argsort(-(col - fl))[:rem]] += 1
        elif rem < 0:
            nz = np.where(fl > 0)[0]
            fl[nz[np.argsort(fl[nz])][: min(-rem, len(nz))]] -= 1
        demand[:, ni] = fl
    return demand, np.concatenate(slot_sizes) if slot_sizes else np.array([])


def fill_greedy(demand, flow_csr, home_idx_of_worker, naics_arr, n_home, n_naics, seed):
    """Step 5 as shipped: descending-demand pair order over shrinking per-(home, NAICS) pools."""
    np.random.seed(seed)
    num_workers = len(naics_arr)
    work_idx = np.full(num_workers, -1, dtype=np.int64)
    assigned = np.zeros(num_workers, dtype=bool)

    valid = np.where(naics_arr >= 0)[0]
    combo = home_idx_of_worker[valid].astype(np.int64) * n_naics + naics_arr[valid].astype(np.int64)
    order = np.lexsort((np.random.random(len(valid)), combo))
    pool_rows = valid[order]
    sc = combo[order]
    uc, cstart, ccount = np.unique(sc, return_index=True, return_counts=True)
    pool_start = np.zeros(n_home * n_naics, dtype=np.int64)
    pool_count = np.zeros(n_home * n_naics, dtype=np.int64)
    pool_start[uc] = cstart
    pool_count[uc] = ccount

    pd_, pn_ = np.where(demand > 0)
    po = np.argsort(-demand[pd_, pn_])
    pd_, pn_ = pd_[po], pn_[po]

    for pi in range(len(pd_)):
        d, ni = int(pd_[pi]), int(pn_[pi])
        s, e = flow_csr.indptr[d], flow_csr.indptr[d + 1]
        d_homes = flow_csr.indices[s:e]
        if len(d_homes) == 0:
            continue
        d_w = flow_csr.data[s:e]
        cidx = d_homes * n_naics + ni
        counts = pool_count[cidx]
        hit = counts > 0
        if not hit.any():
            continue
        starts_m, counts_m, w_m = pool_start[cidx[hit]], counts[hit], d_w[hit]
        total = int(counts_m.sum())
        goff = np.cumsum(counts_m) - counts_m
        offs = np.repeat(starts_m, counts_m) + (np.arange(total) - np.repeat(goff, counts_m))
        cand = pool_rows[offs]
        cw = np.repeat(w_m, counts_m)
        k = min(int(demand[d, ni]), total)
        if k <= 0:
            continue
        chosen = cand[np.random.choice(total, size=k, replace=False, p=cw / cw.sum())]
        work_idx[chosen] = d
        assigned[chosen] = True
        for h in np.unique(home_idx_of_worker[chosen]).tolist():
            ci = h * n_naics + ni
            st, ct = pool_start[ci], pool_count[ci]
            blk = pool_rows[st : st + ct]
            keep = blk[~assigned[blk]]
            pool_rows[st : st + len(keep)] = keep
            pool_count[ci] = len(keep)

    leftover = np.where(~assigned & (naics_arr >= 0))[0]
    flow_csc = flow_csr.tocsc()
    for i in leftover:
        h = home_idx_of_worker[i]
        s, e = flow_csc.indptr[h], flow_csc.indptr[h + 1]
        if e <= s:
            continue
        p = flow_csc.data[s:e]
        work_idx[i] = flow_csc.indices[s:e][np.random.choice(e - s, p=p / p.sum())]
    return work_idx, len(leftover)


def fill_ipf(demand, flow_hd, home_idx_of_worker, naics_arr, n_home, n_dest, seed):
    """Per-NAICS IPF onto the same margins, column-wise TRS, row-repair. Order-independent."""
    sup = np.zeros((n_home, demand.shape[1]), dtype=np.int64)
    np.add.at(sup, (home_idx_of_worker[naics_arr >= 0], naics_arr[naics_arr >= 0]), 1)

    cells_h, cells_d, cells_n, cells_c = [], [], [], []
    unplaceable = 0
    for n in range(demand.shape[1]):
        s = sup[:, n].astype(float)
        t = demand[:, n].astype(float)
        if s.sum() == 0 or t.sum() == 0:
            continue
        ri, ci = np.flatnonzero(s > 0), np.flatnonzero(t > 0)
        P = flow_hd[ri][:, ci].tocsr()
        if P.nnz == 0:
            unplaceable += int(s.sum())
            continue
        # A home whose commute shed reaches no destination wanting this industry cannot be served;
        # likewise a destination no supplying home can reach. Drop both and account for them.
        rk = np.diff(P.indptr) > 0
        unplaceable += int(s[ri][~rk].sum())
        ri = ri[rk]
        P = flow_hd[ri][:, ci].tocsr()
        ck = np.diff(P.tocsc().indptr) > 0
        ci = ci[ck]
        P = flow_hd[ri][:, ci].tocsr()
        if P.nnz == 0:
            unplaceable += int(s[ri].sum())
            continue
        rt, ct = s[ri], t[ci].copy()
        ct = ct * (rt.sum() / ct.sum())
        v, rows, cols = ipf_sparse(P, rt, ct)

        ct_i = np.round(ct).astype(np.int64)
        cnt = np.zeros(len(v), dtype=np.int64)
        o = np.lexsort((ri[rows], cols))
        cs = cols[o]
        b = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1], True])
        for a, z in zip(b[:-1], b[1:]):
            cj = cs[a]
            idx = o[a:z]
            rng = np.random.default_rng([seed, int(ci[cj]), n])
            vals = v[idx]
            whole = np.floor(vals).astype(np.int64)
            frac = vals - whole
            short = int(ct_i[cj]) - int(whole.sum())
            if short > 0 and frac.sum() > 0:
                whole += np.bincount(
                    rng.choice(len(idx), size=short, p=frac / frac.sum()), minlength=len(idx))
            elif short < 0:
                nz = np.flatnonzero(whole > 0)
                if len(nz):
                    whole[nz[rng.choice(len(nz), size=min(-short, len(nz)), replace=False)]] -= 1
            cnt[idx] = whole
        cnt = repair_rows(
            cnt, rows, cols, rt, None,
            key=lambda r, sw, _r=ri: np.random.default_rng([seed, 7, n, int(_r[r]), sw]),
            col_key=ci)
        keep = cnt > 0
        cells_h.append(ri[rows[keep]]); cells_d.append(ci[cols[keep]]); cells_c.append(cnt[keep])
        cells_n.append(np.full(int(keep.sum()), n))
    return (np.concatenate(cells_h), np.concatenate(cells_d),
            np.concatenate(cells_n), np.concatenate(cells_c), unplaceable)


def corr_and_cells(pair_h, pair_d, pair_c, cell_n, homes, dests, lodes):
    gen = (pl.DataFrame({"hi": pair_h, "di": pair_d, "total": pair_c})
           .group_by(["hi", "di"]).agg(pl.col("total").sum()))
    gen = gen.with_columns(
        (pl.Series([homes[i] for i in gen["hi"]]) + "-" + pl.Series([dests[i] for i in gen["di"]]))
        .alias("key"))
    ref = lodes.with_columns((pl.col("h_geocode") + "-" + pl.col("w_geocode")).alias("key"))
    m = (gen.select(["key", "total"])
         .join(ref.select(["key", "count"]), on="key", how="full", coalesce=True)
         .with_columns(pl.col("total").fill_null(0), pl.col("count").fill_null(0)))
    corr = m.select(pl.corr("total", "count")).item()
    cells = (pl.DataFrame({"d": pair_d, "n": cell_n, "c": pair_c})
             .group_by(["d", "n"]).agg(pl.col("c").sum()))["c"].to_numpy()
    return corr, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="/workspaces/ExaEpi/data/UrbanPop/workers_nt_dt.intermediate.csv")
    ap.add_argument("--lodes", default="/workspaces/ExaEpi/data/LODES7/nm_od_main_JT00_2019.csv.gz")
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()
    for p in (args.workers, args.lodes, WG_FILE, EST_FILE):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    w = pl.read_csv(
        args.workers, columns=["home_geoid", "naics"],
        schema_overrides={"home_geoid": pl.Utf8, "naics": pl.Int32},
    ).filter(pl.col("naics") >= 0)
    lodes = load_lodes(args.lodes)
    wg = set(w["home_geoid"].unique().to_list())
    lodes = lodes.filter(pl.col("w_geocode").is_in(wg) & pl.col("h_geocode").is_in(wg))

    homes = sorted(wg); hid = {g: i for i, g in enumerate(homes)}
    dests = sorted(lodes["w_geocode"].unique().to_list()); did = {g: i for i, g in enumerate(dests)}
    nH, nD = len(homes), len(dests)
    n_naics = len(U.categ_types["pr_naics"].categories)

    li = lodes.with_columns(
        pl.col("h_geocode").replace_strict(hid).alias("hi"),
        pl.col("w_geocode").replace_strict(did).alias("di"))
    flow_dh = sp.csr_matrix(
        (li["count"].to_numpy().astype(float), (li["di"].to_numpy(), li["hi"].to_numpy())),
        shape=(nD, nH))
    flow_hd = flow_dh.T.tocsr()

    home_geoid_arr = w["home_geoid"].to_numpy()
    naics_arr = w["naics"].to_numpy()
    hidx = np.array([hid[g] for g in home_geoid_arr])
    home_naics = np.zeros((nH, n_naics))
    np.add.at(home_naics, (hidx, naics_arr), 1.0)

    print(f"{len(w)} workers, {nH} homes, {nD} destinations, {len(lodes)} LODES pairs")
    t0 = time.time()
    demand, slot_sizes = build_demand(
        flow_dh, home_naics, dests, home_geoid_arr, naics_arr, n_naics, args.seed)
    dm = demand[demand > 0]
    print(f"steps 1-4: {time.time() - t0:.1f} s -- {len(dm)} demanded cells, "
          f"{int(demand.sum())} slots-worth of workers")
    print(f"  demanded cell size   median {np.median(dm):.0f}  mean {dm.mean():.1f}  "
          f"p90 {np.quantile(dm, .9):.0f}  max {dm.max()}")
    print(f"  CBP slot sizes drawn median {np.median(slot_sizes):.0f}  mean {slot_sizes.mean():.1f}"
          f"  ({len(slot_sizes)} establishments)\n")

    t0 = time.time()
    gw, gleft = fill_greedy(demand, flow_dh, hidx, naics_arr, nH, n_naics, args.seed)
    t_greedy = time.time() - t0
    ok = gw >= 0
    gcorr, gcells = corr_and_cells(hidx[ok], gw[ok], np.ones(int(ok.sum()), dtype=np.int64),
                                   naics_arr[ok], homes, dests, lodes)

    t0 = time.time()
    ch, cd, cn, cc, unpl = fill_ipf(demand, flow_hd, hidx, naics_arr, nH, nD, args.seed)
    t_ipf = time.time() - t0
    icorr, icells = corr_and_cells(ch, cd, cc, cn, homes, dests, lodes)

    tgt_mean = dm.mean()
    # The realized cell-size distribution is bimodal -- a spike of one-worker cells created by the
    # fallback, plus a bulk near the demanded sizes -- so the median sits on the cliff between them
    # and flips between ~3 and ~24 on a tiny change in the singleton count. Mean, the singleton
    # share and the >=20 share are the statistics that actually describe it.
    print(f"{'fill':8s}{'corr':>8s}{'cells':>8s}{'mean':>7s}{'size-1':>8s}{'>=20':>8s}"
          f"{'fallback':>10s}{'time':>8s}")
    print(f"{'demand':8s}{'--':>8s}{len(dm):8d}{tgt_mean:7.1f}"
          f"{(dm == 1).mean():8.3f}{dm[dm >= 20].sum() / dm.sum():8.3f}"
          f"{'--':>10s}{'--':>8s}   <- the CBP target")
    for name, corr, cells, fb, t in [
            ("greedy", gcorr, gcells, gleft, t_greedy),
            ("ipf", icorr, icells, unpl, t_ipf)]:
        print(f"{name:8s}{corr:8.3f}{len(cells):8d}{cells.mean():7.1f}"
              f"{(cells == 1).mean():8.3f}{cells[cells >= 20].sum() / cells.sum():8.3f}"
              f"{100.0 * fb / len(w):9.2f}%{t:7.1f}s")

    print(f"\n  cell-count vs demand:  greedy {len(gcells) / len(dm):.3f}   ipf {len(icells) / len(dm):.3f}")
    print(f"  mean-size vs demand:   greedy {gcells.mean() / tgt_mean:.3f}   "
          f"ipf {icells.mean() / tgt_mean:.3f}   (1.000 = tracks CBP demand exactly)")
    print(f"  one-worker workplaces: greedy {int((gcells == 1).sum())}   ipf {int((icells == 1).sum())}"
          f"   (a work group of 1 cannot transmit anything)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
