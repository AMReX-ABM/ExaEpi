"""Stage S3: worker destinations (replaces upop_to_exaepi.alloc_workers).

Steps 1-4 keep alloc_workers' design -- size each destination from real inbound LODES flow, give it
establishment slots whose industries come from its own commute shed and whose sizes come from the
CBP distributions, rescale to each industry's true statewide worker count -- with every draw keyed
and every count exact:

  1. flow[d, h] = LODES jobs from home block group h to destination d, restricted to pairs whose
     both ends are worker home block groups; dest_total[d] = row sums. Exact int64.
  2. local[d, n] = sum_h flow[d, h] * workers(h, n): commute-reachable supply of industry n. Exact.
  3. n_slots[d] = max(1, rint(dest_total[d] / avg)), avg = mean workgroup target over workers.
     Slot s of destination d draws its industry with int_cdf over local[d] (key SLOT_NAICS
     (dest, s)) and its size from the CBP table (SLOT_SIZE (dest, s)); implied[d, n] = sum.
  4. Per industry: demand = implied * T / S by integer division, the remainder handed out by
     largest remainder, ties by destination geoid. Column sums are exactly the worker counts.

Step 5 replaces the greedy, order-dependent fill with IPF per industry (experiments/alloc_ipf.py):
rows are homes, columns are demanded destinations, the prior is the LODES flow. Measured on NM with
identical demand: LODES correlation 0.857 vs 0.837, 296 vs 12,471 one-worker workplaces, fallback
0.13% vs 3.10%. Integerization is column-wise TRS (IPF_TRS_ADD / IPF_TRS_TRIM), then row repair
(IPF_REPAIR) restores exact row sums by moving workers only between a home's own LODES cells, then
a deterministic final pass guarantees them. Workers of each (home, industry) are ordered by
WORK_ASSIGN (bg, h, p) and dealt to that row's cells in destination order.

Workers no destination can take (a home reaching no destination demanding its industry, or an
industry with no demand) fall back to a draw over their home's LODES row (WORK_FALLBACK); a home
with no LODES row works in its own block group.

Float sums that affect results are sequential in canonical order (np.bincount / np.cumsum), never
pairwise np.sum, so the C++ port can reproduce them exactly.
"""

import numpy as np
import scipy.sparse as sp

from . import kr64, stages, units

IPF_ITERS = 60
IPF_TOL = 1e-9
REPAIR_SWEEPS = 12


def _seqsum(x):
    return np.cumsum(x)[-1] if len(x) else 0.0


def allocate(b, P, tables, seed, rep):
    """work_geoid per person (home geoid for non-workers) and a stats dict."""
    W = np.flatnonzero(P["employed"])
    home_w = P["bg"][W]
    naics_w = P["naics"][W].astype(np.int64)
    n_naics = len(b["naics.codes.offsets"]) - 1
    work = P["bg"].copy()

    homes = np.unique(home_w)
    hidx_w = np.searchsorted(homes, home_w)

    # --- 1. LODES restricted to worker homes at both ends -------------------------------------
    lh, ld = b["lodes.home_geoid"], b["lodes.dest_geoid"]
    ip = b["lodes.indptr"]
    pair_h = np.repeat(lh, np.diff(ip))
    pair_d = ld[b["lodes.indices"]]
    pair_c = b["lodes.data"].astype(np.int64)
    keep = np.isin(pair_h, homes) & np.isin(pair_d, homes)
    pair_h, pair_d, pair_c = pair_h[keep], pair_d[keep], pair_c[keep]
    dests = np.unique(pair_d)
    ph, pd_ = np.searchsorted(homes, pair_h), np.searchsorted(dests, pair_d)
    flow_hd = sp.csr_matrix((pair_c, (ph, pd_)), shape=(len(homes), len(dests)), dtype=np.int64)
    flow_hd.sort_indices()
    flow_dh = flow_hd.T.tocsr()
    flow_dh.sort_indices()
    dest_total = np.asarray(flow_dh.sum(axis=1)).ravel().astype(np.int64)

    # --- 2. commute-reachable industry supply ----------------------------------------------------
    home_naics = np.zeros((len(homes), n_naics), dtype=np.int64)
    np.add.at(home_naics, (hidx_w, naics_w), 1)
    local = np.asarray(flow_dh @ home_naics, dtype=np.int64)

    # --- 3. slots ------------------------------------------------------------------------------
    state_w = home_w // 10**10
    sn, sn_count = np.unique(np.stack([state_w, naics_w]), axis=1, return_counts=True)
    tsum = sum(int(c) * tables.target(s, n) for (s, n), c in zip(sn.T, sn_count))
    avg = tsum / len(W)
    n_slots = np.where(dest_total > 0, np.maximum(1, np.rint(dest_total / avg)), 0).astype(np.int64)
    implied = np.zeros((len(dests), n_naics), dtype=np.int64)
    for d in units.each(range(len(dests))):
        ns = int(n_slots[d])
        if ns == 0 or local[d].sum() == 0:
            continue
        dg, st = int(dests[d]), int(dests[d] // 10**10)
        slot = np.arange(ns, dtype=np.int64)
        s_naics = kr64.int_cdf(local[d], kr64.draw(kr64.key(seed, rep, stages.SLOT_NAICS, dg, slot), 0))
        for n in np.unique(s_naics):
            sl = slot[s_naics == n]
            if tables.has_est(st, n):
                sizes = tables.sample(st, n, kr64.key(seed, rep, stages.SLOT_SIZE, dg, sl))
                implied[d, n] += int(sizes.sum())
            else:
                implied[d, n] += len(sl) * tables.target(st, n)

    # --- 4. rescale to true industry totals, exactly -------------------------------------------
    true_total = np.bincount(naics_w, minlength=n_naics).astype(np.int64)
    demand = np.zeros_like(implied)
    for n in units.each(range(n_naics)):
        T, S = int(true_total[n]), int(implied[:, n].sum())
        if T == 0 or S == 0:
            continue
        q = implied[:, n] * T
        col = q // S
        rem = q % S
        short = T - int(col.sum())
        if short > 0:
            col[np.lexsort((dests, -rem))[:short]] += 1
        demand[:, n] = col

    # --- 5. IPF fill per industry, then deal workers to cells -----------------------------------
    assigned = np.zeros(len(W), dtype=bool)
    h_w, p_w = P["h"][W], P["p"][W]
    order_key = kr64.draw(kr64.key(seed, rep, stages.WORK_ASSIGN, home_w, h_w, p_w), 0)
    # All workers grouped by (industry, home), keyed order within each group.
    wsort = np.lexsort((p_w, h_w, order_key, hidx_w, naics_w))
    wn_bounds = np.searchsorted(naics_w[wsort], np.arange(n_naics + 1))
    stats = {"unplaceable": 0, "repair_final": 0, "cells": 0, "one_worker_cells": 0}
    for n in units.each(range(n_naics)):
        if true_total[n] == 0 or demand[:, n].sum() == 0:
            continue
        rows_h, cols_d, cnt, unpl = _fill_one(flow_hd, home_naics[:, n], demand[:, n], homes,
                                              dests, n, seed, rep, stats)
        stats["unplaceable"] += unpl
        if len(cnt) == 0:
            continue
        stats["cells"] += len(cnt)
        # Deal each home's workers (keyed order) to its cells in destination-geoid order.
        wn = wsort[wn_bounds[n]:wn_bounds[n + 1]]
        wh = hidx_w[wn]
        co = np.lexsort((dests[cols_d], rows_h))
        rows_s, dest_s, cnt_s = rows_h[co], dests[cols_d[co]], cnt[co]
        cb = np.flatnonzero(np.r_[True, rows_s[1:] != rows_s[:-1], True])
        for a, z in units.each(list(zip(cb[:-1], cb[1:]))):
            h = rows_s[a]
            lo, hi = np.searchsorted(wh, h), np.searchsorted(wh, h, side="right")
            dest_of = np.repeat(dest_s[a:z], cnt_s[a:z])
            pool = wn[lo:lo + len(dest_of)]
            work[W[pool]] = dest_of
            assigned[pool] = True

    # --- fallback: a draw over the home's own LODES row ------------------------------------------
    left = np.flatnonzero(~assigned)
    stats["fallback"] = len(left)
    if len(left):
        fk = kr64.draw(kr64.key(seed, rep, stages.WORK_FALLBACK, P["bg"][W][left], P["h"][W][left],
                                P["p"][W][left]), 0)
        for i, x in units.each(list(zip(left, fk))):
            h = hidx_w[i]
            s, e = flow_hd.indptr[h], flow_hd.indptr[h + 1]
            if e > s:
                work[W[i]] = dests[flow_hd.indices[s:e][kr64.int_cdf(flow_hd.data[s:e], x)]]
    stats["workers"] = len(W)
    return work, stats


def _fill_one(flow_hd, supply, demand, homes, dests, n, seed, rep, stats):
    """IPF + column TRS + row repair for one industry. Returns (home_row, dest_col, count, unplaced)."""
    ri = np.flatnonzero(supply > 0)
    ci = np.flatnonzero(demand > 0)
    P = flow_hd[ri][:, ci].tocsr()
    rk = np.diff(P.indptr) > 0
    unplaceable = int(supply[ri][~rk].sum())
    ri = ri[rk]
    P = flow_hd[ri][:, ci].tocsc()
    ck = np.diff(P.indptr) > 0
    ci = ci[ck]
    P = flow_hd[ri][:, ci].tocsr()
    P.sort_indices()
    if P.nnz == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), \
            unplaceable + int(supply[ri].sum())
    rt = supply[ri].astype(np.float64)
    ct = demand[ci].astype(np.float64) * (float(supply[ri].sum()) / float(demand[ci].sum()))
    rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
    cols = P.indices.astype(np.int64)
    v = P.data.astype(np.float64)
    tol = IPF_TOL * max(1.0, _seqsum(rt))
    for _ in range(IPF_ITERS):
        rs = np.bincount(rows, weights=v, minlength=P.shape[0])
        v = v * np.divide(rt, rs, out=np.zeros_like(rt), where=rs > 0)[rows]
        cs = np.bincount(cols, weights=v, minlength=P.shape[1])
        v = v * np.divide(ct, cs, out=np.zeros_like(ct), where=cs > 0)[cols]
        rs = np.bincount(rows, weights=v, minlength=P.shape[0])
        if _seqsum(np.abs(rs - rt)) < tol:
            break

    # Column-wise TRS: entries of each column in home order (row index order = home geoid order).
    cnt = np.floor(v).astype(np.int64)
    ct_i = np.rint(ct).astype(np.int64)
    oc = np.lexsort((rows, cols))
    bounds = np.flatnonzero(np.r_[True, cols[oc][1:] != cols[oc][:-1], True])
    for a, z in units.each(list(zip(bounds[:-1], bounds[1:]))):
        idx = oc[a:z]
        cj = cols[idx[0]]
        dg = int(dests[ci[cj]])
        short = int(ct_i[cj]) - int(cnt[idx].sum())
        if short > 0:
            frac = v[idx] - np.floor(v[idx])
            if _seqsum(frac) > 0:
                xs = kr64.draw(kr64.key(seed, rep, stages.IPF_TRS_ADD, n, dg,
                                        np.arange(short, dtype=np.int64)), 0)
                cnt[idx] += np.bincount(kr64.float_cdf(frac, xs), minlength=len(idx))
        elif short < 0:
            nz = idx[cnt[idx] > 0]
            hg = homes[ri[rows[nz]]]
            o = kr64.shuffle_order(kr64.key(seed, rep, stages.IPF_TRS_TRIM, n, dg, hg), hg)
            cnt[nz[o[:min(-short, len(nz))]]] -= 1

    # Row repair: a row's cells in destination order; moves stay inside the row.
    orr = np.lexsort((cols, rows))
    rb = np.flatnonzero(np.r_[True, rows[orr][1:] != rows[orr][:-1], True])
    row_cells = {int(rows[orr[a]]): orr[a:z] for a, z in zip(rb[:-1], rb[1:])}
    target = supply[ri].astype(np.int64)
    for sweep in range(REPAIR_SWEEPS):
        delta = target - np.bincount(rows, weights=cnt, minlength=len(ri)).astype(np.int64)
        bad = np.flatnonzero(delta != 0)
        if len(bad) == 0:
            break
        for r in units.each(bad):
            idx = row_cells[int(r)]
            hg = int(homes[ri[r]])
            d = int(delta[r])
            if d > 0:
                xs = kr64.draw(kr64.key(seed, rep, stages.IPF_REPAIR, n, hg, sweep,
                                        np.arange(d, dtype=np.int64)), 0)
                cnt[idx] += np.bincount(kr64.int_cdf(2 * cnt[idx] + 1, xs), minlength=len(idx))
            else:
                w = cnt[idx].copy()
                if w.sum() == 0:
                    continue
                take = min(-d, int(w.sum()))
                xs = kr64.draw(kr64.key(seed, rep, stages.IPF_REPAIR, n, hg, sweep,
                                        np.arange(take, dtype=np.int64)), 0)
                for j in kr64.int_cdf(w, xs):
                    if cnt[idx[j]] > 0:
                        cnt[idx[j]] -= 1
    # Deterministic final pass: whatever repair left, settle on the row's largest cells.
    delta = target - np.bincount(rows, weights=cnt, minlength=len(ri)).astype(np.int64)
    for r in units.each(np.flatnonzero(delta != 0)):
        stats["repair_final"] += 1
        idx = row_cells[int(r)]
        d = int(delta[r])
        while d != 0:
            j = idx[np.argmax(cnt[idx])]          # largest cell, first in destination order
            step = 1 if d > 0 else -1
            if step < 0 and cnt[j] == 0:
                break
            cnt[j] += step
            d -= step
    got = np.bincount(rows, weights=cnt, minlength=len(ri)).astype(np.int64)
    if not np.array_equal(got, target):
        raise RuntimeError(f"industry {n}: row sums not exact after repair")
    keep = cnt > 0
    stats["one_worker_cells"] += int((cnt == 1).sum())
    return ri[rows[keep]], ci[cols[keep]], cnt[keep], unplaceable
