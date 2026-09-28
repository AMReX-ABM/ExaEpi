"""Stage S3: worker destinations (replaces upop_to_exaepi.alloc_workers).

Only commuters are placed: a worker who works from home (travel wfh) keeps their home block group,
which is where ExaEpi keeps them during the day.

Steps 1-4 keep alloc_workers' design -- size each destination from its inbound flow, give it
establishment slots whose industries come from its own commute shed and whose sizes come from the
CBP distributions, rescale to each industry's true statewide count -- with every draw keyed and
every count exact. The flow is not raw LODES but the corrected prior of commute.py: LODES plus a
short-range background, weighted by the CTPP-calibrated reliability r(distance), quantised to
integers (commute.W_SCALE per job).

  1. flow[d, h] = the prior from home h to destination d, both worker homes; dest_total[d] = its
     row sums. Exact int64.
  2. local[d, n] = sum_h flow[d, h] * workers(h, n): commute-reachable supply of industry n.
  3. n_slots[d] = max(1, rint(dest_total[d] / W_SCALE / avg)), avg = mean workgroup target over
     workers. Slot s of destination d draws its industry with int_cdf over local[d] (key
     SLOT_NAICS (dest, s)) and its size from the CBP table (SLOT_SIZE (dest, s)).
  4. Per industry: demand = implied * T / S by integer division, the remainder handed out by
     largest remainder, ties by destination geoid. Column sums are exactly the worker counts.

Step 5 fills each industry with an IPF whose rows are (home, commute-time band) -- the band from
the worker's own reported minutes and mode (commute.band) -- and whose prior at destination d is
flow(h, d) times the band's kernel weight P(band | distance) / P(band) (commute.kern), so a worker
who reports a 5-minute walk is not sent 100 km. Rows are hard (every worker is placed); the
destination demand is soft: the column scaling is b_j = sqrt(ct_j / S_j) with S_j = sum_i q_ij a_i
(unbalanced Sinkhorn, i.e. KL to the prior plus a KL penalty on the column totals with weight 1),
so demand no nearby worker can fill is released rather than filled from far away. Measured on NM:
27% of slot demand moves, and the share of commuters sent > 100 km falls from 20% to 1.3% (CTPP
0.9%) with the work-group sizes S8 builds unchanged.

Integerization: column-wise TRS to the soft column sums rounded by largest remainder (IPF_TRS_ADD
/ IPF_TRS_TRIM), then row repair (IPF_REPAIR) restores exact row sums by moving workers only
between a row's own cells, adding in proportion to the IPF values -- a near-uniform choice here
would scatter single-worker rows over their whole LODES row -- then a deterministic final pass
guarantees them. Workers of each (industry, home, band) are ordered by WORK_ASSIGN (bg, h, p) and
dealt to that row's cells in destination order.

Workers no destination can take fall back to a draw over their home's prior row (WORK_FALLBACK); a
home with no row works in its own block group.

Float sums that affect results are sequential in canonical order (np.bincount / np.cumsum), never
pairwise np.sum, and the only non-arithmetic operation is sqrt, so the C++ port reproduces them.
"""

import numpy as np
import scipy.sparse as sp

from . import commute, kr64, stages, units

IPF_TOL = 1e-9
SOFT_ITERS = 200
REPAIR_SWEEPS = 12


def _seqsum(x):
    return np.cumsum(x)[-1] if len(x) else 0.0


def allocate(b, P, tables, seed, rep):
    """work_geoid per person (home geoid for non-workers and WFH workers) and a stats dict."""
    W = np.flatnonzero(P["employed"] & (P["travel"] != commute.TRAVEL_WFH))
    home_w = P["bg"][W]
    naics_w = P["naics"][W].astype(np.int64)
    n_naics = len(b["naics.codes.offsets"]) - 1
    work = P["bg"].copy()

    homes = np.unique(home_w)
    hidx_w = np.searchsorted(homes, home_w)

    # --- 1. the corrected prior between worker homes ---------------------------------------------
    lh, ld = b["lodes.home_geoid"], b["lodes.dest_geoid"]
    ip = b["lodes.indptr"]
    pair_h = np.repeat(lh, np.diff(ip))
    pair_d = ld[b["lodes.indices"]]
    pair_c = b["lodes.data"].astype(np.int64)
    keep = np.isin(pair_h, homes) & np.isin(pair_d, homes)
    pair_h, pair_d, pair_c = pair_h[keep], pair_d[keep], pair_c[keep]
    dests = np.unique(pair_d)
    ph, pd_ = np.searchsorted(homes, pair_h), np.searchsorted(dests, pair_d)
    o = np.lexsort((pd_, ph))
    ph, pd_, pair_c = ph[o], pd_[o], pair_c[o]
    # LODES rows repeat a (home, dest) pair only if the bundle does; sum them as a CSR would
    key = ph * len(dests) + pd_
    first = np.r_[True, key[1:] != key[:-1]]
    pair_c = np.add.reduceat(pair_c, np.flatnonzero(first))
    ph, pd_ = ph[first], pd_[first]
    qh, qd, qw, qk = commute.prior(b, homes, dests, ph, pd_, pair_c)
    npair = len(qh)
    flow_hd = sp.csr_matrix((qw, (qh, qd)), shape=(len(homes), len(dests)), dtype=np.int64)
    flow_hd.sort_indices()
    pid_hd = sp.csr_matrix((np.arange(1, npair + 1, dtype=np.int64), (qh, qd)), shape=flow_hd.shape)
    pid_hd.sort_indices()
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
    jobs = dest_total.astype(np.float64) / commute.W_SCALE
    n_slots = np.where(dest_total > 0, np.maximum(1, np.rint(jobs / avg)), 0).astype(np.int64)
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

    # --- 5. soft IPF fill per industry over (home, time band) rows, then deal workers -----------
    band_w = commute.bands(b, P["travel"][W], P["jwmnp"][W])
    nrb = commute.row_bands(b)
    rowid_w = hidx_w * nrb + band_w
    kern = b["commute.kern"]
    assigned = np.zeros(len(W), dtype=bool)
    h_w, p_w = P["h"][W], P["p"][W]
    order_key = kr64.draw(kr64.key(seed, rep, stages.WORK_ASSIGN, home_w, h_w, p_w), 0)
    # All workers grouped by (industry, home, band), keyed order within each group.
    wsort = np.lexsort((p_w, h_w, order_key, rowid_w, naics_w))
    wn_bounds = np.searchsorted(naics_w[wsort], np.arange(n_naics + 1))
    stats = {"unplaceable": 0, "repair_final": 0, "cells": 0, "one_worker_cells": 0,
             "demand_moved": 0, "demanded": 0}
    for n in units.each(range(n_naics)):
        if true_total[n] == 0 or demand[:, n].sum() == 0:
            continue
        wn = wsort[wn_bounds[n]:wn_bounds[n + 1]]
        rid = rowid_w[wn]
        rows_u, supply = np.unique(rid, return_counts=True)
        rh, rb = rows_u // nrb, rows_u % nrb
        ci = np.flatnonzero(demand[:, n] > 0)
        Pid = pid_hd[rh][:, ci].tocsr()
        Pid.sort_indices()
        pair = Pid.data.astype(np.int64) - 1
        erow = np.repeat(np.arange(len(rows_u)), np.diff(Pid.indptr))
        v0 = qw[pair].astype(np.float64) * kern[rb[erow], qk[pair]]
        prior = sp.csr_matrix((v0, Pid.indices, Pid.indptr), shape=Pid.shape)
        rr, cc, cnt, unpl = _fill_one(prior, supply, demand[ci, n], homes[rh], rb, nrb, dests[ci], n,
                                      seed, rep, stats)
        stats["unplaceable"] += unpl
        if len(cnt) == 0:
            continue
        stats["cells"] += len(cnt)
        # Deal each row's workers (keyed order) to its cells in destination-geoid order; rows and
        # their workers are both in row order and every placed row is filled exactly.
        co = np.lexsort((dests[ci[cc]], rr))
        rows_s, dest_s, cnt_s = rr[co], dests[ci[cc[co]]], cnt[co]
        placed = np.isin(rid, rows_u[np.unique(rows_s)])
        pool = wn[placed]
        dest_of = np.repeat(dest_s, cnt_s)
        if len(pool) != len(dest_of):
            raise RuntimeError(f"industry {n}: {len(pool)} workers for {len(dest_of)} places")
        work[W[pool]] = dest_of
        assigned[pool] = True

    # --- fallback: a draw over the home's own prior row ------------------------------------------
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
    stats["wfh"] = int((P["employed"] & (P["travel"] == commute.TRAVEL_WFH)).sum())
    return work, stats


def _fill_one(prior, supply, dem, row_home, row_band, nrb, col_geo, n, seed, rep, stats):
    """Soft IPF + column TRS + row repair for one industry on a (rows x demanded destinations)
    prior. Returns (row, col, count, unplaced), indices into the arrays passed in."""
    rk = np.diff(prior.indptr) > 0
    unplaceable = int(supply[~rk].sum())
    ri = np.flatnonzero(rk)
    Pm = prior[ri].tocsc()
    ci = np.flatnonzero(np.diff(Pm.indptr) > 0)
    Pm = prior[ri][:, ci].tocsr()
    Pm.sort_indices()
    if Pm.nnz == 0:
        return (np.zeros(0, np.int64),) * 3 + (unplaceable + int(supply[ri].sum()),)
    rt = supply[ri].astype(np.float64)
    ct = dem[ci].astype(np.float64) * (float(supply[ri].sum()) / float(dem[ci].sum()))
    rows = np.repeat(np.arange(Pm.shape[0]), np.diff(Pm.indptr))
    cols = Pm.indices.astype(np.int64)
    q = Pm.data.astype(np.float64)
    tol = IPF_TOL * max(1.0, _seqsum(rt))

    # v = q a_i b_j: b from the prior's column sums under the current a (not a damped update of
    # the running column factor, whose fixed point is the hard constraint again); a last, so the
    # rows hold exactly.
    a = np.ones(Pm.shape[0])
    prev = None
    for _ in range(SOFT_ITERS):
        S = np.bincount(cols, weights=q * a[rows], minlength=Pm.shape[1])
        bc = np.sqrt(np.divide(ct, S, out=np.ones_like(ct), where=S > 0))
        Tr = np.bincount(rows, weights=q * bc[cols], minlength=Pm.shape[0])
        a = np.divide(rt, Tr, out=np.zeros_like(rt), where=Tr > 0)
        v = q * a[rows] * bc[cols]
        cs = np.bincount(cols, weights=v, minlength=Pm.shape[1])
        if prev is not None and _seqsum(np.abs(cs - prev)) < tol:
            break
        prev = cs
    stats["demanded"] += int(dem[ci].sum())
    stats["demand_moved"] += float(0.5 * _seqsum(np.abs(cs - ct)))

    # Integer column targets: the soft column sums, largest remainder to the supply total.
    ct_i = np.floor(cs).astype(np.int64)
    short = int(supply[ri].sum()) - int(ct_i.sum())
    if short > 0:
        ct_i[np.lexsort((col_geo[ci], -(cs - np.floor(cs))))[:short]] += 1

    # Column-wise TRS: entries of each column in row order.
    cnt = np.floor(v).astype(np.int64)
    rkey = row_home[ri] * nrb + row_band[ri]
    oc = np.lexsort((rows, cols))
    bounds = np.flatnonzero(np.r_[True, cols[oc][1:] != cols[oc][:-1], True])
    for a0, z in units.each(list(zip(bounds[:-1], bounds[1:]))):
        idx = oc[a0:z]
        cj = cols[idx[0]]
        dg = int(col_geo[ci[cj]])
        need = int(ct_i[cj]) - int(cnt[idx].sum())
        if need > 0:
            frac = v[idx] - np.floor(v[idx])
            if _seqsum(frac) > 0:
                xs = kr64.draw(kr64.key(seed, rep, stages.IPF_TRS_ADD, n, dg,
                                        np.arange(need, dtype=np.int64)), 0)
                cnt[idx] += np.bincount(kr64.float_cdf(frac, xs), minlength=len(idx))
        elif need < 0:
            nz = idx[cnt[idx] > 0]
            r = ri[rows[nz]]
            o = kr64.shuffle_order(kr64.key(seed, rep, stages.IPF_TRS_TRIM, n, dg, row_home[r],
                                            row_band[r]), rkey[rows[nz]])
            cnt[nz[o[:min(-need, len(nz))]]] -= 1

    # Row repair: a row's cells in destination order; moves stay inside the row.
    orr = np.lexsort((cols, rows))
    rb = np.flatnonzero(np.r_[True, rows[orr][1:] != rows[orr][:-1], True])
    row_cells = {int(rows[orr[a0]]): orr[a0:z] for a0, z in zip(rb[:-1], rb[1:])}
    target = supply[ri].astype(np.int64)
    for sweep in range(REPAIR_SWEEPS):
        delta = target - np.bincount(rows, weights=cnt, minlength=len(ri)).astype(np.int64)
        bad = np.flatnonzero(delta != 0)
        if len(bad) == 0:
            break
        for r in units.each(bad):
            idx = row_cells[int(r)]
            hg, bd = int(row_home[ri[r]]), int(row_band[ri[r]])
            d = int(delta[r])
            if d > 0:
                xs = kr64.draw(kr64.key(seed, rep, stages.IPF_REPAIR, n, hg, bd, sweep,
                                        np.arange(d, dtype=np.int64)), 0)
                cnt[idx] += np.bincount(kr64.float_cdf(v[idx], xs), minlength=len(idx))
            else:
                w = cnt[idx].copy()
                if w.sum() == 0:
                    continue
                take = min(-d, int(w.sum()))
                xs = kr64.draw(kr64.key(seed, rep, stages.IPF_REPAIR, n, hg, bd, sweep,
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
