#!/usr/bin/env python
"""Is the IPF fill actually order-independent, and what does it cost?

The whole reason to replace the greedy fill is step 7 of the plan: group assignment was moved out
of C++ because per-rank RNG streams made populations depend on rank count. The greedy fill has the
same disease in a worse form -- its own comments record that visiting destinations in a different
order nearly doubled CA's fallback rate (3.96% vs 1.69%). Any replacement has to be provably
insensitive to the order work is done in, or the AMReX port reintroduces exactly the bug the
architecture was changed to avoid.

IPF should be order-independent by construction: it converges to the unique I-projection of the
prior onto the margin constraints, so the answer does not depend on which NAICS, row or column is
touched first. TRS then has to be keyed so that a cell's draw depends on the cell, not on when it
was reached.

This shuffles the processing order of NAICS codes and of the rows/columns within each, keyed per
group rather than per call, and checks the placement comes out bitwise identical. It also times
the solve, since init cost becomes per-run once generation moves in-process.
"""

import argparse
import hashlib
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

from fill_ipf2 import ipf_sparse, load_lodes, repair_rows, trs_group


def keyed_rng(seed: int, group_id: int):
    """A cell's randomness keyed on (seed, group), never on a running stream.

    This is the Python stand-in for Philox keyed on (agent.seed, entity_id): group g always gets
    the same stream no matter how many groups were processed before it, or on which rank.
    """
    return np.random.default_rng([seed, group_id])


def run(seed, naics_order, shuffle_within, homes, dests, flow, sup_by_n, dem_by_n, nH, nD):
    H, D, C = [], [], []
    for n in naics_order:
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
        ri = ri[np.diff(P.indptr) > 0]
        ci = ci[np.diff(P.tocsc().indptr) > 0]
        P = flow[ri][:, ci].tocsr()
        if P.nnz == 0:
            continue

        # Permuting the rows and columns handed to the solver is the strongest form of the test:
        # it changes the memory layout and the sweep order, not just the outer loop.
        if shuffle_within:
            g = np.random.default_rng(12345 + n)
            rp, cp = g.permutation(len(ri)), g.permutation(len(ci))
            inv_r = np.argsort(rp); inv_c = np.argsort(cp)
            P = P[rp][:, cp].tocsr()
            ri_s, ci_s = ri[rp], ci[cp]
        else:
            ri_s, ci_s = ri, ci

        rt, ct = np.zeros(P.shape[0]), np.zeros(P.shape[1])
        rt[:] = s[ri_s]; ct[:] = t[ci_s]
        ct = ct * (rt.sum() / ct.sum())
        v, rows, cols = ipf_sparse(P, rt, ct)

        # Key TRS on the GLOBAL destination index, so a column's draw is the same wherever that
        # column happens to sit in this permutation.
        ct_i = np.round(ct).astype(np.int64)
        cnt = np.zeros(len(v), dtype=np.int64)
        order = np.argsort(cols, kind="stable")
        c_sorted = cols[order]
        bounds = np.flatnonzero(np.r_[True, c_sorted[1:] != c_sorted[:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            cj = c_sorted[a]
            idx = order[a:b]
            rng = keyed_rng(seed, int(ci_s[cj]) * 1000 + n)
            vals = v[idx]
            whole = np.floor(vals).astype(np.int64)
            frac = vals - whole
            short = int(ct_i[cj]) - int(whole.sum())
            if short > 0 and frac.sum() > 0:
                # order by global home index so the draw is permutation-invariant
                gi = np.argsort(ri_s[rows[idx]])
                p = frac[gi] / frac.sum()
                whole[gi] += np.bincount(rng.choice(len(idx), size=short, p=p), minlength=len(idx))
            elif short < 0:
                nz = np.flatnonzero(whole > 0)
                if len(nz):
                    gi = nz[np.argsort(ri_s[rows[idx[nz]]])]
                    take = min(-short, len(gi))
                    whole[gi[rng.choice(len(gi), size=take, replace=False)]] -= 1
            cnt[idx] = whole

        # Key the repair on the GLOBAL home index and the sweep number, so a row's stream is a
        # property of that row rather than of how many rows were repaired before it.
        cnt = repair_rows(
            cnt, rows, cols, rt, None,
            key=lambda ri, sw, _r=ri_s: np.random.default_rng([seed, 7, n, int(_r[ri]), sw]),
            col_key=ci_s,
        )
        keep = cnt > 0
        H.append(ri_s[rows[keep]]); D.append(ci_s[cols[keep]]); C.append(cnt[keep])

    H, D, C = np.concatenate(H), np.concatenate(D), np.concatenate(C)
    # Canonical sort so the digest reflects the placement, not the order it was produced in.
    o = np.lexsort((C, D, H))
    return H[o], D[o], C[o]


def digest(H, D, C):
    h = hashlib.sha256()
    for a in (H, D, C):
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="/workspaces/ExaEpi/data/UrbanPop/workers_nt_dt.intermediate.csv")
    ap.add_argument("--lodes", default="/workspaces/ExaEpi/data/LODES7/nm_od_main_JT00_2019.csv.gz")
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()

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
    codes = sorted(sup_by_n)

    common = (homes, dests, flow, sup_by_n, dem_by_n, nH, nD)

    t0 = time.time()
    a = run(args.seed, codes, False, *common)
    t_base = time.time() - t0

    rev = list(reversed(codes))
    b = run(args.seed, rev, False, *common)

    g = np.random.default_rng(999)
    shuf = list(np.array(codes)[g.permutation(len(codes))])
    c = run(args.seed, shuf, True, *common)

    d = run(args.seed + 1, codes, False, *common)

    print(f"solve + integerize: {t_base:.1f} s for {len(codes)} NAICS, {int(w.height)} workers\n")
    print(f"{'variant':38s}{'digest':>18s}")
    for name, r in [("baseline order, seed 29", a),
                    ("NAICS order reversed", b),
                    ("NAICS shuffled + rows/cols permuted", c),
                    ("baseline order, seed 30", d)]:
        print(f"{name:38s}{digest(*r):>18s}")

    ok = digest(*a) == digest(*b) == digest(*c)
    diff = digest(*a) != digest(*d)
    print()
    print(f"  order-independent: {'YES' if ok else 'NO'}")
    print(f"  seed actually varies the result: {'YES' if diff else 'NO'}")
    if ok and diff:
        print("\n  => The fill is a pure function of (seed, supply, demand, flow). Processing order,")
        print("     and therefore rank decomposition, cannot change the population it produces.")
    return 0 if (ok and diff) else 1


if __name__ == "__main__":
    raise SystemExit(main())
