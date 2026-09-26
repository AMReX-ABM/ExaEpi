#!/usr/bin/env -S python -u
"""Draw a synthetic population from a precompute bundle -- the oracle the AMReX port is checked against.

This is step 4 of the in-process generation plan: implement the runtime sampling path in Python
first, against the same bundle ExaEpi will read, and emit the frame `process_upop` already
consumes. That lets a generated population go straight through the existing converter to a `.bin`
and be compared with the delivered one, before any C++ is written.

What the bundle fixes and what this draws:

    fixed in the bundle   P-MEDM allocation matrices, PUMS donor attributes and household index,
                          per-PUMA population totals, LODES flows, CBP size tables
    drawn here            which donor households land in which block group (TRS), and the dense
                          household ids that follow

Worker destinations, school assignment and every mixing group are deliberately NOT drawn here.
`upop_to_exaepi.py` already does all of that downstream from exactly this frame, and the 70-run
ensemble showed those axes are statistically indistinguishable from disease-seed noise anyway.
The axis that matters, and the only one this adds, is residential placement.

Every draw is keyed on (seed, block group, donor) rather than taken from a running stream, so the
population is a pure function of the seed and does not depend on the order block groups are
visited -- the property step 7 requires of the port, established here first where it is cheap to
test.
"""

import argparse
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_precompute import read_bundle  # noqa: E402

RACE_CATS = ["white", "blk_af_amer", "asian", "native_amer", "pac_island", "other", "mult"]
SEX_CATS = ["female", "male"]
TRAVEL_CATS = ["car_truck_van", "public_transportation", "bicycle", "walked", "motorcycle",
               "taxicab", "other", "wfh"]
VEH_OCC_CATS = ["drove_alone", "carpooled"]
GRADE_CATS = ["childcare", "preschl", "kind", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th",
              "8th", "9th", "10th", "11th", "12th", "undergrad", "grad"]


def trs_column(vals: np.ndarray, target: int, rng) -> np.ndarray:
    """Truncate, replicate, sample: keep whole parts, draw the remainder from the fractions.

    The integerization P-MEDM's own workflow uses. A cell's whole part is a household that is
    certainly there; the fractional parts compete for the seats left over once those are placed.
    """
    whole = np.floor(vals).astype(np.int64)
    short = target - int(whole.sum())
    if short > 0:
        frac = vals - whole
        s = frac.sum()
        if s <= 0:
            return whole
        whole += np.bincount(rng.choice(len(vals), size=short, p=frac / s), minlength=len(vals))
    elif short < 0:
        nz = np.flatnonzero(whole > 0)
        if len(nz):
            take = min(-short, len(nz))
            whole[nz[rng.choice(len(nz), size=take, replace=False)]] -= 1
    return whole


def draw_placements(b: dict, seed: int, rep: int, verbose=True):
    """Which donor households land in which block group.

    Returns (bg_row, donor_row, count) over the bundle's global block-group and donor indices.
    Each block group is drawn independently from a generator keyed on (seed, replicate, geoid),
    so adding a block group, reordering them, or splitting them across ranks cannot change any
    other block group's result.
    """
    values = b["almat.values"]
    scale = b["almat.scale"]
    bg_off = b["almat.puma_bg_offset"]
    don_off = b["almat.puma_donor_offset"]
    bg_index = b["almat.bg_index"]
    donor_index = b["almat.donor_index"]
    bg_pop = b["almat.bg_pop"]
    total_pop = b["almat.total_pop"]
    # Household sizes taken from the donor table's own person offsets, which is the number of
    # people actually available to place. `almat.donor_hh_size` comes from livelike's sporder and
    # can disagree for vacant units.
    hh_off = b["donors.hh_offset"]
    di_all = b["almat.donor_index"]
    true_size = np.where(di_all >= 0, hh_off[np.clip(di_all + 1, 0, len(hh_off) - 1)]
                         - hh_off[np.clip(di_all, 0, len(hh_off) - 1)], 0).astype(np.int64)

    n_reps = values.shape[0]
    if rep >= n_reps:
        raise SystemExit(f"replicate {rep} requested but bundle has {n_reps}")

    out_bg, out_dn, out_ct = [], [], []
    for p in range(len(total_pop)):
        b0, b1 = int(bg_off[p]), int(bg_off[p + 1])
        d0, d1 = int(don_off[p]), int(don_off[p + 1])
        n_bg, n_don = b1 - b0, d1 - d0

        # Undo the per-donor-row uint16 quantisation. pymedm builds the allocation matrix with
        # counts=True, so a cell is ALREADY an expected household count -- checked on NM PUMA
        # 3500100, where the matrix sums to 51,623 households carrying 123,136 people against a
        # published block-group total of 128,325. Multiplying by the PUMA population again, as an
        # occurrence-probability reading of it would, inflated the draw by five orders of
        # magnitude.
        sl = b_slice(p, bg_off, don_off)
        q = values[rep, sl].reshape(n_don, n_bg)
        expect = q.astype(np.float64) / 65535.0 * scale[rep, d0:d1][:, None]

        # Per-block-group household count averaged over ALL replicates. This is the reference the
        # per-replicate target is scaled against, so the ensemble mean lands on published
        # population while each replicate keeps its own deviation.
        ref = np.zeros(n_bg)
        for rr in range(n_reps):
            qq = values[rr, sl].reshape(n_don, n_bg)
            ref += (qq.astype(np.float64) / 65535.0 * scale[rr, d0:d1][:, None]).sum(axis=0)
        ref /= max(n_reps, 1)

        # Only occupied donors can contribute people. Vacant units carry real allocation weight --
        # they satisfy the housing-unit constraints -- but placing them would add households
        # holding nobody, so they are dropped from the person draw.
        occ = np.flatnonzero(true_size[d0:d1] > 0)
        if len(occ) == 0:
            continue

        for j in range(n_bg):
            col = expect[occ, j]
            if col.sum() <= 0:
                continue
            # Anchor the MEAN to published population, but let this replicate's own deviation
            # through. Targeting published population flat would make every replicate produce the
            # same block-group headcount, deleting exactly the variation the replicates exist to
            # carry -- measured, that collapsed the replicate axis to a CV ratio of 0.113 against
            # TRS-only's 0.101, i.e. replicates bought 1.11x over a re-draw. Scaling by this
            # replicate's household count relative to the across-replicate mean keeps the level
            # right and restores the spread.
            #
            # Target the block group's PUBLISHED population, not the column's own household sum.
            #
            # P-MEDM is fitted against `est_ind['population']`, which livelike rescales by the
            # PUMS person/household weight ratio and is therefore FRACTIONAL -- 4.143 persons for
            # a household whose NP is 3. The allocation hits that constraint almost exactly
            # (102,331.5 against a published 102,344 on PUMA 3500804), but expanding each placed
            # household into its literal PUMS person records delivers only NP of them. Statewide
            # that ratio is 3862/4321 = 0.894, and taking the column sum as a household target
            # reproduced it as a 9.45% population shortfall.
            #
            # So convert the published population into a household count through the donors' own
            # true sizes. Household composition still comes from the solve; only how many are
            # placed is re-targeted, which is what keeps the agent count right.
            people = float(bg_pop[b0 + j])
            mean_size = float(col @ true_size[d0:d1][occ]) / col.sum()
            # This replicate's household count for the block group, relative to the mean across
            # replicates. 1.0 for an average replicate; above or below carries its deviation.
            dev = (expect[:, j].sum() / ref[j]) if ref[j] > 0 else 1.0
            target = int(round(people * dev / max(mean_size, 1e-6)))
            if target <= 0:
                continue
            rng = np.random.default_rng([seed, rep, int(bg_index[b0 + j])])
            cnt = trs_column(col, target, rng)
            nz = np.flatnonzero(cnt > 0)
            if len(nz) == 0:
                continue
            out_bg.append(np.full(len(nz), b0 + j, dtype=np.int64))
            out_dn.append(donor_index[d0 + occ[nz]])
            out_ct.append(cnt[nz])
        if verbose:
            print(f"  PUMA {p + 1}/{len(total_pop)}: {n_don} donors x {n_bg} block groups",
                  flush=True)

    return np.concatenate(out_bg), np.concatenate(out_dn), np.concatenate(out_ct)


def b_slice(p, bg_off, don_off):
    """Flat-cell slice for PUMA p inside the concatenated allocation matrix."""
    start = 0
    for i in range(p):
        start += (int(bg_off[i + 1]) - int(bg_off[i])) * (int(don_off[i + 1]) - int(don_off[i]))
    n = (int(bg_off[p + 1]) - int(bg_off[p])) * (int(don_off[p + 1]) - int(don_off[p]))
    return slice(start, start + n)


def expand_to_persons(b, bg_rows, donor_rows, counts):
    """Turn placed households into the person frame process_upop expects.

    Each placement is `count` copies of one donor household, and each copy contributes that
    household's donor persons. Household ids are dense within a block group, which is what
    `adjust_indexes` would otherwise have to recompute.
    """
    hh_off = b["donors.hh_offset"]
    bg_geoid = b["almat.bg_index"]

    # One row per household copy.
    hh_bg = np.repeat(bg_rows, counts)
    hh_donor = np.repeat(donor_rows, counts)
    sizes = (hh_off[hh_donor + 1] - hh_off[hh_donor]).astype(np.int64)

    # Dense household id within each block group, assigned by sorting on the block group only --
    # stable, so it depends on placement order, which is itself a pure function of the seed.
    order = np.argsort(hh_bg, kind="stable")
    hh_id = np.empty(len(hh_bg), dtype=np.int64)
    sb = hh_bg[order]
    bounds = np.flatnonzero(np.r_[True, sb[1:] != sb[:-1], True])
    for a, z in zip(bounds[:-1], bounds[1:]):
        hh_id[order[a:z]] = np.arange(z - a)

    # Ragged expansion: each household contributes its own donor persons, gathered in one pass.
    person_hh = np.repeat(np.arange(len(hh_bg)), sizes)
    within = np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    person_src = hh_off[hh_donor[person_hh]] + within

    return {
        "geoid": bg_geoid[hh_bg[person_hh]],
        "hh_id": hh_id[person_hh],
        "src": person_src,
        "n_hh": len(hh_bg),
    }


def to_frame(b, pers) -> pl.DataFrame:
    """The feather-shaped frame, so the existing converter can consume it unchanged."""
    src = pers["src"]
    naics_codes = np.array(["-1"] + list(_naics_categories()), dtype=object)

    def cat(section, cats):
        """Integer codes back to the feather's category strings, with -1 meaning not applicable.

        Built as a Python list rather than a numpy object array: polars refuses to cast an Object
        column to Utf8, and np.where over a string array with a None branch produces exactly that.
        """
        v = b[section][src].astype(np.int64)
        lookup = list(cats)
        return pl.Series(
            [lookup[i] if 0 <= i < len(lookup) else None for i in v], dtype=pl.Utf8)

    naics = b["donors.naics"][src].astype(np.int64)
    geoid = pers["geoid"]
    hh = pers["hh_id"]
    # p_id / h_id only have to be unique and stable; the converter densifies them anyway.
    h_id = pl.Series([f"{g}-{h}" for g, h in zip(geoid, hh)], dtype=pl.Utf8)
    return pl.DataFrame({
        "p_id": pl.Series([f"{g}-{h}-{i}" for g, h, i in
                           zip(geoid, hh, _within_household(geoid, hh))], dtype=pl.Utf8),
        "h_id": h_id,
        "geoid": pl.Series([str(g) for g in geoid], dtype=pl.Utf8),
        "pr_age": pl.Series(b["donors.age"][src].astype(np.int64), dtype=pl.Int64),
        "pr_sex": cat("donors.sex", SEX_CATS),
        "pr_race": cat("donors.race", RACE_CATS),
        "pr_naics": pl.Series(
            [naics_codes[n + 1] if n >= 0 else None for n in naics], dtype=pl.Utf8),
        "pr_travel": cat("donors.travel", TRAVEL_CATS),
        "pr_veh_occ": cat("donors.veh_occ", VEH_OCC_CATS),
        "pr_grade": cat("donors.grade", GRADE_CATS),
    })


def _within_household(geoid, hh):
    """Sequence number of each person inside its own household copy."""
    key = _group_key(geoid, hh)
    order = np.argsort(key, kind="stable")
    out = np.empty(len(key), dtype=np.int64)
    sk = key[order]
    bounds = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
    for a, z in zip(bounds[:-1], bounds[1:]):
        out[order[a:z]] = np.arange(z - a)
    return out


def _group_key(geoid, hh):
    return geoid.astype(np.int64) * 100000 + hh.astype(np.int64)


def _naics_categories():
    import upop_to_exaepi as U
    return list(U.categ_types["pr_naics"].categories)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replicate", type=int, default=0,
                    help="which allocation-matrix replicate to draw from")
    ap.add_argument("--out", required=True, help="output feather, shaped like the UrbanPop ones")
    ap.add_argument("--compare", default=None,
                    help="glob of delivered feathers to score the generated population against")
    args = ap.parse_args()

    t0 = time.time()
    b = read_bundle(args.bundle)
    if "almat.values" not in b:
        sys.exit("bundle has no allocation matrices -- rebuild it with --pumas")
    print(f"bundle: {b['almat.values'].shape[0]} replicate(s), "
          f"{len(b['almat.total_pop'])} PUMAs, {len(b['bg.geoid'])} block groups")

    bg_rows, donor_rows, counts = draw_placements(b, args.seed, args.replicate)
    pers = expand_to_persons(b, bg_rows, donor_rows, counts)
    df = to_frame(b, pers)
    print(f"\ngenerated {len(df)} persons in {pers['n_hh']} households, "
          f"{df['geoid'].n_unique()} block groups, {time.time() - t0:.1f} s")

    df.write_ipc(args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")

    if args.compare:
        import glob
        ref = pl.concat([pl.read_ipc(p, columns=["geoid", "pr_age"])
                         for p in sorted(glob.glob(args.compare))])
        a = df.group_by("geoid").agg(pl.len().alias("gen"))
        c = ref.group_by("geoid").agg(pl.len().alias("ref"))
        m = a.join(c, on="geoid", how="full", coalesce=True).with_columns(
            pl.col("gen").fill_null(0), pl.col("ref").fill_null(0))
        corr = m.select(pl.corr("gen", "ref")).item()
        tot_g, tot_r = int(m["gen"].sum()), int(m["ref"].sum())
        print(f"\nvs delivered: {tot_g} vs {tot_r} persons ({100.0 * tot_g / tot_r - 100:+.2f}%), "
              f"per-block-group population correlation {corr:.4f}")
        print(f"  mean age generated {df['pr_age'].mean():.2f}  delivered {ref['pr_age'].mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
