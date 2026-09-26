#!/usr/bin/env python
"""Scratch probe: perturb block group population at a chosen coefficient of variation.

This is NOT the production path and does not live in the repo. It exists to answer one question
that extrapolation cannot settle: is a per-block-group population CV of ~0.05 -- what the minimal
constraint set's replicate allocation matrices actually produce -- large enough to move ExaEpi's
epidemic outcomes above the disease-seed noise floor?

Reference points from the 70-run A/B/C ensemble:

    arm A, disease seed only        attack rate sd 0.0008   (the noise floor)
    arm B, assignment re-draw       indistinguishable from A
    arm C, population CV ~0.26      sd_between 0.00217, ICC 0.823

Extrapolating arm C linearly to CV 0.05 predicts sd_between ~0.0004, about half the noise floor --
marginal, hence worth measuring.

`--cv_target` sets the median per-block-group CV directly rather than an inflation factor, so the
probe can be aimed at a measured replicate spread. The draw is lognormal in the relative change:
for the bulk of block groups that is indistinguishable from a normal, but it stays strictly
positive, which matters because 182 of New Mexico's 1445 block groups have an effective CV above
1/3 where a normal draw would take them to zero or below.
"""

import numpy as np
import polars as pl

MOE_Z = 1.645
SIGMA_CLIP = 3.0
ID_REPLICATE_SEP = "+"


def load_moe_table(fname: str) -> tuple[dict[str, float], float]:
    """Return {geoid: relative sd} and the median CV, from read_moe_fit_rates.py output."""
    df = pl.read_csv(fname, schema_overrides={"geoid": pl.Utf8})
    if "constraint" in df.columns:
        df = df.filter(pl.col("constraint") == "population")
    if "se" not in df.columns:
        df = df.with_columns((pl.col("moe") / MOE_Z).alias("se"))
    df = df.filter(pl.col("acs") > 0)
    cv = (df["se"] / df["acs"]).to_numpy()
    return dict(zip(df["geoid"].to_list(), cv.tolist())), float(np.median(cv))


def perturb_households(df: pl.DataFrame, moe_fname: str, rng: np.random.Generator,
                       cv_target: float) -> pl.DataFrame:
    """Resample households within each block group to a perturbed target population."""
    cv_by_geoid, cv_median = load_moe_table(moe_fname)
    # Scale the per-block-group ACS CVs so their median lands on cv_target, preserving the
    # relative pattern of which block groups are well or poorly determined.
    scale = cv_target / cv_median

    households = (
        df.group_by(["home_geoid", "household_id"], maintain_order=True)
        .agg(pl.len().alias("size"))
        .sort(["home_geoid", "household_id"])
    )
    bg = (households.group_by("home_geoid", maintain_order=True)
          .agg(pl.col("size").sum().alias("pop"), pl.col("household_id").max().alias("max_hid"))
          .sort("home_geoid"))
    geoids = bg["home_geoid"].to_list()
    pops = bg["pop"].to_numpy()
    max_hids = bg["max_hid"].to_numpy()

    cv = np.array([cv_by_geoid.get(g, 0.0) for g in geoids]) * scale
    sigma_log = np.sqrt(np.log1p(cv**2))
    draw = np.clip(rng.standard_normal(len(geoids)), -SIGMA_CLIP, SIGMA_CLIP)
    targets = np.rint(pops * np.exp(sigma_log * draw - 0.5 * sigma_log**2)).astype(np.int64)
    targets = np.maximum(targets, np.where(pops > 0, 1, 0))

    hh_geoid = households["home_geoid"].to_numpy()
    hh_id = households["household_id"].to_numpy()
    hh_size = households["size"].to_numpy()
    starts = np.searchsorted(hh_geoid, np.array(geoids), side="left")
    ends = np.searchsorted(hh_geoid, np.array(geoids), side="right")

    add_g, add_src, add_new, add_cp, drop_g, drop_h = [], [], [], [], [], []
    for i, geoid in enumerate(geoids):
        lo, hi = starts[i], ends[i]
        if hi <= lo:
            continue
        sizes_here, ids_here = hh_size[lo:hi], hh_id[lo:hi]
        deficit = int(targets[i]) - int(pops[i])
        if deficit > 0:
            next_hid, added, copies = int(max_hids[i]) + 1, 0, {}
            while added < deficit:
                k = int(rng.integers(0, len(ids_here)))
                src = int(ids_here[k])
                copies[src] = copies.get(src, 0) + 1
                add_g.append(geoid); add_src.append(src)
                add_new.append(next_hid); add_cp.append(copies[src])
                next_hid += 1
                added += int(sizes_here[k])
        elif deficit < 0:
            surplus, removed = -deficit, 0
            for k in rng.permutation(len(ids_here)):
                if removed >= surplus:
                    break
                s = int(sizes_here[k])
                if removed + s > surplus:      # test before dropping, so a block group
                    continue                    # can never be emptied outright
                drop_g.append(geoid); drop_h.append(int(ids_here[k]))
                removed += s

    out = df
    hid_dtype = df.schema["household_id"]
    if drop_h:
        out = out.join(pl.DataFrame({"home_geoid": drop_g,
                                     "household_id": pl.Series(drop_h, dtype=hid_dtype)}),
                       on=["home_geoid", "household_id"], how="anti")
    if add_new:
        adds = pl.DataFrame({
            "home_geoid": add_g,
            "household_id": pl.Series(add_src, dtype=hid_dtype),
            "_new_hid": pl.Series(add_new, dtype=hid_dtype),
            "_copy": pl.Series(add_cp, dtype=pl.Int32),
        })
        new_rows = df.join(adds, on=["home_geoid", "household_id"], how="inner").with_columns([
            (pl.col("id") + pl.lit(ID_REPLICATE_SEP) + pl.col("_copy").cast(pl.Utf8)).alias("id"),
            pl.col("_new_hid").alias("household_id"),
        ]).drop(["_new_hid", "_copy"])
        out = pl.concat([out, new_rows.select(out.columns)], how="vertical")

    b = df.group_by("home_geoid").agg(pl.len().cast(pl.Int64).alias("n0"))
    a = out.group_by("home_geoid").agg(pl.len().cast(pl.Int64).alias("n1"))
    m = b.join(a, on="home_geoid", how="left").with_columns(pl.col("n1").fill_null(0))
    rel = ((m["n1"] - m["n0"]) / m["n0"]).to_numpy()
    print(f"Perturbed (cv_target {cv_target}, scale {scale:.3f}): {len(df)} -> {len(out)} agents "
          f"({len(out) - len(df):+d}, {100 * (len(out) - len(df)) / len(df):+.2f}%)")
    print(f"  per-block-group relative change: sd {rel.std():.4f}  min {rel.min():+.3f}  max {rel.max():+.3f}")
    return out
