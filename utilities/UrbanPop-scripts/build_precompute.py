#!/usr/bin/env -S python -u
"""Build the randomness-free precompute bundle ExaEpi generates a population from.

ExaEpi today reads one delivered UrbanPop realization from a `.bin`. Varying who lives where
between runs -- the axis that dominates epidemic outcomes, ICC 0.82 on attack rate against a
disease-seed noise floor of sd 0.0008 -- means either shipping many `.bin` files or letting ExaEpi
draw a fresh population itself. This script builds the artifact for the second option: everything
in the pipeline that involves no sampling at all, so the only thing left at runtime is the drawing.

The split it encodes:

    precomputed here    P-MEDM allocation matrices, PUMS donor attributes and household index,
                        LODES O-D flows, CBP establishment-size and workgroup-size tables,
                        school capacities, county adjacency, block-group metadata
    drawn in ExaEpi     TRS integerization, worker and student destinations, home neighborhood
                        and household cluster, establishment draw and workgroup split, class
                        partition, dense id assignment

For New Mexico the bundle comes out around the size of the 22.06 MB `.bin` it replaces, so this is
not a storage tradeoff -- it swaps a fixed realization for a generator of about the same size.

Donor attributes are recovered from the delivered feathers rather than re-downloaded from the
Census API. Every synthetic person carries the `pums_id` of the household it was copied from, so
taking one replicate of each `pums_id` reconstructs the donor table exactly, with the exact integer
age and detailed NAICS the API path would need `keep_intermediates=True` to retain.

File layout (little-endian throughout):

    header      magic "UPPB", format version, section count, directory offset
    sections    each deflate-compressed, back to back
    directory   per section: name, dtype, shape, codec, offset, stored and raw byte counts

The directory sits at the end so sections can be streamed out without seeking back. ExaEpi reads
the header, seeks to the directory, and maps the sections it wants -- mirroring how
`readBlockGroupsFile` already handles the `.bin` index.
"""

import argparse
import glob
import json
import os
import struct
import sys
import time
import zlib

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upop_to_exaepi as U  # noqa: E402

# Bumped whenever the section set or any section's layout changes. ExaEpi refuses a mismatch,
# the same way readBlockGroupsFile checks FORMAT_VERSION against the .bin header.
PRECOMPUTE_FORMAT_VERSION = 1
MAGIC = 0x42505055  # "UPPB"

CODEC_RAW = 0
CODEC_DEFLATE = 1
DEFLATE_LEVEL = 6

HEADER = struct.Struct("<4I Q")


class BundleWriter:
    """Accumulates named arrays, then writes them compressed with a trailing directory.

    Sections are kept in memory until `write`, which is fine at state scale (the whole NM bundle
    is tens of MB) and keeps the writer simple. A national build would stream instead.
    """

    def __init__(self):
        self.sections = []

    def add(self, name: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr)
        if arr.dtype.kind not in "iuf":
            raise ValueError(f"section {name}: unsupported dtype {arr.dtype}")
        # Catch a non-finite section at the point it is added, not after an hour of solving.
        # pymedm's solve() returns an all-NaN allocation matrix without raising when LBFGS fails,
        # and a shape-only verify passes straight over it.
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            bad = int((~np.isfinite(arr)).sum())
            raise ValueError(f"section {name}: {bad} of {arr.size} values are not finite")
        self.sections.append((name, arr.dtype.str, arr.shape, arr.tobytes()))

    def add_bytes(self, name: str, blob: bytes) -> None:
        self.sections.append((name, "|u1", (len(blob),), blob))

    def add_json(self, name: str, obj) -> None:
        self.add_bytes(name, json.dumps(obj, indent=1, sort_keys=True).encode())

    def add_strings(self, name: str, values) -> None:
        """A string table as one concatenated blob plus start offsets, Arrow-style.

        Two sections rather than a ragged type, so the C++ side reads both as plain arrays and
        slices the blob -- no string parsing in the reader. String i is
        `blob[offsets[i]:offsets[i+1]]` with no separator to skip; an earlier NUL-joined version
        made offsets[i+1] an END position, so every consumer after the first string silently
        picked up a leading NUL.
        """
        enc = [v.encode() for v in values]
        offs = np.zeros(len(enc) + 1, dtype=np.int64)
        np.cumsum([len(e) for e in enc], out=offs[1:])
        self.add_bytes(f"{name}.blob", b"".join(enc))
        self.add(f"{name}.offsets", offs)

    def write(self, path: str) -> dict:
        stats = {}
        with open(path, "wb") as f:
            f.write(b"\x00" * HEADER.size)
            entries = []
            for name, dtype, shape, raw in self.sections:
                comp = zlib.compress(raw, DEFLATE_LEVEL)
                # Deflate on already-incompressible data can grow it; keep whichever is smaller
                # so a section is never penalised for compressing badly.
                if len(comp) < len(raw):
                    payload, codec = comp, CODEC_DEFLATE
                else:
                    payload, codec = raw, CODEC_RAW
                off = f.tell()
                f.write(payload)
                entries.append((name, dtype, shape, codec, off, len(payload), len(raw)))
                stats[name] = (len(raw), len(payload))

            dir_off = f.tell()
            for name, dtype, shape, codec, off, nstored, nraw in entries:
                nb = name.encode()
                db = dtype.encode()
                f.write(struct.pack("<H", len(nb)))
                f.write(nb)
                f.write(struct.pack("<H", len(db)))
                f.write(db)
                f.write(struct.pack("<B", len(shape)))
                for d in shape:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I QQQ", codec, off, nstored, nraw))

            f.seek(0)
            f.write(HEADER.pack(MAGIC, PRECOMPUTE_FORMAT_VERSION, len(entries), 0, dir_off))
        return stats


def read_bundle(path: str) -> dict:
    """Minimal reader -- the oracle the C++ reader is checked against, and used by --verify."""
    out = {}
    with open(path, "rb") as f:
        magic, version, n_sections, _, dir_off = HEADER.unpack(f.read(HEADER.size))
        if magic != MAGIC:
            raise ValueError(f"not a precompute bundle: magic {magic:#x}")
        if version != PRECOMPUTE_FORMAT_VERSION:
            raise ValueError(f"bundle version {version} != expected {PRECOMPUTE_FORMAT_VERSION}")
        f.seek(dir_off)
        for _ in range(n_sections):
            (nl,) = struct.unpack("<H", f.read(2))
            name = f.read(nl).decode()
            (dl,) = struct.unpack("<H", f.read(2))
            dtype = f.read(dl).decode()
            (nd,) = struct.unpack("<B", f.read(1))
            shape = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            codec, off, nstored, nraw = struct.unpack("<I QQQ", f.read(28))
            out[name] = (dtype, shape, codec, off, nstored, nraw)
        res = {}
        for name, (dtype, shape, codec, off, nstored, nraw) in out.items():
            f.seek(off)
            blob = f.read(nstored)
            if codec == CODEC_DEFLATE:
                blob = zlib.decompress(blob)
            if len(blob) != nraw:
                raise ValueError(f"section {name}: got {len(blob)} bytes, expected {nraw}")
            res[name] = np.frombuffer(blob, dtype=np.dtype(dtype)).reshape(shape)
    return res


def encode_categorical(series: pl.Series, name: str) -> np.ndarray:
    """Map a feather's string column onto the same integer codes upop_to_exaepi uses.

    Going through `categ_types` rather than inventing codes here keeps the bundle's encoding
    identical to the `.bin`'s, so a population generated from the bundle and one converted from a
    feather are directly comparable field by field.
    """
    cats = list(U.categ_types[name].categories)
    lut = {c: i for i, c in enumerate(cats)}
    return series.cast(pl.Utf8).replace_strict(lut, default=-1).to_numpy().astype(np.int16)


def build_lodes(paths, keep_geoids, bw, meta):
    """LODES O-D flows as a home-major CSR over block-group indices.

    CSR because the runtime needs a home's reachable destinations as a contiguous slice -- that is
    the candidate list every worker draw and every IPF sweep walks.
    """
    df = U.get_lodes_groups(paths)
    df = df.filter(
        pl.col("w_geocode").is_in(keep_geoids) & pl.col("h_geocode").is_in(keep_geoids)
    )
    homes = sorted(set(df["h_geocode"].to_list()))
    dests = sorted(set(df["w_geocode"].to_list()))
    hid = {g: i for i, g in enumerate(homes)}
    did = {g: i for i, g in enumerate(dests)}
    idx = df.with_columns(
        pl.col("h_geocode").replace_strict(hid).alias("hi"),
        pl.col("w_geocode").replace_strict(did).alias("di"),
    )
    m = sp.csr_matrix(
        (idx["count"].to_numpy().astype(np.int32),
         (idx["hi"].to_numpy(), idx["di"].to_numpy())),
        shape=(len(homes), len(dests)),
    )
    m.sort_indices()
    bw.add("lodes.indptr", m.indptr.astype(np.int64))
    bw.add("lodes.indices", m.indices.astype(np.int32))
    bw.add("lodes.data", m.data.astype(np.int32))
    bw.add("lodes.home_geoid", np.array([int(g) for g in homes], dtype=np.int64))
    bw.add("lodes.dest_geoid", np.array([int(g) for g in dests], dtype=np.int64))
    meta["lodes"] = {"homes": len(homes), "dests": len(dests), "pairs": int(m.nnz),
                     "jobs": int(m.data.sum())}
    print(f"  LODES: {len(homes)} homes, {len(dests)} destinations, {m.nnz} pairs")


def build_donors(feathers, bw, meta):
    """Reconstruct the PUMS donor table from the delivered synthetic population.

    Every synthetic household is a copy of one PUMS household, tagged with its `pums_id`; the
    replicate index is the middle field of `h_id`. Taking the lexicographically first `h_id` per
    `pums_id` therefore yields exactly one intact copy of each donor household, persons included.

    This is why the bundle does not need a Census API key: the donor attributes it carries are the
    same ones already present in the feathers, at full resolution -- an exact integer age from
    AGEP, not a 5-year cohort, and a detailed NAICS code, not a C24030 sector.
    """
    frames = []
    for p in feathers:
        frames.append(pl.read_ipc(p, columns=[
            "p_id", "pums_id", "h_id", "hh_size", "hh_type",
            "pr_age", "pr_sex", "pr_race", "pr_naics", "pr_travel", "pr_veh_occ", "pr_grade",
        ]))
    df = pl.concat(frames)
    print(f"  read {len(df)} synthetic persons from {len(feathers)} feathers")

    first = df.group_by("pums_id").agg(pl.col("h_id").min().alias("h_id"))
    don = df.join(first, on=["pums_id", "h_id"], how="inner")
    # p_id's trailing field is replicate*100 + person index, so ordering by it inside a household
    # reproduces the donor's own person order.
    don = don.with_columns(
        pl.col("p_id").str.split("-").list.last().cast(pl.Int32).alias("pseq")
    ).sort(["pums_id", "pseq"])

    hh_ids = don["pums_id"].unique(maintain_order=False).sort()
    hh_index = {h: i for i, h in enumerate(hh_ids.to_list())}
    don = don.with_columns(pl.col("pums_id").replace_strict(hh_index).alias("hh"))
    don = don.sort(["hh", "pseq"])

    hh_arr = don["hh"].to_numpy()
    counts = np.bincount(hh_arr, minlength=len(hh_ids))
    offsets = np.zeros(len(hh_ids) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    naics_cats = list(U.categ_types["pr_naics"].categories)
    naics_lut = {c: i for i, c in enumerate(naics_cats)}

    bw.add("donors.hh_offset", offsets)
    bw.add("donors.hh_size", counts.astype(np.int16))
    bw.add("donors.age", don["pr_age"].cast(pl.Int16).to_numpy().astype(np.int8))
    bw.add("donors.sex", encode_categorical(don["pr_sex"], "pr_sex").astype(np.int8))
    bw.add("donors.race", encode_categorical(don["pr_race"], "pr_race").astype(np.int8))
    bw.add("donors.travel", encode_categorical(don["pr_travel"], "pr_travel").astype(np.int8))
    bw.add("donors.veh_occ", encode_categorical(don["pr_veh_occ"], "pr_veh_occ").astype(np.int8))
    bw.add("donors.grade", encode_categorical(don["pr_grade"], "pr_grade").astype(np.int8))
    bw.add("donors.naics",
           don["pr_naics"].cast(pl.Utf8).replace_strict(naics_lut, default=-1)
           .to_numpy().astype(np.int16))
    bw.add_strings("donors.pums_id", hh_ids.to_list())

    meta["donors"] = {"households": len(hh_ids), "persons": len(don)}
    print(f"  donors: {len(hh_ids)} households, {len(don)} persons")
    return hh_index


EST_BANDS = [(1, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 249), (250, 499), (500, 999),
             (1000, None)]


def parse_establishment_bands(fname: str) -> dict:
    """The raw CBP (establishments, employees) pair per size band, as the file states them.

    `load_establishment_size_dists` expands each band into 256 log-uniform draws at load time,
    which is a *derived* representation: it turns 18 integers per (state, NAICS) into 512 floats,
    and storing that expansion made cbp.est_size alone 64 MB raw before compression. The expansion
    is deterministic given its fixed seed, so the bundle carries the 18 integers and the runtime
    reconstructs the distribution -- the same reason the bundle stores an allocation matrix rather
    than the populations drawn from it.
    """
    out = {}
    with open(fname) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2 + 2 * len(EST_BANDS):
                continue
            out[(int(parts[0]), parts[1])] = np.array([int(v) for v in parts[2:]], dtype=np.int32)
    if not out:
        U.raise_err(f"establishment sizes file '{fname}' has no usable rows -- regenerate it with "
                    "compute_workgroup_sizes.py --refresh-cbp-cache")
    return out


# PUMS person variables behind the ten fields ExaEpi reads, and the recodes onto the same
# categorical order `upop_to_exaepi.categ_types` uses. Validated field by field against the
# delivered feathers -- see `--validate_donors`.
PUMS_FEATURES = ["AGEP", "SEX", "RAC1P", "NAICSP", "JWTRNS", "JWRIP", "SCHG", "ESR"]

# PUMS reports NAICSP for anyone who worked in the last five years, but UrbanPop carries an
# industry only for the currently employed. Verified against the delivered feathers: pr_naics is
# non-blank exactly when ESR is 1 or 2 (civilian employed, at work or with a job but not at work),
# and where non-blank it equals NAICSP on 1787 of 1787 overlapping persons. Without this gate the
# bundle would give an industry -- and therefore a workplace and a work group -- to the retired,
# the unemployed and the under-16s, who make up over half the PUMS person records.
ESR_EMPLOYED = {1, 2}

# RAC1P: 1 White, 2 Black, 3/4/5 American Indian and/or Alaska Native, 6 Asian,
# 7 Native Hawaiian and Other Pacific Islander, 8 Some other race, 9 Two or more races.
RAC1P_TO_RACE = {1: 0, 2: 1, 3: 3, 4: 3, 5: 3, 6: 2, 7: 4, 8: 5, 9: 6}

# JWTRNS: 1 car/truck/van; 2-6 bus, streetcar, subway, railroad, ferry (all public transport);
# 7 taxicab; 8 motorcycle; 9 bicycle; 10 walked; 11 worked at home; 12 other.
JWTRNS_TO_TRAVEL = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 5, 8: 4, 9: 2, 10: 3, 11: 7, 12: 6}


def recode_pums(d):
    """Map raw PUMS person records onto the bundle's integer codings.

    Kept as an explicit table rather than inferred, because several of these are not the identity
    anyone would guess: RAC1P splits three codes onto one category, JWTRNS folds five transit modes
    into one, and JWRIP is a passenger COUNT that becomes a two-way drove-alone/carpooled flag.
    SCHG happens to be the identity on 1..16 only because `pr_grade`'s first category, childcare,
    has no PUMS code -- it is assigned later by `set_childcare`.
    """
    naics_lut = {c: i for i, c in enumerate(U.categ_types["pr_naics"].categories)}
    num = lambda c: pd.to_numeric(d[c], errors="coerce").fillna(0).astype(np.int64)

    sex = np.where(num("SEX").to_numpy() == 1, 1, 0)          # categories are [female, male]
    race = num("RAC1P").map(RAC1P_TO_RACE).fillna(-1).to_numpy()
    travel = num("JWTRNS").map(JWTRNS_TO_TRAVEL).fillna(-1).to_numpy()
    jwrip = num("JWRIP").to_numpy()
    veh = np.where(jwrip == 1, 0, np.where(jwrip >= 2, 1, -1))
    schg = num("SCHG").to_numpy()
    grade = np.where((schg >= 1) & (schg <= 16), schg, -1)
    naics = d["NAICSP"].astype(str).map(naics_lut).fillna(-1).to_numpy()
    employed = np.isin(num("ESR").to_numpy(), list(ESR_EMPLOYED))
    naics = np.where(employed, naics, -1)
    return {
        "age": np.clip(num("AGEP").to_numpy(), 0, 127).astype(np.int8),
        "sex": sex.astype(np.int8),
        "race": race.astype(np.int8),
        "travel": travel.astype(np.int8),
        "veh_occ": veh.astype(np.int8),
        "grade": grade.astype(np.int8),
        "naics": naics.astype(np.int16),
    }


def build_donors_pums(pumas, year, key, bw, meta, validate_feathers=None):
    """Donor attributes for EVERY PUMS household the P-MEDM solve can allocate.

    Recovering donors from the delivered feathers instead is tempting and much cheaper -- every
    synthetic person carries its donor's `pums_id` -- but it is wrong: the feathers only contain
    donors the delivered realization happened to use. Measured on NM, that covers 41,848 of the
    50,376 households `est_ind` offers, so **16.9% of allocation rows would have no attributes**,
    and a fresh realization that placed one of them would have nobody to put in the house. The
    whole point of the bundle is to draw allocations the delivered realization did not make.

    So this pulls the PUMS directly, which does require a Census API key.
    """
    # The index comes from the HOUSEHOLD file, not the person file. est_ind includes vacant
    # housing units, which have no person records at all: measured on NM PUMA 3500300, 1,391 of
    # its 4,539 donor rows are vacant (NP=0), and the household extract matches est_ind exactly at
    # 4,539 of 4,539 while the person extract covers only 3,148. Vacant units still carry
    # allocation weight -- they satisfy the housing-unit constraints in the `universe` theme -- so
    # they belong in the donor table as households of size zero, contributing nobody. Building the
    # index from the person file instead drops them and makes 30% of allocation rows look
    # unmatched.
    hframes, pframes = [], []
    for fips in pumas:
        hframes.append(acs_extract_household(fips, year, key))
        pframes.append(acs_extract(fips, year, key))
    h = pd.concat(hframes, ignore_index=True)
    h["SERIALNO"] = h["SERIALNO"].astype(str)
    h = h.drop_duplicates(subset=["SERIALNO"])

    d = pd.concat(pframes, ignore_index=True)
    d = d.drop_duplicates(subset=["SERIALNO", "SPORDER"]).copy()
    d["SERIALNO"] = d["SERIALNO"].astype(str)
    d = d.sort_values(["SERIALNO", "SPORDER"], kind="stable").reset_index(drop=True)

    hh_ids = sorted(h["SERIALNO"].tolist())
    hh_index = {hid: i for i, hid in enumerate(hh_ids)}
    # Keep only person records whose household is in the index, then order persons by household so
    # a household's people are one contiguous slice.
    d = d[d["SERIALNO"].isin(hh_index)].copy()
    d["hh"] = d["SERIALNO"].map(hh_index)
    d = d.sort_values(["hh", "SPORDER"], kind="stable").reset_index(drop=True)

    counts = np.bincount(d["hh"].to_numpy(), minlength=len(hh_ids)).astype(np.int64)
    offsets = np.zeros(len(hh_ids) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    n_vacant = int((counts == 0).sum())

    cols = recode_pums(d)
    bw.add("donors.hh_offset", offsets)
    bw.add("donors.hh_size", counts.astype(np.int16))
    for k, v in cols.items():
        bw.add(f"donors.{k}", v)
    bw.add_strings("donors.pums_id", hh_ids)

    meta["donors"] = {"households": len(hh_ids), "persons": len(d), "vacant": n_vacant,
                      "source": "PUMS"}
    print(f"  donors: {len(hh_ids)} households ({n_vacant} vacant), {len(d)} persons (from PUMS)")

    if validate_feathers:
        validate_donor_recode(d, cols, validate_feathers)
    return hh_index


def acs_extract(fips, year, key):
    from livelike import acs
    return acs.extract_pums_descriptors(
        fips, "person", PUMS_FEATURES, year=year, censusapikey=key)


def acs_extract_household(fips, year, key):
    """Household-level PUMS: the authoritative donor row set, vacant units included."""
    from livelike import acs
    return acs.extract_pums_descriptors(
        fips, "household", ["NP"], year=year, censusapikey=key)


def validate_donor_recode(d, cols, feathers):
    """Check the PUMS recode against the delivered feathers, person by person.

    The feathers are an independent rendering of the same donors, so where the two overlap every
    field must agree exactly. This is the only check that catches a wrong recode table -- a bundle
    built on a mis-mapped RAC1P or JWTRNS would still build, verify and generate, just with the
    wrong people in it.
    """
    ref = pl.concat([
        pl.read_ipc(p, columns=["pums_id", "p_id", "pr_age", "pr_sex", "pr_race", "pr_naics",
                                "pr_travel", "pr_veh_occ", "pr_grade"])
        for p in feathers])
    ref = ref.with_columns(
        pl.col("p_id").str.split("-").list.last().cast(pl.Int32).mod(100).alias("sporder")
    ).unique(subset=["pums_id", "sporder"])

    got = pl.DataFrame({
        "pums_id": d["SERIALNO"].astype(str).to_numpy(),
        "sporder": pd.to_numeric(d["SPORDER"]).to_numpy().astype(np.int32),
        **{k: v for k, v in cols.items()},
    })
    j = got.join(ref, on=["pums_id", "sporder"], how="inner")
    if j.is_empty():
        print("  VALIDATION: no overlap with the delivered feathers -- cannot check the recode")
        return

    naics_cats = list(U.categ_types["pr_naics"].categories)
    checks = [
        ("age", j["age"].cast(pl.Int64), j["pr_age"].cast(pl.Int64)),
        ("sex", j["sex"], j["pr_sex"].cast(pl.Utf8)
         .replace_strict({c: i for i, c in enumerate(SEX_CATS_B)}, default=-1)),
        ("race", j["race"], j["pr_race"].cast(pl.Utf8)
         .replace_strict({c: i for i, c in enumerate(RACE_CATS_B)}, default=-1)),
        ("travel", j["travel"], j["pr_travel"].cast(pl.Utf8)
         .replace_strict({c: i for i, c in enumerate(TRAVEL_CATS_B)}, default=-1)),
        ("veh_occ", j["veh_occ"], j["pr_veh_occ"].cast(pl.Utf8)
         .replace_strict({c: i for i, c in enumerate(VEH_OCC_CATS_B)}, default=-1)),
        ("grade", j["grade"], j["pr_grade"].cast(pl.Utf8)
         .replace_strict({c: i for i, c in enumerate(GRADE_CATS_B)}, default=-1)),
        ("naics", j["naics"], j["pr_naics"].cast(pl.Utf8)
         .replace_strict({c: i for i, c in enumerate(naics_cats)}, default=-1)),
    ]
    print(f"  VALIDATION against {len(j)} overlapping donor persons:")
    worst = 1.0
    for name, a, b in checks:
        agree = float((a.cast(pl.Int64) == b.cast(pl.Int64)).mean())
        worst = min(worst, agree)
        flag = "" if agree > 0.999 else "   <-- MISMATCH"
        print(f"    {name:9s} {agree:7.4f}{flag}")
    if worst <= 0.999:
        raise SystemExit("donor recode disagrees with the delivered feathers -- fix the mapping "
                         "tables before trusting this bundle")


SEX_CATS_B = ["female", "male"]
RACE_CATS_B = ["white", "blk_af_amer", "asian", "native_amer", "pac_island", "other", "mult"]
TRAVEL_CATS_B = ["car_truck_van", "public_transportation", "bicycle", "walked", "motorcycle",
                 "taxicab", "other", "wfh"]
VEH_OCC_CATS_B = ["drove_alone", "carpooled"]
GRADE_CATS_B = ["childcare", "preschl", "kind", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th",
                "8th", "9th", "10th", "11th", "12th", "undergrad", "grad"]


def build_cbp(wg_file, est_file, states, bw, meta):
    """CBP workgroup-size targets and establishment-size bands, restricted to the bundle's states.

    These are what steps 1-4 of the worker allocator draw establishment sizes from. They are pure
    reference data -- identical for every realization -- but they ship nationally, and a bundle
    covering one state has no use for another's rows.
    """
    U.check_size_tables_match(wg_file, est_file)
    wg = U.load_workgroup_targets(wg_file)
    est = parse_establishment_bands(est_file)
    naics_lut = {c: i for i, c in enumerate(U.categ_types["pr_naics"].categories)}

    keys = sorted(k for k in wg if k[1] in naics_lut and k[0] in states)
    bw.add("cbp.wg_state", np.array([k[0] for k in keys], dtype=np.int16))
    bw.add("cbp.wg_naics", np.array([naics_lut[k[1]] for k in keys], dtype=np.int16))
    bw.add("cbp.wg_size", np.array([wg[k] for k in keys], dtype=np.int32))

    ekeys = sorted(k for k in est if k[1] in naics_lut and k[0] in states)
    bw.add("cbp.est_state", np.array([k[0] for k in ekeys], dtype=np.int16))
    bw.add("cbp.est_naics", np.array([naics_lut[k[1]] for k in ekeys], dtype=np.int16))
    bw.add("cbp.est_bands",
           np.stack([est[k] for k in ekeys]) if ekeys else np.zeros((0, 18), np.int32))
    bw.add("cbp.est_band_lo", np.array([lo for lo, _ in EST_BANDS], dtype=np.int32))
    bw.add("cbp.est_band_hi", np.array([-1 if hi is None else hi for _, hi in EST_BANDS],
                                       dtype=np.int32))
    bw.add("cbp.default_workgroup_target", np.array([U.DEFAULT_WORKGROUP_TARGET], dtype=np.int32))

    meta["cbp"] = {"workgroup_rows": len(keys), "establishment_rows": len(ekeys),
                   "bands_per_row": len(EST_BANDS)}
    print(f"  CBP: {len(keys)} workgroup targets, {len(ekeys)} establishment rows "
          f"(states {sorted(states)})")


def build_schools(path, states, bw, meta):
    """School capacities and locations -- administrative, near-exact, and realization-independent.

    Restricted to the bundle's states. Students cross state lines far more rarely than workers do,
    and the commute-shed fallback is county adjacency, which carries its own out-of-state
    neighbours where they exist.
    """
    df = pl.read_csv(path, schema_overrides={"geoid": pl.Utf8, "id": pl.Utf8})
    df = df.filter(pl.col("geoid").str.slice(0, 2).cast(pl.Int32).is_in(sorted(states)))
    levels = sorted(set(df["level"].to_list()))
    lut = {v: i for i, v in enumerate(levels)}
    bw.add("schools.geoid", df["geoid"].cast(pl.Int64).to_numpy())
    bw.add("schools.students", df["students"].cast(pl.Int32).to_numpy())
    bw.add("schools.teachers", df["teachers"].cast(pl.Int32).to_numpy())
    bw.add("schools.level", df["level"].replace_strict(lut).to_numpy().astype(np.int8))
    bw.add_strings("schools.level_names", levels)
    bw.add_strings("schools.id", df["id"].to_list())
    meta["schools"] = {"count": len(df), "levels": levels,
                       "capacity": int(df["students"].sum())}
    print(f"  schools: {len(df)} with {int(df['students'].sum())} student places")


def build_adjacency(path, states, bw, meta):
    """County adjacency as CSR -- the fallback geography when a draw finds no local candidate.

    Kept for counties inside the bundle's states plus their immediate neighbours, including
    out-of-state ones: dropping those would silently close the border and push every edge county's
    fallback back inland.
    """
    df = pl.read_csv(path, schema_overrides={"geoid": pl.Utf8, "neighbor_geoid": pl.Utf8})
    instate = pl.col("geoid").str.slice(0, 2).cast(pl.Int32).is_in(sorted(states))
    df = df.filter(instate)
    counties = sorted(set(df["geoid"].to_list()) | set(df["neighbor_geoid"].to_list()))
    cid = {c: i for i, c in enumerate(counties)}
    idx = df.with_columns(
        pl.col("geoid").replace_strict(cid).alias("a"),
        pl.col("neighbor_geoid").replace_strict(cid).alias("b"),
    )
    m = sp.csr_matrix(
        (np.ones(len(idx), dtype=np.int8), (idx["a"].to_numpy(), idx["b"].to_numpy())),
        shape=(len(counties), len(counties)),
    )
    m.sort_indices()
    bw.add("adjacency.indptr", m.indptr.astype(np.int64))
    bw.add("adjacency.indices", m.indices.astype(np.int32))
    bw.add("adjacency.county", np.array([int(c) for c in counties], dtype=np.int32))
    meta["adjacency"] = {"counties": len(counties), "edges": int(m.nnz)}
    print(f"  adjacency: {len(counties)} counties, {m.nnz} edges")


# Only the constraints that calibrate a field ExaEpi actually reads. The P-MEDM Hessian is
# (dual x dual) with dual = n_constraints x (block groups + tracts + 1), so its memory falls with
# the SQUARE of the constraint count: 123 constraints against up_expanded's 298 is ~8.8 GB rather
# than ~42 GB. Measured on NM PUMA 3500804, the minimal set also fits the retained variables
# slightly BETTER (in_MOE 1.0000 vs 0.9996, median RAE 0.1421 vs 0.1466) -- constraints govern only
# how the population is calibrated in space, while the attribute values come from the matched PUMS
# donor, so dropping a theme coarsens nothing ExaEpi consumes.
EXAEPI_MINIMAL = {
    "universe": True,                    # population/housing guardrails
    # hhtype_hhsize is not optional: homesim.synthesize integerises at the household level and
    # errors out without household type by household size.
    "demographic": ["sex_age", "hhtype_hhsize"],
    "social": ["race"],                  # pr_race
    "worker": ["sexnaics"],              # pr_naics
    "student": ["grade"],                # pr_grade
    "mobility": ["travel", "veh_occ"],   # pr_travel, pr_veh_occ
}


def quantise_rows(a: np.ndarray):
    """Per-donor-row uint16, each row scaled by its own maximum.

    Scaling per row rather than globally gives a row 1/65535 of its OWN range, so a small donor's
    probabilities keep their relative precision next to a large one's. Validated against the thing
    that actually consumes these numbers: TRS only needs a cell's whole and fractional parts to
    survive, and quantising perturbs the resulting population 0.556x as much as simply re-drawing
    TRS with a different seed -- beneath noise the model is already known to ignore.
    """
    scale = a.max(axis=1, keepdims=True)
    scale = np.where(scale <= 0, 1.0, scale)
    q = np.rint(a / scale * 65535.0).astype(np.uint16)
    return q, scale.ravel().astype(np.float32)


def repair_controlled_se(se, est, label):
    """Give controlled ACS estimates a finite standard error.

    The ACS publishes no margin of error for a *controlled* estimate -- one benchmarked to the
    Census Bureau's own population estimates rather than sampled -- and livelike surfaces that as
    NaN. Tract-level total population is controlled, so most PUMAs carry one or two such cells.

    P-MEDM weights each soft constraint by 1/sigma^2, so a single NaN propagates through the whole
    objective: `f(lam)` is NaN at the initial parameters, LBFGS returns immediately, and
    `pmd.almat` comes back entirely NaN without anything raising. Measured on NM, one NaN of 3,321
    tract constraints destroyed a 440,283-cell allocation matrix, and three of eighteen PUMAs were
    silently lost this way.

    A controlled estimate has no sampling error at all, which would mean sigma -> 0 and an infinite
    weight, so it needs a small positive stand-in. Using the tightest standard error the same
    constraint actually exhibits elsewhere keeps it a near-hard constraint without inventing a
    scale that is not in the data.
    """
    se = np.asarray(se, dtype=np.float64).copy()
    bad = ~np.isfinite(se)
    if not bad.any():
        return se, 0
    for c in range(se.shape[1]):
        col = se[:, c]
        m = ~np.isfinite(col)
        if not m.any():
            continue
        good = col[np.isfinite(col) & (col > 0)]
        # Fall back on the whole matrix if this constraint is controlled everywhere, and on a
        # small fraction of the estimate only if nothing finite exists at all.
        if len(good):
            fill = good.min()
        else:
            allgood = se[np.isfinite(se) & (se > 0)]
            fill = allgood.min() if len(allgood) else max(1e-6, 1e-4 * np.abs(est[m, c]).max())
        col[m] = fill
        se[:, c] = col
    return se, int(bad.sum())


# A healthy allocation matrix spreads across donors: NM's PUMAs put ~0.1% of their households in
# their largest single cell. Anything above 1% means the exponential in compute_allocation has run
# away along an ill-conditioned direction.
DEGENERATE_CONCENTRATION = 0.01

# Ridge sizes tried in order, as a multiple of the Hessian's mean absolute diagonal. 0.0 is
# pymedm's own behaviour and is correct for most PUMAs; the rest are escalations for the ones
# whose Hessian is too ill-conditioned to invert safely.
RIDGE_LADDER = (0.0, 1e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0)


def draw_replicates(pmd, n_reps, seed, oversample=2, max_conc=DEGENERATE_CONCENTRATION):
    """Replicate allocation matrices, with a ridge on the Hessian and degenerate draws rejected.

    `pymedm.simulate_allocation_matrix` inverts the Hessian unregularised:

        inv_H = linalg.inv(H);  cov = inv_H / N;  lam ~ MVN(lam_hat, cov)

    and `compute_allocation` then exponentiates `X @ lam`. Where H is near-singular, inv_H has
    enormous eigenvalues along its near-null directions, lam picks up extreme components there,
    and the exponential sends a single (donor, block group) cell to the entire mass. Measured on
    New Mexico, 9 of 18 PUMAs produced at least one such replicate and 3500300 produced 20 of 20,
    at every random seed and under every controlled-standard-error fill strategy. Nothing about
    them looks wrong: they are finite, they sum to exactly the right household total, and a
    shape-only verify passes.

    Two changes fix it:

    * **Tikhonov ridge.** Invert (H + eps * mean|diag(H)| * I), which caps the covariance
      eigenvalues and corresponds to a weakly-informative prior on lam. eps escalates per PUMA
      only as far as needed, so PUMAs that were already fine keep pymedm's own answer -- verified
      on 3500804, where the ridge changes the spread by under 0.005.
    * **Rejection.** Draw `oversample` times as many as requested and keep the healthy ones.
      3500300 still yields ~30% bad draws at the top of the ladder, so the ridge alone is not
      enough for it.

    A large ridge does shrink the spread, so a PUMA rescued high on the ladder carries uncertainty
    that is partly a regularisation choice rather than a measured posterior. The chosen eps is
    recorded per PUMA in the bundle metadata so that is visible rather than buried.
    """
    from scipy import linalg as _linalg

    from pymedm.pmedm import compute_allocation, compute_hessian_matrix

    H0 = np.asarray(compute_hessian_matrix(pmd), dtype=np.float64)
    diag_scale = float(np.mean(np.abs(np.diag(H0))))
    want = n_reps * max(1, oversample)

    for eps in RIDGE_LADDER:
        H = H0 if eps == 0 else H0 + eps * diag_scale * np.eye(H0.shape[0])
        try:
            cov = _linalg.inv(H) / pmd.N
        except Exception as exc:                      # noqa: BLE001 - report and escalate
            print(f"    ridge {eps:g}: Hessian inversion failed ({exc})")
            continue
        rng = np.random.default_rng(seed)
        lam = rng.multivariate_normal(mean=np.asarray(pmd.lam), cov=cov, size=want,
                                      tol=1e-3, method="cholesky")
        good = []
        for lr in lam:
            m = np.asarray(compute_allocation(
                q=pmd.q, X=pmd.X, lam=lr, prob=True, counts=True, reshape=True,
                N=pmd.N, n_obs=pmd.n, n_geo=pmd.n_topo))
            tot = m.sum()
            if np.isfinite(m).all() and tot > 0 and m.max() / tot <= max_conc:
                good.append(m)
                if len(good) == n_reps:
                    return np.stack(good), eps, len(lam)
        print(f"    ridge {eps:g}: only {len(good)} of {want} draws healthy, escalating")

    raise SystemExit(
        f"could not obtain {n_reps} healthy replicates at any ridge in {RIDGE_LADDER}. "
        "The Hessian for this PUMA is too ill-conditioned for the Laplace approximation; "
        "reduce --n_reps, raise --oversample, or exclude it.")


def build_almats(pumas, n_reps, cache_folder, year, bw, meta, hh_index,
                 strict_replicates=True, oversample=2):
    """Solve P-MEDM per PUMA and store the allocation matrices, quantised.

    This is the section that carries "who lives where", and it is the reason the bundle exists:
    the 70-run NM ensemble showed residential variation dominates epidemic outcomes (ICC 0.82)
    while everything ExaEpi can already vary with --rseed is indistinguishable from disease-seed
    noise.

    On n_reps: one allocation matrix plus runtime TRS is NOT enough to vary anything. A TRS
    re-draw moves per-block-group population by a CV of 0.0044 against an ACS sampling CV of
    0.1604 -- 36x too small, and an ensemble built that way came out null. Replicate matrices
    sampled from the Laplace approximation at the converged dual reach CV 0.1425, which is ACS
    scale. So the bundle ships n_reps of them and the runtime picks one by seed.

    Storing the Laplace parameters themselves instead, for truly unlimited realizations, is not
    practical: that needs H^-1, which is (dual x dual) -- ~676 MB per PUMA at 123 constraints
    before any factorisation. Replicates are free once the Hessian exists (n_reps=20 costs the
    same as n_reps=2), so the cost here is per-PUMA and flat in n_reps.
    """
    from livelike import acs, config
    from pymedm import PMEDM

    key = os.environ.get("CENSUS_API_KEY") or None
    if not key:
        print("  WARNING: CENSUS_API_KEY unset -- the Census API answers keyless requests with an "
              "HTTP 200 HTML error page, so this will fail downstream in pd.read_json")

    degenerate = {}
    ridges = {}
    bg_all, don_all, rep_all = [], [], []
    puma_bg_off = [0]
    puma_don_off = [0]
    solved = []
    for i, fips in enumerate(pumas):
        t0 = time.time()
        pup = acs.puma(
            fips, constraints_selection=EXAEPI_MINIMAL,
            constraints_theme_order=config.up_constraints_theme_order,
            year=year, target_zone="bg", cache=True,
            cache_folder=cache_folder, censusapikey=key,
        )
        se_g1, n1 = repair_controlled_se(pup.se_g1, np.asarray(pup.est_g1, dtype=float), "tract")
        se_g2, n2 = repair_controlled_se(pup.se_g2, np.asarray(pup.est_g2, dtype=float), "bg")
        if n1 or n2:
            print(f"    repaired {n1} tract and {n2} block-group controlled standard errors")
            se_g1 = pd.DataFrame(se_g1, index=pup.se_g1.index, columns=pup.se_g1.columns)
            se_g2 = pd.DataFrame(se_g2, index=pup.se_g2.index, columns=pup.se_g2.columns)
        else:
            se_g1, se_g2 = pup.se_g1, pup.se_g2
        pmd = PMEDM(
            pup.year, pup.est_ind.index, pup.wt,
            pup.est_ind, pup.est_g1, pup.est_g2, se_g1, se_g2,
            # 0 always: replicates are drawn by draw_replicates below, which regularises the
            # Hessian and rejects degenerate draws. pymedm's own path does neither.
            n_reps=0, random_state=1,
        )
        pmd.solve()
        if n_reps > 1:
            mats, ridge, tried = draw_replicates(pmd, n_reps, seed=1, oversample=oversample)
            if ridge > 0:
                print(f"    ridge {ridge:g} needed ({tried} draws for {n_reps} healthy)")
            ridges[fips] = ridge
        else:
            mats = np.asarray(pmd.almat)[np.newaxis, ...]
        # pymedm returns an all-NaN allocation matrix rather than raising when the LBFGS solve
        # fails, so a whole PUMA can go missing from a bundle that otherwise builds and verifies
        # cleanly. Name the PUMA here rather than letting it surface as a NaN scale much later.
        if not np.isfinite(mats).all():
            nan_rows = int((~np.isfinite(mats[0])).any(axis=1).sum())
            raise SystemExit(
                f"PUMA {fips}: solve produced a non-finite allocation matrix "
                f"({nan_rows} of {mats.shape[1]} donor rows). The solve did not converge; "
                f"rerun this PUMA alone to see its LBFGS diagnostics.")

        # Replicates can come back DEGENERATE: finite, correctly summing to the right total, and
        # still useless because a single (donor, block group) cell holds nearly all of it. That
        # happens when the Hessian is near-singular, since `simulate_allocation_matrix` inverts it
        # without regularisation and `compute_allocation` then exponentiates the extreme lam
        # components it produces. Measured on NM: 9 of 18 PUMAs had at least one such replicate and
        # 3500300 had 20 of 20, at every random seed -- and none of it showed up in totals, in
        # finiteness, or in the shape-only verify.
        conc = np.array([m.max() / m.sum() for m in mats])
        bad = int((conc > DEGENERATE_CONCENTRATION).sum())
        if bad:
            worst = float(conc.max())
            msg = (f"PUMA {fips}: {bad} of {len(conc)} replicates are degenerate "
                   f"(largest cell holds up to {100 * worst:.1f}% of all allocated households; "
                   f"healthy is ~0.1%).")
            if strict_replicates:
                raise SystemExit(
                    msg + "\n  These would look fine in the bundle -- finite, right totals -- but "
                    "the population\n  drawn from them is nonsense. Rerun with "
                    "--allow_degenerate to write anyway.")
            print(f"    WARNING: {msg}")
            degenerate[fips] = {"bad": bad, "of": len(conc), "worst": round(worst, 4)}
        bgs = [int(g) for g in pup.est_g2.index]
        dons = list(pup.est_ind.index)
        # The allocation matrix holds occurrence weights, not counts: a cell becomes an expected
        # household count only after scaling by the PUMA's own population total. Store that
        # constant and the per-block-group population, or the runtime has weights it cannot turn
        # into people and would have to re-derive the scale it was never given.
        est_g2 = np.asarray(pup.est_g2, dtype=np.float64)
        # fillna(0), not 1: a donor row with no person records is a VACANT housing unit, and it
        # holds nobody. Filling with 1 would have every vacant unit contribute a phantom resident.
        hh_people = (pup.sporder.groupby(level=0).size()
                     .reindex(pup.est_ind.index).fillna(0).to_numpy())
        print(f"  PUMA {fips}: {mats.shape[1]} donors x {mats.shape[2]} bgs, "
              f"{mats.shape[0]} reps, {time.time() - t0:.1f} s")
        solved.append((fips, mats, bgs, dons, est_g2[:, 0], hh_people))

    totals, bgpop, dsize = [], [], []
    for fips, mats, bgs, dons, bg_pop, hh_people in solved:
        totals.append(float(bg_pop.sum()))
        bgpop.append(bg_pop)
        dsize.append(hh_people.astype(np.int16))
        q, scale = quantise_rows(mats.reshape(-1, mats.shape[2]).astype(np.float64))
        rep_all.append((q.reshape(mats.shape), scale.reshape(mats.shape[:2])))
        bg_all.append(np.array(bgs, dtype=np.int64))
        # Donor rows are PUMS household ids; map them onto the bundle's own donor index so the
        # runtime can go straight from an allocation cell to the donor household's persons.
        don_all.append(np.array([hh_index.get(str(d), -1) for d in dons], dtype=np.int32))
        puma_bg_off.append(puma_bg_off[-1] + len(bgs))
        puma_don_off.append(puma_don_off[-1] + len(dons))

    bw.add("almat.total_pop", np.array(totals, dtype=np.float64))
    bw.add("almat.bg_pop", np.concatenate(bgpop).astype(np.float64))
    bw.add("almat.donor_hh_size", np.concatenate(dsize))
    bw.add("almat.puma_bg_offset", np.array(puma_bg_off, dtype=np.int64))
    bw.add("almat.puma_donor_offset", np.array(puma_don_off, dtype=np.int64))
    bw.add("almat.bg_index", np.concatenate(bg_all))
    bw.add("almat.donor_index", np.concatenate(don_all))
    bw.add("almat.values", np.concatenate([q.reshape(q.shape[0], -1) for q, _ in rep_all], axis=1))
    bw.add("almat.scale", np.concatenate([s.reshape(s.shape[0], -1) for _, s in rep_all], axis=1))
    bw.add_strings("almat.puma", [str(p) for p in pumas])

    miss = int(sum((d < 0).sum() for d in don_all))
    cells = int(sum(q[0].size for q, _ in rep_all))
    meta["almat"] = {"pumas": len(pumas), "n_reps": n_reps, "cells_per_rep": cells,
                     "donors_unmatched": miss, "constraints": "EXAEPI_MINIMAL",
                     "degenerate_replicates": degenerate or None,
                     "hessian_ridge": {k: v for k, v in ridges.items() if v > 0} or None}
    print(f"  {len(pumas)} PUMAs, {cells} cells per replicate, {n_reps} replicates")
    if miss:
        print(f"  WARNING: {miss} donor rows did not match the bundle's donor index")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upop_files", required=True,
                    help="glob for the delivered UrbanPop feathers, e.g. 'base/35_NM/syp*.feather'")
    ap.add_argument("--lodes_files", required=True, nargs="+")
    ap.add_argument("--schools_file", required=True)
    ap.add_argument("--county_adjacency_file", required=True)
    ap.add_argument("--workgroup_sizes_file", required=True)
    ap.add_argument("--establishment_sizes_file", required=True)
    ap.add_argument("--out", required=True, help="output bundle path (.upb)")
    ap.add_argument("--verify", action="store_true",
                    help="read the bundle back and check every section round-trips")
    ap.add_argument("--pumas", nargs="*", default=None,
                    help="PUMA FIPS to solve P-MEDM for; omit to skip the allocation matrices")
    ap.add_argument("--n_reps", type=int, default=1,
                    help="replicate allocation matrices per PUMA. 1 gives a single deterministic "
                         "matrix, whose TRS re-draws are 36x below ACS sampling scale and produce "
                         "a null ensemble -- use >= 10 for residential variation that matters")
    ap.add_argument("--acs_year", type=int, default=2019)
    ap.add_argument("--oversample", type=int, default=2,
                    help="draw this many times --n_reps per PUMA and keep the healthy ones; "
                         "some PUMAs yield degenerate draws even with a regularised Hessian")
    ap.add_argument("--allow_degenerate", action="store_true",
                    help="write the bundle even if some replicate allocation matrices collapse "
                         "into a single cell. Off by default: such a bundle is finite, has "
                         "correct totals, verifies cleanly, and produces a nonsense population")
    ap.add_argument("--validate_donors", action="store_true",
                    help="check the PUMS recode field by field against the delivered feathers, "
                         "and refuse to write a bundle whose donors disagree with them")
    ap.add_argument("--cache_folder", default="./livelike_cache_minimal",
                    help="livelike's ACS cache. Its cache key ignores constraints_selection, so "
                         "this MUST NOT be shared with a cache built for a different selection")
    args = ap.parse_args()

    feathers = sorted(glob.glob(args.upop_files))
    if not feathers:
        sys.exit(f"no feathers matched {args.upop_files}")

    t0 = time.time()
    bw = BundleWriter()
    meta = {"format_version": PRECOMPUTE_FORMAT_VERSION,
            "generator": "build_precompute.py",
            "bin_format_version": U.BIN_FORMAT_VERSION}

    U.printgreen("Donors")
    key = os.environ.get("CENSUS_API_KEY") or None
    if args.pumas:
        if not key:
            sys.exit("CENSUS_API_KEY is required to build donor attributes from PUMS. The feather "
                     "shortcut covers only the donors the delivered realization used (41,848 of "
                     "50,376 for NM), which is not enough for a bundle meant to draw new ones.")
        hh_index = build_donors_pums(args.pumas, args.acs_year, key, bw, meta,
                                     validate_feathers=feathers if args.validate_donors else None)
    else:
        hh_index = build_donors(feathers, bw, meta)

    U.printgreen("Block groups")
    geoids = sorted(set().union(*[
        set(pl.read_ipc(p, columns=["geoid"])["geoid"].to_list()) for p in feathers]))
    bw.add("bg.geoid", np.array([int(g) for g in geoids], dtype=np.int64))
    # The states the bundle actually covers, taken from the population rather than a flag, so the
    # national reference tables below are trimmed to what this bundle can possibly use.
    states = sorted({int(g[:2]) for g in geoids})
    meta["block_groups"] = len(geoids)
    meta["states"] = states
    print(f"  {len(geoids)} block groups in states {states}")

    U.printgreen("LODES")
    build_lodes(args.lodes_files, set(geoids), bw, meta)

    U.printgreen("CBP size tables")
    build_cbp(args.workgroup_sizes_file, args.establishment_sizes_file, set(states), bw, meta)

    U.printgreen("Schools")
    build_schools(args.schools_file, set(states), bw, meta)

    U.printgreen("County adjacency")
    build_adjacency(args.county_adjacency_file, set(states), bw, meta)

    if args.pumas:
        U.printgreen(f"P-MEDM allocation matrices ({len(args.pumas)} PUMAs, "
                     f"n_reps={args.n_reps})")
        build_almats(args.pumas, args.n_reps, args.cache_folder, args.acs_year,
                     bw, meta, hh_index, strict_replicates=not args.allow_degenerate,
                     oversample=args.oversample)
    else:
        print("\nNOTE: --pumas not given, so the bundle carries no allocation matrices. Without "
              "them\n      it cannot place anyone: the donor table, flows and size tables are all "
              "here,\n      but 'who lives where' is exactly the missing section.")
        meta["almat"] = None

    bw.add_strings("naics.codes", list(U.categ_types["pr_naics"].categories))
    bw.add_json("meta", meta)

    U.printgreen(f"Writing {args.out}")
    stats = bw.write(args.out)
    total_raw = sum(r for r, _ in stats.values())
    total_st = sum(s for _, s in stats.values())
    size = os.path.getsize(args.out)

    print(f"\n{'section':32s}{'raw':>12s}{'stored':>12s}{'ratio':>8s}")
    for name in sorted(stats, key=lambda n: -stats[n][1]):
        raw, st = stats[name]
        if st < 1024:
            continue
        print(f"{name:32s}{raw / 1e6:11.3f}M{st / 1e6:11.3f}M{raw / max(st, 1):8.2f}")
    print(f"{'TOTAL':32s}{total_raw / 1e6:11.3f}M{total_st / 1e6:11.3f}M"
          f"{total_raw / max(total_st, 1):8.2f}")
    print(f"\nbundle {size / 1e6:.2f} MB written in {time.time() - t0:.1f} s")

    if args.verify:
        U.printgreen("Verifying round trip")
        got = read_bundle(args.out)
        missing = set(stats) - set(got)
        if missing:
            sys.exit(f"sections missing on read back: {sorted(missing)}")
        print(f"  {len(got)} sections read back, all shapes and byte counts consistent")
        print(f"  meta: {json.loads(got['meta'].tobytes().decode())['block_groups']} block groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
