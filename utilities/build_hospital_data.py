#!/usr/bin/env python3
"""Build ExaEpi hospital-capacity data from public sources.

This script produces a small whitespace-delimited ``.dat`` file giving the
acute-care hospital bed supply per U.S. county (FIPS), optionally per census
tract, for use by ExaEpi's medical-workers / hospital model
(``hospital_model.use_HIFLD_HHS_data = true``). The bed supply is tied to a
particular year so the dataset can be regenerated periodically.

Data sources (public)
---------------------
* HHS "COVID-19 Reported Patient Impact and Hospital Capacity by Facility"
  (HealthData.gov, dataset id ``anag-cw7u``) -- the default and recommended source.
  Facility-level **staffed** and **ICU** bed counts reported weekly (2019-2024,
  year-tied), plus a ``geocoded_hospital_address`` point, so it supplies both the
  bed counts and the hospital locations (no separate location source needed).
  Reliable Socrata API, no account. Portal:
  https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u
  (For dates after 2024 the successor is CDC NHSN hospital respiratory data.)
* HIFLD "Hospitals" -- legacy alternative (``--source hifld``), hospital point
  locations + licensed beds. **HIFLD Open is being deprecated** (its layers are
  moving to the account-gated HIFLD Secure / DHS GII), so this path now expects a
  locally supplied CSV (``--hifld-csv``) rather than a download.

We keep only facilities that can admit infectious-disease inpatients. For HHS,
every reporting facility with staffed beds is kept; for HIFLD, ``STATUS == OPEN``
and ``TYPE`` in the acute-care set (general acute care, critical access).

Output format
-------------
A ``.dat`` file with ``#``-comment provenance lines, then a record count, then
one row per geography::

    # provenance ...
    <N>
    FIPS  beds  icu_beds  n_hospitals               (county level)
    FIPS  TRACT  beds  icu_beds  n_hospitals         (tract level)

``FIPS`` is the integer county FIPS (e.g. 6001 for Alameda County, CA), matching
the integer FIPS used in ExaEpi's demographic ``.dat`` files. ``icu_beds`` is 0
when HHS data is not supplied.

Network access
--------------
Endpoint URLs change over time. The script prefers local files
(``--hifld-csv`` / ``--hhs-csv`` / ``--tract-shapefile``); pass ``--download``
to fetch from the documented endpoints (verify they still resolve). Tract-level
output additionally requires ``geopandas``.

Examples
--------
    # county level from a downloaded HIFLD CSV, 2023 vintage
    ./build_hospital_data.py --state CA --year 2023 \\
        --hifld-csv Hospitals.csv --out ../data/HospitalData/CA_hospitals_2023.dat

    # add HHS ICU/staffed beds for a chosen reporting week
    ./build_hospital_data.py --state CA --year 2021 \\
        --hifld-csv Hospitals.csv --hhs-csv hhs_facility.csv \\
        --hhs-week 2021-01-08 --out ../data/HospitalData/CA_hospitals_2021.dat
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# State FIPS lookup (postal code <-> 2-digit state FIPS)
# ---------------------------------------------------------------------------
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}

# Acute-care facility types in HIFLD that admit infectious-disease inpatients.
DEFAULT_ACUTE_TYPES = ["GENERAL ACUTE CARE", "CRITICAL ACCESS"]

# Documented (verify-before-use) download endpoints.
HIFLD_HUB_ID = "6ac5e325468c4cb9b905f1728d6fbf0f"
HIFLD_DOWNLOAD_URL = (
    f"https://hub.arcgis.com/api/v3/datasets/{HIFLD_HUB_ID}_0/downloads/data"
    "?format=csv&spatialRefId=4326"
)
HHS_SODA_URL = "https://healthdata.gov/resource/anag-cw7u.csv"

# Candidate HHS column names (the dataset has changed names over time).
HHS_ICU_COLS = [
    "total_staffed_adult_icu_beds_7_day_avg",
    "total_staffed_adult_icu_beds",
]
HHS_STAFFED_COLS = [
    "inpatient_beds_7_day_avg",
    "all_adult_hospital_inpatient_beds_7_day_avg",
    "total_beds_7_day_avg",
]


def _norm(colnames):
    """Map upper/lower-case column names to the actual dataframe columns."""
    return {c.upper(): c for c in colnames}


def state_fips(state):
    s = state.strip().upper()
    if s == "US":
        return ""  # all states (no state filter)
    if s in STATE_FIPS:
        return STATE_FIPS[s]
    if s.isdigit() and len(s) <= 2:
        return s.zfill(2)
    raise SystemExit(f"Unrecognized --state '{state}' (use a postal code, 2-digit FIPS, or US)")


def load_hifld(args, sfips):
    """Load and filter HIFLD hospitals for the requested state."""
    if args.hifld_csv:
        df = pd.read_csv(args.hifld_csv, dtype=str)
    elif args.download:
        print(f"Downloading HIFLD hospitals from {HIFLD_DOWNLOAD_URL}", file=sys.stderr)
        df = pd.read_csv(HIFLD_DOWNLOAD_URL, dtype=str)
    else:
        raise SystemExit("Provide --hifld-csv PATH (or --download). Get the CSV from "
                         "https://hifld-geoplatform.opendata.arcgis.com/datasets/hospitals")

    cols = _norm(df.columns)
    need = ["TYPE", "STATUS", "COUNTYFIPS", "BEDS", "LATITUDE", "LONGITUDE"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise SystemExit(f"HIFLD CSV is missing expected columns: {missing}")

    df = df.rename(columns={cols[c]: c for c in cols})  # normalize to upper-case
    df["BEDS"] = pd.to_numeric(df["BEDS"], errors="coerce")
    df["COUNTYFIPS"] = df["COUNTYFIPS"].astype(str).str.strip().str.zfill(5)
    df["TYPE"] = df["TYPE"].astype(str).str.strip().str.upper()
    df["STATUS"] = df["STATUS"].astype(str).str.strip().str.upper()

    acute = [t.upper() for t in args.types]
    keep = (
        df["COUNTYFIPS"].str.startswith(sfips)  # "" matches all (US)
        & (df["STATUS"] == "OPEN")
        & df["TYPE"].isin(acute)
        & (df["BEDS"] > 0)
    )
    if args.counties:
        keep &= df["COUNTYFIPS"].isin({c.zfill(5) for c in args.counties})
    df = df[keep].copy()
    if df.empty:
        raise SystemExit("No HIFLD facilities matched the state/type/status filters.")
    print(f"HIFLD: {len(df)} acute-care hospitals, {int(df['BEDS'].sum())} beds", file=sys.stderr)
    return df


def load_hhs_by_county(args, sfips, required):
    """HHS facility-level staffed and ICU beds aggregated per 5-digit county FIPS.

    Returns a DataFrame indexed by FIPS with columns staffed_beds, icu_beds, or
    None when HHS data is neither requested nor required. ``--hhs-week`` selects a
    reporting week (Sunday-dated, e.g. 2021-12-26); without it, all weeks are
    averaged per county."""
    if not args.hhs_csv and not args.download and not required:
        return None
    if args.hhs_csv:
        hhs = pd.read_csv(args.hhs_csv, dtype=str)
    else:
        sel = "fips_code," + ",".join(HHS_STAFFED_COLS[:1] + HHS_ICU_COLS[:1] + ["collection_week"])
        url = HHS_SODA_URL + f"?$select={sel}&$limit=2000000"
        if args.state.upper() != "US":
            url += f"&state={args.state.upper()}"
        if args.hhs_week:
            url += f"&collection_week={args.hhs_week}T00:00:00.000"
        print(f"Downloading HHS facility capacity from {url}", file=sys.stderr)
        hhs = pd.read_csv(url, dtype=str)

    cols = _norm(hhs.columns)
    if "FIPS_CODE" not in cols:
        raise SystemExit("HHS CSV missing fips_code column.")
    icu_col = next((cols[c.upper()] for c in HHS_ICU_COLS if c.upper() in cols), None)
    staffed_col = next((cols[c.upper()] for c in HHS_STAFFED_COLS if c.upper() in cols), None)
    if icu_col is None or staffed_col is None:
        raise SystemExit("HHS CSV is missing expected bed columns; check the data dictionary "
                         f"and update HHS_STAFFED_COLS/HHS_ICU_COLS. Found: {sorted(cols)[:20]}")

    week_col = cols.get("COLLECTION_WEEK")
    if args.hhs_week and week_col:
        hhs = hhs[hhs[week_col].astype(str).str.startswith(args.hhs_week)]
    hhs = hhs.rename(columns={cols["FIPS_CODE"]: "FIPS", icu_col: "icu_beds", staffed_col: "staffed_beds"})
    hhs["FIPS"] = pd.to_numeric(hhs["FIPS"], errors="coerce")
    hhs = hhs.dropna(subset=["FIPS"])  # drop facilities with missing county FIPS
    hhs["FIPS"] = hhs["FIPS"].astype(int).astype(str).str.zfill(5)
    for c in ("icu_beds", "staffed_beds"):
        hhs[c] = pd.to_numeric(hhs[c], errors="coerce")
        hhs.loc[hhs[c] < 0, c] = pd.NA  # -999999 = suppressed small counts
    hhs = hhs[hhs["FIPS"].str.startswith(sfips)]
    if args.counties:
        hhs = hhs[hhs["FIPS"].isin({c.zfill(5) for c in args.counties})]
    out = hhs.groupby("FIPS").agg(
        staffed_beds=("staffed_beds", "sum"), icu_beds=("icu_beds", "sum"), n=("staffed_beds", "size")
    )
    print(f"HHS: staffed/ICU beds for {len(out)} counties", file=sys.stderr)
    return out


def load_hhs_facilities(args, sfips):
    """HHS facility-level records with geocoded location and staffed/ICU beds, for a state/week.
    Returns a DataFrame with COUNTYFIPS, LATITUDE, LONGITUDE, BEDS (staffed inpatient), ICU. Uses
    the ``geocoded_hospital_address`` WKT point, so no separate hospital-location source (HIFLD) is
    needed for tract-level placement."""
    if args.hhs_csv:
        hhs = pd.read_csv(args.hhs_csv, dtype=str)
    else:
        sel = "fips_code,geocoded_hospital_address,collection_week," + ",".join(HHS_STAFFED_COLS[:1] + HHS_ICU_COLS[:1])
        url = HHS_SODA_URL + f"?$select={sel}&$limit=2000000"
        if args.state.upper() != "US":
            url += f"&state={args.state.upper()}"
        if args.hhs_week:
            url += f"&collection_week={args.hhs_week}T00:00:00.000"
        print(f"Downloading HHS facilities from {url}", file=sys.stderr)
        hhs = pd.read_csv(url, dtype=str)

    cols = _norm(hhs.columns)
    icu_col = next((cols[c.upper()] for c in HHS_ICU_COLS if c.upper() in cols), None)
    staffed_col = next((cols[c.upper()] for c in HHS_STAFFED_COLS if c.upper() in cols), None)
    geo_col = cols.get("GEOCODED_HOSPITAL_ADDRESS")
    if not (icu_col and staffed_col and "FIPS_CODE" in cols and geo_col):
        raise SystemExit("HHS CSV missing fips_code / geocoded_hospital_address / bed columns.")
    if args.hhs_week and "COLLECTION_WEEK" in cols:
        hhs = hhs[hhs[cols["COLLECTION_WEEK"]].astype(str).str.startswith(args.hhs_week)]

    df = pd.DataFrame()
    df["COUNTYFIPS"] = pd.to_numeric(hhs[cols["FIPS_CODE"]], errors="coerce")
    df["BEDS"] = pd.to_numeric(hhs[staffed_col], errors="coerce")
    df["ICU"] = pd.to_numeric(hhs[icu_col], errors="coerce")
    pt = hhs[geo_col].astype(str).str.extract(r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)")
    df["LONGITUDE"] = pd.to_numeric(pt[0], errors="coerce")
    df["LATITUDE"] = pd.to_numeric(pt[1], errors="coerce")
    df.loc[df["BEDS"] < 0, "BEDS"] = pd.NA  # -999999 = suppressed
    df.loc[df["ICU"] < 0, "ICU"] = 0
    df = df.dropna(subset=["COUNTYFIPS", "BEDS", "LONGITUDE", "LATITUDE"])
    df = df[df["BEDS"] > 0].copy()
    df["COUNTYFIPS"] = df["COUNTYFIPS"].astype(int).astype(str).str.zfill(5)
    df = df[df["COUNTYFIPS"].str.startswith(sfips)]
    if args.counties:
        df = df[df["COUNTYFIPS"].isin({c.zfill(5) for c in args.counties})]
    if df.empty:
        raise SystemExit("No HHS facilities matched the state/county/week filters.")
    print(f"HHS: {len(df)} facilities, {int(df['BEDS'].sum())} staffed beds", file=sys.stderr)
    return df


def build_tract_records(fac, shapefile, sfips, args):
    """Spatial-join hospital points to TIGER census tracts, then for *every* state tract emit its
    bed supply and the nearest tract that has a hospital (for patient routing, the analog of the
    home->work assignment). ``fac`` is a facilities DataFrame with LATITUDE, LONGITUDE, BEDS and
    optionally ICU columns (from HHS geocoded points, or HIFLD). Returns records (FIPS, TRACT, beds,
    icu_beds, n_hospitals, hosp_FIPS, hosp_TRACT). The census Tract code is int(TRACTCE)
    (e.g. 400 for '000400')."""
    try:
        import numpy as np
        import geopandas as gpd
        from shapely.geometry import Point
        from scipy.spatial import cKDTree
    except ImportError:
        raise SystemExit("--level tract requires geopandas, shapely and scipy "
                         "(pip install -r requirements-hospital-data.txt).")

    tracts = gpd.read_file(shapefile)
    tc = _norm(tracts.columns)
    f_tract = tc.get("TRACTCE") or tc.get("TRACTCE20")
    f_county = tc.get("COUNTYFP") or tc.get("COUNTYFP20")
    f_state = tc.get("STATEFP") or tc.get("STATEFP20")
    if not (f_tract and f_county and f_state):
        raise SystemExit("Tract shapefile missing STATEFP/COUNTYFP/TRACTCE fields.")

    tracts[f_state] = tracts[f_state].astype(str).str.zfill(2)
    if sfips:
        tracts = tracts[tracts[f_state] == sfips].copy()
    tracts["FIPS"] = (tracts[f_state] + tracts[f_county].astype(str).str.zfill(3)).astype(int)
    tracts["TRACT"] = pd.to_numeric(tracts[f_tract], errors="coerce").astype("Int64")
    tracts = tracts.dropna(subset=["TRACT"])
    if args.counties:
        keep = {int(c) for c in args.counties}
        tracts = tracts[tracts["FIPS"].isin(keep)].copy()
    if tracts.empty:
        raise SystemExit("No tracts after the state/county filters.")

    # spatial-join hospital points to the tract polygons; sum beds (and ICU) per tract
    fac = fac.copy()
    if "ICU" not in fac.columns:
        fac["ICU"] = 0
    fac["LATITUDE"] = pd.to_numeric(fac["LATITUDE"], errors="coerce")
    fac["LONGITUDE"] = pd.to_numeric(fac["LONGITUDE"], errors="coerce")
    fac = fac.dropna(subset=["LATITUDE", "LONGITUDE"])
    pts = gpd.GeoDataFrame(
        fac, geometry=[Point(xy) for xy in zip(fac["LONGITUDE"], fac["LATITUDE"])], crs="EPSG:4326"
    ).to_crs(tracts.crs)
    joined = gpd.sjoin(pts, tracts[["FIPS", "TRACT", "geometry"]], how="inner", predicate="within")
    bed = joined.groupby(["FIPS", "TRACT"]).agg(beds=("BEDS", "sum"), icu=("ICU", "sum"), n=("BEDS", "size"))
    bedset = {k: (int(round(v.beds)), int(round(v.icu)), int(v.n)) for k, v in bed.iterrows()}

    # metric centroids (CONUS Albers) for nearest-hospital search (avoid leading-underscore
    # column names, which itertuples() would mangle)
    cen = tracts.to_crs(epsg=5070).geometry.centroid
    tracts = tracts.assign(ctr_x=cen.x.to_numpy(), ctr_y=cen.y.to_numpy())
    key_xy = {(int(r.FIPS), int(r.TRACT)): (r.ctr_x, r.ctr_y) for r in tracts.itertuples()}

    hosp_keys = [k for k, v in bedset.items() if v[0] > 0 and k in key_xy]
    if not hosp_keys:
        raise SystemExit("No hospital tracts after the spatial join.")
    tree = cKDTree(np.array([key_xy[k] for k in hosp_keys]))

    records = []
    for r in tracts.itertuples():
        key = (int(r.FIPS), int(r.TRACT))
        beds_i, icu_i, n_i = bedset.get(key, (0, 0, 0))
        if beds_i > 0:
            hF, hT = key  # has its own hospital
        else:
            hF, hT = hosp_keys[int(tree.query([r.ctr_x, r.ctr_y])[1])]  # nearest hospital tract
            icu_i = 0
        records.append((int(r.FIPS), int(r.TRACT), beds_i, icu_i, n_i, hF, hT))
    records.sort()
    return records


def write_dat(path, records, level, args):
    """Write the ExaEpi hospital-capacity .dat file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# ExaEpi hospital-capacity data -- utilities/build_hospital_data.py\n")
        if args.source == "hhs":
            f.write("# source: HHS COVID-19 Hospital Capacity by Facility (anag-cw7u): "
                    "staffed inpatient beds (beds) + staffed adult ICU beds (icu_beds)\n")
        else:
            f.write("# source: HIFLD Hospitals (acute care: %s; STATUS=OPEN), licensed beds\n" % ", ".join(args.types))
            if args.hhs_csv or args.download:
                f.write("# icu_beds: HHS COVID-19 Hospital Capacity by Facility (anag-cw7u)\n")
        wk = f" week={args.hhs_week}" if args.hhs_week else ""
        f.write(f"# state={args.state.upper()} year={args.year} level={level}{wk}\n")
        if level == "county":
            f.write("# columns: FIPS beds icu_beds n_hospitals\n")
        else:
            f.write("# columns: FIPS TRACT beds icu_beds n_hospitals hosp_FIPS hosp_TRACT\n")
        f.write(f"{len(records)}\n")
        for r in records:
            f.write(" ".join(str(x) for x in r) + "\n")
    print(f"Wrote {len(records)} records to {path}", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True, help="State postal code (e.g. CA) or 2-digit FIPS")
    p.add_argument("--year", required=True, help="Dataset vintage year (tags the output)")
    p.add_argument("--level", choices=["county", "tract"], default="county")
    p.add_argument("--hifld-csv", help="Local HIFLD Hospitals CSV (preferred)")
    p.add_argument("--hhs-csv", help="Local HHS facility-capacity CSV (optional, for ICU/staffed beds)")
    p.add_argument("--hhs-week", help="HHS collection_week prefix to select, e.g. 2021-01-08")
    p.add_argument("--tract-shapefile", help="TIGER tract .shp for --level tract")
    p.add_argument("--source", choices=["hhs", "hifld"], default="hhs",
                   help="Bed source: hhs (geocoded staffed/ICU beds, year-tied, reliable; default) "
                        "or hifld (legacy local CSV -- HIFLD Open is being deprecated)")
    p.add_argument("--counties", nargs="+", help="Restrict to these 5-digit county FIPS (e.g. a metro area)")
    p.add_argument("--types", nargs="+", default=DEFAULT_ACUTE_TYPES, help="HIFLD TYPE values to keep")
    p.add_argument("--download", action="store_true", help="Fetch from documented endpoints (verify URLs)")
    p.add_argument("--out", required=True, help="Output .dat path")
    args = p.parse_args(argv)

    sfips = state_fips(args.state)

    def county_int(fips5):
        return int(fips5)  # 5-digit county FIPS -> int (drops leading zero), matches ExaEpi

    # Tract level: place hospitals from geocoded facility points (HHS by default, or HIFLD).
    if args.level == "tract":
        if not args.tract_shapefile:
            raise SystemExit("--level tract requires --tract-shapefile PATH (TIGER tracts).")
        fac = load_hhs_facilities(args, sfips) if args.source == "hhs" else load_hifld(args, sfips)
        records = build_tract_records(fac, args.tract_shapefile, sfips, args)
        write_dat(args.out, records, "tract", args)
        return

    # County level.
    if args.source == "hhs":
        hhs = load_hhs_by_county(args, sfips, required=True)
        records = []
        for fips5, row in hhs.iterrows():
            beds = int(round(row["staffed_beds"])) if pd.notna(row["staffed_beds"]) else 0
            icu = int(round(row["icu_beds"])) if pd.notna(row["icu_beds"]) else 0
            if beds > 0:
                records.append((county_int(fips5), beds, icu, int(row["n"])))
    else:
        hifld = load_hifld(args, sfips)
        hhs = load_hhs_by_county(args, sfips, required=False)
        g = hifld.groupby("COUNTYFIPS").agg(beds=("BEDS", "sum"), n=("BEDS", "size"))
        records = []
        for fips5, row in g.iterrows():
            icu_beds = 0
            if hhs is not None and fips5 in hhs.index and pd.notna(hhs.loc[fips5, "icu_beds"]):
                icu_beds = int(round(hhs.loc[fips5, "icu_beds"]))
            records.append((county_int(fips5), int(round(row["beds"])), icu_beds, int(row["n"])))
    records.sort()
    write_dat(args.out, records, "county", args)


if __name__ == "__main__":
    main()
