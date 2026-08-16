#!/usr/bin/env python

# Generates the optional density side file consumed by ExaEpi's DensityData (see
# src/DensityData.H), which scales xmit_comm/xmit_hood by local population density. Extracts
# GEOID10 + ALAND10 (land area) for one state from the nationwide Census TIGER/Line block-group
# CSV (data/US_2010_Census_BlockGroups/census_bgs.csv), converts ALAND10 from m^2 to km^2, and
# writes:
#   <num_records>
#   <geoid_1> <area_km2_1>
#   <geoid_2> <area_km2_2>
#   ...
#
# The nationwide CSV is ~1.76GB, almost entirely a `geometry` WKT column this script doesn't need.
# Reading only the needed columns in chunks avoids ever materializing that column or the full file.

import argparse

import pandas as pd

SQ_M_PER_SQ_KM = 1_000_000.0
CHUNK_SIZE = 200_000


def extract_state_areas(input_csv, state_fips):
    usecols = ["STATEFP10", "GEOID10", "ALAND10"]
    dtype = {"STATEFP10": str, "GEOID10": "int64", "ALAND10": "int64"}

    chunks = []
    for chunk in pd.read_csv(input_csv, usecols=usecols, dtype=dtype, chunksize=CHUNK_SIZE):
        matched = chunk.loc[chunk["STATEFP10"] == state_fips]
        if len(matched):
            chunks.append(matched)

    if not chunks:
        raise RuntimeError(f"No records found for STATEFP10='{state_fips}' in {input_csv}")

    df = pd.concat(chunks, ignore_index=True)
    df["area_km2"] = df["ALAND10"] / SQ_M_PER_SQ_KM
    return df[["GEOID10", "area_km2"]]


def write_density_file(df, output_file):
    with open(output_file, "w") as f:
        f.write(f"{len(df)}\n")
        for geoid, area_km2 in zip(df["GEOID10"], df["area_km2"]):
            f.write(f"{geoid} {area_km2:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate an ExaEpi density side file (GEOID -> land area km^2) for one state "
        "from the nationwide Census TIGER/Line block-group CSV."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the nationwide block-group CSV (e.g. data/US_2010_Census_BlockGroups/census_bgs.csv)",
    )
    parser.add_argument(
        "--state-fips",
        "-s",
        required=True,
        help="2-digit state FIPS code, e.g. 35 for New Mexico (matches STATEFP10)",
    )
    parser.add_argument("--output", "-o", required=True, help="Output density side file")
    args = parser.parse_args()

    df = extract_state_areas(args.input, args.state_fips.zfill(2))
    write_density_file(df, args.output)
    print(f"Wrote {len(df)} geoid/area records to {args.output}")


if __name__ == "__main__":
    main()
