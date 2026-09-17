#!/usr/bin/env -S python -u

"""
Extract the small per-day summary that compare_to_epicast.py's plots actually use from one or
more Epicast run.events.bin files.

An events.bin file has one row per agent state-transition event -- tens to hundreds of millions
of rows for a full county/state run -- but the comparison plots only ever need a handful of
per-day aggregate columns (see read_epicast_events.read_epicast_summary). Extracting those once
with this script, instead of having compare_to_epicast.py re-parse the full binary on every plot,
turns a multi-GB, tens-of-seconds-per-file parse into a tiny CSV that loads instantly.

Usage:
    extract_epicast_data.py FILE_OR_GLOB [FILE_OR_GLOB ...]

Each input.bin is written next to itself as input.bin.summary.csv. Pass that file to
compare_to_epicast.py's -e/--epicast_file flag directly; it's recognized by that suffix and read
as-is instead of triggering a raw binary parse.
"""

import sys
import os
import glob
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from read_epicast_events import read_epicast_summary, EPICAST_SUMMARY_SUFFIX


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract the per-day summary compare_to_epicast.py needs from Epicast "
            f"run.events.bin file(s), writing one '<file>{EPICAST_SUMMARY_SUFFIX}' per input."
        ),
    )
    parser.add_argument(
        "files", nargs="+", metavar="FILE_OR_GLOB",
        help="Epicast .events.bin file(s), or glob pattern(s) matching several",
    )
    args = parser.parse_args()

    fnames = []
    for pattern in args.files:
        has_wildcard = any(c in pattern for c in ("*", "?", "["))
        matched = sorted(glob.glob(pattern)) if has_wildcard else [pattern]
        if not matched:
            print(f"Warning: no files matched '{pattern}'", file=sys.stderr)
            continue
        fnames.extend(matched)

    if not fnames:
        sys.exit("No input files found.")

    for fname in fnames:
        if fname.endswith(EPICAST_SUMMARY_SUFFIX):
            print(f"Skipping {fname}: already an extracted summary", file=sys.stderr)
            continue
        out_path = fname + EPICAST_SUMMARY_SUFFIX
        df = read_epicast_summary(fname)
        df.to_csv(out_path, index=False)
        in_size = os.path.getsize(fname)
        out_size = os.path.getsize(out_path)
        print(f"Wrote {out_path}: {len(df)} days, {out_size:,} bytes "
              f"(source file was {in_size:,} bytes)")


if __name__ == "__main__":
    main()
