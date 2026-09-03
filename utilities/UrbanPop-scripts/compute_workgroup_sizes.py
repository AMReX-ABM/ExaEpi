#!/usr/bin/env python

# Computes a per-(state, NAICS-code) work-group target size table for ExaEpi's UrbanPop
# path, from the 2019 Census County Business Patterns (CBP) survey: avg_size =
# employment / establishments per NAICS code per state, capped at 86 (based on a
# workplace-contact-pattern study), following the methodology described in the Epicast 2.0
# paper (related/epicast.pdf) this codebase is modeled on.
#
# Usage:
#   # one-time: download CBP and build the small derived cache checked into the repo
#   python compute_workgroup_sizes.py --refresh-cbp-cache
#
#   # normal use: read the cache, write the table ExaEpi loads at runtime
#   python compute_workgroup_sizes.py
#
# Output is a single table covering every state CBP publishes data for
# (data/UrbanPop/workgroup_sizes_us.txt by default) -- not one file per state -- so it is
# correct to pair with any UrbanPop .bin file regardless of which state(s) it covers,
# including a combined multi-state or national build.
#
# Deliberately stdlib-only (no polars/pandas/requests): this script only needs to read the
# NAICS code list and a small CSV, and shouldn't require the heavier dependencies the main
# UrbanPop-to-ExaEpi conversion pipeline (upop_to_exaepi_polars.py) needs.

import argparse
import csv
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CBP_URL = "https://www2.census.gov/programs-surveys/cbp/datasets/2019/cbp19st.zip"
DEFAULT_CBP_CACHE = REPO_ROOT / "data" / "UrbanPop" / "cbp19st_derived.csv"
DEFAULT_NAICS_HEADER = REPO_ROOT / "src" / "UrbanPopAgentStruct.H"
DEFAULT_OUT = REPO_ROOT / "data" / "UrbanPop" / "workgroup_sizes_us.txt"
DEFAULT_SIZE = 20  # matches Utils.H's workgroup_size default; used when CBP has no
                    # coverage at all for a (state, NAICS) combination (e.g. NAICS 92x
                    # Public Administration, which CBP excludes everywhere)
DEFAULT_CAP = 86    # workplace-contact-pattern-study cap (see epicast.pdf refs [27],[28])

# UrbanPop uses these single-digit placeholders for CBP sector supergroups that get
# suppressed to the 2-digit sector level in state-level CBP data (Manufacturing 31-33,
# Retail Trade 44-45); CBP itself reports these supergroups under their first 2-digit code.
ALIASES = {"3": "31", "4": "44"}


def refresh_cbp_cache(cache_path, url=DEFAULT_CBP_URL):
    """Download the 2019 CBP state-level bulk file (no API key needed, unlike the Census
    API) and write a small filtered/derived cache: just the columns and rows this script
    needs (fipstate, naics, emp, est; lfo == '-' i.e. all legal forms of organization
    combined), stripped of CBP's fixed-width NAICS padding characters ('-', '/')."""
    print(f"Downloading {url} ...", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw_zip = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(raw_zip))
    names = zf.namelist()
    if len(names) != 1:
        raise RuntimeError(f"Expected exactly one file in {url}, found {names}")
    raw_text = zf.read(names[0]).decode("utf-8", errors="replace")

    rows_out = []
    reader = csv.DictReader(io.StringIO(raw_text))
    for row in reader:
        if row["lfo"] != "-":
            continue
        naics = row["naics"].rstrip("-/")
        if not naics:
            continue  # the "------" all-industries total row; not needed per-NAICS
        try:
            emp = int(row["emp"])
            est = int(row["est"])
        except ValueError:
            continue
        rows_out.append((row["fipstate"], naics, emp, est))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fipstate", "naics", "emp", "est"])
        w.writerows(rows_out)
    print(f"Wrote {len(rows_out)} (state, naics) rows to {cache_path}", file=sys.stderr)


def load_cbp_cache(cache_path):
    """Returns {(fipstate, naics_code): (emp, est)}."""
    cbp = {}
    with open(cache_path) as f:
        for row in csv.DictReader(f):
            cbp[(row["fipstate"], row["naics"])] = (int(row["emp"]), int(row["est"]))
    return cbp


def parse_naics_descriptions(header_path):
    """Extracts the NAICS code list from the auto-generated src/UrbanPopAgentStruct.H,
    rather than importing upop_to_exaepi.py's categ_types['pr_naics'] (the two are
    generated from the same source and must agree, but importing that script pulls in
    polars/colorama/psutil for no reason here -- reading the actual header ExaEpi compiles
    against is both lighter-weight and closer to ground truth)."""
    text = header_path.read_text()
    m = re.search(r"NAICS_COUNT\s*=\s*(\d+)", text)
    if not m:
        raise RuntimeError(f"Could not find NAICS_COUNT in {header_path}")
    naics_count = int(m.group(1))
    m2 = re.search(r"naics_descriptions\[NAICS_COUNT\]\s*=\s*\{(.*?)\};", text, re.S)
    if not m2:
        raise RuntimeError(f"Could not find naics_descriptions array in {header_path}")
    codes = re.findall(r'"([^"]*)"', m2.group(1))
    if len(codes) != naics_count:
        raise RuntimeError(
            f"Parsed {len(codes)} NAICS codes from {header_path} but NAICS_COUNT is {naics_count}"
        )
    return codes


def resolve_naics_size(fipstate, naics_code, cbp):
    """Look up the average establishment size for one (state, NAICS code) pair, climbing
    the NAICS hierarchy (truncating trailing digits, floor at 2 digits) if the exact code
    is suppressed at the state level. Returns (avg_size_or_None, note)."""
    code = ALIASES.get(naics_code, naics_code)
    climbed = False
    while True:
        entry = cbp.get((fipstate, code))
        if entry is not None and entry[1] > 0:
            emp, est = entry
            if climbed:
                note = f"fallback:{code}"
            elif code != naics_code:
                note = "alias"
            else:
                note = "exact"
            return emp / est, note
        if len(code) <= 2:
            return None, "no_cbp_coverage"
        code = code[:-1]
        climbed = True


def compute_table(naics_codes, states, cbp, default_size, cap):
    rows = []
    note_counts = {}
    no_coverage_codes = set()
    for fipstate in states:
        for code in naics_codes:
            avg, note = resolve_naics_size(fipstate, code, cbp)
            note_kind = note.split(":", 1)[0]
            note_counts[note_kind] = note_counts.get(note_kind, 0) + 1
            if avg is None:
                size = default_size
                no_coverage_codes.add(code)
            else:
                # a handful of (state, NAICS) pairs report avg establishment size < 0.5
                # (a CBP reporting quirk -- e.g. seasonal/part-time-heavy establishments
                # with nobody on payroll during the March 12 reference week), which would
                # round to 0; floor at 1, since a work-group must contain at least 1 person
                size = max(1, int(round(min(avg, cap))))
            rows.append((fipstate, code, size))
    return rows, note_counts, no_coverage_codes


def write_table(rows, out_path, default_size, cap):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Generated by utilities/UrbanPop-scripts/compute_workgroup_sizes.py\n")
        f.write(f"# source: 2019 CBP state file (cbp19st.txt); fallback default={default_size}, cap={cap}\n")
        f.write("# columns: state_fips  naics_code  workgroup_size\n")
        f.write("# NOTE: rows that used the fallback default (no CBP coverage at any NAICS\n")
        f.write("# level for that state, e.g. most of NAICS 92x Public Administration) bake\n")
        f.write("# that default in at generation time -- overriding agent.workgroup_size at\n")
        f.write("# runtime will NOT change these rows, only (state, NAICS) pairs absent from\n")
        f.write("# this file entirely inherit the live runtime value.\n")
        f.write(f"{len(rows)}\n")
        for fipstate, code, size in rows:
            f.write(f"{int(fipstate)}\t{code}\t{size}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh-cbp-cache", action="store_true",
                    help="download the 2019 CBP bulk file and (re)build the derived cache before generating the table")
    p.add_argument("--cbp-url", default=DEFAULT_CBP_URL)
    p.add_argument("--cbp-cache", type=Path, default=DEFAULT_CBP_CACHE)
    p.add_argument("--naics-header", type=Path, default=DEFAULT_NAICS_HEADER)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--default-size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--cap", type=int, default=DEFAULT_CAP)
    p.add_argument("--state", nargs="*", default=None,
                    help="restrict to these 2-digit state FIPS codes (default: every state found in the cache)")
    args = p.parse_args()

    if args.refresh_cbp_cache:
        refresh_cbp_cache(args.cbp_cache, args.cbp_url)

    if not args.cbp_cache.exists():
        p.error(f"{args.cbp_cache} does not exist -- run with --refresh-cbp-cache first")

    naics_codes = parse_naics_descriptions(args.naics_header)
    cbp = load_cbp_cache(args.cbp_cache)

    states = args.state if args.state else sorted({fipstate for fipstate, _ in cbp.keys()})
    print(f"Computing work-group sizes for {len(states)} state(s) x {len(naics_codes)} NAICS codes", file=sys.stderr)

    rows, note_counts, no_coverage_codes = compute_table(naics_codes, states, cbp, args.default_size, args.cap)
    write_table(rows, args.out, args.default_size, args.cap)

    print(f"Wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    print(f"Resolution breakdown: {note_counts}", file=sys.stderr)
    print(f"{len(no_coverage_codes)} NAICS codes fell back to the default ({args.default_size}) "
          f"in at least one state (no CBP coverage at any level, e.g. public administration):",
          file=sys.stderr)
    print("  " + ", ".join(sorted(no_coverage_codes)), file=sys.stderr)


if __name__ == "__main__":
    main()
