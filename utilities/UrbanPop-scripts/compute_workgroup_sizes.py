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
import hashlib
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
DEFAULT_DIST_OUT = REPO_ROOT / "data" / "UrbanPop" / "establishment_sizes_us.txt"

# CBP's establishment-employment-size bands, as (column suffix, lower bound, upper bound).
# For each band CBP reports n<suffix> establishments holding e<suffix> employees, so the band's
# own mean size is e/n -- far more information than the single emp/est average, and what
# upop_to_exaepi.py's alloc_workers needs to size a workplace by drawing rather than by
# assuming every workplace in an industry is average. The last band is open-ended.
SIZE_BANDS = [("<5", 1, 4), ("5_9", 5, 9), ("10_19", 10, 19), ("20_49", 20, 49), ("50_99", 50, 99),
              ("100_249", 100, 249), ("250_499", 250, 499), ("500_999", 500, 999), ("1000", 1000, None)]

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
        # per-band establishment counts and employment; CBP writes 'N'/'' where a cell is
        # noise-suppressed, which becomes 0 here (band contributes nothing to the distribution)
        bands = []
        for name, _lo, _hi in SIZE_BANDS:
            for prefix in ("n", "e"):
                try:
                    bands.append(int(row[prefix + name]))
                except (ValueError, KeyError):
                    bands.append(0)
        rows_out.append((row["fipstate"], naics, emp, est, *bands))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    band_cols = [p + name for name, _lo, _hi in SIZE_BANDS for p in ("n", "e")]
    with open(cache_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fipstate", "naics", "emp", "est"] + band_cols)
        w.writerows(rows_out)
    print(f"Wrote {len(rows_out)} (state, naics) rows to {cache_path}", file=sys.stderr)


def load_cbp_cache(cache_path):
    """Returns {(fipstate, naics_code): (emp, est)}."""
    cbp = {}
    with open(cache_path) as f:
        for row in csv.DictReader(f):
            cbp[(row["fipstate"], row["naics"])] = (int(row["emp"]), int(row["est"]))
    return cbp


def load_cbp_bands(cache_path):
    """Returns {(fipstate, naics_code): [(n, e), ...]} -- per SIZE_BANDS establishment counts
    and employment. Empty if the cache predates the band columns (refresh it to add them)."""
    bands = {}
    with open(cache_path) as f:
        reader = csv.DictReader(f)
        cols = [(p + name) for name, _lo, _hi in SIZE_BANDS for p in ("n", "e")]
        if not all(c in (reader.fieldnames or []) for c in cols):
            return {}
        for row in reader:
            vals = [int(row[c] or 0) for c in cols]
            bands[(row["fipstate"], row["naics"])] = list(zip(vals[0::2], vals[1::2]))
    return bands


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


def resolve_naics_code(fipstate, naics_code: str, cbp):
    """Which CBP (state, NAICS) row actually backs this NAICS code: the exact code where CBP
    covers it, else the nearest ancestor found by truncating trailing digits (floor at 2).
    Returns (resolved_code_or_None, note). Factored out so the work-group size table and the
    establishment-size distribution table below resolve identically -- a code whose size falls
    back to its parent must draw its size distribution from that same parent."""
    code = ALIASES.get(naics_code, naics_code)
    climbed = False
    while True:
        entry = cbp.get((fipstate, code))
        if entry is not None and entry[1] > 0:
            if climbed:
                return code, f"fallback:{code}"
            return code, "alias" if code != naics_code else "exact"
        if len(code) <= 2:
            return None, "no_cbp_coverage"
        code = code[:-1]
        climbed = True


def resolve_naics_size(fipstate, naics_code: str, cbp):
    """Look up the average establishment size for one (state, NAICS code) pair, climbing
    the NAICS hierarchy (truncating trailing digits, floor at 2 digits) if the exact code
    is suppressed at the state level. Returns (avg_size_or_None, note)."""
    code, note = resolve_naics_code(fipstate, naics_code, cbp)
    if code is None:
        return None, note
    emp, est = cbp[(fipstate, code)]
    return emp / est, note


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


STAMP_PREFIX = "# generation-stamp: "


def generation_stamp(cache_path, naics_codes, default_size, cap):
    """A short hash of everything that determines the contents of both output tables: the CBP
    cache, the NAICS code list, and the two size parameters.

    Written into both files' headers so upop_to_exaepi.py can refuse a mismatched pair -- a stale
    workgroup_sizes_us.txt alongside a fresh establishment_sizes_us.txt (or the reverse) would
    otherwise silently produce a subtly wrong .bin. Derived from content rather than from the
    clock so that regenerating from unchanged inputs still reproduces byte-identical files,
    which keeps these checked-in tables out of the diff unless something really changed.
    """
    h = hashlib.sha256()
    with open(cache_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update("\0".join(naics_codes).encode())
    h.update(f"{default_size}:{cap}".encode())
    return h.hexdigest()[:16]


def compute_dist_table(naics_codes, states, cbp, bands):
    """Per-(state, NAICS) establishment-size distribution, as the CBP size bands themselves.

    This is what the single average size in the work-group table throws away. Establishment
    sizes are heavily right-skewed -- California's hospitals (NAICS 622) average 1110 employees
    but that average covers 27 establishments under 5 people alongside 186 over 1000 -- so
    sizing every workplace at the industry average produces workplaces that are all the same
    size, which is what makes alloc_workers' destination populations pile up at multiples of
    that average. Emitting the bands lets the allocator draw a size instead.
    """
    rows = []
    for fipstate in states:
        for code in naics_codes:
            resolved, _note = resolve_naics_code(fipstate, code, cbp)
            band = bands.get((fipstate, resolved)) if resolved else None
            if not band or not any(n > 0 and e > 0 for n, e in band):
                continue  # no usable distribution; the allocator falls back to the flat size
            rows.append((fipstate, code, band))
    return rows


def write_dist_table(rows, out_path, stamp):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    band_names = " ".join(f"n{name} e{name}" for name, _lo, _hi in SIZE_BANDS)
    with open(out_path, "w") as f:
        f.write("# Generated by utilities/UrbanPop-scripts/compute_workgroup_sizes.py\n")
        f.write("# source: 2019 CBP state file (cbp19st.txt), establishment-employment-size bands\n")
        f.write("# Per (state, NAICS): for each size band, the number of establishments (n) and the\n")
        f.write("# employment they hold (e), so a band's own mean establishment size is e/n. Read by\n")
        f.write("# upop_to_exaepi.py's alloc_workers to size each workplace by drawing from this\n")
        f.write("# distribution rather than assuming every workplace is the industry average.\n")
        f.write("# Rows with no usable CBP band data anywhere up the NAICS hierarchy are omitted.\n")
        f.write(f"# columns: state_fips  naics_code  {band_names}\n")
        f.write(f"{STAMP_PREFIX}{stamp}\n")
        f.write(f"{len(rows)}\n")
        for fipstate, code, band in rows:
            flat = " ".join(f"{n} {e}" for n, e in band)
            f.write(f"{int(fipstate)} {code} {flat}\n")


def write_table(rows, out_path, default_size, cap, stamp):
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
        f.write(f"{STAMP_PREFIX}{stamp}\n")
        f.write(f"{len(rows)}\n")
        for fipstate, code, size in rows:
            f.write(f"{int(fipstate)} {code} {size}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh-cbp-cache", action="store_true",
                    help="download the 2019 CBP bulk file and (re)build the derived cache before generating the table")
    p.add_argument("--cbp-url", default=DEFAULT_CBP_URL)
    p.add_argument("--cbp-cache", type=Path, default=DEFAULT_CBP_CACHE)
    p.add_argument("--naics-header", type=Path, default=DEFAULT_NAICS_HEADER)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dist-out", type=Path, default=DEFAULT_DIST_OUT,
                    help="where to write the per-(state, NAICS) establishment-size distribution "
                         "table (needs a cache refreshed with the size-band columns)")
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

    stamp = generation_stamp(args.cbp_cache, naics_codes, args.default_size, args.cap)
    rows, note_counts, no_coverage_codes = compute_table(naics_codes, states, cbp, args.default_size, args.cap)
    write_table(rows, args.out, args.default_size, args.cap, stamp)

    print(f"Wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    print(f"Resolution breakdown: {note_counts}", file=sys.stderr)
    print(f"{len(no_coverage_codes)} NAICS codes fell back to the default ({args.default_size}) "
          f"in at least one state (no CBP coverage at any level, e.g. public administration):",
          file=sys.stderr)
    print("  " + ", ".join(sorted(no_coverage_codes)), file=sys.stderr)

    bands = load_cbp_bands(args.cbp_cache)
    if not bands:
        print(f"NOTE: {args.cbp_cache} has no size-band columns, so {args.dist_out} was not written "
              "-- re-run with --refresh-cbp-cache to add them", file=sys.stderr)
    else:
        dist_rows = compute_dist_table(naics_codes, states, cbp, bands)
        write_dist_table(dist_rows, args.dist_out, stamp)
        print(f"Wrote {len(dist_rows)} establishment-size distribution rows to {args.dist_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
