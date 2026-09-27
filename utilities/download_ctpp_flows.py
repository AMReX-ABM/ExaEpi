#!/usr/bin/env python

"""Download a CTPP tract-to-tract home->work flow table for one state from the CTPP Data API, as a
CSV with the same src, dst, flow columns as Epicast's commute-flow table (plus moe, the published
90% margin of error), so it can be used anywhere that table can.

CTPP (Census Transportation Planning Products) is built from the ACS question about where each
worker worked the previous week, so unlike LODES (jobs linked to residences from administrative
records) its flows are commutes people reported actually making. The default table, A302100, is
total workers 16+ in the 2012-2016 CTPP, which uses 2010 Census tracts. NOTE that this universe
includes people who worked at home, who appear as a flow from their home tract to itself.

The API needs a key (https://ctppdata.transportation.org/ -> Login -> Manage API Keys), read from
the CTPP_API_KEY environment variable so it never ends up on a command line or in a file.

Usage:
    CTPP_API_KEY=... download_ctpp_flows.py --state 35 -o data/CTPP/nm_ctpp2016_tract_flows.csv
"""

import os
import sys
import csv
import json
import time
import argparse
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://ctppdata.transportation.org/api"
PAGE_SIZE = 1000  # the API's maximum


def fetch_page(api_key, year, params, page, retries=5):
    query = urllib.parse.urlencode({**params, "format": "list", "size": PAGE_SIZE, "page": page}, safe=":*,")
    request = urllib.request.Request(f"{BASE_URL}/data/{year}?{query}",
                                     headers={"X-API-Key": api_key, "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                sys.exit(f"error: page {page}: HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                sys.exit(f"error: page {page}: {e}")
        time.sleep(2 ** attempt)


def tract_geoid(ctpp_geoid):
    """CTPP geoids carry a summary-level prefix, e.g. C1100US35001000107 -> 35001000107."""
    return int(ctpp_geoid.split("US", 1)[1])


def parse_count(value):
    """Estimates come as strings like "1,234"; margins as "+/-74"."""
    return int(value.replace("+/-", "").replace(",", "").strip() or 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", "-s", required=True, help="2-digit state FIPS code, e.g. 06 or 35")
    parser.add_argument("--year", default="2016", help="CTPP dataset year: 2016 (2012-2016, 2010 tracts, the default) "
                        "or 2021 (2017-2021, 2020 tracts)")
    parser.add_argument("--table", default="a302100", help="Flow table ID (default: a302100, total workers)")
    parser.add_argument("--output", "-o", required=True, help="Output CSV (src, dst, flow, moe)")
    args = parser.parse_args()

    api_key = os.environ.get("CTPP_API_KEY")
    if not api_key:
        sys.exit("error: set the CTPP_API_KEY environment variable to your CTPP API key")

    state = args.state.zfill(2)
    table = args.table.lower()
    est, moe = f"{table}_e1", f"{table}_m1"
    params = {"get": f"{est},{moe}", "for": "tract:*", "in": f"state:{state}",
              "d-for": "tract:*", "d-in": f"state:{state}"}

    first = fetch_page(api_key, args.year, params, 1)
    total = int(first["total"])
    n_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    print(f"{total:,} tract pairs in {n_pages} pages")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    rows = 0
    workers = 0
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "dst", "flow", "moe"])
        for page in range(1, n_pages + 1):
            data = first if page == 1 else fetch_page(api_key, args.year, params, page)
            for rec in data["data"]:
                flow = parse_count(rec[est])
                writer.writerow([tract_geoid(rec["origin_geoid"]), tract_geoid(rec["destination_geoid"]), flow,
                                 parse_count(rec[moe])])
                rows += 1
                workers += flow
            print(f"\rpage {page}/{n_pages}", end="", flush=True)
    print()
    if rows != total:
        print(f"WARNING: wrote {rows:,} rows but the API reported {total:,}", file=sys.stderr)
    print(f"Wrote {rows:,} tract pairs, {workers:,} workers, to {args.output}")


if __name__ == "__main__":
    main()
