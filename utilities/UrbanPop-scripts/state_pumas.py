"""PUMA FIPS (state + 2010 PUMA) covering the tracts of a state's delivered UrbanPop feathers.

The list build_precompute.py's --pumas takes. The feathers carry no PUMA column, so tracts are
mapped through livelike's 2010 tract-to-PUMA crosswalk (the geography of the 2019 ACS). For NM this
gives exactly the 18 PUMAs nm_v2.upb was built from; for CA, 265.

    state_pumas.py 'base/06_CA/*.feather'   (run from data/UrbanPop; the list goes to stdout)
"""
import glob
import os
import sys

import livelike
import pandas as pd
import polars as pl

xw = pd.read_csv(os.path.join(os.path.dirname(livelike.__file__), "data", "2010_Census_Tract_to_2010_PUMA.zip"),
                 dtype=str)
xw["tract"] = xw["STATEFP"] + xw["COUNTYFP"] + xw["TRACTCE"]
xw["puma"] = xw["STATEFP"] + xw["PUMA5CE"]
tract_puma = dict(zip(xw["tract"], xw["puma"]))

tracts = set()
for f in sorted(glob.glob(sys.argv[1])):
    tracts |= {g[:11] for g in pl.read_ipc(f, columns=["geoid"])["geoid"].unique().to_list()}
missing = sorted(t for t in tracts if t not in tract_puma)
pumas = sorted({tract_puma[t] for t in tracts if t in tract_puma})
print(f"{len(tracts)} tracts, {len(missing)} not in the 2010 crosswalk {missing[:5]}, {len(pumas)} PUMAs", file=sys.stderr)
print(" ".join(pumas))
