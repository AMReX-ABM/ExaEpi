#!/usr/bin/env python
"""How big is a re-solve bundle -- donors, ACS targets and standard errors -- for NM, and for the US?

If ExaEpi re-solves P-MEDM each run with freshly perturbed ACS targets, the bundle no longer needs
allocation matrices at all. It needs, per PUMA:

    C        donor households x constraints: what each household contributes to each target
    targets  published estimates at block group, tract and PUMA level
    SEs      their standard errors, to draw perturbations from
    q        survey weights (the prior)

plus the person attributes, LODES, schools and CBP tables the bundle already carries.

This measures those sections on New Mexico's 18 PUMAs in the same compressed encodings the bundle
uses. C is stored sparse, because a household only has a handful of nonzero characteristics.
"""

import os
import sys
import zlib

import numpy as np

sys.path.insert(0, "/workspaces/ExaEpi/utilities/UrbanPop-scripts")
from build_precompute import EXAEPI_MINIMAL  # noqa: E402

PUMAS = [f"35{p:05d}" for p in (100, 1001, 1002, 1100, 1200, 200, 300, 400, 500, 600, 700,
                                801, 802, 803, 804, 805, 806, 900)]


def z(a):
    b = np.ascontiguousarray(a).tobytes()
    return len(b), len(zlib.compress(b, 6))


def main():
    from livelike import acs, config

    key = os.environ.get("CENSUS_API_KEY") or None
    tot = {"donors": 0, "nnz": 0, "nonint": 0, "bg": 0, "trt": 0, "puma": 0}
    raw = {"C_idx": 0, "C_val": 0, "C_ptr": 0, "targets": 0, "se": 0, "wt": 0}
    comp = dict.fromkeys(raw, 0)
    K = None
    for f in PUMAS:
        pup = acs.puma(f, constraints_selection=EXAEPI_MINIMAL,
                       constraints_theme_order=config.up_constraints_theme_order,
                       year=2019, target_zone="bg", cache=True,
                       cache_folder="./llcache_minimal", censusapikey=key)
        C = np.asarray(pup.est_ind, dtype=np.float64)
        K = C.shape[1]
        r, c = np.nonzero(C)
        v = C[r, c]
        tot["donors"] += C.shape[0]
        tot["nnz"] += len(v)
        tot["nonint"] += int((np.abs(v - np.rint(v)) > 1e-9).sum())
        ptr = np.zeros(C.shape[0] + 1, np.int32)
        np.cumsum(np.bincount(r, minlength=C.shape[0]), out=ptr[1:])
        for name, arr in (("C_idx", c.astype(np.uint8)), ("C_val", v.astype(np.float32)),
                          ("C_ptr", ptr)):
            a, b = z(arr); raw[name] += a; comp[name] += b

        g1 = np.asarray(pup.est_g1, np.float32); g2 = np.asarray(pup.est_g2, np.float32)
        s1 = np.asarray(pup.se_g1, np.float32); s2 = np.asarray(pup.se_g2, np.float32)
        tot["bg"] += g2.shape[0]; tot["trt"] += g1.shape[0]; tot["puma"] += 1
        for name, arr in (("targets", np.concatenate([g1.ravel(), g2.ravel()])),
                          ("se", np.nan_to_num(np.concatenate([s1.ravel(), s2.ravel()])))):
            a, b = z(arr); raw[name] += a; comp[name] += b
        a, b = z(np.asarray(pup.wt, np.float32)); raw["wt"] += a; comp["wt"] += b

    print(f"NM: {tot['puma']} PUMAs, {tot['donors']:,} donor rows, {K} constraints, "
          f"{tot['bg']:,} block groups, {tot['trt']:,} tracts")
    print(f"  C nonzeros {tot['nnz']:,} ({tot['nnz'] / tot['donors']:.1f} per household, "
          f"{100 * tot['nnz'] / (tot['donors'] * K):.1f}% dense); "
          f"non-integer values {100 * tot['nonint'] / tot['nnz']:.1f}%")
    print(f"\n{'section':12s}{'raw MB':>10s}{'stored MB':>11s}")
    for k in raw:
        print(f"{k:12s}{raw[k] / 1e6:10.3f}{comp[k] / 1e6:11.3f}")
    print(f"{'total':12s}{sum(raw.values()) / 1e6:10.3f}{sum(comp.values()) / 1e6:11.3f}")

    per_donor = (comp["C_idx"] + comp["C_val"] + comp["C_ptr"] + comp["wt"]) / tot["donors"]
    per_zone = (comp["targets"] + comp["se"]) / (tot["bg"] + tot["trt"])
    print(f"\n  per donor row (C + weight): {per_donor:.1f} B stored")
    print(f"  per zone (targets + SE, {K} constraints): {per_zone:.1f} B stored")
    print(f"PER_DONOR={per_donor:.4f} PER_ZONE={per_zone:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
