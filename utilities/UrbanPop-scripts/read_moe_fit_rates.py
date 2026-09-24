#!/usr/bin/env -S python -u

"""Extract the published ACS estimates and margins of error that UrbanPop's P-MEDM run used.

Each state directory under ``data/UrbanPop/base/`` ships a ``moe_fit_rates_<state>.pkl``, the
P-MEDM diagnostic described in section 3.6.1 of ``related/urbanpop.pdf``. It is a dict keyed by
PUMA, each value holding

    Ycomp         a DataFrame indexed by 12-digit block group GEOID, with columns
                  constraint, acs, pmedm, err, moe, in_moe
    moe_fit_rate  the scalar fraction of that PUMA's constraints landing inside the ACS MOE

``acs`` and ``moe`` are the published ACS 5-year estimate and its 90% margin of error, so the ACS
sampling standard error is ``moe / 1.645``. Extracting it here means no Census API call -- and,
better, these are the margins the delivered UrbanPop population was actually fitted against, rather
than a re-fetch that might not match it.

The resulting per-block-group coefficient of variation is the yardstick for how much the resident
population could legitimately differ. For New Mexico it is a median of 0.160 on block group
population. Any candidate source of population variation -- a different TRS draw from one P-MEDM
allocation matrix, or a replicate allocation matrix -- is judged by whether its spread reaches that
scale.

The pickles were written with ``dill``, but only for a numpy array reconstructor; ``_DillShim``
below supplies the two symbols involved so stock ``pickle`` can read them without the dependency.
"""

import argparse
import glob
import os
import pickle
import sys
import types

import numpy as np
import pandas as pd

# The ACS publishes margins of error at 90% confidence: MOE = 1.645 * standard error.
MOE_Z = 1.645


def _install_dill_shim():
    """Make the dill-pickled arrays loadable without dill installed.

    The pickles reference exactly two dill symbols, ``_load_type`` and ``_create_array``, both
    only to rebuild numpy arrays. Reimplementing them is a few lines and avoids requiring dill
    (which pip refuses to install into this container's system Python under PEP 668).
    """
    if "dill" in sys.modules:
        return

    aliases = {"ArrayType": np.ndarray, "NoneType": type(None), "ObjectType": object}

    def _load_type(name):
        import builtins

        if name in aliases:
            return aliases[name]
        for module in (builtins, np):
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(f"dill shim cannot resolve type {name!r}")

    def _create_array(factory, args, state, npdict):
        array = factory(*args)
        array.__setstate__(state)
        if npdict is not None:
            array.__dict__.update(npdict)
        return array

    dill = types.ModuleType("dill")
    inner = types.ModuleType("dill._dill")
    inner._load_type = _load_type
    inner._create_array = _create_array
    dill._dill = inner
    sys.modules["dill"] = dill
    sys.modules["dill._dill"] = inner


def find_fit_rates_file(base_dir: str, state: str) -> str:
    """Locate moe_fit_rates_<state>.pkl, given a 2-digit state FIPS code."""
    pattern = os.path.join(base_dir, f"{state}_*", f"moe_fit_rates_{state}.pkl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"{pattern} is ambiguous: {matches}")
    return matches[0]


def load_constraints(fname: str, constraints: list[str] | None = None) -> pd.DataFrame:
    """Read one state's pickle into a tidy frame.

    Returns columns geoid, puma, constraint, acs, pmedm, err, moe, in_moe, se -- one row per
    (block group, constraint). ``constraints`` filters to a subset by name; None keeps all 266.
    """
    _install_dill_shim()
    with open(fname, "rb") as f:
        by_puma = pickle.load(f)

    wanted = set(constraints) if constraints else None
    frames = []
    for puma, entry in by_puma.items():
        ycomp = entry["Ycomp"]
        if wanted is not None:
            ycomp = ycomp[ycomp["constraint"].isin(wanted)]
            if ycomp.empty:
                continue
        ycomp = ycomp.copy()
        ycomp.insert(0, "puma", str(puma))
        frames.append(ycomp)

    if not frames:
        raise RuntimeError(f"{fname} yielded no rows (constraints={constraints})")

    df = pd.concat(frames)
    df.index.name = "geoid"
    df = df.reset_index()
    df["geoid"] = df["geoid"].astype(str)
    # The sampling standard error behind the published 90% MOE. A handful of constraints are
    # exactly controlled by the Census, for which the MOE is published as zero -- those come
    # through as se == 0, i.e. no room to vary, which is the correct treatment.
    df["se"] = df["moe"] / MOE_Z
    return df[["geoid", "puma", "constraint", "acs", "pmedm", "err", "moe", "se", "in_moe"]]


def summarize(df: pd.DataFrame) -> None:
    """Print the per-constraint coefficient of variation, i.e. the ACS uncertainty scale."""
    print(f"block groups: {df['geoid'].nunique()}   constraints: {df['constraint'].nunique()}")
    print(f"rows: {len(df)}   overall in_moe rate: {df['in_moe'].mean():.4f}")
    for name, group in df.groupby("constraint", sort=True):
        nonzero = group[group["acs"] > 0]
        if nonzero.empty:
            continue
        cv = nonzero["se"] / nonzero["acs"]
        print(
            f"  {name:<28} n={len(nonzero):>6}  CV median={cv.median():.3f} "
            f"mean={cv.mean():.3f} p90={cv.quantile(0.9):.3f} max={cv.max():.3f}"
        )


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract ACS estimates and MOEs from a UrbanPop moe_fit_rates pickle"
    )
    parser.add_argument("--state", "-s", required=True, help="2-digit state FIPS code, e.g. 35")
    parser.add_argument(
        "--base_dir",
        "-b",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                             "UrbanPop", "base"),
        help="directory holding the per-state UrbanPop base data",
    )
    parser.add_argument(
        "--constraints",
        "-c",
        nargs="+",
        default=["population"],
        help="constraint names to keep; pass 'all' for every constraint",
    )
    parser.add_argument("--output", "-o", default=None, help="output CSV (default: stdout summary only)")
    return parser.parse_args()


def main():
    args = get_args()
    constraints = None if args.constraints == ["all"] else args.constraints
    fname = find_fit_rates_file(args.base_dir, args.state)
    print(f"Reading {fname}")
    df = load_constraints(fname, constraints)
    summarize(df)
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
