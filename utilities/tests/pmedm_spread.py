#!/usr/bin/env python
"""Population spread and composition of src/PmedmSolver over many seeds, against the Python oracle.

    pmedm_spread.py BUNDLE REP CHECK_BIN WORKDIR [NSEEDS]

pmedm_check.py shows each C++ solve lands within 0.1% of households of the converged answer. This
checks what the ensemble is for: that across population seeds the C++ solves vary the population
exactly as much, and in the same ways, as the Python production solver (popgen/solver.py) on the
same perturbed inputs. Per PUMA, over seeds 1..NSEEDS (default 20), for both solvers:

    bg_ratio      block-group population CV across draws, median, over the ACS CV of the same
                  block groups (resolve_spread.py's spread measure; ~0.30-0.33 at the design point)
    bg_outside    share of block-group populations outside the published 90% MOE (|z| > 1.645)
    in_moe, cv_*  composition.py's fit, constrained-share and joint-composition measures

CHECK_BIN is bin/popgen_pmedm_check; its output for each seed is written to WORKDIR/cpp_sSEED.bin
(and reused if present). Both solvers see the same draws, so sampling noise is common to them and a
difference measures only the solvers. Pass: every CV and ratio within 2% (relative) of the Python
value and every share within 0.002. Run from anywhere with the livelike environment and
JAX_PLATFORMS=cuda,cpu.
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "UrbanPop-scripts")
sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "experiments"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from popgen import bundle, perturb, solver  # noqa: E402  (sets the deterministic XLA flags)
from popgen.problem import PumaProblem  # noqa: E402
import composition  # noqa: E402
from pmedm_check import read_cpp  # noqa: E402

REL_TOL = 0.02
ABS_TOL = 0.002
Z90 = 1.645


def bg_measures(als, prob, pop):
    """resolve_spread.py's block-group population spread and fit, over draws."""
    bgpop = np.array([a.T @ prob.C[:, pop] for a in als])       # draws x G
    est, se = prob.est_bg[:, pop], prob.se_bg[:, pop]
    m = bgpop.mean(0)
    cv = bgpop.std(0, ddof=1) / np.where(m > 0, m, np.nan)
    acs = se / np.where(est > 0, est, np.nan)
    ok = np.isfinite(cv) & np.isfinite(acs) & (est >= 20)
    z = (bgpop - est) / np.where(se > 0, se, np.nan)
    return {"bg_ratio": float(np.median(cv[ok]) / np.median(acs[ok])),
            "bg_outside": float(np.nanmean(np.abs(z) > Z90))}


def measures(als, prob, cols):
    out = bg_measures(als, prob, cols.index("population"))
    out.update(composition.measure(als, prob.C, cols, prob.A1, prob.est_tract, prob.est_bg,
                                   prob.se_tract, prob.se_bg))
    return out


def is_share(name):
    return name in ("bg_outside", "in_moe")


def main():
    path, rep, check_bin, work = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    nseeds = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    seeds = list(range(1, nseeds + 1))
    os.makedirs(work, exist_ok=True)
    b = bundle.read(path)
    names = bundle.strings(b, "solve.puma")
    cols = bundle.strings(b, "solve.constraints")

    cpp = {}  # puma -> [allocation per seed]
    for s in seeds:
        out = os.path.join(work, f"cpp_s{s}.bin")
        if not os.path.exists(out):
            subprocess.run([check_bin, path, str(s), str(rep), out], check=True, stdout=subprocess.DEVNULL)
        for r in read_cpp(out):
            cpp.setdefault(r["puma"], []).append(r["al"])

    ok = True
    worst = {}
    keys = None
    for puma in names:
        prob = PumaProblem(b, names.index(puma))
        py = []
        for s in seeds:
            Y, _ = perturb.perturbed_targets(prob, s, rep)
            lq = perturb.log_prior(perturb.perturbed_prior(prob, s, rep), prob.G)
            py.append(solver.solve(prob, Y, lq)[0])
        mc, mp = measures(cpp[puma], prob, cols), measures(py, prob, cols)
        if keys is None:
            keys = list(mp)
            print(f"{nseeds} seeds per PUMA; each cell is C++ / Python")
            print(f"{'PUMA':9s}" + "".join(f"{k:>24s}" for k in keys))
        cells = []
        for k in keys:
            d = abs(mc[k] - mp[k]) if is_share(k) else abs(mc[k] - mp[k]) / max(abs(mp[k]), 1e-300)
            bad = d > (ABS_TOL if is_share(k) else REL_TOL)
            ok &= not bad
            worst[k] = max(worst.get(k, 0.0), d)
            cells.append(f"{mc[k]:.4f}/{mp[k]:.4f}{'!' if bad else ' '}")
        print(f"{puma:9s}" + "".join(f"{c:>24s}" for c in cells), flush=True)
    print("worst difference (absolute for shares, relative otherwise):")
    for k in keys:
        print(f"  {k:26s} {worst[k]:.4f}")
    print("pmedm_spread: " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
