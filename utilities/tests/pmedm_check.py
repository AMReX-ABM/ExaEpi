#!/usr/bin/env python
"""Compare src/PmedmSolver (via bin/popgen_pmedm_check) with the Python oracle, PUMA by PUMA.

    popgen_pmedm_check BUNDLE SEED REP cpp.bin
    python utilities/tests/pmedm_check.py BUNDLE SEED REP cpp.bin

For each PUMA in cpp.bin:
  * the perturbation port: C++ Y and log q against popgen/perturb.py for the same key (they use
    log/exp/cos, so they agree to rounding, not bit for bit);
  * the solve: share of households placed differently (half the L1 distance over N) between the
    C++ allocation, the Python production solver (popgen/solver.py) and a float64 solve converged
    to gradient 1e-6 (jaxopt, as experiments/solver_options.py validated against).

Pass: every perturbation agrees to 1e-12 relative, and every C++ allocation is within 0.1% of
households of the converged one. Run with the livelike environment and JAX_PLATFORMS=cuda,cpu.
"""

import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "UrbanPop-scripts")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "experiments"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from popgen import bundle, perturb, solver  # noqa: E402  (sets the deterministic XLA flags)
from popgen.problem import PumaProblem  # noqa: E402

import jax.numpy as jnp  # noqa: E402
from fp32_capped import make_solver  # noqa: E402
from gpu_solve import f  # noqa: E402

TOL_PERTURB = 1e-12
TOL_MOVED = 1e-3


def read_cpp(path):
    out = []
    with open(path, "rb") as fh:
        while True:
            head = fh.read(4)
            if not head:
                break
            (n,) = struct.unpack("<i", head)
            puma = fh.read(n).decode()
            K, T, G, D, iters = struct.unpack("<5i", fh.read(20))
            gnorm, secs = struct.unpack("<2d", fh.read(16))
            nd = K * (1 + T + G)
            Y = np.frombuffer(fh.read(8 * nd), dtype="<f8")
            logq = np.frombuffer(fh.read(8 * D), dtype="<f8")
            al = np.frombuffer(fh.read(8 * D * G), dtype="<f8").reshape(D, G)
            out.append(dict(puma=puma, iters=iters, gnorm=gnorm, secs=secs, Y=Y, logq=logq, al=al))
    return out


def moved(a, b, N):
    return 0.5 * np.abs(a - b).sum() / N


def main():
    path, seed, rep, cpp_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    b = bundle.read(path)
    names = bundle.strings(b, "solve.puma")
    conv = make_solver(f, 50000)
    ok = True
    print(f"{'PUMA':9s}{'Y rel':>10s}{'logq':>10s}{'C++ it':>8s}{'py it':>7s}{'C++ s':>7s}"
          f"{'C++-conv':>10s}{'py-conv':>9s}{'C++-py':>9s}")
    worst = 0.0
    for r in read_cpp(cpp_path):
        prob = PumaProblem(b, names.index(r["puma"]))
        Y, _ = perturb.perturbed_targets(prob, seed, rep)
        lq = perturb.log_prior(perturb.perturbed_prior(prob, seed, rep), prob.G)
        dY = np.max(np.abs(r["Y"] - Y) / np.maximum(np.abs(Y), 1e-300))
        dq = np.max(np.abs(r["logq"] - lq[:, 0]) / np.abs(lq[:, 0]))
        al_py, it_py, _ = solver.solve(prob, Y, lq)
        lam = conv(jnp.zeros(len(Y)), jnp.asarray(prob.C), jnp.asarray(lq), jnp.asarray(prob.A1),
                   jnp.asarray(Y), jnp.asarray(prob.V))[0]
        al_ref = solver.allocation(prob, np.asarray(lam), lq)
        m_cpp, m_py = moved(r["al"], al_ref, prob.N), moved(al_py, al_ref, prob.N)
        m_x = moved(r["al"], al_py, prob.N)
        worst = max(worst, m_cpp)
        good = dY < TOL_PERTURB and dq < TOL_PERTURB and m_cpp < TOL_MOVED
        ok &= good
        print(f"{r['puma']:9s}{dY:10.1e}{dq:10.1e}{r['iters']:8d}{it_py:7d}{r['secs']:7.2f}"
              f"{100 * m_cpp:9.3f}%{100 * m_py:8.3f}%{100 * m_x:8.3f}%{'' if good else '  FAIL'}")
    print(f"worst C++ distance from converged {100 * worst:.3f}% (limit {100 * TOL_MOVED:g}%)")
    print("pmedm_check: " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
