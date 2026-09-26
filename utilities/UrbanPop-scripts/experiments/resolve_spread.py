#!/usr/bin/env python
"""Does perturb-and-re-solve give a calibrated population spread, and does it need to converge?

The Laplace replicates the bundle ships reach only ~0.32 of the ACS block-group CV on the minimal
constraint set, and they need the Hessian, which is what makes the 298-constraint set infeasible.
The alternative is to re-solve P-MEDM each run on perturbed inputs, which needs no Hessian. Two
inputs carry sampling error:

    targets   the ACS block-group and tract counts, published with standard errors
    prior     the PUMS survey weights. The Census publishes their error as 80 replicate weights
              (WGTP1-80 for households, PWGTP1-80 for group-quarters persons, whose person weight
              is livelike's donor weight). A draw with the ACS variance formula's covariance,
              var = 4/80 sum_r (w_r - w)^2, is w + sqrt(4/80) sum_r z_r (w_r - w), z ~ N(0, 1).
              The Bayesian bootstrap (w x Exp(1)) is a second, design-free alternative.

Variants, each over the same draws:

    targets        targets perturbed, prior fixed
    targets_nested block-group targets perturbed, tract targets rebuilt from them (nested_Y)
    prior_rep      prior perturbed by replicate weights, targets fixed
    prior_bb       prior perturbed by Bayesian bootstrap
    both           targets and prior_rep together
    both_nested    targets_nested and prior_rep together

each solved to convergence (gradient 1e-6) from a warm start at the unperturbed solution, and also
from lambda = 0 stopped at a fixed iteration cap. The capped runs test whether an unfinished solve
is a usable source of variation: a cap leaves the answer partway between the prior and the fit.

Spread is measured exactly as in reps_by_constraints.py and theme_curve.py: the per-block-group
population CV across draws, median, over the ACS CV on the same block groups. Fit is measured as
z = (block-group population - published) / published SE, reported as the RMS over draws and
block groups, and as the share of |z| > 1.645 (outside the published 90% margin of error).

Run from data/UrbanPop/experiments with JAX_PLATFORMS=cuda,cpu.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jaxopt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composition  # noqa: E402
from gpu_solve import PUMAS, f, load, perturbed_Y  # noqa: E402

NREP = 80


def replicate_weights(fips, idx, wt, key, cache):
    """(80, n_donors) replicate weights aligned to est_ind, matched to livelike's donor weight."""
    import hashlib
    tag = hashlib.md5("|".join(map(str, idx)).encode()).hexdigest()[:8]
    path = os.path.join(cache, f"repwt_{fips}_{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    from livelike import acs

    def pull(kind, base):
        parts = []
        for lo, hi in ((1, 40), (41, 80)):
            v = [base] + [f"{base}{i}" for i in range(lo, hi + 1)]
            d = acs.extract_pums_descriptors(fips, kind, v, year=2019, censusapikey=key)
            d["SERIALNO"] = d["SERIALNO"].astype(str)
            if kind == "person":
                d = d.sort_values("SPORDER", key=pd.to_numeric)
            parts.append(d.drop_duplicates("SERIALNO").set_index("SERIALNO")[v[1:]])
        return pd.concat(parts, axis=1)

    h, p = pull("household", "WGTP"), pull("person", "PWGTP")
    ids = pd.Index(idx.astype(str))
    hw = h.reindex(ids).astype(float).to_numpy()
    pw = p.reindex(ids).astype(float).to_numpy()
    hw0 = acs.extract_pums_descriptors(fips, "household", ["WGTP"], year=2019, censusapikey=key)
    hw0["SERIALNO"] = hw0["SERIALNO"].astype(str)
    base = hw0.drop_duplicates("SERIALNO").set_index("SERIALNO")["WGTP"].reindex(ids)
    use_h = np.isclose(base.astype(float).to_numpy(), wt)
    reps = np.where(use_h[:, None], hw, pw).T
    assert np.isfinite(reps).all(), f"{fips}: unmatched replicate weights"
    os.makedirs(cache, exist_ok=True)
    np.save(path, reps)
    return reps


def nested_Y(p, rng):
    """Block-group targets perturbed; tract and PUMA targets rebuilt from them.

    perturbed_Y draws tract and block-group counts independently, so a tract's target and the sum
    of its block groups disagree by far more than they do in the published data, and a soft solve
    splits the difference -- which damps the perturbation. Here only block groups are drawn, and
    each tract keeps its published offset from its block groups' sum, so the two stay consistent.
    """
    y2 = np.maximum(p["est2"] + p["se2"] * rng.standard_normal(p["est2"].shape), 0)
    y1 = np.maximum(p["est1"] + p["A1"] @ (y2 - p["est2"]), 0)
    return np.concatenate([y2.sum(axis=0), y1.flatten("F"), y2.flatten("F")]) / p["N"]


def prior_rep(p, rng):
    w, R = p["wt"], p["repwt"]
    z = rng.standard_normal(NREP)
    return np.maximum(w + np.sqrt(4 / NREP) * (z @ (R - w)), 1e-3 * w)


def prior_bb(p, rng):
    return p["wt"] * rng.exponential(size=p["wt"].shape)


def logq_of(w, G):
    return np.log(w / w.sum() / G)[:, None] * np.ones((1, G))


def make_solver(maxiter):
    solver = jaxopt.LBFGS(fun=f, tol=1e-6, maxiter=maxiter, jit=True)

    @jax.jit
    def go(init, C, logq, A1, Y, V):
        r = solver.run(init, C, logq, A1, Y, V)
        return r.params, r.state.iter_num, r.state.error

    return go


def bg_pop(l, C, logq, A1, N):
    K = C.shape[1]
    T, G = A1.shape
    Leff = l[:K, None] + l[K:K + K * T].reshape(K, T) @ A1 + l[K + K * T:].reshape(K, G)
    E = logq - C @ Leff
    al = jnp.exp(E - jax.scipy.special.logsumexp(E)) * N
    return al.T @ C[:, 0], al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pumas", nargs="+", default=["3500804"])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--caps", type=int, nargs="+", default=[250, 500, 1000, 2000])
    ap.add_argument("--variants", nargs="+",
                    default=["targets", "prior_rep", "prior_bb", "both"])
    ap.add_argument("--no_capped", action="store_true")
    ap.add_argument("--selection", choices=["minimal", "expanded"], default="minimal")
    ap.add_argument("--cache", default=None, help="livelike cache (default per selection)")
    ap.add_argument("--moe_csv", default="/workspaces/ExaEpi/data/UrbanPop/acs_moe_35.csv")
    ap.add_argument("--out", default="./resolve_spread.jsonl")
    args = ap.parse_args()

    key = os.environ.get("CENSUS_API_KEY") or None
    dev = jax.devices()[0]
    print(f"device {dev.device_kind}, {args.draws} draws per variant\n", flush=True)
    moe = pd.read_csv(args.moe_csv, dtype={"geoid": str})
    moe = moe[moe.acs > 0].set_index("geoid")
    conv = make_solver(50000)
    capped = {c: make_solver(c) for c in args.caps}

    from livelike import config
    selection = (config.up_expanded_constraints_selection if args.selection == "expanded"
                 else None)
    cache = args.cache or {"minimal": "./llcache_minimal",
                           "expanded": "./llcache_expanded"}[args.selection]
    print(f"constraint set: {args.selection}\n" + composition.header() + "\n", flush=True)

    for fips in args.pumas:
        p = load(fips, key, cache, selection)
        pup = p["pup"]
        print(f"PUMA {fips}: {len(p['cols'])} constraints, dual {len(p['Y'])}", flush=True)
        p["repwt"] = replicate_weights(fips, pup.est_ind.index, p["wt"], key, "./repwt_cache")
        bgs = pup.est_g2.index.astype(str)
        acs_cv = (moe["se"] / moe["acs"]).reindex(bgs).to_numpy()
        pub, pse = p["est2"][:, 0], p["se2"][:, 0]
        G = p["A1"].shape[1]
        put = lambda a: jax.device_put(jnp.asarray(a, jnp.float64), dev)
        C, A1, V = put(p["C"]), put(p["A1"]), put(p["V"])
        lq0 = logq_of(p["wt"], G)

        def solve(fn, init, lq, Y):
            t0 = time.perf_counter()
            lam, it, err = fn(init, C, put(lq), A1, put(Y), V)
            pop, al = bg_pop(lam, C, put(lq), A1, p["N"])
            jax.block_until_ready(pop)
            return (lam, int(it), float(err), np.asarray(pop), np.asarray(al),
                    time.perf_counter() - t0)

        zero = put(np.zeros_like(p["Y"]))
        solve(conv, zero, lq0, p["Y"])  # compile
        lam0, it0, _, pop0, al0, t0 = solve(conv, zero, lq0, p["Y"])
        print(f"PUMA {fips}: {p['C'].shape[0]} donors x {G} block groups; unperturbed solve "
              f"{it0} iterations, {t0:.2f} s", flush=True)

        # Convergence curve on the unperturbed problem: households a cap places differently.
        curve = {}
        for c, fn in capped.items():
            solve(fn, zero, lq0, p["Y"])
            _, _, _, _, al, _ = solve(fn, zero, lq0, p["Y"])
            curve[c] = float(0.5 * np.abs(al - al0).sum() / al0.sum())
        print("  cap -> share of households placed differently from converged: "
              + ", ".join(f"{c}: {100 * v:.2f}%" for c, v in curve.items()), flush=True)

        def summarise(pops, als=None):
            pops = np.asarray(pops)
            m = pops.mean(0)
            cv = pops.std(0, ddof=1) / np.where(m > 0, m, np.nan)
            ok = np.isfinite(cv) & np.isfinite(acs_cv)
            z = (pops - pub) / np.where(pse > 0, pse, np.nan)
            zm = (m - pub) / np.where(pse > 0, pse, np.nan)
            return dict(cv=float(np.median(cv[ok])),
                        ratio=float(np.median(cv[ok]) / np.median(acs_cv[ok])),
                        z_rms=float(np.sqrt(np.nanmean(z ** 2))),
                        outside_moe=float(np.nanmean(np.abs(z) > 1.645)),
                        mean_bias_z=float(np.nanmedian(np.abs(zm))),
                        # households placed differently between consecutive draws, and from the
                        # unperturbed converged allocation
                        turnover=float(np.mean([0.5 * np.abs(a - b).sum() / b.sum()
                                                for a, b in zip(als[1:], als[:-1])]))
                        if als else float("nan"),
                        shift=float(np.mean([0.5 * np.abs(a - al0).sum() / al0.sum()
                                             for a in als])) if als else float("nan"),
                        **(composition.measure(als, p["C"], p["cols"], p["A1"], p["est1"],
                                               p["est2"], p["se1"], p["se2"]) if als else {}))

        base = summarise([pop0, pop0 * (1 + 1e-12)], [al0, al0 * (1 + 1e-12)])
        print(f"  unperturbed fit: z_rms {base['z_rms']:.3f}, outside MOE "
              f"{100 * base['outside_moe']:.1f}%, in_moe (all constraint cells) "
              f"{base['in_moe']:.4f}\n", flush=True)
        print(f"  {'variant':30s}{'CV':>8s}{'ratio':>8s}{'z rms':>8s}{'>MOE':>7s}"
              f"{'|bias| z':>9s}{'turnover':>9s}{'shift':>7s}{'iters':>13s}{'s/draw':>8s}", flush=True)

        draw_fns = {"targets": lambda r: (p["wt"], perturbed_Y(p, r)),
                    "targets_nested": lambda r: (p["wt"], nested_Y(p, r)),
                    "prior_rep": lambda r: (prior_rep(p, r), p["Y"]),
                    "prior_bb": lambda r: (prior_bb(p, r), p["Y"]),
                    "both": lambda r: (prior_rep(p, r), perturbed_Y(p, r)),
                    "both_nested": lambda r: (prior_rep(p, r), nested_Y(p, r))}
        rows = []
        for v in args.variants:
            rng = np.random.default_rng(7)
            draws = [draw_fns[v](rng) for _ in range(args.draws)]
            modes = [("converged, warm", conv, lam0)]
            if v == "targets":
                modes.append(("converged, cold", conv, zero))
            if not args.no_capped:
                modes += [(f"cap {c}, cold", capped[c], zero) for c in args.caps]
            for mode, fn, init in modes:
                res = [solve(fn, init, logq_of(w, G), Y) for w, Y in draws]
                s = summarise([r[3] for r in res], [r[4] for r in res])
                its = [r[1] for r in res]
                s.update(puma=fips, variant=v, mode=mode, iters_min=min(its),
                         iters_max=max(its), s_per_draw=float(np.mean([r[5] for r in res])),
                         worst_grad=max(r[2] for r in res))
                if mode == "converged, cold":
                    warm = rows[-1]["_pops"]
                    s["warm_vs_cold_max_rel"] = float(np.max(np.abs(
                        np.asarray([r[3] for r in res]) - warm) / np.maximum(warm, 1)))
                if mode == "converged, warm":
                    conv_als = [r[4] for r in res]
                else:
                    # households this draw places differently from the same draw solved to
                    # convergence: the cap's own error on a perturbed problem
                    s["self_moved"] = float(np.mean([0.5 * np.abs(r[4] - a).sum() / a.sum()
                                                     for r, a in zip(res, conv_als)]))
                s["_pops"] = np.asarray([r[3] for r in res])
                rows.append(s)
                print(f"  {(v + ': ' + mode):30s}{s['cv']:8.4f}{s['ratio']:8.3f}"
                      f"{s['z_rms']:8.3f}{100 * s['outside_moe']:6.1f}%{s['mean_bias_z']:9.3f}"
                      f"{100 * s['turnover']:8.2f}%{100 * s['shift']:6.2f}%"
                      f"{s['iters_min']:>6d}-{s['iters_max']:<6d}{s['s_per_draw']:8.2f}"
                      + (f"   vs own converged {100 * s['self_moved']:.2f}%"
                         if "self_moved" in s else ""),
                      flush=True)
                print(f"  {'':30s}{composition.line(s)}", flush=True)
                if "warm_vs_cold_max_rel" in s:
                    print(f"  {'':30s}warm vs cold start: max relative difference in block-"
                          f"group population {s['warm_vs_cold_max_rel']:.1e}", flush=True)
            print(flush=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(dict(puma=fips, curve=curve, base=base,
                                     rows=[{k: v for k, v in r.items() if k != "_pops"}
                                           for r in rows])) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
