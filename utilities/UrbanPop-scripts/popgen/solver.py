"""The P-MEDM solver for per-run re-solves: normative reference for src/PmedmSolver.

Solves pymedm's dual

    f(lam) = Y . lam + logsumexp(E) + 1/2 lam . (V * lam),     E = log q - C @ Leff(lam)
    Leff(lam) = lam0[:, None] + lam1 @ A1 + lam2               (K x G)

where lam0 (K), lam1 (K x T) and lam2 (K x G) are the PUMA-, tract- and block-group-level prices,
and returns the allocation N * softmax(E): expected households per (donor, block group).

Chosen settings, each measured on all 18 New Mexico PUMAs (experiments/solver_options.py,
fp32_capped.py):

  * L-BFGS, compact representation (Byrd, Nocedal & Schnabel 1994), history 10.
  * E is held in float64 and updated incrementally along each search direction, E <- E - t D with
    D = C @ Leff_lin(d); recomputed exactly every REFRESH iterations so float32 error in D cannot
    accumulate. The two matrix products per iteration (D, and C^T w for the gradient) are float32
    at full precision -- never TF32.
  * Armijo line search on the objective's DECREASE,
        f(t) - f(0) = t (Y + V lam).d + t^2/2 d.V d + log1p(sum w expm1(-t D)),
    which stays accurate however small the decrease; a float32 absolute objective cannot see
    late-solve decreases at all (pure float32 stalls ~4.5% from converged). The full step is tried
    first (accepted 90-95% of iterations); otherwise the candidates 2^-1 .. 2^-11 at once.
  * Stop when the allocation moved less than TOL_MOVED of households over the last CHECK_EVERY
    iterations, or at MAX_ITER. Every NM PUMA lands within 0.07% of float64 converged, at ~9 s per
    NM realization on an RTX 5070 laptop GPU.
"""

import functools
import os

# XLA autotunes GPU kernels by timing candidates, so two processes can pick different matrix-
# product algorithms, round differently, and -- because a capped solve does not run to the exact
# minimum -- produce different populations from the same seed (measured: 2,073,366 vs 2,073,800
# persons for NM seed 1). Fix the kernel choice and require deterministic reductions. This must be
# set before JAX initialises its GPU backend.
_DETERMINISTIC = "--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
if "xla_gpu_autotune_level" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _DETERMINISTIC).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.scipy.linalg import solve_triangular  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
# Each PUMA shape compiles once (~15 s); keep the compiled solvers across runs.
jax.config.update("jax_compilation_cache_dir",
                  os.environ.get("POPGEN_JAX_CACHE", os.path.expanduser("~/.cache/popgen-jax")))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

HISTORY = 10
REFRESH = 100
C1 = 1e-4
N_STEPS = 12
CHECK_EVERY = 250
TOL_MOVED = 1e-3
MAX_ITER = 6000


@functools.lru_cache(maxsize=None)
def make_solver(K, T, G, check_every=CHECK_EVERY, tol_moved=TOL_MOVED, max_iter=MAX_ITER,
                work_dtype=jnp.float32):
    """A jitted solve(C, logq, A1, Y, V) -> (lam, iterations, |grad|) for one problem shape."""
    f64, wd = jnp.float64, work_dtype

    def leff(v, A1):
        return v[:K, None] + v[K:K + K * T].reshape(K, T) @ A1 + v[K + K * T:].reshape(K, G)

    def exact_E(l, C, logq, A1):
        return logq - C @ leff(l, A1)

    def weights(E):
        z = jnp.exp((E - E.max()).astype(wd))
        return z / z.sum(dtype=f64).astype(wd)

    def grad(l, w, Cw, A1, Y, V):
        dL = (Cw.T @ w).astype(f64)                                     # K x G
        return Y + V * l - jnp.concatenate([dL.sum(1), (dL @ A1.T).ravel(), dL.ravel()])

    def direction(g, S, Yh, n):
        """-H g from the compact L-BFGS form; unused history rows are exactly zero."""
        valid = jnp.arange(HISTORY) >= HISTORY - n
        SY, YY = S @ Yh.T, Yh @ Yh.T
        R = jnp.where(valid[:, None] & valid[None, :], jnp.triu(SY), 0.0)
        R = R + jnp.diag(jnp.where(valid, 0.0, 1.0))
        Dg = jnp.where(valid, jnp.diag(SY), 0.0)
        sy, yy = SY[-1, -1], YY[-1, -1]
        gamma = jnp.where((n > 0) & (yy > 0), sy / jnp.where(yy > 0, yy, 1.0), 1e-3)
        a, b = S @ g, gamma * (Yh @ g)
        u = solve_triangular(R, a, lower=False)
        top = solve_triangular(R.T, Dg * u + gamma * (YY @ u) - b, lower=True)
        return -(gamma * g + S.T @ top - gamma * (Yh.T @ u))

    @jax.jit
    def solve(C, logq, A1, Y, V):
        Cw = C.astype(wd)
        n_dual = Y.shape[0]
        ts = 0.5 ** jnp.arange(N_STEPS, dtype=f64)
        l0 = jnp.zeros(n_dual, f64)
        E = exact_E(l0, C, logq, A1)
        w = weights(E)
        g = grad(l0, w, Cw, A1, Y, V)
        state = (l0, E, w, g, jnp.zeros((HISTORY, n_dual), f64),
                 jnp.zeros((HISTORY, n_dual), f64), jnp.int32(0), jnp.int32(0))

        def step(st):
            l, E, w, g, S, Yh, n, it = st
            d = direction(g, S, Yh, n)
            gd = g @ d
            d = jnp.where(gd < 0, d, -g)                                # fall back to steepest
            gd = jnp.where(gd < 0, gd, -(g @ g))
            D = Cw @ leff(d, A1).astype(wd)                             # donors x G
            lin, quad = (Y + V * l) @ d, 0.5 * (d @ (V * d))

            def decrease(t):
                x = jnp.sum(w * jnp.expm1(-t.astype(wd) * D), dtype=f64)
                return t * lin + t * t * quad + jnp.log1p(x)

            def smaller_steps():
                cand = ts[1:]
                ok = jax.vmap(decrease)(cand) <= C1 * cand * gd
                return jnp.where(ok.any(), cand[jnp.argmax(ok)], cand[-1])

            t = jax.lax.cond(decrease(ts[0]) <= C1 * ts[0] * gd, lambda: ts[0], smaller_steps)
            l_new = l + t * d
            E_new = jax.lax.cond((it + 1) % REFRESH == 0,
                                 lambda: exact_E(l_new, C, logq, A1),
                                 lambda: E - t * D.astype(f64))
            w_new = weights(E_new)
            g_new = grad(l_new, w_new, Cw, A1, Y, V)
            s, y = t * d, g_new - g
            keep = (s @ y) > 1e-12 * jnp.sqrt((s @ s) * (y @ y))
            S = jnp.where(keep, jnp.roll(S, -1, 0).at[-1].set(s), S)
            Yh = jnp.where(keep, jnp.roll(Yh, -1, 0).at[-1].set(y), Yh)
            n = jnp.where(keep, jnp.minimum(n + 1, HISTORY), n)
            return l_new, E_new, w_new, g_new, S, Yh, n, it + 1

        def cond(c):
            st, moved = c
            return (moved > tol_moved) & (st[7] < max_iter)

        def body(c):
            st, _ = c
            w_prev = st[2]
            st = jax.lax.fori_loop(0, check_every, lambda i, s_: step(s_), st)
            # w is normalised, so half its L1 change is the share of households that moved.
            return st, 0.5 * jnp.sum(jnp.abs(st[2] - w_prev), dtype=f64)

        state, _ = jax.lax.while_loop(cond, body, (state, jnp.array(jnp.inf, f64)))
        l, it = state[0], state[7]
        E = exact_E(l, C, logq, A1)
        w = jnp.exp(E - jax.scipy.special.logsumexp(E))
        return l, it, jnp.linalg.norm(grad(l, w, C, A1, Y, V))

    return solve


def allocation(prob, lam, logq):
    """Expected households per (donor, block group), float64: N * softmax(E)."""
    K, T, G = prob.K, prob.T, prob.G
    lam = np.asarray(lam, dtype=np.float64)
    L = lam[:K, None] + lam[K:K + K * T].reshape(K, T) @ prob.A1 + lam[K + K * T:].reshape(K, G)
    E = logq - prob.C @ L
    E -= E.max()
    w = np.exp(E)
    return w / w.sum() * prob.N


def solve(prob, Y, logq, **kw):
    """Solve one PUMA; returns (allocation D x G, iterations, final gradient norm)."""
    fn = make_solver(prob.K, prob.T, prob.G, **kw)
    lam, it, gn = fn(jnp.asarray(prob.C), jnp.asarray(logq), jnp.asarray(prob.A1),
                     jnp.asarray(Y), jnp.asarray(prob.V))
    return allocation(prob, lam, np.asarray(logq)), int(it), float(gn)
