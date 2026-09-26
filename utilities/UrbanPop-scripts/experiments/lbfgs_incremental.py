"""A fixed-iteration L-BFGS for the P-MEDM dual that keeps float32 out of the line search.

jaxopt's L-BFGS re-evaluates the full objective at every line-search trial. In float32 that fails:
late in the solve the decrease per step is below float32 rounding, and the line search retries
(~28 evaluations per iteration measured, against ~2 in float64). Computing only the value in
float64 fixes the accuracy but keeps a float64 matrix product per trial, so it gains just 1.3x.

This exploits the structure instead. The exponent matrix E = logq - C @ Leff(lambda) is linear in
lambda, so along a search direction d

    E(lambda + t d) = E(lambda) - t D,        D = C @ Leff_lin(d)

and the line search needs only the DECREASE of the objective along d, which with the normalised
weights w = softmax(E) is

    f(t) - f(0) = t (Y + V lambda).d + t^2/2 d.V d + log1p( sum w expm1(-t D) )

That form is accurate relative to the decrease itself, however small, so it can be evaluated in
float32 (with float64 accumulation) without the rounding floor that sinks a float32 absolute
objective. All candidate steps 1, 1/2, ..., 2^-11 are evaluated at once and the first satisfying
Armijo is taken.

Per iteration: two float32 matrix products (D, and C^T w for the gradient), float32 exponentials,
and a float64 incremental update E <- E - t D of plain additions, with E recomputed exactly every
`refresh` iterations so float32 error in D cannot accumulate. The L-BFGS direction uses the
compact representation (Byrd, Nocedal & Schnabel 1994) -- a few small dense products rather than
2 x history dot products, each of which is a separate GPU kernel.

Measured on an RTX 5070 laptop GPU, per iteration: float64 GEMM 132 us vs float32 17 us; one
float64 logsumexp 47 us (consumer GPUs run float64 transcendentals slowly too); the unrolled
two-loop recursion 109 us, almost all kernel launches.

This is the shape a compiled port would take. `work_dtype=float64` runs the same algorithm with no
float32 anywhere, to separate the effect of the algorithm from that of the precision.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def split(v, K, T, G):
    return v[:K], v[K:K + K * T].reshape(K, T), v[K + K * T:].reshape(K, G)


def make_incremental_solver(maxiter, K, T, G, work_dtype=jnp.float32, history=10, refresh=100,
                            c1=1e-4, n_steps=12, linesearch="vector", check_every=None,
                            tol_moved=None, diagnostics=False):
    """linesearch: "vector" evaluates all n_steps candidates every iteration; "first" tries the
    full step alone and evaluates the rest only if it is rejected. With check_every set, stop once
    the allocation moves less than tol_moved (share of households) between checks, or at maxiter.
    """
    f64, wd = jnp.float64, work_dtype

    def leff(v, A1):
        v0, v1, v2 = split(v, K, T, G)
        return v0[:, None] + v1 @ A1 + v2

    def exact_E(l, C64, logq, A1):
        return logq - C64 @ leff(l, A1)

    def weights(E):
        z = jnp.exp((E - E.max()).astype(wd))
        return z / z.sum(dtype=f64).astype(wd)

    def grad(l, w, Cw, A1, Y, V):
        dL = (Cw.T @ w).astype(f64)                                     # K x G
        return Y + V * l - jnp.concatenate([dL.sum(1), (dL @ A1.T).ravel(), dL.ravel()])

    def direction(g, S, Yh, n):
        """-H g from the compact L-BFGS form; unused history rows are exactly zero."""
        valid = jnp.arange(history) >= history - n
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
    def run(init, C64, logq, A1, Y, V):
        Cw = C64.astype(wd)
        n_dual = init.shape[0]
        ts = 0.5 ** jnp.arange(n_steps, dtype=f64)
        E = exact_E(init, C64, logq, A1)
        w = weights(E)
        g = grad(init, w, Cw, A1, Y, V)
        # it counts iterations; hist[k] counts iterations that accepted step 2^-k (last bin:
        # none accepted, smallest step taken anyway).
        state = (init, E, w, g, jnp.zeros((history, n_dual), f64),
                 jnp.zeros((history, n_dual), f64), jnp.int32(0), jnp.int32(0),
                 jnp.zeros(n_steps + 1, jnp.int32))

        def step(st):
            l, E, w, g, S, Yh, n, it, hist = st
            d = direction(g, S, Yh, n)
            gd = g @ d
            d = jnp.where(gd < 0, d, -g)                                # fall back to steepest
            gd = jnp.where(gd < 0, gd, -(g @ g))
            D = Cw @ leff(d, A1).astype(wd)                             # donors x G
            lin, quad = (Y + V * l) @ d, 0.5 * (d @ (V * d))

            def decrease(t):
                x = jnp.sum(w * jnp.expm1(-t.astype(wd) * D), dtype=f64)
                return t * lin + t * t * quad + jnp.log1p(x)

            def vector_search(lo):
                # Every candidate step from ts[lo] down, at once; the first satisfying Armijo.
                cand = ts[lo:]
                ok = jax.vmap(decrease)(cand) <= c1 * cand * gd
                k = jnp.where(ok.any(), jnp.argmax(ok), n_steps - lo)
                return (jnp.where(ok.any(), cand[jnp.minimum(k, len(cand) - 1)], cand[-1]),
                        (k + lo).astype(jnp.int32))

            if linesearch == "first":
                # The full step alone first; the other candidates only if it is rejected.
                ok1 = decrease(ts[0]) <= c1 * ts[0] * gd
                t, k = jax.lax.cond(ok1, lambda: (ts[0], jnp.int32(0)),
                                    lambda: vector_search(1))
            else:
                t, k = vector_search(0)
            hist = hist.at[k].add(1)
            l_new = l + t * d
            E_new = jax.lax.cond((it + 1) % refresh == 0,
                                 lambda: exact_E(l_new, C64, logq, A1),
                                 lambda: E - t * D.astype(f64))
            w_new = weights(E_new)
            g_new = grad(l_new, w_new, Cw, A1, Y, V)
            s, y = t * d, g_new - g
            keep = (s @ y) > 1e-12 * jnp.sqrt((s @ s) * (y @ y))
            S = jnp.where(keep, jnp.roll(S, -1, 0).at[-1].set(s), S)
            Yh = jnp.where(keep, jnp.roll(Yh, -1, 0).at[-1].set(y), Yh)
            n = jnp.where(keep, jnp.minimum(n + 1, history), n)
            return l_new, E_new, w_new, g_new, S, Yh, n, it + 1, hist

        if check_every is None:
            state = jax.lax.fori_loop(0, maxiter, lambda i, st: step(st), state)
        else:
            # Stopping rule: every check_every iterations, the share of households the
            # allocation moved since the previous check (w is normalised, so half its L1 change
            # is that share); stop below tol_moved or at maxiter. Depends only on this solve's
            # own arithmetic, so it is as deterministic as a fixed count.
            def cond(c):
                st, moved = c
                return (moved > tol_moved) & (st[7] < maxiter)

            def body(c):
                st, _ = c
                w_prev = st[2]
                st = jax.lax.fori_loop(0, check_every, lambda i, s_: step(s_), st)
                moved = 0.5 * jnp.sum(jnp.abs(st[2] - w_prev), dtype=f64)
                return st, moved

            state, _ = jax.lax.while_loop(cond, body, (state, jnp.array(jnp.inf, f64)))
        l, it, hist = state[0], state[7], state[8]
        # Report the exact float64 gradient norm, not the incrementally tracked one.
        E = exact_E(l, C64, logq, A1)
        w = jnp.exp(E - jax.scipy.special.logsumexp(E))
        gn = jnp.linalg.norm(grad(l, w, C64, A1, Y, V))
        return (l, it, gn, hist) if diagnostics else (l, it, gn)

    return run
