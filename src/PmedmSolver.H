/*! @file PmedmSolver.H
    \brief Per-run P-MEDM re-solve: the solver that turns one PUMA's (perturbed) problem into
    expected households per (donor, block group). The problem and its perturbation are in
    PmedmProblem.H.

    A port of utilities/UrbanPop-scripts/popgen/solver.py, which is normative. The solver does not
    match it bit for bit (different reduction orders and GEMM kernels): it is checked
    statistically, each PUMA's allocation within 0.1% of households of the float64 converged
    solution (utilities/tests/pmedm_check.py).

    The dual solved, as in pymedm:

        f(lam) = Y . lam + logsumexp(E) + 1/2 lam . (V * lam),   E = log q - C Leff(lam)
        Leff(lam)[k, g] = lam0[k] + lam1[k, tract(g)] + lam2[k, g]

    lam0 (K), lam1 (K x T) and lam2 (K x G) are the PUMA-, tract- and block-group-level prices,
    laid out [lam0 ; lam1 row-major ; lam2 row-major], which is also the layout of Y and V
    (constraint-major, pymedm's "F" order). The allocation is N softmax(E).

    Solver settings (popgen/solver.py has the measurements behind each):
      * compact L-BFGS, history 10;
      * E held in float64 and updated along each search direction, E <- E - t D with D = C Leff(d)
        in float32, recomputed exactly (float64) every 100 iterations;
      * float32 GEMMs at full precision (DenseLinAlg; never TF32);
      * Armijo line search on the objective's decrease, t (Y + V lam).d + t^2/2 d.Vd +
        log1p(sum w expm1(-t D)), full step first, else the first of 2^-1 .. 2^-11 that passes;
      * stop once the allocation moves less than tol_moved of households over check_every
        iterations, or at max_iter.

    Everything per iteration stays on the device -- the line-search choice and the 10 x 10 L-BFGS
    algebra run as single-thread kernels -- and every reduction is two-pass with a block count
    fixed by the problem size, so a solve is bitwise reproducible for a given device type and
    shape. The host synchronises once per check_every iterations. On CUDA each iteration is
    recorded once as a graph and replayed. Two choices keep the float64 work small on GPUs with
    slow float64: the history's Gram matrix is updated in place when a pair enters (one row and
    column), and the eleven shorter line-search steps are evaluated only when the full step fails.
*/
#ifndef PMEDM_SOLVER_H_
#define PMEDM_SOLVER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "PmedmProblem.H"

namespace PopGen {

struct PmedmSettings {
    int check_every = 250;   /*!< iterations between convergence checks */
    double tol_moved = 1e-3; /*!< stop when less than this share of households moved per check */
    int max_iter = 6000;
};

struct PmedmResult {
    std::vector<double> allocation; /*!< D x G expected households, float64 */
    int iterations = 0;
    double grad_norm = 0;           /*!< |gradient| at the returned prices, float64 */
};

class PmedmSolver {
  public:
    explicit PmedmSolver (PmedmSettings settings = {}) : m_settings(settings) {}

    /*! Solve one PUMA for targets Y (nDual) and per-donor log prior logq (D). */
    PmedmResult solve(const PmedmProblem& prob, const std::vector<double>& Y, const std::vector<double>& logq) const;

    const PmedmSettings& settings () const noexcept { return m_settings; }

  private:
    PmedmSettings m_settings;
};

} // namespace PopGen

#endif
