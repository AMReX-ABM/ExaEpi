/*! @file PmedmSolver.H
    \brief Per-run P-MEDM re-solve: one PUMA's problem from the bundle, its perturbation, and the
    solver that turns it into expected households per (donor, block group).

    A port of utilities/UrbanPop-scripts/popgen/problem.py, perturb.py and solver.py, which are
    normative. The problem assembly and the keyed perturbation draws follow them exactly; they use
    log, exp, cos and sqrt, so C++ matches Python to rounding rather than bit for bit. The solver
    does not match bit for bit either (different reduction orders and GEMM kernels): it is checked
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
    shape. The host synchronises once per check_every iterations.
*/
#ifndef PMEDM_SOLVER_H_
#define PMEDM_SOLVER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "PopulationBundle.H"

namespace PopGen {

/*! One PUMA's published and survey inputs plus the quantities pymedm derives from them. */
struct PmedmProblem {
    std::string puma;                        /*!< PUMA code as in the bundle, e.g. "3500804" */
    std::int64_t fips = 0;                   /*!< the same code as an integer: the draws' key word */
    int K = 0, T = 0, G = 0, D = 0;          /*!< constraints, tracts, block groups, donors */
    std::vector<std::int64_t> bg_geoid;      /*!< G */
    std::vector<int> bg_tract;               /*!< G: tract index within the PUMA */
    std::vector<int> donor_index;            /*!< D: row in donors.*, -1 if unmatched */
    std::vector<int> donor_hh_size;          /*!< D: livelike's household member count */
    std::vector<double> C;                   /*!< D x K donor constraint matrix */
    std::vector<double> wt;                  /*!< D prior (survey) weights */
    std::vector<double> est_bg, se_bg;       /*!< G x K */
    std::vector<double> est_tract, se_tract; /*!< T x K */
    std::vector<double> V;                   /*!< nDual() target variances, scaled as pymedm does */
    double N = 0;                            /*!< sum of the prior weights */

    PmedmProblem() = default;
    /*! PUMA p of the bundle (problem.py, PumaProblem). */
    PmedmProblem(const PopulationBundle& b, int p);

    int nDual () const noexcept { return K * (1 + T + G); }
};

/*! Number of PUMAs in the bundle. */
int pumaCount(const PopulationBundle& b);

/*! pymedm's Y from block-group (G x K) and tract (T x K) estimates, which may be perturbed. */
std::vector<double> pmedmTargets(const PmedmProblem& prob, const std::vector<double>& est_bg,
                                 const std::vector<double>& est_tract);

/*! Y for this run: block-group estimates redrawn as moment-matched log-normals from their
    standard errors, tracts rebuilt from them (perturb.py, perturbed_targets). If est_bg_out is
    given it receives the perturbed block-group estimates. */
std::vector<double> perturbedTargets(const PmedmProblem& prob, std::int64_t seed, std::int64_t rep,
                                     std::vector<double>* est_bg_out = nullptr);

/*! Per-donor log prior for this run: survey weights times Exp(1), spread evenly over the block
    groups (perturb.py, perturbed_prior and log_prior). log q is the same for every block group. */
std::vector<double> perturbedLogPrior(const PmedmProblem& prob, std::int64_t seed, std::int64_t rep);

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
