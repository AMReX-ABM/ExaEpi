/*! @file PmedmProblem.H
    \brief One PUMA's P-MEDM problem from the population bundle, and its per-run perturbation.

    A port of utilities/UrbanPop-scripts/popgen/problem.py and perturb.py, which are normative. The
    assembly follows them exactly; the keyed perturbation draws use log, exp, cos and sqrt, so C++
    matches Python to rounding rather than bit for bit (utilities/tests/pmedm_check.py: ~1e-15).
    Host-only and AMReX-free, like PopulationBundle, so the population stages that need the problem
    (PopulationGenerator) build and test standalone.
*/
#ifndef PMEDM_PROBLEM_H_
#define PMEDM_PROBLEM_H_

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

} // namespace PopGen

#endif
