/*! @file GeneratePopulation.H
    \brief Generate a synthetic population in-process from a population bundle.

    The whole pipeline, per run: for each PUMA, perturb the P-MEDM inputs within their sampling
    error and re-solve (PmedmProblem, PmedmSolver), place whole households, then stages S0-S10
    (PopulationGenerator). The result is the same block-group index and per-block-group agent
    frames the UrbanPop .bin holds, so UrbanPopData uses it exactly as it would a file.

    Parallel layout: PUMAs are split over ranks by a deterministic longest-processing-time rule
    on donors x block groups; each rank solves and places its own, and the placements are
    all-gathered. Within a rank, PUMAs run concurrently on OpenMP threads: one per thread on
    CPU, and on GPU up to gpu_streams at a time, each on its own stream. Every rank then runs
    S0-S10 on the whole population (deterministic host code), so the population is identical
    whatever the rank count. A given (bundle, seed, rep) always gives the same population on a
    given device type; GPU and CPU solves round differently, so the device type is part of the
    run's configuration, like the seed.
*/
#ifndef GENERATE_POPULATION_H_
#define GENERATE_POPULATION_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "PmedmSolver.H"

namespace PopGen {

struct GenerationSettings {
    std::string bundle;             /*!< population bundle (UPPB v2) */
    std::int64_t seed = 0;          /*!< population seed */
    std::int64_t rep = 0;           /*!< replicate number, a second key word */
    PmedmSettings solver;           /*!< solver stopping rule */
    int gpu_streams = 1;            /*!< GPU builds: PUMAs solved concurrently, each on its own stream */
    std::string inject_allocations; /*!< debug: solved allocations to use instead of solving
                                         (generate_exaepi.py --save-alloc), for bit-for-bit checks */
    int verbose = 0;
};

/*! One block group of the generated population, as in the .bin index plus its agent frame. */
struct GeneratedBlockGroup {
    std::int64_t geoid = 0;
    int home_population = 0;
    std::vector<int> work_populations; /*!< [total workers, workers per NAICS ...] */
    std::vector<char> frame;           /*!< home_population agents, columnar as in a .bin frame */
};

struct GeneratedPopulation {
    /*! Work-only block groups first, then home block groups, each set in ascending geoid -- the
        order upop_to_exaepi.py writes the .bin index in. */
    std::vector<GeneratedBlockGroup> block_groups;
    std::int64_t num_agents = 0;
    std::map<std::string, std::string> digests; /*!< per-stage digests, as generate_exaepi.py's */
    std::string digest;                         /*!< one digest of the whole population's agent records */
};

/*! Generate the population; collective over all MPI ranks. naics_count is the NAICS table size
    ExaEpi expects, checked against the bundle. */
GeneratedPopulation generatePopulation(const GenerationSettings& settings, int naics_count);

/*! Write a generated population as an UrbanPop .bin (format v4, deflated frames) that ExaEpi and
    the UrbanPop tools read like any other. Call on one rank. */
void writePopulationBin(const GeneratedPopulation& pop, const std::string& path, std::uint32_t format_version,
                        std::uint32_t record_size);

} // namespace PopGen

#endif
