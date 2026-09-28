/*! @file PopulationGenerator.H
    \brief From solved P-MEDM allocations to a complete ExaEpi population: placement of whole
    households, and stages S0-S10 (persons, destinations, groups).

    A bit-for-bit port of utilities/UrbanPop-scripts/popgen/ (placement.py, persons.py,
    workers.py, commute.py, cbp.py, students.py, teachers.py, groups.py), which are normative.
    Given the same allocations, every stage's output must equal the Python reference generator's
    exactly; the check compares per-stage digests with generate_exaepi.py's <bin>.digests.json
    (utilities/tests/popgen_stages_check.cpp, with allocations injected via --save-alloc).

    What makes bit-for-bit possible, and must be kept when changing this file:
      * every random decision is a KR64 draw keyed on global identities (KeyedRNG.H), never a
        position in a stream;
      * float sums that affect results run sequentially in the same order as numpy.cumsum /
        numpy.bincount (never pairwise), and the file is compiled with -ffp-contract=off;
      * ties in sorts are broken as numpy's stable lexsort breaks them;
      * counts, roundings and grid cuts are integer arithmetic;
      * the only non-arithmetic operations are floor and sqrt (correctly rounded): anything that
        needs exp, log or trigonometry is tabulated in the bundle (commute.*, geo.xyz).

    Host-only and AMReX-free, so the stage check builds standalone with a plain compiler.
*/
#ifndef POPULATION_GENERATOR_H_
#define POPULATION_GENERATOR_H_

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "KeyedRNG.H"
#include "PmedmProblem.H"
#include "PopulationBundle.H"

namespace PopGen {

/*! Whole households placed in block groups: household count per (block group, donor). Each block
    group's rows are contiguous and in the order placement emits them. */
struct Placements {
    std::vector<std::int64_t> bg;    /*!< block-group geoid */
    std::vector<std::int64_t> donor; /*!< global donor household row (donors.*) */
    std::vector<std::int64_t> count; /*!< copies of that household */

    void append(const Placements& o);
    std::size_t size () const noexcept { return bg.size(); }
};

/*! Placement for one PUMA (placement.py, place_puma): household reweighting, one TRS over the
    PUMA's (household category x block group) matrix, donor draws per (block group, category).
    al is the D x G allocation; cols the bundle's constraint names. */
Placements placePuma(const PopulationBundle& b, const PmedmProblem& prob, const std::vector<double>& al,
                     const std::vector<std::string>& cols, std::int64_t seed, std::int64_t rep);

/*! The person table, in canonical order: sorted by (bg, h, p), agent id = position. */
struct Persons {
    std::vector<std::int64_t> bg, h, p, src;            /*!< identity; src = row in donors.* person columns */
    std::vector<std::int16_t> age, naics, grade, jwmnp; /*!< jwmnp: reported travel time to work, minutes */
    std::vector<std::int8_t> sex, race, travel, veh_occ;
    std::vector<std::uint8_t> employed, student;

    std::size_t size () const noexcept { return bg.size(); }
};

/*! Expansion to persons and stages S0-S2 (placement.expand, persons.build): attributes from the
    donors, childcare (S1), worker/student split (S2). */
Persons buildPersons(const PopulationBundle& b, const Placements& pl, std::int64_t seed, std::int64_t rep);

/*! CBP reference tables (cbp.py, SizeTables): workgroup-size targets and deterministic
    establishment-size tables, 256 log-stratified sizes per (state, NAICS, size band). */
class SizeTables {
  public:
    static constexpr int POINTS = 256;
    explicit SizeTables(const PopulationBundle& b);

    std::int64_t target(std::int64_t state, std::int64_t naics) const;
    bool hasEst(std::int64_t state, std::int64_t naics) const;
    /*! One establishment size for key k: band by int_cdf on draw 0, stratum by index on draw 1. */
    std::int64_t sample(std::int64_t state, std::int64_t naics, const KR64& k) const;
    /*! Exact mean establishment size. */
    double mean(std::int64_t state, std::int64_t naics) const;

  private:
    struct Est {
        std::vector<std::int64_t> cum;    /*!< running sum of establishments per band */
        std::vector<std::int64_t> sizes;  /*!< bands x POINTS */
        std::vector<std::int64_t> counts; /*!< establishments per band */
    };
    std::int64_t m_default = 0;
    std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> m_targets;
    std::map<std::pair<std::int64_t, std::int64_t>, Est> m_est;
};

struct WorkerStats {
    std::int64_t workers = 0, unplaceable = 0, fallback = 0, repair_final = 0, cells = 0, one_worker_cells = 0;
};

/*! How a stage may split its work over MPI ranks (this file itself knows nothing of MPI): this
    rank, the rank count, and an all-gather that returns every rank's buffer concatenated in rank
    order. Collective: every rank must make the same calls. Results never depend on the split. */
struct Partition {
    int rank = 0, nranks = 1;
    std::function<std::vector<std::int64_t>(const std::vector<std::int64_t>&)> allgather;
};

/*! S3 (workers.py, allocate): work geoid per person -- the home geoid for non-workers and for
    workers who work from home. With a Partition, the per-industry fills are split over its ranks
    (collective). */
std::vector<std::int64_t> allocateWorkers(const PopulationBundle& b, const Persons& P, const SizeTables& tables,
                                          std::int64_t seed, std::int64_t rep, WorkerStats* stats = nullptr,
                                          const Partition* part = nullptr);

/*! S4 (students.py, allocate): school row per person (-1 if none). Sets work to the school's
    geoid for placed students; students left unplaced leave school (grade -1). students and
    unplaced, if given, receive the counts per level (P E M H U C). */
std::vector<std::int64_t> allocateStudents(const PopulationBundle& b, Persons& P, std::vector<std::int64_t>& work,
                                           std::int64_t seed, std::int64_t rep,
                                           std::map<std::string, std::pair<std::int64_t, std::int64_t>>* stats = nullptr);

/*! S5 (teachers.py, allocate): teachers to schools that received students; updates school, work
    and grade for the chosen workers. stats, if given, receives (required, placed) per type. */
void allocateTeachers(const PopulationBundle& b, Persons& P, std::vector<std::int64_t>& work, std::vector<std::int64_t>& school,
                      std::int64_t seed, std::int64_t rep,
                      std::map<std::string, std::pair<std::int64_t, std::int64_t>>* stats = nullptr);

/*! Group structure from stages S6-S10 (groups.py), one entry per person. */
struct Groups {
    std::vector<std::int64_t> school_id;             /*!< S6: 1 + dense rank of the school at its block group; 0 = none */
    std::vector<std::int64_t> nborhood, hh_cluster;  /*!< S7 */
    std::vector<std::int64_t> workgroup, work_group; /*!< S8: team within (work bg, NAICS) (0 = none), dense id (-1) */
    std::vector<std::int64_t> school_class, school_class_group; /*!< S9 (0 / -1 for anyone not at a school) */
    std::vector<std::int64_t> work_nborhood;                    /*!< S10: dense id per daytime block group */
};

/*! Stages S6-S10 (groups.py: school_ids, home_groups, work_groups, school_groups,
    day_neighborhoods). If digests is given, each stage's digest is added under its name. */
Groups assignGroups(const PopulationBundle& b, const Persons& P, const std::vector<std::int64_t>& work,
                    const std::vector<std::int64_t>& school, const SizeTables& tables, std::int64_t seed, std::int64_t rep,
                    std::map<std::string, std::string>* digests = nullptr);

/*! The digests generate_exaepi.py writes to <bin>.digests.json, by the same names. */
std::string placementDigest(const Placements& pl);
std::string personsDigest(const Persons& P);

} // namespace PopGen

#endif
