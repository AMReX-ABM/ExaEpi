/*! @file Utils.H
    \brief Defines the #ExaEpi::Utils namespace
*/

#ifndef UTILS_H_
#define UTILS_H_

#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <vector>

#include <map>
#include <string>

namespace ExaEpi {

/*! \brief Structure with various simulation parameters; these are read in
    for the input file provided in the command line.
*/
struct TestParams {
    amrex::IntVect size;                                  /*!< domain size (see ExaEpi::Utils::getGeometry) */
    int max_box_size = 16;                                /*!< box size */
    int nsteps = 1;                                       /*!< number of simulation steps */
    int plot_int = -1;                                    /*!< plot interval (see ExaEpi::IO::writePlotFile) */
    int check_int = -1;                                   /*!< checkpoint interval (see ExaEpi::IO::writeCheckpointFile) */
    int random_travel_int = -1;                           /*!< steps between random travel events
                                                               (see AgentContainer::moveRandomTravel) */
    amrex::Real random_travel_prob = amrex::Real(0.0001); /*!< probability of an agent going on random travel */
    int air_travel_int = -1;
    int weather_int = -1;

    int num_diseases = 1;                   /*!< Number of diseases to track */
    std::vector<std::string> disease_names; /*!< Names of the diseases */

    std::string air_traffic_filename;
    std::string airports_filename;
    std::string weather_filename;
    std::string startdate;

    /*! UrbanPop data filename.
     */
    std::string urbanpop_filename;

    /*! Optional per-(state, NAICS-code) work-group target size table (see
        utilities/UrbanPop-scripts/compute_workgroup_sizes.py). File format: first non-comment
        line is a row count, followed by "<state_fips> <naics_code> <int size>" lines
        (order-independent, matched by state FIPS and NAICS code string, not position). If empty
        (the default), every state/NAICS combination falls back to the flat #workgroup_size,
        exactly as before this feature existed -- fully backward compatible. */
    std::string workgroup_size_filename;

    /*! Population-size-based scale factor (see computeCommunitySizeScale / computeCommunityWorkSizeScale).
        home_population and UrbanPopData::day_population are always available, no side file needed.
        When true, this populates AgentContainer::comm_density_scale (nighttime, keyed on
        home_population) and comm_density_scale_work (daytime, keyed on day_population -- the true
        daytime headcount, not BlockGroup::work_populations[0], which structurally excludes
        non-commuting residents and students). Both share the same clip bounds (compile-time
        constants in UrbanPopData.cpp, not user-tunable) -- a decoupled sweep (fixing home, tuning
        work independently) found no fit improvement over keeping them equal, so a single set is
        kept rather than duplicating home/work variants. The disease-specific overall magnitude of
        community transmission is a separate knob -- see DiseaseParm::xmit_comm_scale.

        There is no beta exponent to tune: both mechanisms hard-code a 1/population correction (see
        computeCommunitySizeScale). InteractionModComm.H/InteractionModNborhood.H count the *raw
        number* of infectious agents in a susceptible's entire community/neighborhood, so without
        correction, a susceptible's force of infection from these venues scales with the absolute
        size of their community (density-dependent transmission) rather than with local disease
        prevalence (frequency-dependent transmission, the standard and more realistic assumption --
        a person's number of meaningful community-level contacts doesn't grow just because their
        census block group happens to have more people in it). Dividing by population[c] exactly
        cancels that artifact: num_infected ~= population[c] * prevalence, so scale[c] ~ 1/population[c]
        makes num_infected * scale[c] ~= prevalence, independent of population[c]. This was
        discovered empirically as size_beta fitting to -1.0 across every calibration sweep, but it's
        a correction for how the interaction code counts contacts, not a real epidemiological
        hypothesis about large communities transmitting differently -- so it's built in rather than
        left tunable. */
    bool size_scale_enabled = true;

    /*! Split each (community, school_id, grade) group into classes (see
        AgentContainer::assignSchoolClasses) so school transmission happens within a class
        rather than across an entire grade at a school -- a grade-at-school group averages
        ~62 students statewide (NM UrbanPop), with a long tail into the thousands for colleges,
        far larger than a real classroom. The number of classes is driven primarily by the raw
        group's actual teacher headcount (one class per teacher, at most, mirroring real
        classroom staffing -- any teachers beyond that go to a separate non-classroom "admin"
        bucket), clamped so no class averages fewer than #school_class_size_min or more than
        #school_class_size_max students. #school_class_size itself is now only the fallback
        target used when a raw group has students but zero identified teachers. */
    int school_class_size = 20;     /*!< fallback target students-per-class for raw groups with no identified teachers */
    int school_class_size_min = 5;  /*!< floor on average class size (bounds class count from above) */
    int school_class_size_max = 50; /*!< cap on average class size (bounds class count from below) */

    /*! Correction factor applied to the raw teacher headcount for college-level (grade>17,
        SchoolType::college) raw groups specifically, before it drives class count. College
        "teacher" counts (utilities/UrbanPop-scripts/get_schools.py) are sourced from HIFLD's
        TOT_EMP field -- total college employment (faculty, admin, facilities, research staff,
        everyone), not a faculty-specific count -- unlike K-12, which does have a genuine "Full
        Time Teachers" field, and unlike the rest of the raw HIFLD college data (checked directly:
        TOT_EMP is the only employment-related field available, so there's no better source signal
        to fall back on). Without this correction, that inflated headcount routinely exceeds
        student_count/school_class_size_min, forcing nearly every over-staffed college raw group
        to collapse to the identical class count regardless of its actual data -- visible as a
        single dominant spike in the "School class size" histogram (see
        ExaEpi::Utils::printHistogram) rather than a distribution that varies by institution.
        Scaling the headcount down by this fraction *before* the school_class_size_min/max clamp
        lets each college's derived class size reflect its own reported staffing level again
        (validated empirically: with fraction=0.1, only 2 of 68 NM college raw groups still hit
        the clamp, versus the previous approach where effectively all of them did). */
    amrex::Real college_instructional_fraction = amrex::Real(0.1);

    int aggregated_diag_int = -1;                 /*!< interval for writing aggregated diagnostic data
                                                       (see: ExaEpi::IO::writeFIPSData) */
    std::string aggregated_diag_prefix = "cases"; /*!< filename prefix for diagnostic data, used only if
                                                       aggregated_diag_int >= 0 (see: ExaEpi::IO::writeFIPSData) */

    std::string restart_chkfile = "";             /*!< the name of the checkpoint to restart from */

    int shelter_start = -1;
    int shelter_length = 0;

    int nborhood_size = 360;
    int workgroup_size = 20;

    bool fast = false;         /*!< Use the fast implementations for various operations, e.g. binning, at the cost of
                                   exact reproducibility for a given random seed. */

    bool context_diag = false; /*!< If true, attribute infections to interaction contexts via snapshotProbs /
                                    sumContextInfections and write per-context columns to the output file. */
};

/*! \brief Namespace with utility functions */
namespace Utils {

void getTestParams(ExaEpi::TestParams& params, const std::string& prefix);

/*! \brief Print an ASCII text histogram to the output trace (IOProcessor-only; a no-op on
    every other rank).

    \p value_counts must already be fully merged/reduced -- see #gatherHistogramCounts for
    combining rank-local partial counts first. If \p log_scale is true, values are grouped into
    up to \p max_distinct_buckets geometrically-spaced (doubling-ish) range bins instead of
    equal-width ones -- use this for distributions spanning multiple orders of magnitude (e.g. a
    community-size histogram with a long tail into the thousands), where equal-width bins would
    otherwise waste nearly all their resolution on a handful of large outliers while cramming the
    bulk of the data into the first bucket or two; adjacent bins that round to the same integer
    range at the small end are merged, so the actual bucket count may come out lower than \p
    max_distinct_buckets. Otherwise (\p log_scale false, the default), if \p bin_width is 0
    bucket width is chosen automatically: if the observed value range fits within \p
    max_distinct_buckets, each distinct integer value gets its own bar; otherwise values are
    grouped into \p max_distinct_buckets equal-width range bins. If \p bin_width is nonzero,
    every bucket uses that fixed width instead (overriding \p max_distinct_buckets) -- useful
    when a distribution's natural bucket width (1 or 2, from the auto sizing above) is too
    fine-grained to be legible. In every mode, runs of 2+ consecutive empty buckets are collapsed
    into a single "..." gap line noting the skipped range, so a wide bin_width or a log-scaled
    tail doesn't blow up the output with empty rows. Bar length is normalized so the largest
    bucket is at most \p max_bar_width characters wide (noting the scale used, if any, so the raw
    counts printed alongside each bar remain the source of truth). */
void printHistogram(const std::string& label, const std::map<amrex::Long, amrex::Long>& value_counts,
                    int max_distinct_buckets = 50, int max_bar_width = 60, amrex::Long bin_width = 0, bool log_scale = false);

/*! \brief Merge per-rank-local value->count maps (e.g. a histogram tallied only from the
    agents/data a rank happens to own) into a single global map.

    Meaningful result on IOProcessor only, by design (matches how #printHistogram is also
    called IOProcessor-only) -- the return value on every other rank is unused. */
std::map<amrex::Long, amrex::Long> gatherHistogramCounts(const std::map<amrex::Long, amrex::Long>& local_counts);
} // namespace Utils
} // namespace ExaEpi

#endif
