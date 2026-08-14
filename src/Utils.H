/*! @file Utils.H
    \brief Defines the #ExaEpi::Utils namespace
*/

#ifndef UTILS_H_
#define UTILS_H_

#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <vector>

#include "DemographicData.H"

#include <string>

namespace ExaEpi {

/*! \brief Structure with various simulation parameters; these are read in
    for the input file provided in the command line.
*/
struct TestParams {
    amrex::IntVect size;                                  /*!< domain size; (used only for ic_type = demo)
                                                               (see ExaEpi::Utils::getGeometry) */
    int max_box_size;                                     /*!< box size */
    int nsteps = 1;                                       /*!< number of simulation steps */
    int plot_int = -1;                                    /*!< plot interval (see ExaEpi::IO::writePlotFile) */
    int check_int = -1;                                   /*!< checkpoint interval (see ExaEpi::IO::writeCheckpointFile) */
    int random_travel_int = -1;                           /*!< steps between random travel events
                                                               (see AgentContainer::moveRandomTravel) */
    amrex::Real random_travel_prob = amrex::Real(0.0001); /*!< probability of an agent going on random travel */
    int air_travel_int = -1;
    short ic_type;                                        /*!< initialization type (see ExaEpi::ICType) */
    int weather_int = -1;

    int num_diseases = 1;                                 /*!< Number of diseases to track */
    std::vector<std::string> disease_names;               /*!< Names of the diseases */

    /*! Census data filename (DemographicData::initFromFile)

        The file contents are:
        + The first line of the file contains the number of units.
        + The following lines have the following data:
          + ID (US-wide census tract ID)
          + Population
          + Number of day workers
          + FIPS code
          + Census tract number
          + Numbers of people in age groups: under 5, 5-17, 18-29, 30-64, and 65+
          + Number of households with: 1, 2, 3, 4, 5, 6, and 7 member(s)
    */
    std::string census_filename;
    std::string air_traffic_filename;
    std::string airports_filename;
    std::string weather_filename;
    std::string startdate;

    /*! UrbanPop data filename.
     */
    std::string urbanpop_filename;

    /*! Population-size-based scale factor (see computeCommunitySizeScale / computeCommunityWorkSizeScale).
        home_population and work_populations[0] are always available, no side file needed. When true,
        this populates AgentContainer::comm_density_scale (nighttime, keyed on home_population) and
        comm_density_scale_work (daytime, keyed on work_populations[0]). comm_hood_scale/min_scale/
        max_scale below are shared by both -- a decoupled sweep (fixing home, tuning work
        independently) found no fit improvement over keeping them equal, so a single set is kept
        rather than duplicating home/work variants.

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
    amrex::Real comm_hood_scale = 1.0; /*!< population-weighted average scale factor applied only to
                                            xmit_comm/xmit_hood (see InteractionModComm.H /
                                            InteractionModNborhood.H) -- the overall calibrated
                                            magnitude of community/neighborhood transmission once
                                            the 1/population correction above has removed the
                                            community-size artifact. Does not affect household,
                                            workgroup, school, or any other venue. The clip bounds
                                            on the per-community scale factor are compile-time
                                            constants in UrbanPopData.cpp, not user-tunable. */

    /*! Split each (community, school_id, grade) group into fixed-size classes (see
        AgentContainer::assignSchoolClasses) so school transmission happens within a class (~15
        classmates) rather than across an entire grade at a school -- a grade-at-school group averages
        ~62 students statewide (NM UrbanPop), with a long tail into the thousands for colleges, far
        larger than a real classroom. Educators teaching that grade are spread uniformly across its
        classes the same way regular workers are spread across a workplace's workgroup buckets (see
        agent.workgroup_size). */
    int school_class_size = 15; /*!< target number of students per class */

    /*! Worker flow filename (ExaEpi::Initialization::readWorkerflow):
        It is a binary file that contains 3 x (number of work patthers) unsigned integer
        data. The 3 integers are: from, to, and the number of workers with this from and to.
        The from and to are the IDs from the first column of the census data file
        (#DemographicData::myID).
    */
    std::string workerflow_filename;

    int aggregated_diag_int = -1;       /*!< interval for writing aggregated diagnostic data
                                             (see: ExaEpi::IO::writeFIPSData) */
    std::string aggregated_diag_prefix; /*!< filename prefix for diagnostic data
                                             (see: ExaEpi::IO::writeFIPSData) */

    std::string restart_chkfile = "";   /*!< the name of the checkpoint to restart from */

    int shelter_start = -1;
    int shelter_length = 0;

    int nborhood_size = 500;
    int workgroup_size = 20;

    bool fast = false;         /*!< Use the fast implementations for various operations, e.g. binning, at the cost of
                                   exact reproducibility for a given random seed. */

    bool context_diag = false; /*!< If true, attribute infections to interaction contexts via snapshotProbs /
                                    sumContextInfections and write per-context columns to the output file. */
};

/**
 * \brief enum for the different initial condition options.\n
 *        census reads in census data.\n
 */
struct ICType {
    enum {
        Census = 0,  /*!< Census data. Default */
        UrbanPop = 1 /*!< Data generated by UrbanPop */
    };
};

/*! \brief Namespace with utility functions */
namespace Utils {

void getTestParams(ExaEpi::TestParams& params, const std::string& prefix);
}
} // namespace ExaEpi

#endif
