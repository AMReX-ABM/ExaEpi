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

    /*! Optional side file mapping UrbanPop block-group GEOIDs to land area (km^2), used to scale
        xmit_comm/xmit_hood by local population density (see DensityData). "" (default) disables
        density scaling entirely -- UrbanPop only; has no effect on the deprecated Census path. */
    std::string density_filename = "";
    amrex::Real density_ref_density = -1.0; /*!< reference density; <=0 means "auto": population-weighted
                                                 mean density over matched communities */
    amrex::Real density_beta = 0.3;         /*!< exponent reshaping transmission by density; renormalized so it
                                                 doesn't change the population-weighted average scale (see
                                                 DensityData::computeCommunityScale) -- placeholder, not validated */
    amrex::Real density_global_scale = 1.0; /*!< population-weighted average density scale factor (decoupled
                                                 from density_beta's shape); beta=0 gives a flat scale equal to
                                                 this value everywhere -- placeholder, not yet empirically validated */
    amrex::Real density_min_scale = 0.5;    /*!< lower clip bound on the density scale factor -- placeholder */
    amrex::Real density_max_scale = 2.0;    /*!< upper clip bound on the density scale factor -- placeholder */
    bool density_diag = false;              /*!< if true, dump a per-community density/scale diagnostic CSV */

    /*! Population-size-based alternative to density scaling (see DensityData::computeCommunitySizeScale
        / computeCommunityWorkSizeScale). No side file needed -- home_population and work_populations[0]
        are always available. When true, this takes over populating AgentContainer::comm_density_scale
        (nighttime, keyed on home_population) and comm_density_scale_work (daytime, keyed on
        work_populations[0]) instead of the density side file; mutually exclusive with density scaling
        (used to isolate the size effect from the density effect). global_scale/min_scale/max_scale
        below are shared by both mechanisms -- a decoupled sweep (fixing home, tuning work
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
    bool size_scale_enabled = false;
    amrex::Real size_global_scale = 1.0;    /*!< population-weighted average size scale factor -- the
                                                 overall calibrated magnitude of community/neighborhood
                                                 transmission once the 1/population correction above
                                                 has removed the community-size artifact */
    amrex::Real size_min_scale = 0.05;      /*!< lower clip bound on the size scale factor -- placeholder */
    amrex::Real size_max_scale = 20.0;      /*!< upper clip bound on the size scale factor -- placeholder */

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
