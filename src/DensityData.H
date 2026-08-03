/*! @file DensityData.H
    \brief #DensityData class definition
*/

#ifndef DENSITYDATA_H_
#define DENSITYDATA_H_

#include <map>
#include <string>

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include "AgentContainer.H"
#include "UrbanPopData.H"
#include "Utils.H"

/*! \brief Optional side-file data mapping UrbanPop block-group GEOIDs to land area, used to
    scale xmit_comm / xmit_hood by local population density. If no file is provided, or the file
    cannot be read, #has_data stays false and #computeCommunityScale returns a flat 1.0 scale for
    every community -- the feature is a no-op unless the data is explicitly supplied. */
struct DensityData {
    bool has_data = false;
    std::map<int64_t, amrex::Real> geoid_to_area_km2;

    /*! \brief Read geoid/area_km2 pairs from fname. Never aborts: an empty fname or an unreadable
        file leaves #has_data false and logs why via amrex::Print(). Returns #has_data. */
    bool readDataFromFile(const std::string& fname);

    /*! \brief Compute a per-community density scale factor, sized and indexed identically to
        block_groups. Communities whose GEOID isn't found in #geoid_to_area_km2 (or has near-zero
        area) get a scale of 1.0. Safe to call when !has_data (returns all 1.0). */
    amrex::Vector<amrex::Real> computeCommunityScale(const amrex::Vector<BlockGroup>& block_groups,
                                                     const ExaEpi::TestParams& params) const;
};

/*! \brief Compute a per-community population-size scale factor that converts the community/
    neighborhood interaction models from density-dependent to frequency-dependent transmission
    (a fixed 1/population correction, not a tunable exponent -- see the .cpp for why), keyed on
    home_population directly instead of density (no side file needed: population is always
    available from the UrbanPop binary). */
amrex::Vector<amrex::Real> computeCommunitySizeScale(const amrex::Vector<BlockGroup>& block_groups,
                                                     const ExaEpi::TestParams& params);

/*! \brief Daytime analog of computeCommunitySizeScale, keyed on each community's *work* population
    (BlockGroup::work_populations[0]) instead of home_population. Meant to populate a separate
    scale array used only for daytime interactions, since home_population is not a meaningful
    covariate for the population physically present in a community during the day (see
    AgentContainer::comm_density_scale_work). */
amrex::Vector<amrex::Real> computeCommunityWorkSizeScale(const amrex::Vector<BlockGroup>& block_groups,
                                                         const ExaEpi::TestParams& params);

#endif
