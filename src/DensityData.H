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

#endif
