/*! @file DensityData.cpp
    \brief Function implementations of #DensityData class
*/

#include "DensityData.H"

#include <AMReX_BLassert.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

using namespace amrex;

/*! \brief Read geoid/area_km2 pairs from a plain-text side file:
 *    <num_records>
 *    <geoid_1> <area_km2_1>
 *    <geoid_2> <area_km2_2>
 *    ...
 *  geoid is the 12-digit Census block-group GEOID (matches BlockGroup::geoid); area_km2 is land
 *  area only. A missing filename or unreadable file is not an error: has_data is left false and
 *  the caller falls back to a flat 1.0 density scale everywhere. */
bool DensityData::readDataFromFile (const std::string& fname) {
    has_data = false;
    geoid_to_area_km2.clear();

    if (fname.empty()) {
        amrex::Print() << "DensityData: no density_filename provided; xmit_comm/xmit_hood density "
                          "scaling disabled.\n";
        return has_data;
    }

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr, /*bExitOnError=*/false);
    if (fileCharPtr.empty()) {
        amrex::Print() << "DensityData: could not open density file '" << fname
                       << "'; xmit_comm/xmit_hood density scaling disabled.\n";
        return has_data;
    }

    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line;
    std::getline(is, line);
    int num_records = std::stoi(line);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(num_records >= 0, "DensityData: number of records can't be negative");

    for (int i = 0; i < num_records; ++i) {
        AMREX_ALWAYS_ASSERT(is.good());
        std::getline(is, line);
        std::istringstream lis(line);
        int64_t geoid;
        amrex::Real area_km2;
        lis >> geoid >> area_km2;
        geoid_to_area_km2[geoid] = area_km2;
    }

    has_data = !geoid_to_area_km2.empty();
    amrex::Print() << "DensityData: read " << geoid_to_area_km2.size() << " geoid/area records from '" << fname << "'\n";
    return has_data;
}

/*! \brief Compute a per-community density scale factor: scale[c] = clip((density[c]/ref_density)
 *  ^beta, min_scale, max_scale), where density[c] = home_population[c] / area_km2[c]. Communities
 *  whose GEOID has no matching area record (or a near-zero area) default to scale=1.0. If
 *  params.density_ref_density <= 0 ("auto"), ref_density is the population-weighted mean density
 *  over matched communities, so a sensible baseline needs no manual calibration. */
amrex::Vector<amrex::Real> DensityData::computeCommunityScale (const amrex::Vector<BlockGroup>& block_groups,
                                                               const ExaEpi::TestParams& params) const {
    Vector<Real> scale(block_groups.size(), 1.0_rt);
    if (!has_data) { return scale; }

    constexpr Real area_eps = 1e-6_rt;

    Vector<Real> density(block_groups.size(), -1.0_rt);
    long matched = 0;
    Real weighted_density_sum = 0.0_rt;
    Real weight_sum = 0.0_rt;
    for (int c = 0; c < (int)block_groups.size(); ++c) {
        auto it = geoid_to_area_km2.find(block_groups[c].geoid);
        if (it == geoid_to_area_km2.end() || it->second < area_eps) { continue; }
        Real pop = (Real)block_groups[c].home_population;
        density[c] = pop / it->second;
        ++matched;
        weighted_density_sum += pop * density[c];
        weight_sum += pop;
    }

    Real ref_density = params.density_ref_density;
    if (ref_density <= 0.0_rt) { ref_density = (weight_sum > 0.0_rt) ? (weighted_density_sum / weight_sum) : 1.0_rt; }

    for (int c = 0; c < (int)block_groups.size(); ++c) {
        if (density[c] < 0.0_rt) { continue; }
        Real s = std::pow(density[c] / ref_density, params.density_beta);
        scale[c] = std::max(params.density_min_scale, std::min(params.density_max_scale, s));
    }

    amrex::Print() << "DensityData: matched " << matched << " / " << block_groups.size()
                   << " communities to a density record (ref_density=" << ref_density << ")\n";

    return scale;
}
