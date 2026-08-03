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

/*! \brief Compute a per-community density scale factor, decoupling *how much* total transmission
 *  (density_global_scale) from *how it's redistributed* by density (density_beta):
 *    raw[c]   = (density[c]/ref_density)^beta                  -- shape only
 *    scale[c] = clip(global_scale * raw[c]/mean(raw), min_scale, max_scale)
 *  where density[c] = home_population[c]/area_km2[c] and mean(raw) is the population-weighted
 *  mean of raw[] over matched communities. Renormalizing by mean(raw) pins the population-weighted
 *  average of scale[] to global_scale regardless of beta, so beta can be tuned to reshape
 *  transmission by density without also changing the overall transmission level -- beta=0 gives a
 *  flat scale[c]=global_scale everywhere. Communities whose GEOID has no matching area record (or
 *  a near-zero area) default to scale=1.0, unaffected by global_scale. If params.density_ref_density
 *  <= 0 ("auto"), ref_density is the population-weighted mean density over matched communities, so
 *  a sensible baseline needs no manual calibration. */
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

    Vector<Real> raw(block_groups.size(), 1.0_rt);
    Real weighted_raw_sum = 0.0_rt;
    for (int c = 0; c < (int)block_groups.size(); ++c) {
        if (density[c] < 0.0_rt) { continue; }
        raw[c] = std::pow(density[c] / ref_density, params.density_beta);
        weighted_raw_sum += (Real)block_groups[c].home_population * raw[c];
    }
    Real mean_raw = (weight_sum > 0.0_rt) ? (weighted_raw_sum / weight_sum) : 1.0_rt;

    for (int c = 0; c < (int)block_groups.size(); ++c) {
        if (density[c] < 0.0_rt) { continue; }
        Real s = params.density_global_scale * raw[c] / mean_raw;
        scale[c] = std::max(params.density_min_scale, std::min(params.density_max_scale, s));
    }

    amrex::Print() << "DensityData: matched " << matched << " / " << block_groups.size()
                   << " communities to a density record (ref_density=" << ref_density
                   << ", global_scale=" << params.density_global_scale << ")\n";

    return scale;
}

/*! \brief Compute a per-community population-size scale factor, decoupling *how much* total
 *  transmission (size_global_scale) from *how it's redistributed* by community population
 *  (size_beta), exactly mirroring DensityData::computeCommunityScale:
 *    raw[c]   = (population[c]/ref_population)^beta               -- shape only
 *    scale[c] = clip(global_scale * raw[c]/mean(raw), min_scale, max_scale)
 *  where mean(raw) is the population-weighted mean of raw[] over all communities. No side file or
 *  GEOID matching is needed -- home_population is always available -- so every community
 *  participates (unlike density's matched/unmatched split). If params.size_ref_population <= 0
 *  ("auto"), ref_population is the population-weighted mean community population. */
amrex::Vector<amrex::Real> computeCommunitySizeScale (const amrex::Vector<BlockGroup>& block_groups,
                                                      const ExaEpi::TestParams& params) {
    Vector<Real> scale(block_groups.size(), 1.0_rt);
    if (block_groups.empty()) { return scale; }

    Real weighted_pop_sum = 0.0_rt;
    Real weight_sum = 0.0_rt;
    for (const auto& bg : block_groups) {
        Real pop = (Real)bg.home_population;
        weighted_pop_sum += pop * pop;
        weight_sum += pop;
    }

    Real ref_population = params.size_ref_population;
    if (ref_population <= 0.0_rt) { ref_population = (weight_sum > 0.0_rt) ? (weighted_pop_sum / weight_sum) : 1.0_rt; }

    Vector<Real> raw(block_groups.size(), 1.0_rt);
    Real weighted_raw_sum = 0.0_rt;
    for (int c = 0; c < (int)block_groups.size(); ++c) {
        Real pop = (Real)block_groups[c].home_population;
        raw[c] = std::pow(pop / ref_population, params.size_beta);
        weighted_raw_sum += pop * raw[c];
    }
    Real mean_raw = (weight_sum > 0.0_rt) ? (weighted_raw_sum / weight_sum) : 1.0_rt;

    for (int c = 0; c < (int)block_groups.size(); ++c) {
        Real s = params.size_global_scale * raw[c] / mean_raw;
        scale[c] = std::max(params.size_min_scale, std::min(params.size_max_scale, s));
    }

    amrex::Print() << "SizeScale: " << block_groups.size() << " communities (ref_population=" << ref_population
                   << ", global_scale=" << params.size_global_scale << ")\n";

    return scale;
}

/*! \brief Compute a per-community work-population scale factor, decoupling *how much* total
 *  daytime transmission (work_size_global_scale) from *how it's redistributed* by community work
 *  population (work_size_beta), exactly mirroring computeCommunitySizeScale but keyed on
 *  work_populations[0] (total workers whose workplace is this community) instead of
 *  home_population:
 *    raw[c]   = (work_population[c]/ref_work_population)^beta        -- shape only
 *    scale[c] = clip(global_scale * raw[c]/mean(raw), min_scale, max_scale)
 *  where mean(raw) is the work-population-weighted mean of raw[] over all communities. Communities
 *  with zero work population (no one's workplace is there) get scale=1.0, unaffected by
 *  global_scale, and don't contribute to the weighted mean. If params.work_size_ref_population <=
 *  0 ("auto"), ref_work_population is the work-population-weighted mean community work
 *  population. */
amrex::Vector<amrex::Real> computeCommunityWorkSizeScale (const amrex::Vector<BlockGroup>& block_groups,
                                                          const ExaEpi::TestParams& params) {
    Vector<Real> scale(block_groups.size(), 1.0_rt);
    if (block_groups.empty()) { return scale; }

    Real weighted_pop_sum = 0.0_rt;
    Real weight_sum = 0.0_rt;
    for (const auto& bg : block_groups) {
        Real pop = (Real)bg.work_populations[0];
        if (pop <= 0.0_rt) { continue; }
        weighted_pop_sum += pop * pop;
        weight_sum += pop;
    }

    Real ref_population = params.work_size_ref_population;
    if (ref_population <= 0.0_rt) { ref_population = (weight_sum > 0.0_rt) ? (weighted_pop_sum / weight_sum) : 1.0_rt; }

    Vector<Real> raw(block_groups.size(), 1.0_rt);
    Real weighted_raw_sum = 0.0_rt;
    for (int c = 0; c < (int)block_groups.size(); ++c) {
        Real pop = (Real)block_groups[c].work_populations[0];
        if (pop <= 0.0_rt) { continue; }
        raw[c] = std::pow(pop / ref_population, params.work_size_beta);
        weighted_raw_sum += pop * raw[c];
    }
    Real mean_raw = (weight_sum > 0.0_rt) ? (weighted_raw_sum / weight_sum) : 1.0_rt;

    for (int c = 0; c < (int)block_groups.size(); ++c) {
        if (block_groups[c].work_populations[0] <= 0) { continue; }
        Real s = params.work_size_global_scale * raw[c] / mean_raw;
        scale[c] = std::max(params.work_size_min_scale, std::min(params.work_size_max_scale, s));
    }

    amrex::Print() << "WorkSizeScale: " << block_groups.size() << " communities (ref_work_population=" << ref_population
                   << ", global_scale=" << params.work_size_global_scale << ")\n";

    return scale;
}
