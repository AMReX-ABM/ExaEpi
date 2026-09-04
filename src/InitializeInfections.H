/*! @file InitializeInfections.H
    \brief #Routines to implement the initial infections
*/

#pragma once

#include <AMReX_GpuContainers.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include <string>

#include "AgentContainer.H"
#include "CaseData.H"
#include "Utils.H"

namespace ExaEpi {
namespace Initialization {

/*! \brief Build a per-unit cumulative population distribution over its communities.

    For each unit u (e.g. county), computes the prefix sum of community_population over the
    communities in [unit_community_start[u], unit_community_start[u+1]), normalized so that
    slice ends at exactly 1.0. Used to draw a community with probability proportional to its
    population (matching Epicast's population-proportional index-case seeding), rather than
    uniformly over communities. If a unit's total population is 0, its slice is left at 0.0 as
    a sentinel so that community selection can fall back to a uniform draw for that unit. */
amrex::Vector<float> buildCommunityCumProb(const Vector<int>& unit_community_start, const Vector<int>& community_population);

} // namespace Initialization
} // namespace ExaEpi

void setInitialCasesFromFile(AgentContainer& pc, CaseData& cases, const std::string& name, int d_idx,
                             const Vector<int>& FIPS_codes, const Vector<int>& unit_community_start,
                             const Vector<float>& community_cum_prob, iMultiFab& comm_mf, const bool fast_bin);

void setInitialCasesRandom(AgentContainer& pc, int num_cases, const std::string& name, int d_idx, const Vector<int>& FIPS_codes,
                           const Vector<int>& unit_community_start, const Vector<float>& community_cum_prob, iMultiFab& comm_mf,
                           const bool fast_bin);
