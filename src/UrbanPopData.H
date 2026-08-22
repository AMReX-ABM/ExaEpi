/*! @file UrbanPopData.H
    \brief #UrbanPopData class definition
*/

#pragma once

#include <iostream>
#include <string>

#include <AMReX_GpuContainers.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include "CaseData.H"
#include "UrbanPopAgentStruct.H"
#include "Utils.H"

class AgentContainer;

struct AgentExtras {
    amrex::IntVect home_xy;
    amrex::IntVect work_xy;
    int naics_population;
    int work_population;
    int home_population;
    int work_state_fips;     /*!< 2-digit state FIPS of the agent's workplace (from work_geoid),
                                  used to index UrbanPopData::workgroup_size_table. Only
                                  meaningful when naics_population > 0 (i.e. the agent is
                                  employed); left unset otherwise. */
    int home_num_households; /*!< number of distinct households in the agent's home block group
                                  (BlockGroup::num_households), used to size the modulus for
                                  hh_cluster assignment -- see UrbanPopData::initAgents. */
};

struct BlockGroup {
    int64_t geoid;
    size_t file_offset;
    int block_i;
    int x = -1;
    int y = -1;
    int home_population;
    amrex::Vector<int> work_populations;
    int num_households = 0;
    int num_employed = 0;
    int num_students = 0;
    int num_educators = 0;

    bool read(std::istream& f);
    bool readAgents(std::ifstream& f, amrex::Vector<UrbanPop::UrbanPopAgent>& agents, amrex::Vector<AgentExtras>& agents_extras,
                    const std::map<int64_t, int>& geoid_to_block_groups, const amrex::Vector<BlockGroup>& block_groups);
};

/*! \brief Compute a per-community population-size scale factor that converts the community/
    neighborhood interaction models from density-dependent to frequency-dependent transmission
    (a fixed 1/population correction, not a tunable exponent -- see the .cpp for why), keyed on
    home_population (population is always available from the UrbanPop binary). */
amrex::Vector<amrex::Real> computeCommunitySizeScale(const amrex::Vector<BlockGroup>& block_groups,
                                                     const ExaEpi::TestParams& params);

/*! \brief Daytime analog of computeCommunitySizeScale, keyed on each community's true daytime
    headcount (UrbanPopData::day_population) instead of home_population. Meant to populate a
    separate scale array used only for daytime interactions, since home_population is not a
    meaningful covariate for the population physically present in a community during the day (see
    AgentContainer::comm_density_scale_work).

    day_population, not BlockGroup::work_populations[0], is the right covariate here:
    work_populations[0] only counts agents with naics != -1 whose job is physically located there,
    which structurally excludes every non-commuting resident (retirees, unemployed, work-from-home)
    who stays home during the day, and every student at a school located there (naics == -1).
    Empirically (NM UrbanPop), the true daytime headcount exceeds work_populations[0] by a median
    of 4x and by >20x at the 90th percentile, so using work_populations[0] here would systematically
    push the 1/population correction toward its size_max_scale clip for the most heavily
    residential/school-heavy communities -- exactly backwards, since those are typically among the
    *largest* daytime-occupied communities, not the smallest. */
amrex::Vector<amrex::Real> computeCommunityWorkSizeScale(const amrex::Vector<amrex::Real>& day_population,
                                                         const ExaEpi::TestParams& params);

struct UrbanPopData {
    // these three data structures are used for setting the initial cases
    amrex::Vector<int> FIPS_codes;
    amrex::Vector<int> County_on_proc;
    amrex::Vector<int> fips_community_start;
    amrex::Vector<BlockGroup> block_groups;

    /*! Real US state/territory FIPS codes top out at 78 (US Virgin Islands); this gives
        headroom without a remapping table -- workgroup_size_table is indexed directly by
        raw state FIPS. */
    static constexpr int MAX_STATE_FIPS = 100;
    amrex::Vector<int> workgroup_size_table; /*!< flat, size MAX_STATE_FIPS * NAICS_COUNT;
                                                   see UrbanPop-scripts/compute_workgroup_sizes.py */
    amrex::Gpu::DeviceVector<int> workgroup_size_table_d;

    /*! True daytime headcount per community (index-aligned with block_groups), tallied from every
        agent's effective daytime location (naics==-1 or work-from-home -> home block group;
        otherwise -> work block group) during init(), via a full, independent scan of the raw
        UrbanPop agent file done identically on every rank (cheap -- pure host-side counting, no
        particle/GPU work) rather than during initAgents()'s per-rank-partial tile loop.
        Deliberately NOT computed in initAgents(): that only runs on a fresh start (restarts load
        agents from a checkpoint instead and skip it entirely -- see main.cpp), so day_population
        would be left empty on a restarted run if it depended on initAgents() having run. See
        computeCommunityWorkSizeScale's doc comment for why this is used instead of
        BlockGroup::work_populations[0]. */
    amrex::Vector<amrex::Real> day_population;

    amrex::iMultiFab geoid_mf;     /*!< FIPS code (component 0) and census block group number (component 1) of each block group */
    amrex::iMultiFab community_mf; /*!< Index for community == block group */

    UrbanPopData () {}

    void init(ExaEpi::TestParams& params, amrex::Geometry& geom, amrex::BoxArray& ba, amrex::DistributionMapping& dm);

    void initAgents(AgentContainer& pc, const ExaEpi::TestParams& params);

  private:
    void fillGridMetadataOnHost();

    std::map<int64_t, int> geoid_to_block_groups;
    std::map<amrex::IntVect, int> xy_to_block_groups;
    std::ifstream urbanpop_file;
};
