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
    int work_state_fips; /*!< 2-digit state FIPS of the agent's workplace (from work_geoid),
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

/*! \brief Daytime analog of computeCommunitySizeScale, keyed on each community's *work* population
    (BlockGroup::work_populations[0]) instead of home_population. Meant to populate a separate
    scale array used only for daytime interactions, since home_population is not a meaningful
    covariate for the population physically present in a community during the day (see
    AgentContainer::comm_density_scale_work). */
amrex::Vector<amrex::Real> computeCommunityWorkSizeScale(const amrex::Vector<BlockGroup>& block_groups,
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
    amrex::Vector<int> workgroup_size_table;             /*!< flat, size MAX_STATE_FIPS * NAICS_COUNT;
                                                               see UrbanPop-scripts/compute_workgroup_sizes.py */
    amrex::Gpu::DeviceVector<int> workgroup_size_table_d;

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
