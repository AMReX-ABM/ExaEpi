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

struct UrbanPopData {
    // these three data structures are used for setting the initial cases
    amrex::Vector<int> FIPS_codes;
    amrex::Vector<int> County_on_proc;
    amrex::Vector<int> fips_community_start;
    amrex::Vector<BlockGroup> block_groups;

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
