/*! @file UrbanPopData.H
    \brief #UrbanPopData class definition
*/

#pragma once

#include <string>
#include <iostream>

#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include "Utils.H"
#include "CaseData.H"
#include "UrbanPopAgentStruct.H"


struct BlockGroup {
    int64_t geoid;
    amrex::Real lng;
    amrex::Real lat;
    size_t file_offset;
    int block_i;
    int x;
    int y;
    int home_population;
    amrex::Vector<int> work_populations;
    int num_households;
    int num_employed;
    int num_students;
    int num_educators;

    bool read(std::istringstream &iss);
    bool read_agents(std::ifstream &f, amrex::Vector<UrbanPop::UrbanPopAgent> &agents, amrex::Vector<int>& group_work_population,
                     amrex::Vector<int>& group_home_population, const std::map<IntVect, BlockGroup> &xy_to_block_groups,
                     const LngLatToGrid &lnglat_to_grid, const GridToLngLat &grid_to_lnglat);
};

struct UrbanPopData {
    amrex::Real min_lng, min_lat, max_lng, max_lat;
    amrex::Real gspacing_x, gspacing_y;

    amrex::Vector<int> FIPS_codes;   /*!< FIPS codes for each unit */
    amrex::Vector<int> unit_community_start;  /*!< Starting community number (block group) for each unit */

    amrex::iMultiFab FIPS_mf;        /*!< FIPS code (component 0) and census tract number (component 1) of each community */
    amrex::iMultiFab comm_mf;        /*!< Community number */

    int num_communities;

    UrbanPopData () {}

    void init(ExaEpi::TestParams &params, amrex::Geometry &geom, amrex::BoxArray &ba, amrex::DistributionMapping &dm);

    void initAgents(AgentContainer &pc, const ExaEpi::TestParams &params);

  private:

    std::map<IntVect, BlockGroup> xy_to_block_groups;
};
