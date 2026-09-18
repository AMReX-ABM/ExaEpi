/*! @file UrbanPopData.H
    \brief #UrbanPopData class definition
*/

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <AMReX_GpuContainers.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include "CaseData.H"
#include "UrbanPopAgentStruct.H"
#include "Utils.H"

class AgentContainer;

/*! \brief Per-agent grid coordinates, the one thing initAgents needs that is not already in the
    agent record itself. The group-structure sizes that used to live here (block group and
    per-NAICS populations, workplace state FIPS, household count) were only ever inputs to the
    neighborhood/work-group/household-cluster draws, which now happen in preprocessing -- see
    UrbanPop-scripts/group_assignment.py. */
struct AgentExtras {
    amrex::IntVect home_xy;
    amrex::IntVect work_xy;
};

/*! Codec used for the per-block-group agent frames, from the .bin header. The field exists in the
    format so a future codec can be added without another version bump -- see CODEC_* in
    UrbanPop-scripts/upop_to_exaepi.py. */
enum UrbanPopCodec : uint32_t { CODEC_NONE = 0, CODEC_DEFLATE = 1 };

/*! \brief Scratch buffers for inflating one block-group frame at a time.

    Owned by the caller and reused across block groups so the per-tile loops don't reallocate for
    every frame. `raw` is a std::vector so its storage is 8-byte aligned, which is what lets
    UrbanPop::AgentFrame point its columns straight into it. */
struct FrameScratch {
    std::vector<char> compressed;
    std::vector<char> raw;
};

struct BlockGroup {
    int64_t geoid;
    size_t frame_offset;   /*!< byte offset of this block group's compressed agent frame */
    uint32_t frame_nbytes; /*!< compressed size of that frame; it inflates to home_population * record_size() */
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
    /*! Inflate this block group's frame into `scratch` and append its agents to `agents`. */
    bool readAgents(std::ifstream& f, uint32_t codec, FrameScratch& scratch, amrex::Vector<UrbanPop::UrbanPopAgent>& agents,
                    amrex::Vector<AgentExtras>& agents_extras, const std::map<int64_t, int>& geoid_to_block_groups,
                    const amrex::Vector<BlockGroup>& block_groups);
};

/*! \brief Compute a per-community population-size scale factor that converts the community/
    neighborhood interaction models from density-dependent to frequency-dependent transmission
    (a fixed 1/population correction, not a tunable exponent -- see the .cpp for why), keyed on
    home_population (population is always available from the UrbanPop binary). */
amrex::Vector<amrex::Real> computeCommunitySizeScale(const amrex::Vector<BlockGroup>& block_groups, const bool verbose = false);

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
                                                         const bool verbose = false);

struct UrbanPopData {
    // these three data structures are used for setting the initial cases
    amrex::Vector<int> FIPS_codes;
    amrex::Vector<int> County_on_proc;
    amrex::Vector<int> fips_community_start;
    amrex::Vector<BlockGroup> block_groups;

    /*! Total residential population per FIPS county, keyed by FIPS code -- feeds
        AirTravelFlow::readAirports (see AirTravelDemoData), which needs the population served by
        each airport's county. */
    std::map<int, int> CountyPop;
    /*! Residential population per FIPS unit, index-aligned with FIPS_codes/fips_community_start
        -- feeds AirTravelFlow::computeTravelProbs (see AirTravelDemoData::Population). For
        UrbanPop, a "unit" is exactly one FIPS county, so this holds the same values as
        CountyPop[FIPS_codes[i]], just index-aligned for device use. */
    amrex::Vector<int> Population;
    /*! Device copy of fips_community_start, for AgentContainer::setAirTravel's per-agent kernel
        (see AirTravelDemoData::Start_d). */
    amrex::Gpu::DeviceVector<int> fips_community_start_d;

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
    /*! Compact FIPS-unit index (0..FIPS_codes.size()-1, index-aligned with FIPS_codes/
        fips_community_start) of each grid cell -- for AgentContainer::setAirTravel/moveAirTravel
        (see AirTravelDemoData), which need a per-cell index into the same per-unit arrays that
        AirTravelFlow::readAirports/computeTravelProbs build from FIPS_codes. */
    amrex::iMultiFab unit_mf;

    UrbanPopData () {}

    void init(ExaEpi::TestParams& params, amrex::Geometry& geom, amrex::BoxArray& ba, amrex::DistributionMapping& dm);

    void initAgents(AgentContainer& pc, const ExaEpi::TestParams& params);

  private:
    void fillGridMetadataOnHost();

    std::map<int64_t, int> geoid_to_block_groups;
    std::map<amrex::IntVect, int> xy_to_block_groups;
    std::ifstream urbanpop_file;
    uint32_t codec = CODEC_DEFLATE; /*!< how urbanpop_file's agent frames are compressed */
};
