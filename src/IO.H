/*! @file IO.H
    \brief Defines the #ExaEpi::IO namespace
*/

#ifndef IO_H_
#define IO_H_

#include <string>
#include <vector>

#include <AMReX_iMultiFab.H>

#include "AgentContainer.H"
#include "UrbanPopData.H"

#include <string>

namespace ExaEpi {

/*! \brief Namespace for IO functions; note that this namespace contains that write
           plotfiles and diagnostics data */
namespace IO {

void writePlotFile(const AgentContainer& pc, const MFPtrVec& a_disease_stats, const iMultiFab* unit_mf_ptr,
                   const iMultiFab* FIPS_mf_ptr, const iMultiFab* comm_mf_ptr, const int num_diseases,
                   const std::vector<std::string>& disease_names, const Real cur_time, const int step,
                   const int verbose = ExaEpi::Verbosity::quiet);

void readCheckpointFile(const std::string restart_chkfile, AgentContainer& pc, MFPtrVec& a_disease_stats, iMultiFab* unit_mf_ptr,
                        iMultiFab* FIPS_mf_ptr, iMultiFab* comm_mf_ptr, Real& cur_time, int& step);

void writeCheckpointFile(const AgentContainer& pc, const MFPtrVec& a_disease_stats, const iMultiFab* unit_mf_ptr,
                         const iMultiFab* FIPS_mf_ptr, const iMultiFab* comm_mf_ptr, const int num_diseases,
                         const std::vector<std::string>& disease_names, const Real cur_time, const int step);

void writeAggregatedData(const AgentContainer& pc, const UrbanPopData& urbanpopData, const std::string& prefix,
                         const int num_diseases, const std::vector<std::string>& disease_names, const int step);

/*! Per-community population counts (everyone / workers / students), index-aligned with
    UrbanPopData::block_groups -- see AgentContainer::generatePopulationBreakdown() and
    computePopulationBreakdownCsvData() below. Valid (non-empty) on the IOProcessor only. */
struct PopulationBreakdown {
    std::vector<amrex::Long> total, workers, students;
};

/*! \brief Compute PopulationBreakdown from pc's agents' CURRENT positions (see
    AgentContainer::generatePopulationBreakdown()'s doc comment for the home/work timing
    requirement this relies on), collapsed to one row per community via urbanpopData.community_mf,
    exactly like writeAggregatedData's own collapse. */
PopulationBreakdown computePopulationBreakdownCsvData(const AgentContainer& pc, const UrbanPopData& urbanpopData);

/*! \brief Write the static (run-long-constant), once-per-run aggregated diagnostics: a
    <prefix>_day_night_population.csv (GEOID,night_total,night_workers,night_students,day_total,
    day_workers,day_students, one row per community) and <prefix>_workgroup_sizes.txt /
    _class_sizes.txt / _school_sizes.txt / _nborhood_sizes.txt / _nborhoods_per_community.txt /
    _work_nborhood_sizes.txt / _work_nborhoods_per_community.txt (plain text, one integer per line,
    matching Epicast's own reference-file format).

    `nborhoods` and `day_nborhoods` are the night (home) and day neighborhood distributions; the
    _work_-prefixed files are the day ones, named for the work_nborhood attribute they come from.
    IOProcessor only. */
void writeStaticAggregatedData(const PopulationBreakdown& day, const PopulationBreakdown& night,
                               const GroupSizeAggregates& groups, const NborhoodSizeAggregates& nborhoods,
                               const NborhoodSizeAggregates& day_nborhoods, const UrbanPopData& urbanpopData,
                               const std::string& prefix);
} // namespace IO
} // namespace ExaEpi

#endif
