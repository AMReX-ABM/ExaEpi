/*! @file IO.H
    \brief Defines the #ExaEpi::IO namespace
*/

#ifndef IO_H_
#define IO_H_

#include <string>
#include <vector>

#include <AMReX_iMultiFab.H>

#include "AgentContainer.H"
#include "CensusData.H"
#include "UrbanPopData.H"

#include <string>

namespace ExaEpi {

/*! \brief Namespace for IO functions; note that this namespace contains that write
           plotfiles and diagnostics data */
namespace IO {

void writePlotFile(const AgentContainer& pc, const MFPtrVec& a_disease_stats, const iMultiFab* unit_mf_ptr,
                   const iMultiFab* FIPS_mf_ptr, const iMultiFab* comm_mf_ptr, const int num_diseases,
                   const std::vector<std::string>& disease_names, const Real cur_time, const int step);

void readCheckpointFile(const std::string restart_chkfile, AgentContainer& pc, MFPtrVec& a_disease_stats, iMultiFab* unit_mf_ptr,
                        iMultiFab* FIPS_mf_ptr, iMultiFab* comm_mf_ptr, Real& cur_time, int& step);

void writeCheckpointFile(const AgentContainer& pc, const MFPtrVec& a_disease_stats, const iMultiFab* unit_mf_ptr,
                         const iMultiFab* FIPS_mf_ptr, const iMultiFab* comm_mf_ptr, const int num_diseases,
                         const std::vector<std::string>& disease_names, const Real cur_time, const int step);

void writeFIPSData(const AgentContainer& pc, const CensusData& censusData, const std::string& prefix, const int num_diseases,
                   const std::vector<std::string>& disease_names, const int step);

void writeAggregatedData(const AgentContainer& pc, const UrbanPopData& urbanpopData, const std::string& prefix,
                         const int num_diseases, const std::vector<std::string>& disease_names, const int step);
} // namespace IO
} // namespace ExaEpi

#endif
