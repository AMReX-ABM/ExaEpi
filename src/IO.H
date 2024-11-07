/*! @file IO.H
    \brief Defines the #ExaEpi::IO namespace
*/

#ifndef IO_H_
#define IO_H_

#include <vector>
#include <string>

#include <AMReX_iMultiFab.H>

#include "AgentContainer.H"
#include "CensusData.H"

#include <string>

namespace ExaEpi
{

/*! \brief Namespace for IO functions; note that this namespace contains that write
           plotfiles and diagnostics data */
namespace IO
{

    void writePlotFile (    const AgentContainer& pc,
                            const CensusData& censusData,
                            const int num_diseases,
                            const std::vector<std::string>& disease_names,
                            const amrex::Real cur_time,
                            const int step);

    void writeFIPSData (    const AgentContainer& pc,
                            const CensusData& censusData,
                            const std::string& prefix,
                            const int num_diseases,
                            const std::vector<std::string>& disease_names,
                            const int step);
}
}

#endif
