/*! @file CensusData.H
    \brief #CensusData class definition
*/

#pragma once

#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_iMultiFab.H>

#include <string>

#include "Utils.H"
#include "DemographicData.H"
#include "AgentContainer.H"
#include "CaseData.H"

/*! \brief A class encapsulating all data relevant to the ICType Census */
struct CensusData
{

    DemographicData demo;

    amrex::iMultiFab num_residents_mf; /*!< Number of residents in each community (grid cell);
                                            component 0: age under 5,
                                            component 1: age group 5-17,
                                            component 2: age group 18-29,
                                            component 3: age group 30-64,
                                            component 4: age group 65+,
                                            component 4: total. */
    amrex::iMultiFab unit_mf;          /*!< Unit number of each community */
    amrex::iMultiFab FIPS_mf;          /*!< FIPS code (component 0) and
                                            census tract number (component 1)
                                            of each community */
    amrex::iMultiFab comm_mf;          /*!< Community number */

    CensusData () {}

    void init (ExaEpi::TestParams &params, amrex::Geometry &geom, amrex::BoxArray &ba, amrex::DistributionMapping &dm);

    void initAgents (AgentContainer &pc, const int nborhood_size);

    void read_workerflow (AgentContainer& pc, const std::string &workerflow_filename, const int workgroup_size);

  //private:
    void assignTeachersAndWorkgroup (AgentContainer& pc, const int workgroup_size);

};


