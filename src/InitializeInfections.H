/*! @file InitializeInfections.H
    \brief #Routines to implement the initial infections
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


void setInitialCasesFromFile (AgentContainer& pc,
                              CaseData& cases,
                              const std::string& name,
                              int d_idx,
                              const Vector<int> &FIPS_codes,
                              const Vector<int> &unit_community_start,
                              iMultiFab &comm_mf,
                              const bool fast_bin);

void setInitialCasesRandom (AgentContainer& pc,
                            int num_cases,
                            const std::string& name,
                            int d_idx,
                            const Vector<int> &unit_community_start,
                            iMultiFab &comm_mf,
                            const bool fast_bin);



