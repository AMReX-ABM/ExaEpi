/*! @file CensusData.H
    \brief #CensusData class definition
*/

#pragma once

#include <AMReX_GpuContainers.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include <string>

#include "AgentContainer.H"
#include "CaseData.H"
#include "DemographicData.H"
#include "NAICS.H"
#include "Utils.H"

/*! \brief A class encapsulating all data relevant to the ICType Census */
struct CensusData {

    DemographicData demo;

    amrex::iMultiFab num_residents_mf; /*!< Number of residents in each community (grid cell);
                                            component 0: age under 5,
                                            component 1: age group 5-17,
                                            component 2: age group 18-29,
                                            component 3: age group 30-64,
                                            component 4: age group 65+,
                                            component 5: total. */
    amrex::iMultiFab unit_mf;          /*!< Unit number of each community */
    amrex::iMultiFab FIPS_mf;          /*!< FIPS code (component 0) and
                                            census tract number (component 1)
                                            of each community */
    amrex::iMultiFab comm_mf;          /*!< Community number */

    CensusData () {}

    void init(ExaEpi::TestParams& params, amrex::Geometry& geom, amrex::BoxArray& ba, amrex::DistributionMapping& dm);

    void initAgents(AgentContainer& pc, const int nborhood_size);

    void readWorkerflow(AgentContainer& pc, const std::string& workerflow_filename, const int workgroup_size);

    void assignTeachersAndWorkgroup(AgentContainer& pc);
    void assignMedicalWorkers(AgentContainer& pc);

    int countWorkersByComm(AgentContainer& pc, Gpu::HostVector<int>& a_array, const int a_naics = -1);
    int setAsMedicalWorkers(AgentContainer& pc, Gpu::HostVector<int>& a_array, bool a_local);
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void assignSchool(int* school_grade, int* school_id, const int age_group, const int nborhood, const RandomEngine& engine);

template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void setAgentDataAndAssignSchool (PTDType& ptd, int ip, int i, int j, int k, const amrex::GpuArray<amrex::Real, 2>& dx,
                                  amrex::Long id, int cpu, int age_group, int family, int nborhood, int n_disease,
                                  amrex::Array4<int> const& nr_arr, amrex::Array4<int> const& student_counts_arr,
                                  const amrex::RandomEngine& engine) {
    setAgentData(ptd, ip, i, j, dx, id, cpu, age_group, family, nborhood, n_disease);
    Gpu::Atomic::AddNoRet(&nr_arr(i, j, k, age_group), 1);

    assignSchool(&ptd.m_idata[IntIdx::school_grade][ip], &ptd.m_idata[IntIdx::school_id][ip], age_group, nborhood, engine);
    ptd.m_idata[IntIdx::school_closed][ip] = 0;

    // Increment the appropriate student counter based on the school assignment
    if (ptd.m_idata[IntIdx::school_id][ip] >= SchoolCensusIDType::daycare_5) {
        Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolCensusIDType::daycare_5 - 1), 1);
    } else if (ptd.m_idata[IntIdx::school_id][ip] > SchoolCensusIDType::none) {
        Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, ptd.m_idata[IntIdx::school_id][ip] - 1), 1);
    }
}
