/*! @file DiseaseStatus.H
 *  \brief Defines class for disease status updates */

#ifndef _DISEASE_STATUS_H_
#define _DISEASE_STATUS_H_

#include <vector>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>

#include "AgentDefinitions.H"

using namespace amrex;

using MFPtr = std::unique_ptr<MultiFab>;
using MFPtrVec = std::vector<MFPtr>;

/*! \brief Assigns a hospital community to an agent: right now
 *  it is the same as the home community */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void assign_hospital( const int a_i, /*!< agent index */
                      int* const a_hosp_i, /*!< hospital index i */
                      int* const a_hosp_j, /*!< hospital index j */
                      const PTDType& a_ptd /*!< Particle tile data */ )
{
    auto home_i_ptr = a_ptd.m_idata[IntIdx::home_i];
    auto home_j_ptr = a_ptd.m_idata[IntIdx::home_j];
    a_hosp_i[a_i] = home_i_ptr[a_i];
    a_hosp_j[a_i] = home_j_ptr[a_i];
}

/*! \brief Disease status and its updates for each agent
 *
 *  Contains data and functions for updating disease status for agents.
 */
template <  typename AC /*!< agent container type */,
            typename ACT, /*!< agent container tile type */
            typename ACTD, /*!< agent container tile data type */
            typename A /*!< agent type */>
class DiseaseStatus
{
    public:

        /*! \brief constructor */
        DiseaseStatus<AC,ACT,ACTD,A>() { }

        /*! \brief default destructor */
        virtual ~DiseaseStatus<AC,ACT,ACTD,A>() = default;

        /*! \brief Updates disease status of each agent at a given step and
         *   also updates a MultiFab that tracks disease statistics (hospitalization,
         *   ICU, ventilator, and death) in a community. */
        virtual void updateAgents(AC&, MFPtrVec&) const;

    protected:

};

/*! At a given step, update the disease status of each agent based on the following overall logic:
    + If agent status is #Status::never or #Status::susceptible, do nothing
    + If agent status is #Status::infected, then
      + Increment its counter by 1 day
      + If counter is within latent period, do nothing more
      + Else, use hospitalization probabilities (by age group)
        to decide if agent is hospitalized. If yes, use age group to set hospital timer. Also, use
        age-group-wise probabilities to move agent to ICU and then to ventilator. Adjust timer
        accordingly.
      + Update the community-wise disease stats tracker MultiFab according to hospitalization/ICU/vent
        status (using the agent's home community)
      + Else (beyond 3 days), count down hospital timer if agent is hospitalized. At end of hospital
        stay, determine if agent is #Status dead or #Status::immune. For non-hospitalized agents,
        set them to #Status::immune after latent length + infection length days.

    The input argument is a MultiFab with 4 components corresponding to "hospitalizations", "ICU",
    "ventilator", and "death". It contains the cumulative totals of these quantities for each
    community as the simulation progresses.
*/
template<typename AC, typename ACT, typename ACTD, typename A>
void DiseaseStatus<AC,ACT,ACTD,A>::updateAgents(AC& a_agents, /*!< Agent containter */
                                                MFPtrVec& a_stats /*!< MultiFab to store disease stats */ ) const
{
    BL_PROFILE("DiseaseStatus::updateAgents");
    int n_disease = a_agents.numDiseases();

    for (int lev = 0; lev <= a_agents.finestLevel(); ++lev)
    {
        auto& plev  = a_agents.GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();

            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
            auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            auto symptomatic_withdraw = a_agents.symptomaticWithdraw();
            auto symptomatic_withdraw_compliance = a_agents.symptomaticWithdrawCompliance();

            // fraction of age group 30-64 who are under 50 (assuming uniform distribution)
            Real u50frac = 0.57_rt;

            int i_RT = IntIdx::nattribs;
            int r_RT = RealIdx::nattribs;

            Gpu::DeviceVector<int> flag_hosp, flag_ICU, flag_vent;
            flag_hosp.resize(np);
            flag_ICU.resize(np);
            flag_vent.resize(np);
            auto flag_hosp_ptr = flag_hosp.data();
            auto flag_ICU_ptr = flag_ICU.data();
            auto flag_vent_ptr = flag_vent.data();
            ParallelFor( np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                flag_hosp_ptr[i] = 0;
                flag_ICU_ptr[i] = 0;
                flag_vent_ptr[i] = 0;
            });
            Gpu::synchronize();

            for (int d = 0; d < n_disease; d++ ) {

                auto status_ptr = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::status).data();
                auto symptomatic_ptr = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::symptomatic).data();

                auto timer_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::treatment_timer).data();
                auto counter_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::disease_counter).data();
                auto prob_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::prob).data();
                auto latent_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::latent_period).data();
                auto infectious_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::infectious_period).data();
                auto incubation_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::incubation_period).data();

                auto* disease_parm_d = a_agents.getDiseaseParameters_d(d);
                auto* disease_parm_h = a_agents.getDiseaseParameters_h(d);
                auto immune_length_mean = disease_parm_h->immune_length_mean;
                auto immune_length_std = disease_parm_h->immune_length_std;

                ParallelForRNG( np,
                                [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
                {
                    prob_ptr[i] = 1.0_rt;
                    if ( status_ptr[i] == Status::never ||
                         status_ptr[i] == Status::susceptible ) {
                        return;
                    }
                    else if (status_ptr[i] == Status::immune) {
                        counter_ptr[i] -= 1.0_rt;
                        if (counter_ptr[i] < 0.0_rt) {
                            counter_ptr[i] = 0.0_rt;
                            timer_ptr[i] = 0.0_rt;
                            status_ptr[i] = Status::susceptible;
                            return;
                        }
                    }
                    else if (status_ptr[i] == Status::infected) {
                        counter_ptr[i] += 1;
                        if (counter_ptr[i] == 1) {
                            if (Random(engine) < disease_parm_d->p_asymp[0]) {
                                symptomatic_ptr[i] = SymptomStatus::asymptomatic;
                            } else {
                                symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                            }
                        }
                        if (counter_ptr[i] == Math::floor(incubation_period_ptr[i])) {
                            if (symptomatic_ptr[i] != SymptomStatus::asymptomatic) {
                                symptomatic_ptr[i] = SymptomStatus::symptomatic;
                            }
                            if (    (symptomatic_ptr[i] == SymptomStatus::symptomatic)
                                &&  (symptomatic_withdraw)
                                &&  (Random(engine) < symptomatic_withdraw_compliance)) {
                                withdrawn_ptr[i] = 1;
                            }

                            if (symptomatic_ptr[i] == SymptomStatus::symptomatic) {
                                int flag_ICU_i = 0, flag_vent_i = 0;
                                Real num_days = 0;
                                disease_parm_d->check_hospitalization( num_days,
                                                                       flag_ICU_i,
                                                                       flag_vent_i,
                                                                       age_group_ptr[i],
                                                                       u50frac,
                                                                       engine );
                                timer_ptr[i] = ParticleReal(num_days);
                                if (timer_ptr[i] > 0) { flag_hosp_ptr[i] = 1; }
                                if (flag_ICU_i) { flag_ICU_ptr[i] = 1; }
                                if (flag_vent_i) { flag_vent_ptr[i] = 1; }
                            }
                        }
                        if (!isHospitalized(i,ptd)) {
                            if (counter_ptr[i] >= (latent_period_ptr[i] + infectious_period_ptr[i])) {
                                status_ptr[i] = Status::immune;
                                counter_ptr[i] = amrex::RandomNormal(immune_length_mean, immune_length_std, engine);
                                symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                                withdrawn_ptr[i] = 0;
                            }
                        }
                    }
                });

                Gpu::synchronize();
            }

            ParallelFor( np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                if (flag_hosp_ptr[i] == 1) {
                    AMREX_ALWAYS_ASSERT(!isHospitalized(i, ptd));
                    assign_hospital( i, hosp_i_ptr, hosp_j_ptr, ptd);
                }
            });

            for (int d = 0; d < n_disease; d++) {
                auto ds_arr = (*a_stats[d])[mfi].array();
                ParallelFor( np, [=] AMREX_GPU_DEVICE (int i) noexcept
                {
                    if (flag_hosp_ptr[i] == 1) {
                        Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                         home_j_ptr[i],
                                                         0,
                                                         DiseaseStats::hospitalization ),
                                               1.0_rt );
                    }
                    if (flag_ICU_ptr[i] == 1) {
                        Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                         home_j_ptr[i],
                                                         0,
                                                         DiseaseStats::ICU ),
                                               1.0_rt );
                    }
                    if (flag_vent_ptr[i] == 1) {
                        Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                         home_j_ptr[i],
                                                         0,
                                                         DiseaseStats::ventilator ),
                                               1.0_rt );
                    }
                });
            }

            Gpu::synchronize();
        }

    }
}
#endif
