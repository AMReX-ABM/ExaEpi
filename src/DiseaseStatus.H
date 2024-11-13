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
        for (MFIter mfi = a_agents.MakeMFIter(lev); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            if (np == 0) continue;

            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
            auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            auto symptomatic_withdraw_compliance = a_agents.symptomaticWithdrawCompliance();

            int i_RT = IntIdx::nattribs;
            int r_RT = RealIdx::nattribs;

            Gpu::DeviceVector<int> marked_for_hosp(np, 0);
            Gpu::DeviceVector<int> marked_for_ICU(np, 0);
            Gpu::DeviceVector<int> marked_for_vent(np, 0);
            auto marked_for_hosp_ptr = marked_for_hosp.data();
            auto marked_for_ICU_ptr = marked_for_ICU.data();
            auto marked_for_vent_ptr = marked_for_vent.data();

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
                    if (status_ptr[i] == Status::never || status_ptr[i] == Status::susceptible) {
                        return;
                    } else if (status_ptr[i] == Status::immune) {
                        counter_ptr[i] -= 1.0_prt;
                        if (counter_ptr[i] < 0.0_prt) {
                            counter_ptr[i] = 0.0_prt;
                            timer_ptr[i] = 0.0_prt;
                            status_ptr[i] = Status::susceptible;
                            return;
                        }
                    } else if (status_ptr[i] == Status::infected) {
                        counter_ptr[i] += 1;
                        if (counter_ptr[i] == 1) {
                            // just infected, check to see if this agent will be asymptomatic
                            if (Random(engine) < disease_parm_d->p_asymp) {
                                symptomatic_ptr[i] = SymptomStatus::asymptomatic;
                            } else {
                                symptomatic_ptr[i] = SymptomStatus::presymptomatic;
                            }
                        } else if (counter_ptr[i] == Math::floor(incubation_period_ptr[i])) {
                            AMREX_ASSERT(symptomatic_ptr[i] != SymptomStatus::symptomatic);
                            // at end of incubation period, symptoms start to show unless asymptomatic
                            if (symptomatic_ptr[i] == SymptomStatus::presymptomatic) {
                                symptomatic_ptr[i] = SymptomStatus::symptomatic;
                                if (symptomatic_withdraw_compliance > 0.0_rt && (Random(engine) < symptomatic_withdraw_compliance)) {
                                    withdrawn_ptr[i] = 1;
                                }
                                disease_parm_d->check_hospitalization(&(timer_ptr[i]),
                                                                      &(marked_for_ICU_ptr[i]),
                                                                      &(marked_for_vent_ptr[i]),
                                                                      age_group_ptr[i],
                                                                      engine);
                                if (timer_ptr[i] > 0) { marked_for_hosp_ptr[i] = 1; }
                            }
                        } else if (!inHospital(i,ptd)) {
                            if (counter_ptr[i] >= (latent_period_ptr[i] + infectious_period_ptr[i])) {
                                status_ptr[i] = Status::immune;
                                counter_ptr[i] =
                                    static_cast<ParticleReal>(RandomNormal(immune_length_mean, immune_length_std, engine));
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
                // check if not in hospital because this agent could have already been assigned a hospital for another disease
                if (marked_for_hosp_ptr[i] == 1 && !inHospital(i, ptd)) {
                    assign_hospital( i, hosp_i_ptr, hosp_j_ptr, ptd);
                }
            });

            for (int d = 0; d < n_disease; d++) {
                auto ds_arr = (*a_stats[d])[mfi].array();
                ParallelFor( np, [=] AMREX_GPU_DEVICE (int i) noexcept
                {
                    if (marked_for_hosp_ptr[i] == 1) {
                        Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                         home_j_ptr[i],
                                                         0,
                                                         DiseaseStats::hospitalization ),
                                               1.0_rt );
                    }
                    if (marked_for_ICU_ptr[i] == 1) {
                        Gpu::Atomic::AddNoRet( &ds_arr(  home_i_ptr[i],
                                                         home_j_ptr[i],
                                                         0,
                                                         DiseaseStats::ICU ),
                                               1.0_rt );
                    }
                    if (marked_for_vent_ptr[i] == 1) {
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
