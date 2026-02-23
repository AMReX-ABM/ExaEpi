/*! @file HospitalModel.H
 * \brief Contains the class describing agent interactions at a hospital
 */

#ifndef _HOSPITAL_MODEL_H_
#define _HOSPITAL_MODEL_H_

#include "AgentDefinitions.H"
#include "DiseaseParm.H"
#include "InteractionModel.H"
#include "Utils.H"

using namespace amrex;

/*! \brief Class describing agent interactions and treatment at hospital */
template <typename PCType, typename PTDType, typename PType>
class HospitalModel : public InteractionModel<PCType, PTDType, PType> {
  public:
    /*! \brief null constructor */
    HospitalModel (bool _fast_bin) : InteractionModel<PCType, PTDType, PType>(_fast_bin) {}

    /*! \brief default destructor */
    virtual ~HospitalModel() = default;

    /*! \brief Simulate agent interaction at hospital */
    virtual void interactAgents (PCType&, MultiFab&) override {
        // not yet implemented
    }

    using MFPtr = std::unique_ptr<MultiFab>;
    using MFPtrVec = std::vector<MFPtr>;

    /*! \brief Simulate treatment of agents in a hospital */
    virtual void treatAgents(PCType&, MFPtrVec&);

  protected:
  private:
};

/*! Simulate the treatment of agents and interactions between agents at a hospital
   and compute the infection probability for each agent:

    + For each agent *i*:
      + If the agent is not hospitalized, do nothing.
      + Else: implement the treatment
              discharge once treatment is complete
*/
template <typename PCType, typename PTDType, typename PType>
void HospitalModel<PCType, PTDType, PType>::treatAgents (PCType& a_agents, /*!< Agent container */
                                                         MFPtrVec& a_dstats /*!< disease stats */) {
    BL_PROFILE("HospitalModel::interactAgents");
    int n_disease = a_agents.numDiseases();

    for (int lev = 0; lev < a_agents.numLevels(); ++lev) {
        const auto dx = a_agents.Geom(lev).CellSizeArray();
        auto& plev = a_agents.GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = a_agents.MakeMFIter(lev); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& soa = ptile.GetStructOfArrays();
            auto& aos = ptile.GetArrayOfStructs();
            PType* pstruct = &(aos[0]);
            const auto np = ptile.numParticles();
            if (np == 0) { continue; }

            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
            auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();

            int i_RT = IntIdx::nattribs;
            int r_RT = RealIdx::nattribs;

            GpuArray<int*, ExaEpi::max_num_diseases> status_ptrs, symptomatic_ptrs;
            GpuArray<ParticleReal*, ExaEpi::max_num_diseases> counter_ptrs, timer_ptrs, incubation_per_ptrs, hospital_delay_ptrs;
            for (int d = 0; d < n_disease; d++) {
                status_ptrs[d] = soa.GetIntData(i_RT + i0(d) + IntIdxDisease::status).data();
                symptomatic_ptrs[d] = soa.GetIntData(i_RT + i0(d) + IntIdxDisease::symptomatic).data();
                counter_ptrs[d] = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::disease_counter).data();
                timer_ptrs[d] = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::treatment_timer).data();
                incubation_per_ptrs[d] = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::incubation_period).data();
                hospital_delay_ptrs[d] = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::hospital_delay).data();
            }

            Gpu::DeviceVector<int> is_alive(np);
            auto is_alive_ptr = is_alive.data();

            Gpu::DeviceVector<int> died_today(np, 0);
            Gpu::DeviceVector<int> exit_hosp(np * n_disease, -1);
            Gpu::DeviceVector<int> exit_ICU(np * n_disease, -1);
            Gpu::DeviceVector<int> exit_vent(np * n_disease, -1);
            auto died_today_ptr = died_today.data();
            auto exit_hosp_ptr = exit_hosp.data();
            auto exit_ICU_ptr = exit_ICU.data();
            auto exit_vent_ptr = exit_vent.data();

            ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                // if status_ptr for any one disease is dead, they should all be dead
                if (status_ptrs[0][i] == Status::dead) {
                    is_alive_ptr[i] = 0;
                    for (int d = 1; d < n_disease; d++) {
                        AMREX_ALWAYS_ASSERT(status_ptrs[d][i] == Status::dead);
                    }
                } else {
                    is_alive_ptr[i] = 1;
                    for (int d = 1; d < n_disease; d++) {
                        AMREX_ALWAYS_ASSERT(status_ptrs[d][i] != Status::dead);
                    }
                }
            });
            Gpu::synchronize();

            for (int d = 0; d < n_disease; d++) {

                auto* disease_parm_h = a_agents.getDiseaseParameters_h(d);
                auto immune_length_alpha = disease_parm_h->immune_length_alpha;
                auto immune_length_beta = disease_parm_h->immune_length_beta;
                auto* disease_parm_d = a_agents.getDiseaseParameters_d(d);

                ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
                    if (!inHospital(i, ptd)) {
                        // agent is not in hospital
                        return;
                    }
                    if (counter_ptrs[d][i] == Math::floor(incubation_per_ptrs[d][i]) + Math::floor(hospital_delay_ptrs[d][i])) {
                        // agent just started treatment
                        return;
                    }
                    if (timer_ptrs[d][i] == 0) {
                        // agent has recovered/died from disease d
                        return;
                    }
                    if (is_alive_ptr[i] == 0) {
                        // agent is dead
                        return;
                    }

                    AMREX_ALWAYS_ASSERT(status_ptrs[d][i] == Status::infected);
                    // check for exit conditions
                    if (timer_ptrs[d][i] > 2 * disease_parm_d->m_t_hosp_offset + 1) {
                        exit_vent_ptr[d * np + i] = 0;
                        exit_ICU_ptr[d * np + i] = 0;
                        exit_hosp_ptr[d * np + i] = 0;
                    } else if (timer_ptrs[d][i] == 2 * disease_parm_d->m_t_hosp_offset + 1) {
                        exit_vent_ptr[d * np + i] = 1;
                        exit_ICU_ptr[d * np + i] = 1;
                        exit_hosp_ptr[d * np + i] = 1;
                    } else if (timer_ptrs[d][i] > disease_parm_d->m_t_hosp_offset + 1) {
                        exit_ICU_ptr[d * np + i] = 0;
                        exit_hosp_ptr[d * np + i] = 0;
                    } else if (timer_ptrs[d][i] == disease_parm_d->m_t_hosp_offset + 1) {
                        exit_ICU_ptr[d * np + i] = 1;
                        exit_hosp_ptr[d * np + i] = 1;
                    } else if (timer_ptrs[d][i] > 1) {
                        exit_hosp_ptr[d * np + i] = 0;
                    } else if (timer_ptrs[d][i] == 1) {
                        exit_hosp_ptr[d * np + i] = 1;
                    }
                    // decrement days in hospital
                    timer_ptrs[d][i] -= 1.0_prt;
                    int status_d = 0;
                    if (timer_ptrs[d][i] == 0) {
                        // finished hospitalization period
                        status_d = DiseaseStats::hospitalization + 1;
                    } else if (timer_ptrs[d][i] == disease_parm_d->m_t_hosp_offset) {
                        // finished ICU hospitalization period, counted down to m_t_hosp_offset
                        status_d = DiseaseStats::ICU + 1;
                    } else if (timer_ptrs[d][i] == 2 * disease_parm_d->m_t_hosp_offset) {
                        // finished ventilator hospitalization period, counted down to 2 * m_t_hosp_offset
                        status_d = DiseaseStats::ventilator + 1;
                    }

                    if (status_d > 0) {
                        // Check if hospitalized patient recovers or dies
                        if (Random(engine) < disease_parm_d->m_hospToDeath[status_d - 1][age_group_ptr[i]]) {
                            is_alive_ptr[i] = 0;
                            died_today_ptr[i] = 1;
                            status_ptrs[d][i] = Status::dead;
                        } else {
                            // If alive, hospitalized patient recovers
                            status_ptrs[d][i] = Status::immune;
                            counter_ptrs[d][i] =
                                    static_cast<ParticleReal>(RandomGamma(immune_length_alpha, immune_length_beta, engine));
                            symptomatic_ptrs[d][i] = SymptomStatus::presymptomatic;
                            // don't withdraw yet since the agent could still have another disease
                            // withdrawn_ptr[i] = 0;
                            timer_ptrs[d][i] = 0.0_prt;
                        }
                    }
                });
                Gpu::synchronize();
            }

            Gpu::DeviceVector<int> rm_from_hosp(np, 0);
            Gpu::DeviceVector<int> rm_from_ICU(np, 0);
            Gpu::DeviceVector<int> rm_from_vent(np, 0);
            auto rm_from_hosp_ptr = rm_from_hosp.data();
            auto rm_from_ICU_ptr = rm_from_ICU.data();
            auto rm_from_vent_ptr = rm_from_vent.data();

            ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                if (!inHospital(i, ptd)) { return; }

                rm_from_hosp_ptr[i] = 1;
                rm_from_ICU_ptr[i] = 1;
                rm_from_vent_ptr[i] = 1;
                int count_h = 0;
                int count_i = 0;
                int count_v = 0;
                for (int d = 0; d < n_disease; d++) {
                    if (exit_hosp_ptr[d * np + i] > 0) {
                        rm_from_hosp_ptr[i] *= exit_hosp_ptr[d * np + i];
                        count_h++;
                    }
                    if (exit_ICU_ptr[d * np + i] > 0) {
                        rm_from_ICU_ptr[i] *= exit_ICU_ptr[d * np + i];
                        count_i++;
                    }
                    if (exit_vent_ptr[d * np + i] > 0) {
                        rm_from_vent_ptr[i] *= exit_vent_ptr[d * np + i];
                        count_v++;
                    }
                }
                if (count_h == 0) { rm_from_hosp_ptr[i] = 0; }
                if (count_i == 0) { rm_from_ICU_ptr[i] = 0; }
                if (count_v == 0) { rm_from_vent_ptr[i] = 0; }

                if (is_alive_ptr[i] == 0) {
                    // agent has died
                    for (int d = 0; d < n_disease; d++) {
                        status_ptrs[d][i] = Status::dead;
                    }
                    hosp_i_ptr[i] = -1;
                    hosp_j_ptr[i] = -1;
                    withdrawn_ptr[i] = 0;
                } else {
                    // check if agent can be discharged from hospital
                    ParticleReal sum_timers = 0;
                    for (int d = 0; d < n_disease; d++) {
                        sum_timers += timer_ptrs[d][i];
                    }
                    if (sum_timers == 0) {
                        // discharge patient
                        hosp_i_ptr[i] = -1;
                        hosp_j_ptr[i] = -1;
                        withdrawn_ptr[i] = 0;
                        PType& p = pstruct[i];
                        p.pos(0) = static_cast<ParticleReal>((home_i_ptr[i] + 0.5_rt) * dx[0]);
                        p.pos(1) = static_cast<ParticleReal>((home_j_ptr[i] + 0.5_rt) * dx[1]);
                    } else {
                        AMREX_ALWAYS_ASSERT(!rm_from_hosp_ptr[i]);
                    }
                }
            });
            Gpu::synchronize();

            // accumulate statistics
            for (int d = 0; d < n_disease; d++) {
                auto ds_arr = (*a_dstats[d])[mfi].array();
                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (died_today_ptr[i]) {
                        Gpu::Atomic::AddNoRet(&ds_arr(home_i_ptr[i], home_j_ptr[i], 0, DiseaseStats::death), 1.0_rt);
                    }
                    if (rm_from_hosp_ptr[i]) {
                        Gpu::Atomic::AddNoRet(&ds_arr(home_i_ptr[i], home_j_ptr[i], 0, DiseaseStats::hospitalization), -1.0_rt);
                    }
                    if (rm_from_ICU_ptr[i]) {
                        Gpu::Atomic::AddNoRet(&ds_arr(home_i_ptr[i], home_j_ptr[i], 0, DiseaseStats::ICU), -1.0_rt);
                    }
                    if (rm_from_vent_ptr[i]) {
                        Gpu::Atomic::AddNoRet(&ds_arr(home_i_ptr[i], home_j_ptr[i], 0, DiseaseStats::ventilator), -1.0_rt);
                    }
                });
                Gpu::synchronize();
            }
        }
    }
}

#endif
