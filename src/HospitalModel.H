/*! @file HospitalModel.H
 * \brief Contains the class describing agent interactions at a hospital
 */

#ifndef _HOSPITAL_MODEL_H_
#define _HOSPITAL_MODEL_H_

#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_REAL.H>

#include "AgentDefinitions.H"
#include "DiseaseParm.H"
#include "InteractionModel.H"
#include "NAICS.H"
#include "Utils.H"

using namespace amrex;

struct HospMod {
    enum {
        score = 0,  // hospital score (treatment quality)
        capacity,   // patient capacity
        r_a2t_mw,   // ratio of active to total medical workers
        load,       // hospital load (total patients / capacity)
        n_patients, // number of patients
        n_a_mw,     // number of active medical workers
        n_t_mw,     // total number of medical workers
        ncomps
    };
};

#ifndef FAST_INTERACTIONS
/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent in a hospital. */
template <typename PTDType>
struct BinaryInteractionHospWorkers {
    AMREX_GPU_HOST_DEVICE
    ParticleReal operator()(const int infectious_i, const int susceptible_i, const PTDType& a_ptd,
                            const DiseaseParm* const a_lparm, const Real a_hosp_scale) const noexcept {
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::naics][infectious_i] == a_ptd.m_idata[IntIdx::naics][susceptible_i]);
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::naics][infectious_i] == NAICSCodes::NAICS::med_sca);
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::workgroup][infectious_i] == a_ptd.m_idata[IntIdx::workgroup][susceptible_i]);
        AMREX_ALWAYS_ASSERT(a_ptd.m_idata[IntIdx::work_i][infectious_i] == a_ptd.m_idata[IntIdx::work_i][susceptible_i] &&
                            a_ptd.m_idata[IntIdx::work_j][infectious_i] == a_ptd.m_idata[IntIdx::work_j][susceptible_i]);
        // infect *= i_mask;
        // infect *= j_mask;
        return a_lparm->xmit_hosp * a_hosp_scale;
    }
};
#endif

template <typename PTDType>
struct HospWorkerCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator()(const int idx, const PTDType& ptd) const noexcept {
        return !inHospital(idx, ptd) && ptd.m_idata[IntIdx::work_i][idx] >= 0 && ptd.m_idata[IntIdx::workgroup][idx] > 0 &&
               !ptd.m_idata[IntIdx::withdrawn][idx] && ptd.m_idata[IntIdx::naics][idx] == NAICSCodes::NAICS::med_sca &&
               ptd.m_idata[IntIdx::air_travel][idx] < 0 && ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};

/*! \brief Class describing agent interactions and treatment at hospital */
template <typename PCType, typename PTDType, typename PType>
class HospitalModel : public InteractionModel<PCType, PTDType, PType> {
  public:
    /*! \brief Constructor - reads in inputs */
    HospitalModel (bool _fast_bin, const std::string& a_prefix) : InteractionModel<PCType, PTDType, PType>(_fast_bin) {
        ParmParse pp(a_prefix);
        pp.query("num_patients_per_doctor", m_num_patients_per_doctor);
        pp.query("score_minimum", m_sc_floor);
        pp.query("halfscore_load", m_sc_half_load);
        pp.query("write_pltfiles", m_write_pltfiles);

        m_very_large_number = 4.0_rt * m_sc_half_load;

        ParmParse pp_agent("agent");
        pp_agent.query("plot_int", m_plot_int);
        if (m_plot_int < 0) { m_write_pltfiles = false; }

        Print() << "Initialized hospital model...\n";
        Print() << "  Number of patients per doctor: " << m_num_patients_per_doctor << "\n";
        Print() << "  Minimum score for overloaded hospitals: " << m_sc_floor << "\n";
        Print() << "  Load for score=0.5: " << m_sc_half_load << "\n";
        Print() << "  Write hospital data plotfiles: " << (m_write_pltfiles ? "yes" : "no") << "\n";
    }

    /*! \brief default destructor */
    virtual ~HospitalModel() = default;

    /*! \brief Simulate agent interaction at hospital */
    virtual void interactAgents (PCType& a_agents, MultiFab&) override {
#ifdef FAST_INTERACTIONS
        fastInteractHospWorkers(a_agents);
#else
        interactAgentsImpl<HospitalModel<PCType, PTDType, PType>, PCType, PTDType, HospWorkerCandidate<PTDType>,
                           BinaryInteractionHospWorker<PTDType>>(*this, a_agents, IntIdx::workgroup);
#endif
    }

    using MFPtr = std::unique_ptr<MultiFab>;
    using MFPtrVec = std::vector<MFPtr>;

    /*! \brief Simulate treatment of agents in a hospital */
    virtual void treatAgents(PCType&, MFPtrVec&, int);

    /*! \brief Interactions between hospital workers */
    void fastInteractHospWorkers(PCType&);

    /*! \brief Allocate hospital score multifab */
    void initHospitalScoreMF (const BoxArray& a_ba, const DistributionMapping& a_dm) {
        m_data = std::make_unique<MultiFab>(a_ba, a_dm, HospMod::ncomps, 0);
        m_data->setVal(1.0, 0, 1);
        m_data->setVal(0.0, 1, m_data->nComp() - 1);
    }

    /*! \brief Update patient capacity of hospitals in each community */
    void updateCapacities(const MFPtr&);

    /*! \brief Compute hospital scores (treatment quality) in each community */
    void computeHospitalScores(const PCType&, int);

  protected:
    /*! MultiFab with hospital data - see HospMod for components */
    MFPtr m_data;

    /*! Number of patients that a doctor can see in a day */
    Real m_num_patients_per_doctor = 50.0_rt;

    /*! Minimum hospital score even at very high loading */
    Real m_sc_floor = 0.1_rt;

    /*! Load at which hospital score falls to 0.5 */
    Real m_sc_half_load = 5.0_rt;

    /*! Number of hospitals operating beyond capacity */
    Long m_num_overloaded_hospitals = Long(0);

    /*! Write out hospital data to AMReX plot files */
    bool m_write_pltfiles = false;

    /*! Number of underserved patients */
    int m_num_underserved_patients;

    /*! Plotting interval */
    int m_plot_int = 1;

    /*! "Very large" finite number to set hospital load to when zero active medical workers */
    Real m_very_large_number = 4.0_rt * m_sc_half_load;

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
void HospitalModel<PCType, PTDType, PType>::treatAgents (PCType& a_agents,   /*!< Agent container */
                                                         MFPtrVec& a_dstats, /*!< disease stats */
                                                         const int a_iter) {
    BL_PROFILE("HospitalModel::interactAgents");
    int n_disease = a_agents.numDiseases();

    computeHospitalScores(a_agents, a_iter);
    m_num_underserved_patients = 0;

    for (int lev = 0; lev < a_agents.numLevels(); ++lev) {
        auto& plev = a_agents.GetParticles(lev);
        const auto& geom = a_agents.Geom(lev);
        const auto dx = geom.CellSizeArray();
        const auto plo = geom.ProbLoArray();
        const auto dxi = geom.InvCellSizeArray();
        const auto domain = geom.Domain();

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
            auto hosp_data_arr = m_data->const_array(mfi);

            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
            auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();

            int i_RT = IntIdx::nattribs;
            int r_RT = RealIdx::nattribs;

            GpuArray<int*, ExaEpi::max_num_diseases> status_ptrs, symptomatic_ptrs;
            GpuArray<ParticleReal*, ExaEpi::max_num_diseases> counter_ptrs, timer_ptrs, incubation_per_ptrs, treat_q_ptrs;
            for (int d = 0; d < n_disease; d++) {
                status_ptrs[d] = soa.GetIntData(i_RT + i0(d) + IntIdxDisease::status).data();
                symptomatic_ptrs[d] = soa.GetIntData(i_RT + i0(d) + IntIdxDisease::symptomatic).data();
                counter_ptrs[d] = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::disease_counter).data();
                timer_ptrs[d] = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::treatment_timer).data();
                treat_q_ptrs[d] = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::treatment_quality).data();
                incubation_per_ptrs[d] = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::incubation_period).data();
            }

            Gpu::DeviceVector<int> is_alive(np);
            auto is_alive_ptr = is_alive.data();

            Gpu::DeviceVector<int> flag_status(np);
            auto flag_status_ptr = flag_status.data();

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
                flag_status_ptr[i] = 0; // 0: nothing changed on this day
            });
            Gpu::synchronize();

            Gpu::Buffer<int> underserved_agents({0});
            auto* underserved_agents_ptr = underserved_agents.data();

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
                    if (counter_ptrs[d][i] == Math::floor(incubation_per_ptrs[d][i])) {
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

                    // decrement days in hospital
                    timer_ptrs[d][i] -= 1.0_prt;
                    // accumulate treatement quality
                    PType& p = pstruct[i];
                    auto iv = getParticleCell(p, plo, dxi, domain);
                    AMREX_ALWAYS_ASSERT((iv[0] == ptd.m_idata[IntIdx::hosp_i][i]) && (iv[1] == ptd.m_idata[IntIdx::hosp_j][i]));
                    treat_q_ptrs[d][i] *= hosp_data_arr(iv, HospMod::score);
                    if ((d == 0) && (treat_q_ptrs[d][i] < 1.0_prt)) {
                        Gpu::Atomic::Add(underserved_agents_ptr, 1);
                    }

                    if (timer_ptrs[d][i] == 0) {
                        // finished hospitalization period
                        flag_status_ptr[i] = DiseaseStats::hospitalization + 1;
                    } else if (timer_ptrs[d][i] == disease_parm_d->m_t_hosp_offset) {
                        // finished ICU hospitalization period, counted down to m_t_hosp_offset
                        flag_status_ptr[i] = DiseaseStats::ICU + 1;
                    } else if (timer_ptrs[d][i] == 2 * disease_parm_d->m_t_hosp_offset) {
                        // finished ventilator hospitalization period, counted down to 2 * m_t_hosp_offset
                        flag_status_ptr[i] = DiseaseStats::ventilator + 1;
                    }
                    if (flag_status_ptr[i] > 0) {
                        // Check if hospitalized patient recovers or dies
                        auto p_survive = 1.0_rt - disease_parm_d->m_hospToDeath[flag_status_ptr[i] - 1][age_group_ptr[i]];
                        p_survive *= treat_q_ptrs[d][i];
                        if (Random(engine) >= p_survive) {
                            // patient dies
                            is_alive_ptr[i] = 0;
                            flag_status_ptr[i] *= -1;
                            status_ptrs[d][i] = Status::dead;
                        } else {
                            // If alive, hospitalized patient recovers
                            status_ptrs[d][i] = Status::immune;
                            counter_ptrs[d][i] =
                                    static_cast<ParticleReal>(RandomGamma(immune_length_alpha, immune_length_beta, engine));
                            symptomatic_ptrs[d][i] = SymptomStatus::presymptomatic;
                            withdrawn_ptr[i] = 0;
                            timer_ptrs[d][i] = 0.0_prt;
                        }
                    }
                });
            }

            Gpu::synchronize();
            m_num_underserved_patients += *(underserved_agents.copyToHost());

            ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                if (!inHospital(i, ptd)) { return; }

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
                    }
                }
            });
            Gpu::synchronize();

            for (int d = 0; d < n_disease; d++) {
                auto ds_arr = (*a_dstats[d])[mfi].array();
                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (flag_status_ptr[i] < 0) {
                        Gpu::Atomic::AddNoRet(&ds_arr(home_i_ptr[i], home_j_ptr[i], 0, DiseaseStats::death), 1.0_rt);
                    }
                    if (std::abs(flag_status_ptr[i]) > DiseaseStats::hospitalization) {
                        Gpu::Atomic::AddNoRet(&ds_arr(home_i_ptr[i], home_j_ptr[i], 0, DiseaseStats::hospitalization), -1.0_rt);
                    }
                    if (std::abs(flag_status_ptr[i]) > DiseaseStats::ICU) {
                        Gpu::Atomic::AddNoRet(&ds_arr(home_i_ptr[i], home_j_ptr[i], 0, DiseaseStats::ICU), -1.0_rt);
                    }
                    if (std::abs(flag_status_ptr[i]) > DiseaseStats::ventilator) {
                        Gpu::Atomic::AddNoRet(&ds_arr(home_i_ptr[i], home_j_ptr[i], 0, DiseaseStats::ventilator), -1.0_rt);
                    }
                });
                Gpu::synchronize();
            }
        }
    }

    ParallelDescriptor::ReduceIntSum( &m_num_underserved_patients, 1 );

    amrex::Print() << "Day " << a_iter << ": " << m_num_overloaded_hospitals << " hospitals over capacity, " << m_num_underserved_patients << " underserved hospitalized agents\n";

}

/*! Interaction between medical workers at a hospital */
template <typename PCType, typename PTDType, typename PType>
void HospitalModel<PCType, PTDType, PType>::fastInteractHospWorkers (PCType& a_agents /*!< Agent container */) {
    BL_PROFILE(__func__);
    int n_disease = a_agents.numDiseases();

    HospWorkerCandidate<PTDType> isCandidate;
    Vector<Gpu::DeviceVector<int>> infected_group_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_group_asymp_d(OMP_MAX_THREADS);

    for (int lev = 0; lev < a_agents.numLevels(); ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = a_agents.MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = a_agents.ParticlesAt(lev, mfi);

            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            if (np == 0) { continue; }

            auto& soa = ptile.GetStructOfArrays();
            auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();

            GetCommunityIndex<PTDType> getCommunityIndex;
            getCommunityIndex.init(a_agents.Geom(lev), mfi.tilebox(), a_agents.comm_mf[mfi].array());
            auto plo = getCommunityIndex.plo;
            auto dxi = getCommunityIndex.dxi;
            auto domain = getCommunityIndex.domain;
            auto valid_box = getCommunityIndex.valid_box;
            auto bin_size = getCommunityIndex.bin_size;
            auto comm_to_local_index_d = getCommunityIndex.comm_to_local_index_d;


            int max_communities = getCommunityIndex.max();
            int max_group = a_agents.getMaxGroup(IntIdx::workgroup) + 1;

            infected_group_d[OMP_THREAD_NUM].resize(max_communities * max_group);
            auto infected_group_d_ptr = infected_group_d[OMP_THREAD_NUM].data();

            infected_group_asymp_d[OMP_THREAD_NUM].resize(max_communities * max_group);
            auto infected_group_asymp_d_ptr = infected_group_asymp_d[OMP_THREAD_NUM].data();

            for (int d = 0; d < n_disease; d++) {
                devMemset(infected_group_d_ptr, 0, infected_group_d[OMP_THREAD_NUM].size() * sizeof(int));
                devMemset(infected_group_asymp_d_ptr, 0, infected_group_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));

                auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                auto lparm = a_agents.getDiseaseParameters_d(d);
                auto lparm_h = a_agents.getDiseaseParameters_h(d);
                Real scale = 1.0_rt; // TODO this should vary based on cell
                Real infect = 1.0_rt - lparm_h->vac_eff;
                auto d_ptr = getCommunityIndex.comm_to_local_index_d.data();

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (isInfectious(i, ptd, d) && isCandidate(i, ptd)) {
                        Box tbx;
                        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                        auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                        auto community = d_ptr[tidx];
                        int group_i = community * max_group + workgroup_ptr[i];
                        if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                            Gpu::Atomic::AddNoRet(&infected_group_asymp_d_ptr[group_i], 1);
                        } else {
                            Gpu::Atomic::AddNoRet(&infected_group_d_ptr[group_i], 1);
                        }
                    }
                });
                Gpu::synchronize();

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (isSusceptible(i, ptd, d) && isCandidate(i, ptd)) {
                        Box tbx;
                        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                        auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                        auto community = d_ptr[tidx];
                        int group_i = community * max_group + workgroup_ptr[i];
                        int num_infected_group = infected_group_d_ptr[group_i];
                        Real group_prob = 1.0_prt - infect * lparm->xmit_hosp * scale;
                        prob_ptr[i] *= static_cast<ParticleReal>(std::pow(group_prob, num_infected_group));
                        int num_infected_group_asymp = infected_group_asymp_d_ptr[group_i];
                        Real group_prob_asymp = 1.0_prt - lparm->asymp_relative_inf * infect * lparm->xmit_hosp * scale;
                        prob_ptr[i] *= static_cast<ParticleReal>(std::pow(group_prob_asymp, num_infected_group_asymp));
                    }
                });
                Gpu::synchronize();
            }
        }
    }
}

/*! Given a MultiFab with the number of active medical workers in each community, updates the
 *  hospital capacities for each community. The input argument is a MultiFab with two components
 *  - the total number of medical workers and the currently-active number of medical workers. */
template <typename PCType, typename PTDType, typename PType>
void HospitalModel<PCType, PTDType, PType>::updateCapacities (const MFPtr& a_num_medworkers /*!< Number of medical workers */) {
    BL_PROFILE("HospitalModel::updateCapacities");

    for (MFIter mfi(*m_data); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.tilebox();
        const auto n_mw_arr = a_num_medworkers->const_array(mfi);
        auto data_arr = m_data->array(mfi);
        auto p_per_d = m_num_patients_per_doctor;

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            data_arr(i, j, k, HospMod::n_t_mw) = n_mw_arr(i, j, k, 0);
            data_arr(i, j, k, HospMod::n_a_mw) = n_mw_arr(i, j, k, 1);
            data_arr(i, j, k, HospMod::capacity) = n_mw_arr(i, j, k, 1) * p_per_d;

            if (n_mw_arr(i, j, k, 0) == 0) {
                AMREX_ALWAYS_ASSERT(n_mw_arr(i, j, k, 1) == 0);
                data_arr(i, j, k, HospMod::r_a2t_mw) = 0.0_rt;
            } else {
                data_arr(i, j, k, HospMod::r_a2t_mw) = n_mw_arr(i, j, k, 1) / n_mw_arr(i, j, k, 0);
            }
        });
    }
}

/*! Computes the hospital scores (treatment quality) for each community based on the number of
 *  patients in that community that are in the hospital and the capacity of the hospital based
 *  on the available number of medical workers. The "load" is the ratio of the number of patients
 *  and the capacity. For load < 1, the hospital score is 1.0. For loads > 1, the hospital score
 *  deteriorates. */
template <typename PCType, typename PTDType, typename PType>
void HospitalModel<PCType, PTDType, PType>::computeHospitalScores (const PCType& a_agents, /*!< Agent container */
                                                                   const int a_iter /*!< Day/iteration */) {
    BL_PROFILE("HospitalModel::computeHospitalScores");

    const int lev = 0;
    AMREX_ASSERT(a_agents.OK());
    AMREX_ASSERT(numParticlesOutOfRange(a_agents, 0) == 0);

    const auto& geom = a_agents.Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const auto domain = geom.Domain();
    auto& mf = *m_data;
    const auto very_large_number = m_very_large_number;

    // Compute the number of patients in a hospital for each community
    mf.setVal(0.0, HospMod::n_patients, 1);
    ParticleToMesh(
            a_agents, mf, lev,
            [=] AMREX_GPU_DEVICE (const typename PCType::ParticleTileType::ConstParticleTileDataType& ptd, int i,
                                 Array4<Real> const& mf_arr) {
                auto p = ptd.m_aos[i];
                auto iv = getParticleCell(p, plo, dxi, domain);
                if (inHospital(i, ptd)) {
                    AMREX_ALWAYS_ASSERT((iv[0] == ptd.m_idata[IntIdx::hosp_i][i]) && (iv[1] == ptd.m_idata[IntIdx::hosp_j][i]));
                    Gpu::Atomic::AddNoRet(&mf_arr(iv, HospMod::n_patients), 1.0_rt);
                }
            },
            false);

    // Compute hospital loads and scores
    Gpu::Buffer<Long> overloaded_hospitals({0});
    auto* overloaded_hosp_ptr = overloaded_hospitals.data();
    for (MFIter mfi(*m_data); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.tilebox();
        auto data_arr = m_data->array(mfi);

        const auto sc_floor = m_sc_floor;
        const auto sc_half_load = m_sc_half_load;

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Compute load
            if (data_arr(i, j, k, HospMod::n_t_mw) == 0) {
                data_arr(i, j, k, HospMod::load) = 0.0_prt; // exclude from hospital score modeling
            } else  if (data_arr(i, j, k, HospMod::capacity) == 0) {
                if (data_arr(i, j, k, HospMod::n_patients) == 0) {
                    data_arr(i, j, k, HospMod::load) = 0.0_prt;
                } else {
                    data_arr(i, j, k, HospMod::load) = very_large_number;
                }
            } else {
                data_arr(i, j, k, HospMod::load) = data_arr(i, j, k, HospMod::n_patients) / data_arr(i, j, k, HospMod::capacity);
            }
            // Compute score
            if (data_arr(i, j, k, HospMod::load) <= 1.0) {
                data_arr(i, j, k, HospMod::score) = 1.0_rt;
            } else {
                auto x = data_arr(i, j, k, HospMod::load) - 1.0_rt;
                data_arr(i, j, k, HospMod::score) =
                        1.0_rt - (1.0_rt - sc_floor) * std::exp(x) / (std::exp(sc_half_load) + std::exp(x));
                Gpu::Atomic::Add(overloaded_hosp_ptr, Long(1));
            }
        });
    }
    Gpu::synchronize();

    m_num_overloaded_hospitals = *(overloaded_hospitals.copyToHost());
    ParallelDescriptor::ReduceLongSum(&m_num_overloaded_hospitals, 1);

    if (m_write_pltfiles && (a_iter%m_plot_int == 0)) {
        char iter_str[12];
        sprintf(iter_str, "%05d", a_iter);
        {
            std::string op_fname = "hospital_data_" + std::string(iter_str);
            Print() << "Writing " << op_fname << "\n";
            Vector<std::string> varnames =
                    {"hospital_score", "patient_capacity",           "active_to_total_medworkers", "hospital_load",
                     "num_patients",   "num_active_medical_workers", "num_total_medical_workets"};
            WriteSingleLevelPlotfile(op_fname, *m_data, varnames, geom, a_iter, a_iter);
        }
    }
}

#endif
