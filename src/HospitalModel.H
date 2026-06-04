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
        score = 0,     // hospital score (treatment quality)
        capacity,      // patient capacity (staffed beds, scaled by workforce availability)
        load,          // hospital load (total patients / capacity)
        n_patients,    // number of patients
        n_avail_mw,    // number of currently-available frontline medical workers
        n_baseline_mw, // frontline medical workforce at full strength (fixed at init)
        bed_supply,    // staffed-bed supply at full strength (fixed at init)
        ncomps
    };
};

template <typename PTDType>
struct HospWorkerCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator()(const int idx, const PTDType& ptd) const noexcept {
        return !inHospital(idx, ptd) && ptd.m_idata[IntIdx::work_i][idx] >= 0 && ptd.m_idata[IntIdx::workgroup][idx] > 0 &&
               !ptd.m_idata[IntIdx::withdrawn][idx] && ptd.m_idata[IntIdx::naics][idx] == NAICSCodes::NAICS::med_sca &&
               ptd.m_idata[IntIdx::air_travel][idx] < 0 && ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};

template <typename PTDType>
struct HospPatientCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator()(const int idx, const PTDType& ptd) const noexcept { return inHospital(idx, ptd); }
};

template <typename PTDType>
struct HospAgentCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator()(const int idx, const PTDType& ptd) const noexcept {
        HospWorkerCandidate<PTDType> isMW;
        HospPatientCandidate<PTDType> isP;
        return (isMW(idx, ptd) || isP(idx, ptd));
    }
};

#ifndef FAST_INTERACTIONS
/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent in a hospital. */
template <typename PTDType>
struct BinaryInteractionHospAgents {
    AMREX_GPU_HOST_DEVICE
    ParticleReal operator()(const int infectious_i, const int susceptible_i, const PTDType& a_ptd,
                            const DiseaseParm* const a_lparm, const Real a_hosp_scale) const noexcept {
        // infect *= i_mask;
        // infect *= j_mask;
        HospWorkerCandidate<PTDType> isMW;
        HospPatientCandidate<PTDType> isP;
        if (isMW(infectious_i, a_ptd) && isMW(susceptible_i, a_ptd)) {
            return a_lparm->xmit_hosp_d2d * a_hosp_scale;
        } else if (isP(infectious_i, a_ptd) && isMW(susceptible_i, a_ptd)) {
            return a_lparm->xmit_hosp_p2d * a_hosp_scale;
        } else if (isMW(infectious_i, a_ptd) && isP(susceptible_i, a_ptd)) {
            return a_lparm->xmit_hosp_d2p * a_hosp_scale;
        } else if (isP(infectious_i, a_ptd) && isP(susceptible_i, a_ptd)) {
            return a_lparm->xmit_hosp_p2p * a_hosp_scale;
        }
    }
};
#endif

/*! \brief Class describing agent interactions and treatment at hospital */
template <typename PCType, typename PTDType, typename PType>
class HospitalModel : public InteractionModel<PCType, PTDType, PType> {
  public:
    /*! \brief Constructor - reads in inputs */
    HospitalModel (bool _fast_bin, const std::string& a_prefix, bool a_model_medical_workers)
        : InteractionModel<PCType, PTDType, PType>(_fast_bin), m_model_medical_workers(a_model_medical_workers) {
        ParmParse pp(a_prefix);
        pp.query("staffed_beds_per_1000", m_beds_per_1000);
        pp.query("use_HHS_data", m_use_hhs_data);
        pp.query("hospital_data_file", m_hospital_data_file);
        pp.query("score_minimum", m_sc_floor);
        pp.query("halfscore_load", m_sc_half_load);
        pp.query("write_pltfiles", m_write_pltfiles);

        m_very_large_number = 4.0_rt * m_sc_half_load;

        ParmParse pp_agent("agent");
        pp_agent.query("plot_int", m_plot_int);
        if (m_plot_int < 0) { m_write_pltfiles = false; }

        if (m_model_medical_workers) {
            Print() << "Initialized hospital model...\n";
            if (m_use_hhs_data) {
                Print() << "  Bed supply from HHS data file: " << m_hospital_data_file << "\n";
            } else {
                Print() << "  Staffed beds per 1000 population (at full strength): " << m_beds_per_1000 << "\n";
            }
            Print() << "  Minimum score for overloaded hospitals: " << m_sc_floor << "\n";
            Print() << "  Load for score=0.5: " << m_sc_half_load << "\n";
            Print() << "  Write hospital data plotfiles: " << (m_write_pltfiles ? "yes" : "no") << "\n";
        }
    }

    /*! \brief default destructor */
    virtual ~HospitalModel() = default;

    /*! \brief Simulate agent interaction at hospital */
    virtual void interactAgents (PCType& a_agents, MultiFab&) override {
#ifdef FAST_INTERACTIONS
        fastInteractHospital(a_agents);
#else
        interactAgentsImpl<HospitalModel<PCType, PTDType, PType>, PCType, PTDType, HospAgentCandidate<PTDType>,
                           BinaryInteractionHospAgents<PTDType>>(*this, a_agents, IntIdx::workgroup);
#endif
    }

    using MFPtr = std::unique_ptr<MultiFab>;
    using MFPtrVec = std::vector<MFPtr>;

    /*! \brief Simulate treatment of agents in a hospital */
    virtual void treatAgents(PCType&, MFPtrVec&, int);

    /*! \brief Interactions between agents in hospital */
    void fastInteractHospital(PCType&);

    /*! \brief Allocate hospital score multifab */
    void initHospitalScoreMF (MultiFab* const a_data) {
        AMREX_ASSERT(a_data != nullptr);
        m_data = a_data;
        m_data->setVal(1.0, 0, 1);
        m_data->setVal(0.0, 1, m_data->nComp() - 1);
    }

    /*! \brief Set the fixed staffed-bed supply per community from the residential population */
    void initBedSupply(const MultiFab&);

    /*! \brief Set the fixed staffed-bed supply per community directly (e.g. from HHS data) */
    void setBedSupply(const MultiFab&);

    /*! \brief whether to set the bed supply from real HHS county data */
    bool useHHSData () const { return m_use_hhs_data; }

    /*! \brief path to the per-county hospital bed-supply data file */
    const std::string& hospitalDataFile () const { return m_hospital_data_file; }

    /*! \brief Update patient capacity of hospitals in each community */
    void updateCapacities(const MFPtr&);

    /*! \brief Compute hospital scores (treatment quality) in each community */
    void computeHospitalScores(const PCType&, int);

  protected:
    /*! MultiFab with hospital data - see HospMod for components */
    MultiFab* m_data = nullptr;

    /*! Whether the medical-workers / hospital-capacity model is active. When false,
     *  treatment uses the load-independent baseline mortality (reproduces development). */
    bool m_model_medical_workers = false;

    /*! Staffed hospital beds per 1000 population at full workforce strength.
     *  Sets the per-community bed supply: bed_supply = (m_beds_per_1000/1000) x population.
     *  Ignored when m_use_hhs_data is true. */
    Real m_beds_per_1000 = 2.4_rt;

    /*! Whether to set the per-community bed supply from real HHS county data
     *  (built by utilities/build_hospital_data.py) instead of the per-capita formula. */
    bool m_use_hhs_data = false;

    /*! Path to the per-county hospital bed-supply data file (census initialization only). */
    std::string m_hospital_data_file;

    /*! Whether the fixed per-community baseline (bed supply, full-strength workforce)
     *  has been recorded; set on the first call to updateCapacities. */
    bool m_capacities_initialized = false;

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
    const bool model_mw = m_model_medical_workers;

    if (model_mw) { computeHospitalScores(a_agents, a_iter); }
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
            GpuArray<ParticleReal*, ExaEpi::max_num_diseases> counter_ptrs, timer_ptrs, incubation_per_ptrs, hospital_delay_ptrs,
                    treat_q_ptrs;
            for (int d = 0; d < n_disease; d++) {
                status_ptrs[d] = soa.GetIntData(i_RT + i0(d) + IntIdxDisease::status).data();
                symptomatic_ptrs[d] = soa.GetIntData(i_RT + i0(d) + IntIdxDisease::symptomatic).data();
                counter_ptrs[d] = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::disease_counter).data();
                timer_ptrs[d] = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::treatment_timer).data();
                treat_q_ptrs[d] = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::treatment_quality).data();
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

            Gpu::Buffer<int> underserved_agents({0});
            auto* underserved_agents_ptr = underserved_agents.data();

            for (int d = 0; d < n_disease; d++) {

                auto* disease_parm_h = a_agents.getDiseaseParameters_h(d);
                auto immune_length_alpha = disease_parm_h->immune_length_alpha;
                auto immune_length_beta = disease_parm_h->immune_length_beta;
                auto immune_length_loc = disease_parm_h->immune_length_loc;
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
                    // check for exit conditions (multi-disease hospital-stat fix, development #146)
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
                    // accumulate treatment quality (medical-workers model); when the model is
                    // off, treatment quality stays 1 and mortality is the load-independent baseline
                    if (model_mw) {
                        PType& p = pstruct[i];
                        auto iv = getParticleCell(p, plo, dxi, domain);
                        AMREX_ALWAYS_ASSERT((iv[0] == ptd.m_idata[IntIdx::hosp_i][i]) &&
                                            (iv[1] == ptd.m_idata[IntIdx::hosp_j][i]));
                        treat_q_ptrs[d][i] *= hosp_data_arr(iv, HospMod::score);
                        if ((d == 0) && (treat_q_ptrs[d][i] < 1.0_prt)) { Gpu::Atomic::Add(underserved_agents_ptr, 1); }
                    }
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
                        // Check if hospitalized patient recovers or dies. The baseline severity is
                        // indexed by status_d (development #146 multi-disease fix). With the
                        // medical-workers model active, capacity-limited treatment quality lowers
                        // survival; without it, mortality is exactly the baseline (reproduces development).
                        amrex::Real p_death = disease_parm_d->m_hospToDeath[status_d - 1][age_group_ptr[i]];
                        if (model_mw) { p_death = 1.0_rt - treat_q_ptrs[d][i] * (1.0_rt - p_death); }
                        if (Random(engine) < p_death) {
                            // patient dies
                            is_alive_ptr[i] = 0;
                            died_today_ptr[i] = 1;
                            status_ptrs[d][i] = Status::dead;
                        } else {
                            // If alive, hospitalized patient recovers
                            status_ptrs[d][i] = Status::immune;
                            counter_ptrs[d][i] =
                                    static_cast<ParticleReal>(RandomGamma(immune_length_alpha, immune_length_beta, engine)) +
                                    static_cast<ParticleReal>(immune_length_loc);
                            symptomatic_ptrs[d][i] = SymptomStatus::presymptomatic;
                            // don't withdraw yet since the agent could still have another disease
                            // withdrawn_ptr[i] = 0;
                            timer_ptrs[d][i] = 0.0_prt;
                        }
                    }
                });
            }

            Gpu::synchronize();
            m_num_underserved_patients += *(underserved_agents.copyToHost());

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

    if (model_mw) {
        ParallelDescriptor::ReduceIntSum(&m_num_underserved_patients, 1);
        amrex::Print() << "Day " << a_iter << ": " << m_num_overloaded_hospitals << " hospitals over capacity, "
                       << m_num_underserved_patients << " underserved hospitalized agents\n";
    }
}

/*! Interaction between medical workers at a hospital */
template <typename PCType, typename PTDType, typename PType>
void HospitalModel<PCType, PTDType, PType>::fastInteractHospital (PCType& a_agents /*!< Agent container */) {
    BL_PROFILE(__func__);
    int n_disease = a_agents.numDiseases();

    HospWorkerCandidate<PTDType> isMW; // is a medical worker
    HospPatientCandidate<PTDType> isP; // is a patient

    Vector<Gpu::DeviceVector<int>> inf_mw_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> inf_mw_asymp_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> inf_p_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> inf_p_asymp_d(OMP_MAX_THREADS);

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

            inf_mw_d[OMP_THREAD_NUM].resize(max_communities * max_group);
            auto inf_mw_d_ptr = inf_mw_d[OMP_THREAD_NUM].data();

            inf_mw_asymp_d[OMP_THREAD_NUM].resize(max_communities * max_group);
            auto inf_mw_asymp_d_ptr = inf_mw_asymp_d[OMP_THREAD_NUM].data();

            inf_p_d[OMP_THREAD_NUM].resize(max_communities);
            auto inf_p_d_ptr = inf_p_d[OMP_THREAD_NUM].data();

            inf_p_asymp_d[OMP_THREAD_NUM].resize(max_communities);
            auto inf_p_asymp_d_ptr = inf_mw_asymp_d[OMP_THREAD_NUM].data();

            for (int d = 0; d < n_disease; d++) {
                devMemset(inf_mw_d_ptr, 0, inf_mw_d[OMP_THREAD_NUM].size() * sizeof(int));
                devMemset(inf_mw_asymp_d_ptr, 0, inf_mw_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));
                devMemset(inf_p_d_ptr, 0, inf_p_d[OMP_THREAD_NUM].size() * sizeof(int));
                devMemset(inf_p_asymp_d_ptr, 0, inf_p_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));

                auto d_ptr = getCommunityIndex.comm_to_local_index_d.data();

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (isInfectious(i, ptd, d) && isMW(i, ptd)) {
                        Box tbx;
                        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                        auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                        auto community = d_ptr[tidx];
                        int group_i = community * max_group + workgroup_ptr[i];
                        if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                            Gpu::Atomic::AddNoRet(&inf_mw_asymp_d_ptr[group_i], 1);
                        } else {
                            Gpu::Atomic::AddNoRet(&inf_mw_d_ptr[group_i], 1);
                        }
                    }
                    if (isInfectious(i, ptd, d) && isP(i, ptd)) {
                        Box tbx;
                        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                        auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                        auto community = d_ptr[tidx];
                        if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                            Gpu::Atomic::AddNoRet(&inf_p_asymp_d_ptr[community], 1);
                        } else {
                            Gpu::Atomic::AddNoRet(&inf_p_d_ptr[community], 1);
                        }
                    }
                });
                Gpu::synchronize();

                auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                auto lparm = a_agents.getDiseaseParameters_d(d);
                auto lparm_h = a_agents.getDiseaseParameters_h(d);
                Real scale = 1.0_rt; // TODO this should vary based on cell
                Real infect = 1.0_rt - lparm_h->vac_eff;

                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    if (isSusceptible(i, ptd, d) && isMW(i, ptd)) {
                        Box tbx;
                        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                        auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                        auto community = d_ptr[tidx];
                        int group_i = community * max_group + workgroup_ptr[i];

                        {
                            int num_infected_group = inf_mw_d_ptr[group_i];
                            Real group_prob = 1.0_prt - infect * lparm->xmit_hosp_d2d * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(group_prob, num_infected_group));
                            int num_infected_group_asymp = inf_mw_asymp_d_ptr[group_i];
                            Real group_prob_asymp = 1.0_prt - lparm->asymp_relative_inf * infect * lparm->xmit_hosp_d2d * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(group_prob_asymp, num_infected_group_asymp));
                        }
                        {
                            int num_infected_group = inf_p_d_ptr[community];
                            Real group_prob = 1.0_prt - infect * lparm->xmit_hosp_p2d * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(group_prob, num_infected_group));
                            int num_infected_group_asymp = inf_p_asymp_d_ptr[community];
                            Real group_prob_asymp = 1.0_prt - lparm->asymp_relative_inf * infect * lparm->xmit_hosp_p2d * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(group_prob_asymp, num_infected_group_asymp));
                        }
                    }
                    if (isSusceptible(i, ptd, d) && isP(i, ptd)) {
                        Box tbx;
                        auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                        auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                        auto community = d_ptr[tidx];

                        for (int wg = 0; wg < max_group; wg++) {
                            int group_i = community * max_group + wg;
                            int num_infected_group = inf_mw_d_ptr[group_i];
                            Real group_prob = 1.0_prt - infect * lparm->xmit_hosp_d2p * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(group_prob, num_infected_group));
                            int num_infected_group_asymp = inf_mw_asymp_d_ptr[group_i];
                            Real group_prob_asymp = 1.0_prt - lparm->asymp_relative_inf * infect * lparm->xmit_hosp_d2p * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(group_prob_asymp, num_infected_group_asymp));
                        }
                        {
                            int num_infected_group = inf_p_d_ptr[community];
                            Real group_prob = 1.0_prt - infect * lparm->xmit_hosp_p2p * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(group_prob, num_infected_group));
                            int num_infected_group_asymp = inf_p_asymp_d_ptr[community];
                            Real group_prob_asymp = 1.0_prt - lparm->asymp_relative_inf * infect * lparm->xmit_hosp_p2p * scale;
                            prob_ptr[i] *= static_cast<ParticleReal>(std::pow(group_prob_asymp, num_infected_group_asymp));
                        }
                    }
                });
                Gpu::synchronize();
            }
        }
    }
}

/*! Sets the fixed per-community staffed-bed supply from the residential population:
 *  bed_supply = (staffed_beds_per_1000 / 1000) x population. Called once at initialization,
 *  when agents are at their home communities, so the count is the residential population the
 *  community hospital serves. */
template <typename PCType, typename PTDType, typename PType>
void HospitalModel<PCType, PTDType, PType>::initBedSupply (const MultiFab& a_population /*!< residential population */) {
    BL_PROFILE("HospitalModel::initBedSupply");
    const auto beds_per_capita = m_beds_per_1000 / 1000.0_rt;
    for (MFIter mfi(*m_data); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.tilebox();
        const auto pop_arr = a_population.const_array(mfi);
        auto data_arr = m_data->array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            data_arr(i, j, k, HospMod::bed_supply) = beds_per_capita * pop_arr(i, j, k, 0);
        });
    }
}

/*! Sets the fixed per-community staffed-bed supply directly from a MultiFab (component 0),
 *  e.g. the real per-county bed supply apportioned by population from HHS data. */
template <typename PCType, typename PTDType, typename PType>
void HospitalModel<PCType, PTDType, PType>::setBedSupply (const MultiFab& a_bed_supply /*!< bed supply per community */) {
    BL_PROFILE("HospitalModel::setBedSupply");
    for (MFIter mfi(*m_data); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.tilebox();
        const auto src = a_bed_supply.const_array(mfi);
        auto data_arr = m_data->array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            data_arr(i, j, k, HospMod::bed_supply) = src(i, j, k, 0);
        });
    }
}

/*! Given a MultiFab with the current frontline medical-worker counts in each community, updates
 *  the hospital capacities. The input MultiFab has two components: (0) current total frontline
 *  medical workers (the full-strength workforce, recorded once on the first call), and (1) the
 *  currently-available frontline medical workers. The staffed-bed supply (fixed per community)
 *  must already have been set by initBedSupply. On every call the capacity is the bed supply
 *  scaled by workforce availability: capacity = bed_supply x (available / baseline). Communities
 *  with no baseline workforce carry no capacity limit. */
template <typename PCType, typename PTDType, typename PType>
void HospitalModel<PCType, PTDType, PType>::updateCapacities (const MFPtr& a_num_medworkers /*!< frontline worker counts */) {
    BL_PROFILE("HospitalModel::updateCapacities");

    const bool first_call = !m_capacities_initialized;

    for (MFIter mfi(*m_data); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.tilebox();
        const auto n_mw_arr = a_num_medworkers->const_array(mfi);
        auto data_arr = m_data->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // record the full-strength frontline workforce once (bed supply is set by initBedSupply)
            if (first_call) { data_arr(i, j, k, HospMod::n_baseline_mw) = n_mw_arr(i, j, k, 0); }
            data_arr(i, j, k, HospMod::n_avail_mw) = n_mw_arr(i, j, k, 1);

            const auto n_baseline = data_arr(i, j, k, HospMod::n_baseline_mw);
            if (n_baseline == 0) {
                // no hospital workforce in this community -> no capacity limit
                data_arr(i, j, k, HospMod::capacity) = 0.0_rt;
            } else {
                data_arr(i, j, k, HospMod::capacity) =
                        data_arr(i, j, k, HospMod::bed_supply) * data_arr(i, j, k, HospMod::n_avail_mw) / n_baseline;
            }
        });
    }

    m_capacities_initialized = true;
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
            if (data_arr(i, j, k, HospMod::n_baseline_mw) == 0) {
                data_arr(i, j, k, HospMod::load) = 0.0_prt; // no hospital workforce: exclude from score modeling
            } else if (data_arr(i, j, k, HospMod::capacity) == 0) {
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

    if (m_write_pltfiles && (a_iter % m_plot_int == 0)) {
        char iter_str[12];
        sprintf(iter_str, "%05d", a_iter);
        {
            std::string op_fname = "hospital_data_" + std::string(iter_str);
            Print() << "Writing " << op_fname << "\n";
            Vector<std::string> varnames = {"hospital_score",
                                            "patient_capacity",
                                            "hospital_load",
                                            "num_patients",
                                            "num_available_frontline_medworkers",
                                            "baseline_frontline_medworkers",
                                            "staffed_bed_supply"};
            WriteSingleLevelPlotfile(op_fname, *m_data, varnames, geom, a_iter, a_iter);
        }
    }
}

#endif
