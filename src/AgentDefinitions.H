/*! @file AgentDefinitions.H
    \brief Contains some basic definitions for #AgentContainer
*/

#ifndef _AGENT_DEF_H_
#define _AGENT_DEF_H_

#include "AgentEnums.H"
#include <AMReX_Particles.H>

namespace ExaEpi {
/*! Maximum number of diseases */
const int max_num_diseases = 10;
} // namespace ExaEpi

/*! \brief Real-type SoA attributes of agent */
struct RealIdx {
    enum {
        nattribs = 0 /*!< number of real-type attribute*/
    };
};

/*! \brief School types used only in initializing census data approach */
struct SchoolCensusIDType {
    enum {
        none,      /*!< Not in school */
        high_1,    /*!< High School */
        middle_2,  /*!< Middle School */
        elem_3,    /*!< Elementary School, nborhoods 1 and 2 */
        elem_4,    /*!< Elementary School, nborhoods 3 and 4 */
        daycare_5, /*!< Day care + Playground */
        total
    };
};
/*! \brief Disease symptom status */
struct SymptomStatus {
    enum {
        presymptomatic = 0, /*!< asymptomatic but will develop symptoms */
        symptomatic,        /*!< Infectious and symptomatic */
        asymptomatic        /*!< asymptomatic and will remain so */
    };
};

struct OutputStatus {
    enum {
        /* Total number of agents in each category are written to summary table each step */
        Su = 0,  /*!< Susceptible */
        PS_PI,   /*!< Presymptomatic/Preinfectious */
        S_PI_NH, /*!< Symptomatic/Preinfectious/Not Hospitalized */
        S_PI_H,  /*!< Symptomatic/Preinfectious/Hospitalized */
        PS_I,    /*!< Presymptomatic/Infectious */
        S_I_NH,  /*!< Symptomatic/Infectious/Not Hospitalized */
        S_I_H,   /*!< Symptomatic/Infectious/Hospitalized */
        A_PI,    /*!< Asymptomatic/Preinfectious */
        A_I,     /*!< Asymptomatic/Infectious */
        H_NI,    /*!< Hospitalized/Noninfectious */
        H_I,     /*!< Hostitalized/Infectious */
        ICU,     /*!< In the ICU */
        V,       /*!< On ventilator */
        R,       /*!< Recovered */
        D,       /*!< Dead */
        NewI,    /*!< Became infected this step */
        NewS,    /*!< Became symptomatic this step */
        NewH,    /*!< Became hospitalized this step */
        NewA,    /*!< Became asymptomatic this step */
        NewP,    /*!< Became presymptomatic this step */
        nattribs
    };
};

/*! \brief Get the total number of exposed agents (in or out of hospital) */
AMREX_FORCE_INLINE
amrex::Long totalInfected (const std::array<amrex::Long, OutputStatus::nattribs> counts) {
    return counts[OutputStatus::PS_PI] + counts[OutputStatus::S_PI_NH] + counts[OutputStatus::S_PI_H] +
           counts[OutputStatus::S_I_NH] + counts[OutputStatus::S_I_H] + counts[OutputStatus::PS_I] + counts[OutputStatus::A_PI] +
           counts[OutputStatus::A_I] + counts[OutputStatus::H_NI] + counts[OutputStatus::H_I];
}

/*! \brief Is an agent symptomatic? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isSymptomatic (const int a_idx,      /*!< Agent index */
                    const PTDType& a_ptd, /*!< Particle tile data */
                    const int a_d /*!< Disease index */) {
    return a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::symptomatic][a_idx] == SymptomStatus::symptomatic;
}

/*! \brief Did the agent just become infected this step? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isNewlyInfected (const int a_idx,      /*!< Agent index */
                      const PTDType& a_ptd, /*!< Particle tile data */
                      const int a_d /*!< Disease index */) {
    int status = a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx];
    int disease_counter = (int)a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::disease_counter][a_idx];
    return status == Status::infected && disease_counter <= 1;
}

/*! \brief Did the agent just become symptomatic this step? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isNewlySymptomatic (const int a_idx,      /*!< Agent index */
                         const PTDType& a_ptd, /*!< Particle tile data */
                         const int a_d /*!< Disease index */) {
    int status = a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx];
    int symptom_status = a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::symptomatic][a_idx];
    int disease_counter = (int)a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::disease_counter][a_idx];
    int incb_period = (int)amrex::Math::floor(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::incubation_period][a_idx]);
    return status == Status::infected && disease_counter == incb_period && symptom_status == SymptomStatus::symptomatic;
}

/*! \brief Did the agent just become asymptomatic this step? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isNewlyAsymptomatic (const int a_idx,      /*!< Agent index */
                          const PTDType& a_ptd, /*!< Particle tile data */
                          const int a_d /*!< Disease index */) {
    int status = a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx];
    int symptom_status = a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::symptomatic][a_idx];
    int disease_counter = (int)a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::disease_counter][a_idx];
    int incb_period = (int)amrex::Math::floor(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::incubation_period][a_idx]);
    return status == Status::infected && disease_counter == incb_period && symptom_status == SymptomStatus::asymptomatic;
}

/*! \brief Did the agent just become presymptomatic this step? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isNewlyPresymptomatic (const int a_idx,      /*!< Agent index */
                            const PTDType& a_ptd, /*!< Particle tile data */
                            const int a_d /*!< Disease index */) {
    int status = a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx];
    int symptom_status = a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::symptomatic][a_idx];
    int disease_counter = (int)a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::disease_counter][a_idx];
    int incb_period = (int)amrex::Math::floor(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::incubation_period][a_idx]);
    // presymptomatic for one day only, just before becoming asymp/symp
    return status == Status::infected && (disease_counter == incb_period - 1) && symptom_status == SymptomStatus::presymptomatic;
}

/*! \brief Is an agent asymptomatic (does not currently have symptoms and will not develop them)? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isAsymptomatic (const int a_idx,      /*!< Agent index */
                     const PTDType& a_ptd, /*!< Particle tile data */
                     const int a_d /*!< Disease index */) {
    return a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::symptomatic][a_idx] == SymptomStatus::asymptomatic;
}

/*! \brief Is an agent presymptomatic (does not currently have symptoms but will develop them)? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isPresymptomatic (const int a_idx,      /*!< Agent index */
                       const PTDType& a_ptd, /*!< Particle tile data */
                       const int a_d /*!< Disease index */) {
    return a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::symptomatic][a_idx] == SymptomStatus::presymptomatic;
}

/*! \brief Is an agent infected but not infectious? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool notInfectiousButInfected (const int a_idx,      /*!< Agent index */
                               const PTDType& a_ptd, /*!< Particle tile data */
                               const int a_d /*!< Disease index */) {
    return ((a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx] == Status::infected) &&
            (a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::disease_counter][a_idx] <=
             amrex::Math::floor(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::latent_period][a_idx])));
}

/*! \brief Is an agent infectious? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isInfectious (const int a_idx,      /*!< Agent index */
                   const PTDType& a_ptd, /*!< Particle tile data */
                   const int a_d /*!< Disease index */) {
    return ((a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx] == Status::infected) &&
            (a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::disease_counter][a_idx] >=
             amrex::Math::floor(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::latent_period][a_idx])));
}

/*! \brief Is an agent susceptible? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isSusceptible (const int a_idx,      /*!< Agent index */
                    const PTDType& a_ptd, /*!< Particle tile data */
                    const int a_d /*!< Disease index */) {
    return ((a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx] != Status::immune) &&
            (a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx] != Status::dead) &&
            (a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx] != Status::infected));
}

/*! \brief Is an agent not susceptible (i.e., dead, immune, or already infected)? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool notSusceptible (const int a_idx,      /*!< Agent index */
                     const PTDType& a_ptd, /*!< Particle tile data */
                     const int a_d /*!< Disease index */) {
    return ((a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx] == Status::immune) ||
            (a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx] == Status::dead) ||
            (a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx] == Status::infected));
}

/*! \brief Is an agent hospitalized? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool inHospital (const int a_idx, /*!< Agent index */
                 const PTDType& a_ptd /*!< Particle tile data */) {
    return ((a_ptd.m_idata[IntIdx::hosp_i][a_idx] >= 0) && (a_ptd.m_idata[IntIdx::hosp_j][a_idx] >= 0));
}

/*! \brief Did the agent just become hospitalized this step? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isNewlyHospitalized (const int a_idx,      /*!< Agent index */
                          const PTDType& a_ptd, /*!< Particle tile data */
                          const int a_d /*!< Disease index */) {
    if (!inHospital(a_idx, a_ptd)) { return false; }
    int disease_counter = (int)(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::disease_counter][a_idx]);
    int incb_period = (int)(amrex::Math::floor(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::incubation_period][a_idx]));
    int hospital_delay = (int)(amrex::Math::floor(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::hospital_delay][a_idx]));
    return disease_counter == incb_period + hospital_delay;
}

// Include DiseaseParm after basic definitions to avoid circular dependency
#include "DiseaseParm.H"

template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void setAgentData (PTDType& ptd, int ip, int i, int j, const amrex::GpuArray<amrex::Real, 2>& dx, amrex::Long id, int cpu,
                   int age_group, int family, int nborhood, int n_disease, const DiseaseParm** disease_parms,
                   const amrex::RandomEngine& engine) {
    using namespace amrex::literals;

    ptd[ip].pos(0) = static_cast<amrex::ParticleReal>((i + 0.5_rt) * dx[0]);
    ptd[ip].pos(1) = static_cast<amrex::ParticleReal>((j + 0.5_rt) * dx[1]);
    ptd[ip].id() = id;
    ptd[ip].cpu() = cpu;

    for (int d = 0; d < n_disease; d++) {
        // Check if agent should be initially immune for this disease
        if (amrex::Random(engine) < disease_parms[d]->initial_immunity_fraction) {
            ptd.m_runtime_idata[i0(d) + IntIdxDisease::status][ip] = Status::immune;
            // Set immune counter to random point in immunity period
            // Sample full immune duration, then pick random point within it
            amrex::ParticleReal full_immune_duration = static_cast<amrex::ParticleReal>(
                    amrex::RandomGamma(disease_parms[d]->immune_length_alpha, disease_parms[d]->immune_length_beta, engine));
            ptd.m_runtime_rdata[r0(d) + RealIdxDisease::disease_counter][ip] =
                    amrex::Random(engine) * full_immune_duration;
        } else {
            ptd.m_runtime_idata[i0(d) + IntIdxDisease::status][ip] = Status::never;
            ptd.m_runtime_rdata[r0(d) + RealIdxDisease::disease_counter][ip] = 0.0_prt;
        }
        ptd.m_runtime_rdata[r0(d) + RealIdxDisease::treatment_timer][ip] = 0.0_prt;
        ptd.m_runtime_rdata[r0(d) + RealIdxDisease::prob][ip] = 0.0_prt;
        ptd.m_runtime_rdata[r0(d) + RealIdxDisease::latent_period][ip] = 0.0_prt;
        ptd.m_runtime_rdata[r0(d) + RealIdxDisease::infectious_period][ip] = 0.0_prt;
        ptd.m_runtime_rdata[r0(d) + RealIdxDisease::incubation_period][ip] = 0.0_prt;
        ptd.m_runtime_rdata[r0(d) + RealIdxDisease::hospital_delay][ip] = 0.0_prt;
        ptd.m_runtime_rdata[r0(d) + RealIdxDisease::hospital_random][ip] = 0.0_prt;
    }

    ptd.m_idata[IntIdx::age_group][ip] = age_group;
    ptd.m_idata[IntIdx::family][ip] = family;
    ptd.m_idata[IntIdx::home_i][ip] = i;
    ptd.m_idata[IntIdx::home_j][ip] = j;
    ptd.m_idata[IntIdx::work_i][ip] = i;
    ptd.m_idata[IntIdx::work_j][ip] = j;
    ptd.m_idata[IntIdx::trav_i][ip] = i;
    ptd.m_idata[IntIdx::trav_j][ip] = j;
    ptd.m_idata[IntIdx::hosp_i][ip] = -1;
    ptd.m_idata[IntIdx::hosp_j][ip] = -1;
    ptd.m_idata[IntIdx::nborhood][ip] = nborhood;
    ptd.m_idata[IntIdx::hh_cluster][ip] = family / 4;
    ptd.m_idata[IntIdx::work_nborhood][ip] = nborhood;
    ptd.m_idata[IntIdx::workgroup][ip] = 0;
    ptd.m_idata[IntIdx::naics][ip] = 0;
    ptd.m_idata[IntIdx::random_travel][ip] = -1;
    ptd.m_idata[IntIdx::air_travel][ip] = -1;
}

/*! \brief Is agent an adult? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isAnAdult (const int a_idx, /*!< Agent index */
                const PTDType& a_ptd /*!< Particle tile data */) {
    return a_ptd.m_idata[IntIdx::age_group][a_idx] > AgeGroups::a5to17;
}

// School grade is actually age of student
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
static int getSchoolType (const int grade) {
    if (grade == 3) { return SchoolType::daycare; }
    if (grade > 3 && grade <= 10) { return SchoolType::elem; }
    if (grade > 10 && grade <= 13) { return SchoolType::middle; }
    if (grade > 13 && grade <= 17) { return SchoolType::high; }
    if (grade > 17) { return SchoolType::college; }
    return SchoolType::none;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static int schoolTypeToGrade (const int school_type) {
    switch (school_type) {
    case SchoolType::daycare: return 3;
    case SchoolType::elem: return 5;
    case SchoolType::middle: return 11;
    case SchoolType::high: return 15;
    case SchoolType::college: return 18;
    default: return -1;
    }
}

/*! \brief Set computational domain, i.e., number of cells in each direction, from the number of communities.
 *
 *  + The domain is a 2D square, where the total number of cells is the lowest square of an
 *    integer that is greater than #DemographicData::Ncommunity
 *  + The physical size is 1.0 in each dimension.
 *
 *  A periodic Cartesian grid is defined.
 */
inline amrex::Geometry getGeometry (const int ncommunity) {
    int is_per[BL_SPACEDIM];
    for (int i = 0; i < BL_SPACEDIM; i++) {
        is_per[i] = true;
    }

    amrex::RealBox real_box;
    amrex::Box base_domain;
    amrex::Geometry geom;

    amrex::IntVect iv;
    iv[0] = iv[1] = (int)std::floor(std::sqrt((double)ncommunity));
    while (iv[0] * iv[1] <= ncommunity) {
        ++iv[0];
    }
    base_domain = amrex::Box(amrex::IntVect(AMREX_D_DECL(0, 0, 0)), iv - 1);

    for (int n = 0; n < BL_SPACEDIM; n++) {
        real_box.setLo(n, 0.0);
        real_box.setHi(n, 1.0);
    }

    geom.define(base_domain, &real_box, amrex::CoordSys::cartesian, is_per);
    return geom;
}

#endif
