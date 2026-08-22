/*! @file AgentDefinitions.H
    \brief Contains some basic definitions for #AgentContainer
*/

#ifndef _AGENT_DEF_H_
#define _AGENT_DEF_H_

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

/*! \brief Disease-specific Real-type Runtime-SoA attributes of agent */
struct RealIdxDisease {
    enum {
        /* Disease counter starts after infection. */
        treatment_timer = 0, /*!< Timer since hospital admission */
        disease_counter,     /*!< Counter since start of infection */
        prob,                /*!< Probability of infection */
        latent_period,       /*!< Time until infectious, which could be before symptoms appear */
        infectious_period,   /*!< Length of time infectious */
        incubation_period,   /*!< Time until symptoms appear */
        hospital_delay,      /*!< Delay after symptom onset until hospital treatment is sought */
        hospital_random,     /*!< Random number that decides hospitaliation */
        nattribs             /*!< number of real-type attribute*/
    };
};

/*! \brief Integer-type SoA attributes of agent */
struct IntIdx {
    enum {
        age_group = 0, /*!< Age group (under 5, 5-17, 18-29, 30-64, 65+) */
        family,        /*!< Family ID */
        home_i,        /*!< home location index */
        home_j /*!< home location index */,
        work_i /*!< work location index */,
        work_j /*!< work location index */,
        hosp_i /*!< hosp location index */,
        hosp_j /*!< hosp location index */,
        trav_i /*!< air travel location index */,
        trav_j /*!< air travel location index */,
        nborhood,           /*!< home neighborhood ID */
        hh_cluster,         /*!< household cluster ID */
        school_grade,       /*!< school grade, including universities */
        school_id,          /*!< ID for a given school */
        school_closed,      /*!< 0 for open, 1 for closed */
        school_class,       /*!< 0-based class index within this agent's (school_id, grade) group at its
                                 work community, or <= -2 (-2, -3, -4, ...) for a teacher in one of
                                 that raw group's non-classroom "admin" groups -- one negative
                                 sentinel value per admin group in the raw group (consumed by
                                 InteractionModSchool.H to identify teacher-teacher contact, in
                                 addition to being diagnostic) -- see
                                 AgentContainer::assignSchoolClasses / school_class_group below */
        school_class_group, /*!< compact, densely-packed ID for this agent's actual (community,
                                 school_id, grade, school_class) mixing bucket -- see
                                 AgentContainer::assignSchoolClasses / InteractionModSchool.H. -1 for
                                 agents not enrolled (school_id == 0) */
        naics,              /*!< industry NAICS code for business employed at */
        workgroup,          /*!< workgroup ID */
        work_nborhood,      /*!< work neighborhood ID */
        withdrawn,          /*!< quarantine status */
        random_travel,      /*!< on long distance travel? */
        air_travel,         /*!< on long distance travel by Air? */
        weatherLookup,      /*!< index for current weather variables, incremented weekly */
        nattribs            /*!< number of integer-type attribute */
    };
};

/*! \brief Disease-specific Integer-type Runtime-SoA attributes of agent */
struct IntIdxDisease {
    enum {
        status = 0,  /*!< Disease status (#Status) */
        symptomatic, /*!< currently symptomatic? 0: no, but will be, 1: yes, 2: no, and will remain so until recovered */
        nattribs     /*!< number of integer-type attribute */
    };
};

/*! \brief School Type  */
struct SchoolType {
    enum {
        none, /*!< Not in school */
        college,
        high,
        middle,
        elem,
        daycare,
        total
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

/*! \brief Age Group */
struct AgeGroups {
    enum {
        u5 = 0,  /*!< Under 5 */
        a5to17,  /*!< 5-17 */
        a18to29, /*!< 18-29 */
        a30to49, /*!< 30-49 */
        a50to64,
        o65,     /*!< over 65 */
        total    /*!< number of age groups */
    };
};

/*! \brief Disease status */
struct Status {
    enum {
        never = 0,   /*!< never infected */
        infected,    /*!< infected */
        immune,      /*!< no longer infected, immune. lasts 6 months. */
        susceptible, /*!< no longer infected, no longer immnune */
        dead         /*!< passed away */
    };
};

/*! \brief Disease statistics */
struct DiseaseStats {
    enum {
        hospitalization = 0, /*!< number of current hospitalizations */
        ICU,                 /*!< number of current ICU cases */
        ventilator,          /*!< number of current ventilator cases */
        death,               /*!< number of deaths */
        new_cases            /*!< number of new infections each day */
    };
};

/*! \brief Compute index offsets for runtime int-type disease attributes */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int i0 (const int a_d /*!< Disease index */) {
    return a_d * IntIdxDisease::nattribs;
}

/*! \brief Compute index offsets for runtime real-type disease attributes */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int r0 (const int a_d /*!< Disease index */) {
    return a_d * RealIdxDisease::nattribs;
}

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
    int incb_period = (int)amrex::Math::round(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::incubation_period][a_idx]);
    return status == Status::infected && disease_counter == incb_period && symptom_status == SymptomStatus::symptomatic;
}

/*! \brief Is agent on travel? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isOnTravel (const int a_idx, /*!< Agent index */
                 const PTDType& a_ptd /*!< Particle tile data */) {
    return (a_ptd.m_idata[IntIdx::random_travel][a_idx] >= 0 || a_ptd.m_idata[IntIdx::air_travel][a_idx] >= 0);
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
    int incb_period = (int)amrex::Math::round(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::incubation_period][a_idx]);
    return status == Status::infected && disease_counter == incb_period && symptom_status == SymptomStatus::asymptomatic;
}

/*! \brief Did the agent just transition out of presymptomatic this step (i.e., just became symptomatic or asymptomatic)?
 *  This fires on the same step as isNewlySymptomatic / isNewlyAsymptomatic, so that
 *  sum(NewP) == sum(NewS) + sum(NewA) over a complete run. */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isNewlyPresymptomatic (const int a_idx,      /*!< Agent index */
                            const PTDType& a_ptd, /*!< Particle tile data */
                            const int a_d /*!< Disease index */) {
    int status = a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx];
    int symptom_status = a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::symptomatic][a_idx];
    int disease_counter = (int)a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::disease_counter][a_idx];
    int incb_period = (int)amrex::Math::round(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::incubation_period][a_idx]);
    // fires on the same step as isNewlySymptomatic/isNewlyAsymptomatic (counter == incb_period, after transition)
    return status == Status::infected && (disease_counter == incb_period) &&
           (symptom_status == SymptomStatus::symptomatic || symptom_status == SymptomStatus::asymptomatic);
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
             amrex::Math::round(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::latent_period][a_idx])));
}

/*! \brief Is an agent infectious? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isInfectious (const int a_idx,      /*!< Agent index */
                   const PTDType& a_ptd, /*!< Particle tile data */
                   const int a_d /*!< Disease index */) {
    return ((a_ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][a_idx] == Status::infected) &&
            (a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::disease_counter][a_idx] >=
             amrex::Math::round(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::latent_period][a_idx])));
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
    int incb_period = (int)(amrex::Math::round(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::incubation_period][a_idx]));
    int hospital_delay = (int)(amrex::Math::round(a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::hospital_delay][a_idx]));
    return disease_counter == incb_period + hospital_delay;
}

template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void setAgentData (PTDType& ptd, int ip, int i, int j, const amrex::GpuArray<amrex::Real, 2>& dx, amrex::Long id, int cpu,
                   int age_group, int family, int nborhood, int n_disease) {
    using namespace amrex::literals;

    ptd[ip].pos(0) = static_cast<amrex::ParticleReal>((i + 0.5_rt) * dx[0]);
    ptd[ip].pos(1) = static_cast<amrex::ParticleReal>((j + 0.5_rt) * dx[1]);
    ptd[ip].id() = id;
    ptd[ip].cpu() = cpu;

    for (int d = 0; d < n_disease; d++) {
        ptd.m_runtime_idata[i0(d) + IntIdxDisease::status][ip] = 0;
        ptd.m_runtime_rdata[r0(d) + RealIdxDisease::treatment_timer][ip] = 0.0_prt;
        ptd.m_runtime_rdata[r0(d) + RealIdxDisease::disease_counter][ip] = 0.0_prt;
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
    ptd.m_idata[IntIdx::school_class][ip] = 0;
    ptd.m_idata[IntIdx::school_class_group][ip] = -1;
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
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
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
