/*! @file AgentDefinitions.H
    \brief Contains some basic definitions for #AgentContainer
*/

#ifndef _AGENT_DEF_H_
#define _AGENT_DEF_H_

#include <AMReX_Particles.H>

namespace ExaEpi
{
    /*! Maximum number of diseases */
    const int max_num_diseases = 10;
}

/*! \brief Real-type SoA attributes of agent */
struct RealIdx
{
    enum {
        nattribs = 0 /*!< number of real-type attribute*/
    };
};

/*! \brief Disease-specific Real-type Runtime-SoA attributes of agent */
struct RealIdxDisease
{
    enum {
        /* Disease counter starts after infection. */
        treatment_timer = 0,    /*!< Timer since hospital admission */
        disease_counter,        /*!< Counter since start of infection */
        prob,                   /*!< Probability of infection */
        latent_period,          /*!< Time until infectious, which could be before symptoms appear */
        infectious_period,      /*!< Length of time infectious */
        incubation_period,      /*!< Time until symptoms appear */
        nattribs                /*!< number of real-type attribute*/
    };
};

/*! \brief Integer-type SoA attributes of agent */
struct IntIdx
{
    enum {
        age_group = 0,  /*!< Age group (under 5, 5-17, 18-29, 30-64, 65+) */
        family,         /*!< Family ID */
        home_i,         /*!< home location index */
        home_j          /*!< home location index */,
        work_i          /*!< work location index */,
        work_j          /*!< work location index */,
        hosp_i          /*!< hosp location index */,
        hosp_j          /*!< hosp location index */,
        trav_i          /*!< air travel location index */,
        trav_j          /*!< air travel location index */,
        nborhood,       /*!< home neighborhood ID */
        school_grade,   /*!< school grade, including universities */
        school_id,      /*!< ID for a given school */
        school_closed,  /*!< 0 for open, 1 for closed */
        naics,          /*!< industry NAICS code for business employed at */
        workgroup,      /*!< workgroup ID */
        work_nborhood,  /*!< work neighborhood ID */
        withdrawn,      /*!< quarantine status */
        random_travel,  /*!< on long distance travel? */
        air_travel,     /*!< on long distance travel by Air? */
        nattribs        /*!< number of integer-type attribute */
    };
};

/*! \brief Disease-specific Integer-type Runtime-SoA attributes of agent */
struct IntIdxDisease
{
    enum {
        status = 0,     /*!< Disease status (#Status) */
        symptomatic,    /*!< currently symptomatic? 0: no, but will be, 1: yes, 2: no, and will remain so until recovered */
        nattribs        /*!< number of integer-type attribute */
    };
};

/*! \brief School Type  */
struct SchoolType
{
    enum {
        none,          /*!< Not in school */
        college,
        high,
        middle,
        elem,
        daycare,
        total
    };
};

/*! \brief School types used only in initializing census data approach */
struct SchoolCensusIDType
{
    enum {
        none,           /*!< Not in school */
        high_1,         /*!< High School */
        middle_2,       /*!< Middle School */
        elem_3,         /*!< Elementary School, nborhoods 1 and 2 */
        elem_4,         /*!< Elementary School, nborhoods 3 and 4 */
        daycare_5,      /*!< Day care + Playground */
        total
    };
};



/*! \brief Age Group */
struct AgeGroups
{
    enum {
        u5 = 0, /*!< Under 5 */
        a5to17,  /*!< 5-17 */
        a18to29, /*!< 18-29 */
        a30to49, /*!< 30-49 */
        a50to64,
        o65,    /*!< over 65 */
        total   /*!< number of age groups */
    };
};

/*! \brief Age Groups for Hospitalization Days */
struct AgeGroups_Hosp
{
    enum {
        u50 = 0, /*!< Under 50 */
        a50to64, /*!< Ages 50-64 */
        o65,     /*!< 65 and over */
        total    /*!< number of age groups */
    };
};

/*! \brief Disease status */
struct Status
{
    enum {
        never = 0,   /*!< never infected */
        infected,    /*!< infected */
        immune,      /*!< no longer infected, immune. lasts 6 months. */
        susceptible, /*!< no longer infected, no longer immnune */
        dead         /*!< passed away */
    };
};

/*! \brief Disease statistics */
struct DiseaseStats
{
    enum {
        hospitalization = 0, /*!< number of hospitalizations */
        ICU,                 /*!< number of ICU cases */
        ventilator,          /*!< number of ventilator cases */
        death                /*!< number of deaths */
    };
};

/*! \brief Compute index offsets for runtime int-type disease attributes */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int i0 ( const int a_d /*!< Disease index */)
{
    return a_d*IntIdxDisease::nattribs;
}

/*! \brief Compute index offsets for runtime real-type disease attributes */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int r0 ( const int a_d /*!< Disease index */)
{
    return a_d*RealIdxDisease::nattribs;
}

/*! \brief Disease symptom status */
struct SymptomStatus
{
    enum {
        presymptomatic = 0, /*!< asymptomatic but will develop symptoms */
        symptomatic,        /*!< Infectious and symptomatic */
        asymptomatic        /*!< asymptomatic and will remain so */
    };
};

/*! \brief Is an agent infected but not infectious? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool notInfectiousButInfected ( const int      a_idx, /*!< Agent index */
                                const PTDType& a_ptd, /*!< Particle tile data */
                                const int      a_d    /*!< Disease index */ )
{
    return (    (a_ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::status][a_idx] == Status::infected)
             && (a_ptd.m_runtime_rdata[r0(a_d)+RealIdxDisease::disease_counter][a_idx]
                 <= a_ptd.m_runtime_rdata[r0(a_d)+RealIdxDisease::latent_period][a_idx]) );
}

/*! \brief Is an agent infectious? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isInfectious ( const int      a_idx, /*!< Agent index */
                    const PTDType& a_ptd, /*!< Particle tile data */
                    const int      a_d    /*!< Disease index */ )
{
    return (   (a_ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::status][a_idx] == Status::infected)
            && (a_ptd.m_runtime_rdata[r0(a_d)+RealIdxDisease::disease_counter][a_idx]
                >= a_ptd.m_runtime_rdata[r0(a_d)+RealIdxDisease::latent_period][a_idx]) );
}

/*! \brief Is an agent susceptible? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isSusceptible ( const int      a_idx, /*!< Agent index */
                     const PTDType& a_ptd, /*!< Particle tile data */
                     const int      a_d    /*!< Disease index */ )
{
    return (   (a_ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::status][a_idx] != Status::immune)
            && (a_ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::status][a_idx] != Status::dead)
            && (a_ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::status][a_idx] != Status::infected) );
}

/*! \brief Is an agent not susceptible (i.e., dead, immune, or already infected)? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool notSusceptible ( const int      a_idx, /*!< Agent index */
                      const PTDType& a_ptd, /*!< Particle tile data */
                      const int       a_d    /*!< Disease index */ )
{
    return (   (a_ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::status][a_idx] == Status::immune)
            || (a_ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::status][a_idx] == Status::dead)
            || (a_ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::status][a_idx] == Status::infected) );
}

/*! \brief Is an agent hospitalized? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool inHospital ( const int      a_idx, /*!< Agent index */
                  const PTDType& a_ptd  /*!< Particle tile data */ )
{
    return (   (a_ptd.m_idata[IntIdx::hosp_i][a_idx] >= 0)
            && (a_ptd.m_idata[IntIdx::hosp_j][a_idx] >= 0) );
}

/*! \brief Is agent an adult? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool isAnAdult (const int      a_idx, /*!< Agent index */
                const PTDType& a_ptd  /*!< Particle tile data */ )
{
    return a_ptd.m_idata[IntIdx::age_group][a_idx] > AgeGroups::a5to17;
}

// grade definitions
// 0: "childcare",
// 1: "k12pub_preschl", 2: "k12pub_kind", 3: "k12pub_1st", 4: "k12pub_2nd", 5: "k12pub_3rd", 6: "k12pub_4th", 7: "k12pub_5th",
// 8: "k12pub_6th", 9: "k12pub_7th", 10: "k12pub_8th",
// 11: "k12pub_9th", 12: "k12pub_10th", 13: "k12pub_11th", 14: "k12pub_12th",
// 15: "k12pv_preschl", 16: "k12pv_kind", 17: "k12pv_1st", 18: "k12pv_2nd", 19: "k12pv_3rd", 20: "k12pv_4th", 21: "k12pv_5th",
// 22: "k12pv_6th", 23: "k12pv_7th", 24: "k12pv_8th",
// 25: "k12pv_9th", 26: "k12pv_10th", 27: "k12pv_11th", 28: "k12pv_12th",
// 29: "undergrad", 30: "grad"

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
static int getSchoolType(const int grade) {
    if (grade == 0) return SchoolType::daycare;
    if ((grade >= 1 && grade <= 7) || (grade >= 15 && grade <= 21)) return SchoolType::elem;
    if ((grade >= 8 && grade <= 10) || (grade >= 22 && grade <= 24)) return SchoolType::middle;
    if ((grade >= 11 && grade <= 14) || (grade >= 25 && grade <= 28)) return SchoolType::high;
    if (grade >= 29) return SchoolType::college;
    return SchoolType::none;
}

#endif
