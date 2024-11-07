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
        latent_period,          /*!< Time until infectious */
        infectious_period,      /*!< Length of time infectious */
        incubation_period,      /*!< Time until symptoms would develop */
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
        school,         /*!< school type (elementary, middle, high, none) */
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
        strain,         /*!< virus strain */
        symptomatic,    /*!< currently symptomatic? 0: no, but will be, 1: yes, 2: no, and will remain so until recovered */
        nattribs        /*!< number of integer-type attribute */
    };
};

/*! \brief School Type  */
struct SchoolType
{
    enum {
        total = 0,   /*!< All School type */
        high,    /*!< High School */
        middle,      /*!< Middle School */
        elem_3,      /*!< Elementary School Neighbordhood 1 & 2 */
        elem_4,      /*!< Elementary School Neighbordhood 3 & 4 */
        day_care,     /*!< Day care + Playground */
        total_school_type
    };
};

/*! \brief Age Group */
struct AgeGroups
{
    enum {
        u5 = 0, /*!< Under 5 */
        a5to17,  /*!< 5-17 */
        a18to29, /*!< 18-29 */
        a30to64, /*!< 30-64 */
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
bool isHospitalized ( const int      a_idx, /*!< Agent index */
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
    return a_ptd.m_idata[IntIdx::age_group][a_idx] > 1;
}


#endif
