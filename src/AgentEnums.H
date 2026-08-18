/*! @file AgentEnums.H
    \brief Contains basic enum definitions used across agent and disease code
*/

#ifndef _AGENT_ENUMS_H_
#define _AGENT_ENUMS_H_

#include <AMReX_GpuQualifiers.H>

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

/*! \brief Age Group */
struct AgeGroups {
    enum {
        u5 = 0,  /*!< Under 5 */
        a5to17,  /*!< 5-17 */
        a18to29, /*!< 18-29 */
        a30to49, /*!< 30-49 */
        a50to64, /*!< 50-64 */
        o65,     /*!< Over 65 */
        total
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
        nborhood,      /*!< home neighborhood ID */
        hh_cluster,    /*!< household cluster ID */
        school_grade,  /*!< school grade, including universities */
        school_id,     /*!< ID for a given school */
        school_closed, /*!< 0 for open, 1 for closed */
        naics,         /*!< industry NAICS code for business employed at */
        workgroup,     /*!< workgroup ID */
        work_nborhood, /*!< work neighborhood ID */
        withdrawn,     /*!< quarantine status */
        random_travel, /*!< on long distance travel? */
        air_travel,    /*!< on long distance travel by Air? */
        weatherLookup, /*!< index for current weather variables, incremented weekly */
        nattribs       /*!< number of integer-type attribute */
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

/*! \brief Disease symptom status */
struct SymptomStatus {
    enum {
        presymptomatic = 0, /*!< asymptomatic but will develop symptoms */
        symptomatic,        /*!< Infectious and symptomatic */
        asymptomatic        /*!< asymptomatic and will remain so */
    };
};

/*! \brief Disease status */
struct Status {
    enum {
        never = 0,   /*!< Never infected */
        infected,    /*!< Currently infected */
        immune,      /*!< Immune (recovered or vaccinated) */
        susceptible, /*!< Susceptible (waned immunity) */
        dead         /*!< Dead */
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

/*! \brief Compute index offsets for runtime integer-type disease attributes */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int i0 (const int a_d /*!< Disease index */) {
    return a_d * IntIdxDisease::nattribs;
}

/*! \brief Compute index offsets for runtime real-type disease attributes */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int r0 (const int a_d /*!< Disease index */) {
    return a_d * RealIdxDisease::nattribs;
}

#endif
