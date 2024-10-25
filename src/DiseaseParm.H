/*! @file DiseaseParm.H
    \brief #DiseaseParm struct definition
*/

#ifndef DISEASE_PARM_H_
#define DISEASE_PARM_H_

#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

#include "AgentDefinitions.H"

/*! \brief Disease parameters

    Structure containing disease parameters.
*/
struct DiseaseParm
{
    // Default contact matrix scaling factors
    amrex::Real pSC = amrex::Real(0.2);     /*!< Contact matrix scaling factor - school */
    amrex::Real pHSC = amrex::Real(0.3);    /*!< Contact matrix scaling factor - high school */
    amrex::Real pMSC = amrex::Real(0.3);    /*!< Contact matrix scaling factor - middle school */
    amrex::Real pESC = amrex::Real(0.3);    /*!< Contact matrix scaling factor - elementary school */
    amrex::Real pCO = amrex::Real(1.45);    /*!< Contact matrix scaling factor - community */
    amrex::Real pNH = amrex::Real(1.45);    /*!< Contact matrix scaling factor - neighborhood */
    amrex::Real pWO = amrex::Real(0.5);     /*!< Contact matrix scaling factor - workplace */
    amrex::Real pHC = amrex::Real(0.5);     /*!< Contact matrix scaling factor - ?? */
    amrex::Real pHCC0 = amrex::Real(0.5);   /*!< Contact matrix scaling factor - ?? */
    amrex::Real pHCC1 = amrex::Real(0.5);   /*!< Contact matrix scaling factor - ?? */
    amrex::Real pFA =  amrex::Real(1.0);    /*!< Contact matrix scaling factor - family */
    amrex::Real pBAR = amrex::Real(-1.0);   /*!< Contact matrix scaling factor - bar or other large social gatherings (<0 turns this off) */

    int nstrain = 2;  /*!< Number of strains */
    amrex::Real p_trans[2] = {amrex::Real(0.20), amrex::Real(0.30)};     /*!< probability of transimission given contact */
    amrex::Real p_asymp[2] = {amrex::Real(0.40), amrex::Real(0.40)};     /*!< fraction of cases that are asymptomatic */
    amrex::Real reduced_inf[2] = {amrex::Real(0.75), amrex::Real(0.75)}; /*!< relative infectiousness of asymptomatic individuals */

    // these are computed from the above
    /*! Community transmission probabilities for each age group (<5, 5-17, 18-29, 30-64, 65+) */
    amrex::Real xmit_comm[AgeGroups::total] = { amrex::Real(.0000125*pCO),
                                                amrex::Real(.0000375*pCO),
                                                amrex::Real(.00010*pCO),
                                                amrex::Real(.00010*pCO),
                                                amrex::Real(.00010*pCO),
                                                amrex::Real(.00015*pCO) };
    /*! Neighborhood transmission probabilities for each age group (<5, 5-17, 18-29, 30-64, 65+) */
    amrex::Real xmit_hood[AgeGroups::total] = { amrex::Real(.00005*pNH),
                                                amrex::Real(.00015*pNH),
                                                amrex::Real(.00040*pNH),
                                                amrex::Real(.00040*pNH),
                                                amrex::Real(.00040*pNH),
                                                amrex::Real(.00060*pNH) };
    /*! Transmission probabilities of adults for each age group (<5, 5-17, 18-29, 30-64, 65+) for neighborhood cluster */
    amrex::Real xmit_nc_adult[AgeGroups::total] = { amrex::Real(.08*pHC),
                                                    amrex::Real(.08*pHC),
                                                    amrex::Real(.1*pHC),
                                                    amrex::Real(.1*pHC),
                                                    amrex::Real(.1*pHC),
                                                    amrex::Real(.1*pHC) };
    /*! Transmission probabilities of children for each age group (<5, 5-17, 18-29, 30-64, 65+) for neighborhood cluster */
    amrex::Real xmit_nc_child[AgeGroups::total] = { amrex::Real(.15*pHCC0),
                                                    amrex::Real(.15*pHCC1),
                                                    amrex::Real(.08*pHC),
                                                    amrex::Real(.08*pHC),
                                                    amrex::Real(.08*pHC),
                                                    amrex::Real(.08*pHC) };
    /*! Work transmission probabilities for each age group (<5, 5-17, 18-29, 30-64, 65+) */
    amrex::Real xmit_work = amrex::Real(0.115*pWO);

    /*! School transmission probabilities for each school type (High, Middle, Elem3/4 , Daycare/Playgroups */
    amrex::Real xmit_college = amrex::Real(.105*pHSC);
    amrex::Real xmit_high_school = amrex::Real(.105*pHSC);
    amrex::Real xmit_middle_school = amrex::Real(.125*pMSC);
    amrex::Real xmit_elem_school = amrex::Real(.145*pESC);
    amrex::Real xmit_daycare = amrex::Real(.15);
    amrex::Real xmit_playgroup = amrex::Real(.35);
    /*!< Transmission probability in school - child to adult */
    amrex::Real xmit_high_school_c2a = amrex::Real(.105*pHSC);
    amrex::Real xmit_middle_school_c2a = amrex::Real(.125*pMSC);
    amrex::Real xmit_elem_school_c2a = amrex::Real(.145*pESC);
    amrex::Real xmit_daycare_c2a = amrex::Real(.15);
    amrex::Real xmit_playgroup_c2a = amrex::Real(.35);
    /*!< Transmission probability in school - adult to child */
    amrex::Real xmit_high_school_a2c = amrex::Real(.105*pHSC);
    amrex::Real xmit_middle_school_a2c = amrex::Real(.125*pMSC);
    amrex::Real xmit_elem_school_a2c = amrex::Real(.145*pESC);
    amrex::Real xmit_daycare_a2c = amrex::Real(.15);
    amrex::Real xmit_playgroup_a2c = amrex::Real(.35);

    /* Within family, from child or adult, for 5 age groups (here Thai paper, US:0.6->0.8)*/
    /*! Transmission probability within family for each age-group of adult */
    amrex::Real xmit_adult[AgeGroups::total] = {  amrex::Real(.3*pFA),
                                                  amrex::Real(.3*pFA),
                                                  amrex::Real(.4*pFA),
                                                  amrex::Real(.4*pFA),
                                                  amrex::Real(.4*pFA),
                                                  amrex::Real(.4*pFA) };
    /*! Transmission probability within family for each age-group of child */
    amrex::Real xmit_child[AgeGroups::total] = {  amrex::Real(.6*pFA),
                                                  amrex::Real(.6*pFA),
                                                  amrex::Real(.3*pFA),
                                                  amrex::Real(.3*pFA),
                                                  amrex::Real(.3*pFA),
                                                  amrex::Real(.3*pFA) };

    amrex::Real xmit_child_SC[AgeGroups::total],     /*!< Transmission probability of child - school closed */
                xmit_adult_SC[AgeGroups::total],     /*!< Transmission probability of adult - school closed */
                xmit_nc_child_SC[AgeGroups::total],  /*!< Transmission probability for neighborhood cluster (child) - school closed ?? */
                xmit_nc_adult_SC[AgeGroups::total];  /*!< Transmission probability for neighborhood cluster (adult) - school closed ?? */
    amrex::Real xmit_comm_SC[AgeGroups::total],      /*!< Transmission probability in community - school closed */
                xmit_hood_SC[AgeGroups::total];      /*!< Transmission probability in neighborhood - school closed ?? */

    amrex::Real infect = amrex::Real(1.0); /*!< Overall multiplier for agent's infectiousness */

    amrex::Real reinfect_prob = amrex::Real(0.0); /*!< Reinfection probability */

    amrex::Real vac_eff = amrex::Real(0.0); /*!< Vaccine efficacy */

    amrex::Real Child_compliance, /*!< Child compliance with masking ?? */
                Child_HH_closure; /*!< Multiplier for household contacts during school closure */

    amrex::Real latent_length_mean = 3.0;   /*!< mean time (in days) until infectious*/
    amrex::Real infectious_length_mean = 6.0;   /*!< mean time (in days) agents are infectious */
    amrex::Real incubation_length_mean = 5.0;   /*!< mean time (in days) until symptoms show */

    amrex::Real latent_length_std = 1.0;   /*!< std dev (in days) for the above*/
    amrex::Real infectious_length_std = 1.0;   /*!< std dev (in days) for the above */
    amrex::Real incubation_length_std = 1.0;   /*!< std dev (in days) for the above */

    amrex::Real immune_length_mean = 30*6;  /*! mean immunity time in days*/
    amrex::Real immune_length_std = 30*2;  /*! spread in immunity time in days*/

    /*! number of hospitalization days by age group (#AgeGroups_Hosp); note that the
     *  age groups here are under 50, 50-64, and over 65, and *not* the age groups
     *  used in other parts of the code (#AgeGroups) */
    amrex::Real m_t_hosp[AgeGroups_Hosp::total] = { amrex::Real(3), /* under 50 */
                                                    amrex::Real(8), /* ages 50-64 */
                                                    amrex::Real(7)  /* over 65 */ };
    /*! Offset to separate the timers for hospital, ICU, and ventilator;
     * needs to be greater than the maximum of #DiseaseParm::m_t_hosp */
    amrex::Real m_t_hosp_offset = 10;

    /*! sick -> hospital probabilities */
    amrex::Real m_CHR[AgeGroups::total] = { amrex::Real(.0104),
                                            amrex::Real(.0104),
                                            amrex::Real(.070),
                                            amrex::Real(.28),
                                            amrex::Real(.28),
                                            amrex::Real(1.0) };
    /*! hospital -> ICU probabilities */
    amrex::Real m_CIC[AgeGroups::total] = { amrex::Real(.24),
                                            amrex::Real(.24),
                                            amrex::Real(.24),
                                            amrex::Real(.36),
                                            amrex::Real(.36),
                                            amrex::Real(.35) };
    /*! ICU -> ventilator probabilities */
    amrex::Real m_CVE[AgeGroups::total] = { amrex::Real(.12),
                                            amrex::Real(.12),
                                            amrex::Real(.12),
                                            amrex::Real(.22),
                                            amrex::Real(.22),
                                            amrex::Real(.22) };
    /*! Probabilites of dying while in hospital, for various ages and conditions (ICU, ventilator, neither) */
    amrex::Real m_HospToDeath[DiseaseStats::death][AgeGroups::total] = {
                                                    // from only hospitalized
                                                   {amrex::Real(0),
                                                    amrex::Real(0),
                                                    amrex::Real(0),
                                                    amrex::Real(0),
                                                    amrex::Real(0),
                                                    amrex::Real(0)},
                                                    // from ICU
                                                   {amrex::Real(0),
                                                    amrex::Real(0),
                                                    amrex::Real(0),
                                                    amrex::Real(0),
                                                    amrex::Real(0),
                                                    amrex::Real(0.26)},
                                                    // from ventilator
                                                   {amrex::Real(0.20),
                                                    amrex::Real(0.20),
                                                    amrex::Real(0.20),
                                                    amrex::Real(0.45),
                                                    amrex::Real(0.45),
                                                    amrex::Real(1.0)}};


    void readContact ();

    void readInputs (const std::string&);

    void Initialize ();

    void printMatrix ();

    /*! \brief Given age group, decide if hospitalized or not;
     *  if so, compute number of hospitalization days and check if
     *  moved to ICU and ventilator */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void check_hospitalization( amrex::Real& a_t_hosp, /*!< number of hospitalization days */
                                int& a_ICU, /*!< moved to ICU ? */
                                int& a_ventilator, /*!< moved to ventilator? */
                                const int a_age_group, /*!< age group */
                                const amrex::RandomEngine& a_reng /*!< random engine */) const
    {
        a_t_hosp = 0.0;
        a_ICU = 0;
        a_ventilator = 0;
        amrex::Real p_hosp = m_CHR[a_age_group];
        if (amrex::Random(a_reng) < p_hosp) {
            if (a_age_group == AgeGroups::o65) a_t_hosp = m_t_hosp[AgeGroups_Hosp::o65];
            else if (a_age_group == AgeGroups::a50to64) a_t_hosp = m_t_hosp[AgeGroups_Hosp::a50to64];
            else a_t_hosp = m_t_hosp[AgeGroups_Hosp::u50];
            if (amrex::Random(a_reng) < m_CIC[a_age_group]) {
                a_t_hosp += m_t_hosp_offset;  // move to ICU, adds 10 days (m_t_hosp_offset)
                a_ICU = 1;
                if (amrex::Random(a_reng) < m_CVE[a_age_group]) {
                    a_t_hosp += m_t_hosp_offset;  // put on ventilator, adds another 10 days
                    a_ventilator = 1;
                }
            }
        }
    }

};
#endif
