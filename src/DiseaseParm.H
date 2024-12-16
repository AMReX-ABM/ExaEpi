/*! @file DiseaseParm.H
    \brief #DiseaseParm struct definition
*/

#ifndef DISEASE_PARM_H_
#define DISEASE_PARM_H_

#include <AMReX_Random.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

#include "AgentDefinitions.H"

using amrex::Real;
using amrex::ParticleReal;
using amrex::Vector;

struct CaseTypes
{
    enum {
        rnd = 0, /*!< randomly allocate */
        file        /*!< data read from file */
    };
};

/*! \brief Disease parameters

    Structure containing disease parameters.
*/
struct DiseaseParm
{
    /*! Initial case type (random or read from file) */
    int initial_case_type = CaseTypes::rnd;
    /*! Number of initial cases (in case of random initialization) */
    int num_initial_cases = 0;
    /*! Name of disease. This is a char array because this whole structure is memcpy'd to the device */
    char disease_name[50];
    /*! Initial cases filename (CaseData::InitFromFile):
        The case data file is an ASCII text file with three columns of numbers:
        FIPS code, current number of cases, and cumulative number of cases till date. */
    char case_filename[255];

    // Transmission probabilities based on age group of receiver (0-4, 5-17, 18-29, 30-49, 50-64, 65+)
    /*! community transmission, both home and work */
    Real xmit_comm[AgeGroups::total] = {Real(0.000018125), Real(0.000054375), Real(0.000145), Real(0.000145), Real(0.000145), Real(0.0002175)};
    /*! neighborhood transmission, both home and work */
    Real xmit_hood[AgeGroups::total] = {Real(0.0000725), Real(0.0002175), Real(0.00058), Real(0.00058), Real(0.00058), Real(0.00087)};
    /*! within household transmission, where transmitter is an adult */
    Real xmit_hh_adult[AgeGroups::total] = {Real(0.3), Real(0.3), Real(0.4), Real(0.4), Real(0.4), Real(0.4)};
    /*! within household transmission, where transmitter is a child */
    Real xmit_hh_child[AgeGroups::total] = {Real(0.6), Real(0.6), Real(0.3), Real(0.3), Real(0.3), Real(0.3)};
    /*! neighborhood cluster transmission, where transmitter is an adult */
    Real xmit_nc_adult[AgeGroups::total] = {Real(0.04), Real(0.04), Real(0.05), Real(0.05), Real(0.05), Real(0.05)};
    /*! neighborhood cluster transmission, where transmitter is a child */
    Real xmit_nc_child[AgeGroups::total] = {Real(0.075), Real(0.075), Real(0.04), Real(0.04), Real(0.04), Real(0.04)};
    /// probabilities for school groups: none, college, high, middle, elementary, and daycare
    /*! child-to-child */
    Real xmit_school[SchoolType::total] = {Real(0), Real(0.0315), Real(0.0315), Real(0.0375), Real(0.0435), Real(0.15)};
    /*! for adult to child */
    Real xmit_school_a2c[SchoolType::total] = {Real(0), Real(0.0315), Real(0.0315), Real(0.0375), Real(0.0435), Real(0.15)};
    /*! for child to adult */
    Real xmit_school_c2a[SchoolType::total] = {Real(0), Real(0.0315), Real(0.0315), Real(0.0375), Real(0.0435), Real(0.15)};


    /*! Probabilities when school is closed */
    Real xmit_comm_SC[AgeGroups::total];
    Real xmit_hood_SC[AgeGroups::total];
    Real xmit_hh_adult_SC[AgeGroups::total];
    Real xmit_hh_child_SC[AgeGroups::total];
    Real xmit_nc_adult_SC[AgeGroups::total];
    Real xmit_nc_child_SC[AgeGroups::total];

    /*! probability for transmission within a workgroup, independent of age group currently */
    Real xmit_work = Real(0.0575);

    Real p_trans = Real(0.20);     /*!< probability of transimission given contact */
    Real p_asymp = Real(0.40);     /*!< fraction of cases that are asymptomatic */
    Real asymp_relative_inf = Real(0.75); /*!< relative infectiousness of asymptomatic individuals */

    Real vac_eff = Real(0.0); /*!< Vaccine efficacy */

    Real child_compliance; /*!< Child compliance with masking ?? */
    Real child_HH_closure; /*!< Multiplier for household contacts during school closure */

    Real immune_length_alpha = Real(9.0);  /*! alpha parameter for gamma distribution*/
    Real immune_length_beta = Real(20.0);  /*! beta parameter for gamma distribution*/

    Real latent_length_alpha = Real(9.0);  /*! alpha parameter for gamma distribution*/
    Real latent_length_beta = Real(0.33);  /*! beta parameter for gamma distribution*/

    Real infectious_length_alpha = Real(36.0);  /*! alpha parameter for gamma distribution*/
    Real infectious_length_beta = Real(0.17);   /*! beta parameter for gamma distribution*/

    Real incubation_length_alpha = Real(25.0);  /*! alpha parameter for gamma distribution*/
    Real incubation_length_beta = Real(0.2);    /*! beta parameter for gamma distribution*/

    /*! number of hospitalization days by age group (#AgeGroups_Hosp); note that the
     *  age groups here are under 50, 50-64, and over 65, and *not* the age groups
     *  used in other parts of the code (#AgeGroups) */
    Real m_t_hosp[AgeGroups_Hosp::total] = {Real(3), Real(8), Real(7)};
    /*! Offset to separate the timers for hospital, ICU, and ventilator;
     * needs to be greater than the maximum of #DiseaseParm::m_t_hosp
     * Set automatically when the hospital days are set. */
    Real m_t_hosp_offset = 10;

    /*! sick -> hospital probabilities */
    Real m_CHR[AgeGroups::total] = {Real(.0104), Real(.0104), Real(.070), Real(.28), Real(.28), Real(1.0)};
    /*! hospital -> ICU probabilities */
    Real m_CIC[AgeGroups::total] = {Real(.24), Real(.24), Real(.24), Real(.36), Real(.36), Real(.35)};
    /*! ICU -> ventilator probabilities */
    Real m_CVE[AgeGroups::total] = {Real(.12), Real(.12), Real(.12), Real(.22), Real(.22), Real(.22)};
    /*! Probabilites of dying while in hospital, for age groups, in hosp, icu and ventilated */
    Real m_hospToDeath[AgeGroups_Hosp::total][AgeGroups::total] = {
        {Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)},
        {Real(0), Real(0), Real(0), Real(0), Real(0), Real(0.26)},
        {Real(0.20), Real(0.20), Real(0.20), Real(0.45), Real(0.45), Real(1.0)}};


    DiseaseParm(const std::string& dname) {
        memset(disease_name, 0, 50);
        strncpy(disease_name, dname.c_str(), 49);
        memset(case_filename, 0, 255);
    }

    void readInputs (const std::string&);

    void Initialize ();

    /*! \brief Given age group, decide if hospitalized or not;
     *  if so, compute number of hospitalization days and check if
     *  moved to ICU and ventilator */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void check_hospitalization (ParticleReal* a_t_hosp, /*!< number of hospitalization days */
                                int* a_ICU, /*!< moved to ICU ? */
                                int* a_ventilator, /*!< moved to ventilator? */
                                const int a_age_group, /*!< age group */
                                const amrex::RandomEngine& a_reng /*!< random engine */) const
    {
        *a_t_hosp = ParticleReal(0);
        *a_ICU = 0;
        *a_ventilator = 0;
        if (amrex::Random(a_reng) < m_CHR[a_age_group]) {
            if (a_age_group == AgeGroups::o65) *a_t_hosp = m_t_hosp[AgeGroups_Hosp::o65];
            else if (a_age_group == AgeGroups::a50to64) *a_t_hosp = m_t_hosp[AgeGroups_Hosp::a50to64];
            else *a_t_hosp = m_t_hosp[AgeGroups_Hosp::u50];
            if (amrex::Random(a_reng) < m_CIC[a_age_group]) {
                *a_t_hosp += m_t_hosp_offset;  // move to ICU, adds 10 days (m_t_hosp_offset)
                *a_ICU = 1;
                if (amrex::Random(a_reng) < m_CVE[a_age_group]) {
                    *a_t_hosp += m_t_hosp_offset;  // put on ventilator, adds another 10 days
                    *a_ventilator = 1;
                }
            }
        }
    }

};


/*! \brief Set this agent to infected status, and initialize disease periods. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void setInfected ( int* status,
                   amrex::ParticleReal* counter,
                   amrex::ParticleReal* latent_period,
                   amrex::ParticleReal* infectious_period,
                   amrex::ParticleReal* incubation_period,
                   amrex::RandomEngine const& engine,
                   const DiseaseParm* lparm)
{
    *status = Status::infected;
    *counter = ParticleReal(0);
    *latent_period =
        static_cast<ParticleReal>(amrex::RandomGamma(lparm->latent_length_alpha, lparm->latent_length_beta, engine));
    *infectious_period =
        static_cast<ParticleReal>(amrex::RandomGamma(lparm->infectious_length_alpha, lparm->infectious_length_beta, engine));
    *incubation_period =
        static_cast<ParticleReal>(amrex::RandomGamma(lparm->incubation_length_alpha, lparm->incubation_length_beta, engine));
    if (*latent_period < 0) { *latent_period = amrex::Real(0);}
    if (*infectious_period < 0) { *infectious_period = amrex::Real(0);}
    if (*incubation_period < 0) { *incubation_period = amrex::Real(0);}
    if (*incubation_period > (*infectious_period + *latent_period)) {
        *incubation_period = std::floor(*infectious_period + *latent_period);
    }
}

void queryArray(amrex::ParmParse &pp, const std::string& s, amrex::Real* a, int n);
void queryArray(amrex::ParmParse &pp, const std::string& s, int* a, int n);


#endif
