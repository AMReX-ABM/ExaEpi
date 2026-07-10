/*! @file DiseaseParm.H
    \brief #DiseaseParm struct definition
*/

#ifndef DISEASE_PARM_H_
#define DISEASE_PARM_H_

#include <AMReX_GpuMemory.H>
#include <AMReX_REAL.H>
#include <AMReX_Random.H>

#include "AgentDefinitions.H"

using amrex::ParticleReal;
using amrex::Real;
using amrex::Vector;

struct CaseTypes {
    enum {
        rnd = 0, /*!< randomly allocate */
        file     /*!< data read from file */
    };
};

struct HospitalStayType {
    enum {
        Constant = 0, /*!< Agents in the same age group all have the same hospital stay duration. */
        Random        /*!< Agents have a random hospital day duration, with the distribution varying by age group. */
    };
};

/*! \brief Disease parameters

    Structure containing disease parameters.
*/
struct DiseaseParm {
    /*! Initial case type (random or read from file) */
    int initial_case_type = CaseTypes::rnd;
    /*! Number of initial cases (in case of random initialization) */
    int num_initial_cases = 0;
    /*! Name of disease. This is a char array because this whole structure is memcpy'd to the device */
    char disease_name[50];
    /*! Initial cases filename (CaseData::initFromFile):
        The case data file is an ASCII text file with three columns of numbers:
        FIPS code, current number of cases, and cumulative number of cases till date. */
    char case_filename[255];

    // Transmission probabilities based on age group of receiver (0-4, 5-17, 18-29, 30-49, 50-64, 65+)
    /*! community transmission, both home and work */
    Real xmit_comm[AgeGroups::total] = {Real(0.000021), Real(0.000062), Real(0.000165),
                                        Real(0.000165), Real(0.000165), Real(0.000247)};
    /*! neighborhood transmission, both home and work */
    Real xmit_hood[AgeGroups::total] = {Real(0.000082), Real(0.000247), Real(0.00066), Real(0.00066), Real(0.00066), Real(0.001)};
    /*! within household transmission, where transmitter is an adult */
    Real xmit_hh_adult[AgeGroups::total] = {Real(0.3), Real(0.3), Real(0.4), Real(0.4), Real(0.4), Real(0.4)};
    /*! within household transmission, where transmitter is a child */
    Real xmit_hh_child[AgeGroups::total] = {Real(0.6), Real(0.6), Real(0.3), Real(0.3), Real(0.3), Real(0.3)};
    /*! neighborhood cluster transmission, where transmitter is an adult */
    Real xmit_nc_adult[AgeGroups::total] = {Real(0.132), Real(0.132), Real(0.165), Real(0.165), Real(0.165), Real(0.165)};
    /*! neighborhood cluster transmission, where transmitter is a child */
    Real xmit_nc_child[AgeGroups::total] = {Real(0.25), Real(0.25), Real(0.132), Real(0.132), Real(0.132), Real(0.132)};
    /// probabilities for school groups: none, college, high, middle, elementary, and daycare
    /*! child-to-child */
    Real xmit_school[SchoolType::total] = {Real(0), Real(0.0002), Real(0.0024), Real(0.0027), Real(0.0088), Real(0.0125)};
    /*! for adult to child */
    Real xmit_school_a2c[SchoolType::total] = {Real(0), Real(0.001), Real(0.01), Real(0.013), Real(0.043), Real(0.025)};
    /*! for child to adult */
    Real xmit_school_c2a[SchoolType::total] = {Real(0), Real(0.001), Real(0.01), Real(0.013), Real(0.043), Real(0.025)};
    /*! scale factor applied to all xmit_school* values; single knob to adjust all school transmission */
    Real xmit_school_scale = Real(1.0);

    /*! Probabilities when school is closed */
    Real xmit_comm_SC[AgeGroups::total];
    Real xmit_hood_SC[AgeGroups::total];
    Real xmit_hh_adult_SC[AgeGroups::total];
    Real xmit_hh_child_SC[AgeGroups::total];
    Real xmit_nc_adult_SC[AgeGroups::total];
    Real xmit_nc_child_SC[AgeGroups::total];

    /*! probability for transmission within a workgroup, independent of age group currently */
    Real xmit_work = Real(0.0575);

    Real p_trans = Real(0.20);                 /*!< probability of transimission given contact */
    Real p_asymp = Real(0.30);                 /*!< fraction of cases that are asymptomatic */
    Real asymp_relative_inf = Real(0.7);       /*!< relative infectiousness of asymptomatic individuals */

    Real vac_eff = Real(0.0);                  /*!< Vaccine efficacy */

    Real child_compliance;                     /*!< Child compliance with masking ?? */
    Real child_HH_closure;                     /*!< Multiplier for household contacts during school closure */

    Real immune_length_alpha = Real(1.0);      /*! alpha parameter for gamma distribution*/
    Real immune_length_beta = Real(1.0);       /*! beta parameter for gamma distribution*/
    Real immune_length_loc = Real(10000.0);    /*! location (shift) parameter for gamma distribution*/

    Real latent_length_alpha = Real(6.0);      /*! alpha parameter for gamma distribution (unused - see latent_period_cdf)*/
    Real latent_length_beta = Real(0.73);      /*! beta parameter for gamma distribution (unused - see latent_period_cdf)*/

    Real infectious_length_alpha = Real(3.54); /*! alpha parameter for gamma distribution (unused - see infectious_period_cdf)*/
    Real infectious_length_beta = Real(1.22);  /*! beta parameter for gamma distribution (unused - see infectious_period_cdf)*/
    Real infectious_length_loc = Real(2.5);    /*! location (shift) parameter for gamma distribution */

    Real incubation_length_alpha = Real(6.0);  /*! alpha parameter for gamma distribution (unused - see incubation_period_cdf)*/
    Real incubation_length_beta = Real(0.73);  /*! beta parameter for gamma distribution (unused - see incubation_period_cdf)*/
    Real incubation_length_loc = Real(1.0);    /*! location (shift) parameter for gamma distribution */

    Real hospital_delay_length_alpha = Real(0.1); /*! alpha parameter for gamma distribution*/
    Real hospital_delay_length_beta = Real(0.1);  /*! beta parameter for gamma distribution*/
    Real hospital_delay_length_loc = Real(1.0);   /*! location (shift) parameter for gamma distribution*/

    int m_hospital_stay_type = HospitalStayType::Constant;

    /*! number of hospitalization days by age group (#AgeGroups) */
    Real m_t_hosp_days[AgeGroups::total] = {Real(3), Real(3), Real(3), Real(3), Real(8), Real(7)};
    /*! Offset to separate the timers for hospital, ICU, and ventilator;
     * All agents just in hospital will have a timer counting up from 0, all agents in ICU will
     * have a timer counting up from m_t_hosp_offset, and all agents ventilated will have a timer
     * counting up from 2 * m_t_hosp_offset
     * For this reason, this value needs to be greater than the maximum of #DiseaseParm::m_t_hosp_days,
     * or whatvere value is generated from the gamma distribution for random length stays.
     * The default here is greater than a human lifespan, and so never needs to be changed.
     */
    Real m_t_hosp_offset = 100000.0;

    /*! Parameters for gamma distribution for hospital stay lengths. These are used if
     *  m_hospital_stay_type = HospitalStayType::Random */
    Real m_t_hosp_days_alpha[AgeGroups::total] = {Real(1.33), Real(1.33), Real(1.33), Real(1.33), Real(0.5), Real(0.57)};
    Real m_t_hosp_days_beta[AgeGroups::total] = {Real(1.5), Real(1.5), Real(1.5), Real(1.5), Real(4.0), Real(3.5)};

    /*! sick -> hospital probabilities */
    Real m_CHR[AgeGroups::total] = {Real(.0104), Real(.0104), Real(.070), Real(.28), Real(.28), Real(1.0)};
    /*! hospital -> ICU probabilities */
    Real m_CIC[AgeGroups::total] = {Real(.24), Real(.24), Real(.24), Real(.36), Real(.36), Real(.35)};
    /*! ICU -> ventilator probabilities */
    Real m_CVE[AgeGroups::total] = {Real(.12), Real(.12), Real(.12), Real(.22), Real(.22), Real(.22)};
    /*! Probabilites of dying while in hospital, for age groups, in hospital, ICU and ventilated */
    Real m_hospToDeath[3][AgeGroups::total] = {{Real(0), Real(0), Real(0), Real(0), Real(0), Real(0)},
                                               {Real(0), Real(0), Real(0), Real(0), Real(0), Real(0.26)},
                                               {Real(0.20), Real(0.20), Real(0.20), Real(0.45), Real(0.45), Real(1.0)}};

    DiseaseParm (const std::string& dname) {
        memset(disease_name, 0, 50);
        strncpy(disease_name, dname.c_str(), 49);
        memset(case_filename, 0, 255);
    }

    void readInputs(const std::string&);

    void initialize();

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    bool checkHospitalization (const amrex::Real a_random_draw, /*!< random number */
                               const int a_age_group /*!< age group */) const {
        return (a_random_draw < m_CHR[a_age_group]);
    }

    /*! \brief Given age group, decide if hospitalized or not;
     *  if so, compute number of hospitalization days and check if
     *  moved to ICU and ventilator */
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void checkHospitalization (ParticleReal* a_t_hosp_days,     /*!< number of hospitalization days */
                               int* a_ICU,                      /*!< moved to ICU ? */
                               int* a_ventilator,               /*!< moved to ventilator? */
                               const int a_age_group,           /*!< age group */
                               const amrex::Real a_random_draw, /*!< random number */
                               const amrex::RandomEngine& a_reng /*!< random engine */) const {
        *a_t_hosp_days = ParticleReal(0);
        // check the agent will be hospitalized (m_CHR prob)
        if (a_random_draw < m_CHR[a_age_group]) {
            // determine how long agent will stay in hospital
            if (m_hospital_stay_type == HospitalStayType::Constant) {
                *a_t_hosp_days = m_t_hosp_days[a_age_group];
            } else { // Random
                int num_days = static_cast<int>(
                        amrex::RandomGamma(m_t_hosp_days_alpha[a_age_group], m_t_hosp_days_beta[a_age_group], a_reng));
                // the number of days must always be less than the offset, because the offset is used to separate out the
                // only-hospitalized from ICU and ventilated
                if (num_days > m_t_hosp_offset - 1.0) { num_days = (int)(m_t_hosp_offset - 1.0); }
                if (num_days < 1) { num_days = 1; }
                *a_t_hosp_days = static_cast<ParticleReal>(num_days);
            }
            // check if agent will be in ICU
            if (amrex::Random(a_reng) < m_CIC[a_age_group]) {
                *a_t_hosp_days += m_t_hosp_offset; // move to ICU, treatment timer now runs starting at m_t_hosp_offset
                *a_ICU = 1;
                // check if agent will be on ventilator
                if (amrex::Random(a_reng) < m_CVE[a_age_group]) {
                    *a_t_hosp_days += m_t_hosp_offset; // put on ventilator, treatment timer starts at 2 * m_t_hosp_offset
                    *a_ventilator = 1;
                }
            }
        }
    }
};

#ifdef COMPARE_TO_EPICAST
/*! Fixed cumulative distribution for latent period from Epicast (fig 6)
 *  Entry i gives P(transition by day i). Days 0-7. */
static constexpr int latent_period_cdf_size = 8;
AMREX_GPU_DEVICE AMREX_GPU_CONSTANT
constexpr Real latent_period_cdf[latent_period_cdf_size] = {Real(0.0),  Real(0.1),  Real(0.25), Real(0.55),
                                                            Real(0.66), Real(0.78), Real(0.85), Real(1.0)};

/*! Fixed cumulative distribution for incubation period from Epicast (fig 6)
 *  Entry i gives P(transition by day i). Days 0-8. */
static constexpr int incubation_period_cdf_size = 9;
AMREX_GPU_DEVICE AMREX_GPU_CONSTANT
constexpr Real incubation_period_cdf[incubation_period_cdf_size] = {Real(0.0),  Real(0.0),  Real(0.1),  Real(0.25), Real(0.55),
                                                                    Real(0.66), Real(0.78), Real(0.85), Real(1.0)};

/*! Fixed cumulative distribution for infectious period from Epicast (fig 6)
 *  Entry i gives P(transition by day i). Days 0-10. */
static constexpr int infectious_period_cdf_size = 11;
AMREX_GPU_DEVICE AMREX_GPU_CONSTANT
constexpr Real infectious_period_cdf[infectious_period_cdf_size] = {Real(0.0),  Real(0.0),  Real(0.0), Real(0.0),
                                                                    Real(0.1),  Real(0.3),  Real(0.5), Real(0.7),
                                                                    Real(0.85), Real(0.95), Real(1.0)};
#endif

/*! \brief Will a symptomatic agent be hospitalized? */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
bool willBeHospitalized (const int a_idx,      /*!< Agent index */
                         const PTDType& a_ptd, /*!< Particle tile data */
                         const int a_d,        /*!< Disease index */
                         const DiseaseParm& a_parm /*!< disease parameters*/) {
    amrex::Real draw = a_ptd.m_runtime_rdata[r0(a_d) + RealIdxDisease::hospital_random][a_idx];
    return a_parm.checkHospitalization(draw, a_ptd.m_idata[IntIdx::age_group][a_idx]);
}

/*! \brief Sample a period (in days) from a fixed cumulative distribution array.
 *
 *  Draws a uniform random number and returns the first day index where the
 *  cumulative probability exceeds the draw (inverse CDF / quantile method).
 *  If the draw exceeds all entries the last valid day index is returned.
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int sampleFromCDF (const amrex::Real* cdf, int cdf_size, amrex::RandomEngine const& engine) {
    amrex::Real u = amrex::Random(engine);
    for (int i = 0; i < cdf_size; ++i) {
        if (u <= cdf[i]) { return i; }
    }
    return cdf_size - 1;
}

/*! \brief Set this agent to infected status, and initialize disease periods. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void setInfected (int* status, int* symptomatic, amrex::ParticleReal* counter, amrex::ParticleReal* latent_period,
                  amrex::ParticleReal* infectious_period, amrex::ParticleReal* incubation_period,
                  amrex::ParticleReal* hospital_delay, amrex::ParticleReal* hospital_random, amrex::RandomEngine const& engine,
                  const DiseaseParm* lparm) {
    *status = Status::infected;
    *symptomatic = SymptomStatus::presymptomatic;
    *counter = ParticleReal(0);

#ifdef COMPARE_TO_EPICAST
    // Sample latent, incubation, and infectious periods from fixed Epicast CDFs
    *latent_period = static_cast<ParticleReal>(sampleFromCDF(latent_period_cdf, latent_period_cdf_size, engine));
    *incubation_period = *latent_period + 1.0;
    *infectious_period = static_cast<ParticleReal>(sampleFromCDF(infectious_period_cdf, infectious_period_cdf_size, engine));
    *hospital_delay = amrex::Real(2);
#else
    *latent_period = static_cast<ParticleReal>(amrex::RandomGamma(lparm->latent_length_alpha, lparm->latent_length_beta, engine));
    *incubation_period =
            static_cast<ParticleReal>(amrex::RandomGamma(lparm->incubation_length_alpha, lparm->incubation_length_beta, engine)) +
            static_cast<ParticleReal>(lparm->incubation_length_loc);
    // Hack to match EPICAST hospitalization rates better
    //*incubation_period = *latent_period + 1.0;
    *infectious_period =
            static_cast<ParticleReal>(amrex::RandomGamma(lparm->infectious_length_alpha, lparm->infectious_length_beta, engine)) +
            static_cast<ParticleReal>(lparm->infectious_length_loc);

    *hospital_delay = static_cast<ParticleReal>(
                              amrex::RandomGamma(lparm->hospital_delay_length_alpha, lparm->hospital_delay_length_beta, engine)) +
                      static_cast<ParticleReal>(lparm->hospital_delay_length_loc);
    if (*latent_period < 1) { *latent_period = amrex::Real(1); }
    if (*incubation_period < 1) { *incubation_period = amrex::Real(1); }
    if (*infectious_period < 1) { *infectious_period = amrex::Real(1); }
#endif
    if (*incubation_period >= (*infectious_period + *latent_period)) {
        *incubation_period = std::floor(*infectious_period + *latent_period) - amrex::Real(1);
    }

    *hospital_random = static_cast<ParticleReal>(amrex::Random(engine));
}

template <typename T>
void queryArray (amrex::ParmParse& pp, const std::string& s, T* a, int n) {
    Vector<T> tmp(n, 0);
    for (int i = 0; i < n; i++) {
        tmp[i] = a[i];
    }
    pp.queryarr(s.c_str(), tmp, 0, n);
    for (int i = 0; i < n; i++) {
        a[i] = tmp[i];
    }
}

template <typename T, int N>
void queryGpuArray (amrex::ParmParse& pp, const std::string& s, amrex::GpuArray<T, N>& a) {
    T tmp_arr[N];
    for (unsigned int i = 0; i < N; i++) {
        tmp_arr[i] = a[i];
    }

    queryArray(pp, s, tmp_arr, N);
    for (unsigned int i = 0; i < N; ++i) {
        a[i] = tmp_arr[i];
    }
}

#endif
