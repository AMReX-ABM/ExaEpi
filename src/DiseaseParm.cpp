/*! @file DiseaseParm.cpp
    \brief Function implementations for #DiseaseParm class
*/

#include "DiseaseParm.H"

#include "AMReX_Print.H"

using namespace amrex;

void queryArray(ParmParse &pp, const std::string& s, Real* a, int n) {
    Vector<Real> tmp(n, 0);
    for (int i = 0; i < n; i++) {
        tmp[i] = a[i];
    }
    pp.queryarr(s.c_str(), tmp, 0, n);
    for (int i = 0; i < n; i++) {
        a[i] = tmp[i];
    }
}

void queryArray(ParmParse &pp, const std::string& s, int* a, int n) {
    Vector<int> tmp(n, 0);
    for (int i = 0; i < n; i++) {
        tmp[i] = a[i];
    }
    pp.queryarr(s.c_str(), tmp, 0, n);
    for (int i = 0; i < n; i++) {
        a[i] = tmp[i];
    }
}

/*! \brief Read disease inputs from input file */
void DiseaseParm::readInputs ( const std::string& a_pp_str /*!< Parmparse string */)
{
    ParmParse pp(a_pp_str);

    std::string initial_case_type_str = (initial_case_type == CaseTypes::rnd ? "random" : "file");
    pp.query("initial_case_type", initial_case_type_str);
    if (initial_case_type_str == "file") {
        initial_case_type = CaseTypes::file;
        /*! Initial cases filename (CaseData::InitFromFile):
        The case data file is an ASCII text file with three columns of numbers:
        FIPS code, current number of cases, and cumulative number of cases till date. */
        std::string case_filename_str(case_filename);
        if (pp.contains("case_filename")) pp.get("case_filename", case_filename_str);
        strncpy(case_filename, case_filename_str.c_str(), 254);
    } else if (initial_case_type_str == "random") {
        initial_case_type = CaseTypes::rnd;
        pp.query("num_initial_cases", num_initial_cases);
    } else {
        amrex::Abort("initial case type not recognized");
    }

    queryArray(pp, "xmit_comm", xmit_comm, AgeGroups::total);
    queryArray(pp, "xmit_hood", xmit_hood, AgeGroups::total);
    queryArray(pp, "xmit_hh_adult", xmit_hh_adult, AgeGroups::total);
    queryArray(pp, "xmit_hh_child", xmit_hh_child, AgeGroups::total);
    queryArray(pp, "xmit_nc_adult", xmit_nc_adult, AgeGroups::total);
    queryArray(pp, "xmit_nc_child", xmit_nc_child, AgeGroups::total);

    queryArray(pp, "xmit_school", xmit_school, SchoolType::total);
    queryArray(pp, "xmit_school_a2c", xmit_school_a2c, SchoolType::total);
    queryArray(pp, "xmit_school_c2a", xmit_school_c2a, SchoolType::total);

    pp.query("p_trans", p_trans);
    pp.query("p_asymp", p_asymp);
    pp.query("asymp_relative_inf", asymp_relative_inf);

    pp.query("vac_eff", vac_eff);
    // no support yet for vaccinations
    AMREX_ALWAYS_ASSERT(vac_eff == 0);

    pp.query("child_compliance", child_compliance);
    pp.query("child_hh_closure", child_HH_closure);

    pp.query("latent_length_alpha", latent_length_alpha);
    pp.query("infectious_length_alpha", infectious_length_alpha);
    pp.query("incubation_length_alpha", incubation_length_alpha);

    pp.query("latent_length_beta", latent_length_beta);
    pp.query("infectious_length_beta", infectious_length_beta);
    pp.query("incubation_length_beta", incubation_length_beta);

    pp.query("immune_length_alpha", immune_length_alpha);
    pp.query("immune_length_beta", immune_length_beta);

    m_t_hosp_offset = 0;
    queryArray(pp, "hospitalization_days", m_t_hosp, AgeGroups_Hosp::total);
    for (int i = 0; i < AgeGroups_Hosp::total; i++) {
        if (m_t_hosp[i] > m_t_hosp_offset) m_t_hosp_offset = m_t_hosp[i] + 3;
    }

    queryArray(pp, "CHR", m_CHR, AgeGroups::total);
    queryArray(pp, "CIC", m_CIC, AgeGroups::total);
    queryArray(pp, "CVE", m_CVE, AgeGroups::total);
    queryArray(pp, "hospCVF", m_hospToDeath[DiseaseStats::hospitalization], AgeGroups::total);
    queryArray(pp, "icuCVF", m_hospToDeath[DiseaseStats::ICU], AgeGroups::total);
    queryArray(pp, "ventCVF", m_hospToDeath[DiseaseStats::ventilator], AgeGroups::total);
}


/*! \brief Initialize disease parameters

    Compute transmission probabilities for various situations based on disease
    attributes.
*/
void DiseaseParm::Initialize ()
{
    // Optimistic scenario: 50% reduction in external child contacts during school dismissal
    //   or remote learning, and no change in household contacts
    child_compliance = 0.5_rt;
    child_HH_closure = 2.0_rt;
    // Pessimistic scenario: 30% reduction in external child contacts during school dismissal
    //   or remote learning, and 2x increase in household contacts
    //  sch_compliance=0.3; sch_effect=2.0;

    // Multiply contact rates by transmission probability given contact
    xmit_work *= p_trans;

    for (int i = 0; i < AgeGroups::total; i++) {
        xmit_comm[i] *= p_trans;
        xmit_hood[i] *= p_trans;
        xmit_nc_adult[i] *= p_trans;
        xmit_nc_child[i] *= p_trans;
        xmit_hh_adult[i] *= p_trans;
        xmit_hh_child[i] *= p_trans;
    }

    for (int i = 0; i < 5; i++) {
        xmit_school[i] *= p_trans;
        xmit_school_a2c[i] *= p_trans;
        xmit_school_c2a[i] *= p_trans;
    }

    /*
      Double household contact rate involving children, and reduce
      other child-related contacts (neighborhood cluster, neigborhood,
      and community) by the compliance rate, child_compliance
    */
    for (int i = 0; i < AgeGroups::total; i++) {
        xmit_hh_child_SC[i] = xmit_hh_child[i] * child_HH_closure;
        xmit_nc_child_SC[i] = xmit_nc_child[i] * (1.0_rt - child_compliance);
    }
    // if receiver is a child
    for (int i = 0; i < AgeGroups::a18to29; i++) {
        xmit_hh_adult_SC[i] = xmit_hh_adult[i] * child_HH_closure;
        xmit_nc_adult_SC[i] = xmit_nc_adult[i] * (1.0_rt - child_compliance);
        xmit_comm_SC[i] = xmit_comm[i] * (1.0_rt - child_compliance);
        xmit_hood_SC[i] = xmit_hood[i] * (1.0_rt - child_compliance);
    }
    // if receiver is an adult, contacts remain unchanged
    for (int i = AgeGroups::a18to29; i < AgeGroups::total; i++) {
        xmit_hh_adult_SC[i] = xmit_hh_adult[i];
        xmit_nc_adult_SC[i] = xmit_nc_adult[i];
        xmit_comm_SC[i] = xmit_comm[i];
        xmit_hood_SC[i] = xmit_hood[i];
    }

}

