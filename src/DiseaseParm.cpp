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

/*! \brief Read disease inputs from input file */
void DiseaseParm::readInputs ( const std::string& a_pp_str /*!< Parmparse string */)
{
    ParmParse pp(a_pp_str);

    queryArray(pp, "xmit_comm", xmit_comm, AgeGroups::total);
    queryArray(pp, "xmit_hood", xmit_hood, AgeGroups::total);
    queryArray(pp, "xmit_hh_adult", xmit_hh_adult, AgeGroups::total);
    queryArray(pp, "xmit_hh_child", xmit_hh_child, AgeGroups::total);
    queryArray(pp, "xmit_nc_adult", xmit_nc_adult, AgeGroups::total);
    queryArray(pp, "xmit_nc_child", xmit_nc_child, AgeGroups::total);

    queryArray(pp, "xmit_school", xmit_school, SchoolType::total);
    queryArray(pp, "xmit_school_a2c", xmit_school_a2c, SchoolType::total);
    queryArray(pp, "xmit_school_c2a", xmit_school_c2a, SchoolType::total);

    pp.query("nstrain", nstrain);
    AMREX_ASSERT(nstrain <= 2);
    pp.query("reinfect_prob", reinfect_prob);

    queryArray(pp, "p_trans", p_trans, nstrain);
    queryArray(pp, "p_asymp", p_asymp, nstrain);
    queryArray(pp, "reduced_inf", reduced_inf, nstrain);

    pp.query("infect", infect);
    pp.query("reinfect", reinfect_prob);
    pp.query("vac_eff", vac_eff);

    pp.query("child_compliance", Child_compliance);
    pp.query("child_hh_closure", Child_HH_closure);

    pp.query("latent_length_mean", latent_length_mean);
    pp.query("infectious_length_mean", infectious_length_mean);
    pp.query("incubation_length_mean", incubation_length_mean);

    pp.query("latent_length_std", latent_length_std);
    pp.query("infectious_length_std", infectious_length_std);
    pp.query("incubation_length_std", incubation_length_std);

    pp.query("immune_length_mean", immune_length_mean);
    pp.query("immune_length_std", immune_length_std);

    queryArray(pp, "hospitalization_days", m_t_hosp, AgeGroups_Hosp::total);
    for (int i = 0; i < AgeGroups_Hosp::total; i++) {
        if (m_t_hosp[i] > m_t_hosp_offset) m_t_hosp_offset = m_t_hosp[i] + 1;
    }

    queryArray(pp, "CHR", m_CHR, AgeGroups::total);
    queryArray(pp, "CIC", m_CIC, AgeGroups::total);
    queryArray(pp, "CVE", m_CVE, AgeGroups::total);
    queryArray(pp, "hospCVF", m_HospToDeath[DiseaseStats::hospitalization], AgeGroups::total);
    queryArray(pp, "icuCVF", m_HospToDeath[DiseaseStats::ICU], AgeGroups::total);
    queryArray(pp, "ventCVF", m_HospToDeath[DiseaseStats::ventilator], AgeGroups::total);
}


/*! \brief Initialize disease parameters

    Compute transmission probabilities for various situations based on disease
    attributes.
*/
void DiseaseParm::Initialize ()
{
    // Optimistic scenario: 50% reduction in external child contacts during school dismissal
    //   or remote learning, and no change in household contacts
    Child_compliance=0.5_rt; Child_HH_closure=1.0_rt;
    // Pessimistic scenario: 30% reduction in external child contacts during school dismissal
    //   or remote learning, and 2x increase in household contacts
    //  sch_compliance=0.3; sch_effect=2.0;

    // Multiply contact rates by transmission probability given contact
    xmit_work *= p_trans[0];

    for (int i = 0; i < AgeGroups::total; i++) {
        xmit_comm[i] *= p_trans[0];
        xmit_hood[i] *= p_trans[0];
        xmit_nc_adult[i] *= p_trans[0];
        xmit_nc_child[i] *= p_trans[0];
        xmit_hh_adult[i] *= p_trans[0];
        xmit_hh_child[i] *= p_trans[0];
    }

    for (int i = 0; i < 5; i++) {
        xmit_school[i] *= p_trans[0];
        xmit_school_a2c[i] *= p_trans[0];
        xmit_school_c2a[i] *= p_trans[0];
    }

    /*
      Double household contact rate involving children, and reduce
      other child-related contacts (neighborhood cluster, neigborhood,
      and community) by the compliance rate, Child_compliance
    */
    for (int i = 0; i < AgeGroups::total; i++) {
        xmit_hh_child_SC[i] = xmit_hh_child[i] * Child_HH_closure;
        xmit_nc_child_SC[i] = xmit_nc_child[i] * (1.0_rt - Child_compliance);
    }
    // if receiver is a child
    for (int i = 0; i < AgeGroups::a18to29; i++) {
        xmit_hh_adult_SC[i] = xmit_hh_adult[i] * Child_HH_closure;
        xmit_nc_adult_SC[i] = xmit_nc_adult[i] * (1.0_rt - Child_compliance);
        xmit_comm_SC[i] = xmit_comm[i] * (1.0_rt - Child_compliance);
        xmit_hood_SC[i] = xmit_hood[i] * (1.0_rt - Child_compliance);
    }
    // if receiver is an adult, contacts remain unchanged
    for (int i = AgeGroups::a18to29; i < AgeGroups::total; i++) {
        xmit_hh_adult_SC[i] = xmit_hh_adult[i];
        xmit_nc_adult_SC[i] = xmit_nc_adult[i];
        xmit_comm_SC[i] = xmit_comm[i];
        xmit_hood_SC[i] = xmit_hood[i];
    }

    infect = 1.0_rt;
}

/*! \brief Print disease parameters */
void DiseaseParm::printMatrix () {
    /*
    Print() << "xmit_comm: " << " ";
    for (int i = 0; i < AgeGroups::total; ++i) {
        Print() << xmit_comm[i] << " ";
    }
    Print() << "\n";

    Print() << "xmit_hood: " <<  " ";
    for (int i = 0; i < AgeGroups::total; ++i) {
        Print() << xmit_hood[i] << " ";
    }
    Print() << "\n";

    Print() << "xmit_nc_adult: " << " ";
    for (int i = 0; i < AgeGroups::total; ++i) {
        Print() << xmit_nc_adult[i] << " ";
    }
    Print() << "\n";

    Print() << "xmit_nc_child: " << " ";
    for (int i = 0; i < AgeGroups::total; ++i) {
        Print() << xmit_nc_child[i] << " ";
    }
    Print() << "\n";

    Print() << "xmit_work: " << " ";
    Print() << xmit_work << "\n";

    Print() << "xmit_child_SC: " << " ";
    for (int i = 0; i < AgeGroups::total; ++i) {
        Print() << xmit_child_SC[i] << " ";
    }
    Print() << "\n";

    Print() << "xmit_nc_child_SC: " << " ";
    for (int i = 0; i < AgeGroups::total; ++i) {
        Print() << xmit_nc_child_SC[i] << " ";
    }
    Print() << "\n";

    Print() << "xmit_adult_SC: " << " ";
    for (int i = 0; i < 2; ++i) {
        Print() << xmit_adult_SC[i] << " ";
    }
    Print() << "\n";

    Print() << "xmit_nc_adult_SC: " << " ";
    for (int i = 0; i < 2; ++i) {
        Print() << xmit_nc_adult_SC[i] << " ";
    }
    Print() << "\n";

    Print() << "xmit_hood_SC: " << " ";
    for (int i = 0; i < 2; ++i) {
        Print() << xmit_hood_SC[i] << " ";
    }
    Print() << "\n";

    Print() << "xmit_comm_SC: " << " ";
    for (int i = 0; i < 2; ++i) {
        Print() << xmit_comm_SC[i] << " ";
    }
    Print() << "\n";
    */
}
