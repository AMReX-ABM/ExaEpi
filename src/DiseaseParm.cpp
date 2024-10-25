/*! @file DiseaseParm.cpp
    \brief Function implementations for #DiseaseParm class
*/

#include "DiseaseParm.H"

#include "AMReX_Print.H"

using namespace amrex::literals;

/*! \brief Read contact coefficients  from input file */
void DiseaseParm::readContact ()
{
    std::string key = "contact";
    amrex::ParmParse pp(key);
    pp.query("pSC", pSC);
    pp.query("pCO", pCO);
    pp.query("pNH", pNH);
    pp.query("pWO", pWO);
    pp.query("pFA", pFA);
    pp.query("pBAR", pBAR);
}

/*! \brief Read disease inputs from input file */
void DiseaseParm::readInputs ( const std::string& a_pp_str /*!< Parmparse string */)
{
    amrex::ParmParse pp(a_pp_str);
    pp.query("nstrain", nstrain);
    AMREX_ASSERT(nstrain <= 2);
    pp.query("reinfect_prob", reinfect_prob);

    amrex::Vector<amrex::Real> t_p_trans(nstrain);
    amrex::Vector<amrex::Real> t_p_asymp(nstrain);
    amrex::Vector<amrex::Real> t_reduced_inf(nstrain);

    // set correct default
    for (int i = 0; i < nstrain; i++) {
        t_p_trans[i] = p_trans[i];
        t_p_asymp[i] = p_asymp[i];
        t_reduced_inf[i] = reduced_inf[i];
    }

    pp.queryarr("p_trans", t_p_trans, 0, nstrain);
    pp.queryarr("p_asymp", t_p_asymp, 0, nstrain);
    pp.queryarr("reduced_inf", t_reduced_inf, 0, nstrain);

    for (int i = 0; i < nstrain; ++i) {
        p_trans[i] = t_p_trans[i];
        p_asymp[i] = t_p_asymp[i];
        reduced_inf[i] = t_reduced_inf[i];
    }

    pp.query("vac_eff", vac_eff);

    pp.query("latent_length_mean", latent_length_mean);
    pp.query("infectious_length_mean", infectious_length_mean);
    pp.query("incubation_length_mean", incubation_length_mean);

    pp.query("latent_length_std", latent_length_std);
    pp.query("infectious_length_std", infectious_length_std);
    pp.query("incubation_length_std", incubation_length_std);

    pp.query("immune_length_mean", immune_length_mean);
    pp.query("immune_length_std", immune_length_std);

    amrex::Vector<amrex::Real> t_hosp(AgeGroups_Hosp::total);
    for (int i = 0; i < AgeGroups_Hosp::total; i++) {
        t_hosp[i] = m_t_hosp[i];
    }
    pp.queryarr("hospitalization_days", t_hosp, 0, AgeGroups_Hosp::total);
    for (int i = 0; i < AgeGroups_Hosp::total; i++) {
        m_t_hosp[i] = t_hosp[i];
        if (t_hosp[i] > m_t_hosp_offset) {
            m_t_hosp_offset = t_hosp[i] + 1;
        }
    }

    amrex::Vector<amrex::Real> CHR(AgeGroups::total);
    amrex::Vector<amrex::Real> CIC(AgeGroups::total);
    amrex::Vector<amrex::Real> CVE(AgeGroups::total);
    amrex::Vector<amrex::Real> hospCVF(AgeGroups::total);
    amrex::Vector<amrex::Real> icuCVF(AgeGroups::total);
    amrex::Vector<amrex::Real> ventCVF(AgeGroups::total);
    for (int i = 0; i < AgeGroups::total; i++) {
        CHR[i] = m_CHR[i];
        CIC[i] = m_CIC[i];
        CVE[i] = m_CVE[i];
        hospCVF[i] = m_HospToDeath[0][i];
        icuCVF[i] = m_HospToDeath[1][i];
        ventCVF[i] = m_HospToDeath[2][i];
    }
    pp.queryarr("CHR", CHR, 0, AgeGroups::total);
    pp.queryarr("CIC", CHR, 0, AgeGroups::total);
    pp.queryarr("CVE", CHR, 0, AgeGroups::total);
    pp.queryarr("hospCVF", hospCVF, 0, AgeGroups::total);
    pp.queryarr("icuCVF", icuCVF, 0, AgeGroups::total);
    pp.queryarr("ventCVF", ventCVF, 0, AgeGroups::total);
    for (int i = 0; i < AgeGroups::total; i++) {
        m_CHR[i] = CHR[i];
        m_CIC[i] = CIC[i];
        m_HospToDeath[0][i] = hospCVF[i];
        m_HospToDeath[1][i] = icuCVF[i];
        m_HospToDeath[2][i] = ventCVF[i];
    }
}


/*! \brief Initialize disease parameters

    Compute transmission probabilities for various situations based on disease
    attributes.
*/
void DiseaseParm::Initialize ()
{
    xmit_comm[AgeGroups::u5] = .0000125_rt*pCO;
    xmit_comm[AgeGroups::a5to17] = .0000375_rt*pCO;
    xmit_comm[AgeGroups::a18to29] = .00010_rt*pCO;
    xmit_comm[AgeGroups::a30to49] = .00010_rt*pCO;
    xmit_comm[AgeGroups::a50to64] = .00010_rt*pCO;
    xmit_comm[AgeGroups::o65] = .00015_rt*pCO;

    xmit_hood[AgeGroups::u5] = .00005_rt*pNH;
    xmit_hood[AgeGroups::a5to17] = .00015_rt*pNH;
    xmit_hood[AgeGroups::a18to29] = .00040_rt*pNH;
    xmit_hood[AgeGroups::a30to49] = .00040_rt*pNH;
    xmit_hood[AgeGroups::a50to64] = .00040_rt*pNH;
    xmit_hood[AgeGroups::o65] = .00060_rt*pNH;

    xmit_nc_adult[AgeGroups::u5] = .08_rt*pHC;
    xmit_nc_adult[AgeGroups::a5to17] = .08_rt*pHC;
    xmit_nc_adult[AgeGroups::a18to29] = .1_rt*pHC;
    xmit_nc_adult[AgeGroups::a30to49] = .1_rt*pHC;
    xmit_nc_adult[AgeGroups::a50to64] = .1_rt*pHC;
    xmit_nc_adult[AgeGroups::o65] = .1_rt*pHC;

    xmit_nc_child[AgeGroups::u5] = .15_rt*pHC;
    xmit_nc_child[AgeGroups::a5to17] = .15_rt*pHC;
    xmit_nc_child[AgeGroups::a18to29] = .08_rt*pHC;
    xmit_nc_child[AgeGroups::a30to49] = .08_rt*pHC;
    xmit_nc_child[AgeGroups::a50to64] = .08_rt*pHC;
    xmit_nc_child[AgeGroups::o65] = .08_rt*pHC;

    xmit_work = 0.115_rt*pWO;

    // Optimistic scenario: 50% reduction in external child contacts during school dismissal
    //   or remote learning, and no change in household contacts
    Child_compliance=0.5_rt; Child_HH_closure=1.0_rt;
    // Pessimistic scenario: 30% reduction in external child contacts during school dismissal
    //   or remote learning, and 2x increase in household contacts
    //  sch_compliance=0.3; sch_effect=2.0;

    /*
      Double household contact rate involving children, and reduce
      other child-related contacts (neighborhood cluster, neigborhood,
      and community) by the compliance rate, Child_compliance
    */
    for (int i = 0; i < AgeGroups::total; i++) {
        xmit_child_SC[i] = xmit_child[i] * Child_HH_closure;
        xmit_nc_child_SC[i] = xmit_nc_child[i] * (1.0_rt - Child_compliance);
    }
    for (int i = 0; i < 2; i++) {
        xmit_adult_SC[i] = xmit_adult[i] * Child_HH_closure;
        xmit_nc_adult_SC[i] = xmit_nc_adult[i] * (1.0_rt - Child_compliance);
        xmit_comm_SC[i] = xmit_comm[i] * (1.0_rt - Child_compliance);
        xmit_hood_SC[i] = xmit_hood[i] * (1.0_rt - Child_compliance);
    }
    for (int i = 2; i < AgeGroups::total; i++) {
        xmit_adult_SC[i] = xmit_adult[i];
        xmit_nc_adult_SC[i] = xmit_nc_adult[i];    // Adult-only contacts remain unchanged
        xmit_comm_SC[i] = xmit_comm[i];
        xmit_hood_SC[i] = xmit_hood[i];
    }

    // Multiply contact rates by transmission probability given contact
    xmit_work *= p_trans[0];

    for (int i = 0; i < AgeGroups::total; i++) {
        xmit_comm[i] *= p_trans[0];
        xmit_hood[i] *= p_trans[0];
        xmit_nc_adult[i] *= p_trans[0];
        xmit_nc_child[i] *= p_trans[0];
        xmit_adult[i] *= p_trans[0];
        xmit_child[i] *= p_trans[0];
    }

    xmit_college *= p_trans[0];
    xmit_high_school *= p_trans[0];
    xmit_middle_school *= p_trans[0];
    xmit_elem_school *= p_trans[0];
    xmit_daycare *= p_trans[0];
    xmit_playgroup *= p_trans[0];
    xmit_high_school_c2a *= p_trans[0];
    xmit_middle_school_c2a *= p_trans[0];
    xmit_elem_school_c2a *= p_trans[0];
    xmit_daycare_c2a *= p_trans[0];
    xmit_playgroup_c2a *= p_trans[0];
    xmit_high_school_a2c *= p_trans[0];
    xmit_middle_school_a2c *= p_trans[0];
    xmit_elem_school_a2c *= p_trans[0];
    xmit_daycare_a2c *= p_trans[0];
    xmit_playgroup_a2c *= p_trans[0];

    for (int i = 1; i < AgeGroups::total; i++) {
        xmit_child_SC[i] *= p_trans[0];
        xmit_adult_SC[i] *= p_trans[0];
        xmit_nc_child_SC[i] *= p_trans[0];
        xmit_nc_adult_SC[i] *= p_trans[0];
        xmit_comm_SC[i] *= p_trans[0];
        xmit_hood_SC[i] *= p_trans[0];
    }

    infect = 1.0_rt;
}

/*! \brief Print disease parameters */
void DiseaseParm::printMatrix () {
    amrex::Print() << "xmit_comm: " << " ";
    for (int i = 0; i < AgeGroups::total; ++i) {
        amrex::Print() << xmit_comm[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_hood: " <<  " ";
    for (int i = 0; i < AgeGroups::total; ++i) {
        amrex::Print() << xmit_hood[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_nc_adult: " << " ";
    for (int i = 0; i < AgeGroups::total; ++i) {
        amrex::Print() << xmit_nc_adult[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_nc_child: " << " ";
    for (int i = 0; i < AgeGroups::total; ++i) {
        amrex::Print() << xmit_nc_child[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_work: " << " ";
    amrex::Print() << xmit_work << "\n";

    amrex::Print() << "xmit_child_SC: " << " ";
    for (int i = 0; i < AgeGroups::total; ++i) {
        amrex::Print() << xmit_child_SC[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_nc_child_SC: " << " ";
    for (int i = 0; i < AgeGroups::total; ++i) {
        amrex::Print() << xmit_nc_child_SC[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_adult_SC: " << " ";
    for (int i = 0; i < 2; ++i) {
        amrex::Print() << xmit_adult_SC[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_nc_adult_SC: " << " ";
    for (int i = 0; i < 2; ++i) {
        amrex::Print() << xmit_nc_adult_SC[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_hood_SC: " << " ";
    for (int i = 0; i < 2; ++i) {
        amrex::Print() << xmit_hood_SC[i] << " ";
    }
    amrex::Print() << "\n";

    amrex::Print() << "xmit_comm_SC: " << " ";
    for (int i = 0; i < 2; ++i) {
        amrex::Print() << xmit_comm_SC[i] << " ";
    }
    amrex::Print() << "\n";
}
