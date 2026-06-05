/*! @file HospitalData.H
    \brief #HospitalData class definition

    Encapsulates the optional "real hospital placement" feature of the medical-workers /
    hospital-capacity model: parsing the HHS data file, depositing the residential population,
    writing the per-community staffed-bed supply, and (for tract-level data) building the
    per-community hospital assignment used to route patients and retarget medical-worker commute.

    All of this implementation -- including its CUDA device kernels and STL parsing -- lives in
    HospitalData.cpp. AgentContainer owns a HospitalData through a std::unique_ptr and forward
    declaration only, so none of it is compiled into the AgentContainer translation unit.
*/

#ifndef HOSPITALDATA_H_
#define HOSPITALDATA_H_

#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>

#include <string>

class AgentContainer;
struct DemographicData;

/*! \brief Bed-supply and patient-routing data for the medical-workers / hospital-capacity model. */
class HospitalData {
  public:
    HospitalData () = default;

    /*! \brief Destructor (defined out-of-line in HospitalData.cpp so that owners holding a
     *  std::unique_ptr<HospitalData> need not instantiate the iMultiFab member's destructor). */
    ~HospitalData ();

    /*! \brief Records the fixed per-community staffed-bed supply (into a_hosp_data, component
     *  HospMod::bed_supply) and, for tract-level data, the per-community hospital assignment used to
     *  route patients. By default the supply is the residential population times the per-capita bed
     *  density (a_beds_per_1000). With a_use_hhs_data (census only) it is set from real hospital data:
     *  county-level data apportions each county's beds to its communities by population; tract-level
     *  data places beds at hospital tracts and routes each community's patients to its nearest
     *  hospital tract (the analog of work_i/work_j).
     *
     *  Must be called once before the time loop, when agents are at their home communities, so the
     *  population deposited per community is the residential population. */
    void initialize (AgentContainer& a_pc,
                     amrex::MultiFab& a_hosp_data,
                     const amrex::iMultiFab* a_fips_mf,
                     const DemographicData* a_demo,
                     amrex::Real a_beds_per_1000,
                     bool a_use_hhs_data,
                     const std::string& a_hospital_data_file);

    /*! \brief Tract-level routing: send each hospitalized agent to its assigned (nearest) hospital
     *  community -- the work_i/work_j analog -- using the per-community assignment built at init. */
    void rerouteHospitalized (AgentContainer& a_pc) const;

    /*! \brief Whether hospitalized patients are routed to a nearest hospital community (tract-level
     *  data). When false, patients are treated in their home community (the default). */
    bool tractRouting () const { return m_tract_routing; }

  private:
    /*! Per-community assigned hospital community (component 0,1 = hospital i,j). Built at init for
     *  tract-level routing; agents are sent here on hospitalization, the same way work_i/work_j sends
     *  them to work. */
    amrex::iMultiFab m_assignment;

    bool m_tract_routing = false;
};

#endif
