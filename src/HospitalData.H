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

#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#include <string>
#include <vector>

class AgentContainer;
struct DemographicData;

/*! \brief Bed-supply and patient-routing data for the medical-workers / hospital-capacity model. */
class HospitalData {
  public:
    HospitalData() = default;

    /*! \brief Destructor (defined out-of-line in HospitalData.cpp so that owners holding a
     *  std::unique_ptr<HospitalData> need not instantiate the iMultiFab member's destructor). */
    ~HospitalData();

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
    void initialize(AgentContainer& a_pc, amrex::MultiFab& a_hosp_data, const amrex::iMultiFab* a_fips_mf,
                    const DemographicData* a_demo, amrex::Real a_beds_per_1000, bool a_use_hhs_data,
                    const std::string& a_hospital_data_file);

    /*! \brief Tract-level routing: send each hospitalized agent to its assigned (nearest) hospital
     *  community -- the work_i/work_j analog -- using the per-community assignment built at init. When
     *  patient transfer is active, a just-admitted agent whose nearest hospital is over capacity is
     *  instead routed to the lowest-load hospital in that hospital's county (a_hosp_data supplies the
     *  current per-hospital load, lagged one day). a_iter is the current day, used only to label the
     *  optional transfer log. */
    void rerouteHospitalized(AgentContainer& a_pc, const amrex::MultiFab& a_hosp_data, int a_iter);

    /*! \brief Whether hospitalized patients are routed to a nearest hospital community (tract-level
     *  data). When false, patients are treated in their home community (the default). */
    bool tractRouting () const { return m_tract_routing; }

    // The next two are internal helpers of rerouteHospitalized, public only because CUDA forbids an
    // extended __device__ lambda inside a private or protected member function.

    /*! For each hospital's county, find the lowest-load hospital and record, per hospital, the cell of
     *  the hospital a just-admitted patient is routed to: the nearest hospital itself if it is within
     *  capacity (load <= 1) or its county has a single hospital, else the lowest-load county peer.
     *  Fills m_transfer_i/j_d and the per-hospital target index m_transfer_target from the current
     *  per-hospital load in a_hosp_data (HospMod::load), and resets the per-day transfer counter. */
    void computeTransferTargets (const amrex::MultiFab& a_hosp_data);

    /*! Append the current day's patient transfers to the transfer log (one line per source hospital
     *  that sent patients away): day, from FIPS/tract, to FIPS/tract, and the patient count. Gathers
     *  the per-source counts across ranks; writes on the I/O rank only. */
    void writeTransferLog (int a_iter);

  private:
    /*! Per-community assigned hospital community (component 0,1 = hospital i,j; component 2 = the
     *  global index of that hospital, or -1 for communities with no hospital). Built at init for
     *  tract-level routing; agents are sent here on hospitalization, the same way work_i/work_j sends
     *  them to work. */
    amrex::iMultiFab m_assignment;

    bool m_tract_routing = false;

    /*! Whether to transfer a just-admitted patient from an over-capacity nearest hospital to the
     *  lowest-load hospital in the same county (no transfer if that county has a single hospital). */
    bool m_patient_transfer = false;

    /*! Global hospital tables, identical on every rank (built from the global demographic data at
     *  init). A "hospital" is a tract with a positive bed supply. */
    int m_nhosp = 0;
    amrex::Gpu::DeviceVector<int> m_hosp_i_d;        /*!< hospital cell i, per hospital */
    amrex::Gpu::DeviceVector<int> m_hosp_j_d;        /*!< hospital cell j, per hospital */
    std::vector<int> m_hosp_i_h, m_hosp_j_h;         /*!< host copies of the hospital cells */
    std::vector<int> m_hosp_county;                  /*!< county FIPS, per hospital */
    std::vector<int> m_hosp_tract;                   /*!< census tract, per hospital (for the transfer log) */
    std::vector<std::vector<int>> m_county_hosps;    /*!< hospital indices grouped by county */
    std::vector<int> m_hosp_county_slot;             /*!< index into m_county_hosps, per hospital */

    /*! Per-hospital routing target cell for the current day (filled by computeTransferTargets). */
    amrex::Gpu::DeviceVector<int> m_transfer_i_d;
    amrex::Gpu::DeviceVector<int> m_transfer_j_d;

    /*! Optional plain-text transfer log: per day, the patients moved from each over-capacity hospital
     *  to its lowest-load county peer (set hospital_model.transfer_output_file to enable). */
    bool m_write_transfers = false;
    std::string m_transfer_file;
    std::vector<int> m_transfer_target;              /*!< per-hospital routing target index, current day */
    amrex::Gpu::DeviceVector<int> m_transfer_count_d; /*!< per-source-hospital transfer count, current day */
};

#endif
