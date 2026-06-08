/*! @file HospitalData.cpp
    \brief #HospitalData implementation, plus the thin AgentContainer glue that drives it.

    This translation unit holds all of the medical-workers hospital-data device code and STL parsing.
    It is kept out of AgentContainer.cpp (and the AgentContainer/HospitalModel headers), which is why
    AgentContainer refers to HospitalData only through a forward declaration and a std::unique_ptr.
*/

#include "HospitalData.H"

#include "AgentContainer.H"
#include "AgentDefinitions.H"
#include "DemographicData.H"
#include "HospitalModel.H"
#include "NAICS.H"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace amrex;
using namespace ExaEpi::Utils;

namespace {

/*! Bed supply and assigned (nearest) hospital tract for one census tract. */
struct TractHospInfo {
    Real beds = 0.0_rt;  /*!< staffed beds at this tract's hospital (0 if none) */
    int hosp_fips = -1;  /*!< FIPS of the tract this tract's patients are routed to */
    int hosp_tract = -1; /*!< census tract this tract's patients are routed to */
};

/*! Returns "county" only if the data-file header marks "level=county"; otherwise
 *  defaults to "tract". Tract-level placement (real hospital locations + patient
 *  routing) is the default when HHS hospital data is used; county-level
 *  apportionment is opt-in via an explicit "level=county" header. */
std::string readHospitalDataLevel (const std::string& a_fname) {
    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(a_fname, fileCharPtr);
    std::string s(fileCharPtr.dataPtr());
    return (s.find("level=county") != std::string::npos) ? std::string("county") : std::string("tract");
}

/*! Reads a per-county hospital bed-supply data file (built by utilities/build_hospital_data.py):
 *  '#'-comment lines, then a record count, then rows "FIPS beds icu n_hospitals". Returns a map
 *  from integer county FIPS to the staffed bed supply. */
std::map<int, Real> readHospitalBedData (const std::string& a_fname) {
    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(a_fname, fileCharPtr);
    std::string fileStr(fileCharPtr.dataPtr());
    std::istringstream is(fileStr, std::istringstream::in);

    std::map<int, Real> beds;
    std::string line;
    bool have_count = false;
    while (std::getline(is, line)) {
        const auto p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos) { continue; } // blank
        if (line[p] == '#') { continue; }         // comment
        std::istringstream ls(line);
        if (!have_count) {
            have_count = true;                    // first non-comment line is the record count
            continue;
        }
        int fips = -1;
        Real b = 0.0_rt;
        if (ls >> fips >> b) { beds[fips] += b; }
    }
    return beds;
}

/*! Reads a per-tract hospital data file: rows "FIPS TRACT beds icu n hosp_FIPS hosp_TRACT".
 *  Returns a map from (FIPS, tract) to its bed supply and assigned (nearest) hospital tract. */
std::map<std::pair<int, int>, TractHospInfo> readHospitalTractData (const std::string& a_fname) {
    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(a_fname, fileCharPtr);
    std::string fileStr(fileCharPtr.dataPtr());
    std::istringstream is(fileStr, std::istringstream::in);

    std::map<std::pair<int, int>, TractHospInfo> data;
    std::string line;
    bool have_count = false;
    while (std::getline(is, line)) {
        const auto p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos) { continue; }
        if (line[p] == '#') { continue; }
        std::istringstream ls(line);
        if (!have_count) {
            have_count = true;
            continue;
        }
        int fips = -1, tract = -1, icu = 0, nh = 0, hF = -1, hT = -1;
        Real beds = 0.0_rt;
        if (ls >> fips >> tract >> beds >> icu >> nh >> hF >> hT) { data[{fips, tract}] = TractHospInfo{beds, hF, hT}; }
    }
    return data;
}

} // namespace

HospitalData::~HospitalData() = default;

void HospitalData::initialize (AgentContainer& a_pc, MultiFab& a_hosp_data, const iMultiFab* a_fips_mf,
                               const DemographicData* a_demo, Real a_beds_per_1000, bool a_use_hhs_data,
                               const std::string& a_hospital_data_file) {
    BL_PROFILE("HospitalData::initialize");

    const int lev = 0;
    const auto& geom = a_pc.Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const auto domain = geom.Domain();

    MultiFab population(a_hosp_data.boxArray(), a_hosp_data.DistributionMap(), 1, 0);
    population.setVal(0.0);
    ParticleToMesh(
            a_pc, population, lev,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd, int i,
                                 Array4<Real> const& mf_arr) {
                auto p = ptd.m_aos[i];
                auto iv = getParticleCell(p, plo, dxi, domain);
                Gpu::Atomic::AddNoRet(&mf_arr(iv, 0), 1.0_rt);
            },
            false);

    if (!(a_use_hhs_data && (a_fips_mf != nullptr) && (a_demo != nullptr))) {
        // uniform per-capita bed density: bed_supply = (staffed_beds_per_1000/1000) x population.
        const Real beds_per_capita = a_beds_per_1000 / 1000.0_rt;
        for (MFIter mfi(a_hosp_data); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto pop = population.const_array(mfi);
            auto hd = a_hosp_data.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                hd(i, j, k, HospMod::bed_supply) = beds_per_capita * pop(i, j, k, 0);
            });
        }
        return;
    }

    if (readHospitalDataLevel(a_hospital_data_file) == "tract") {
        // Place beds at hospital tracts and route each community's patients to its nearest hospital
        // tract. Everything is deterministic from the global demographic data and the domain raster
        // (community number == domain.index(cell)); no inter-rank communication is needed.
        auto tract_data = readHospitalTractData(a_hospital_data_file);
        const int Nunit = a_demo->Nunit;
        const int Ncommunity = a_demo->Ncommunity;

        std::map<std::pair<int, int>, int> tract2unit;
        for (int u = 0; u < Nunit; ++u) {
            tract2unit[{a_demo->FIPS[u], a_demo->Tract[u]}] = u;
        }

        // per-unit bed supply, assigned (nearest) hospital cell = first community cell of the hospital
        // unit, and the unit of that assigned hospital (for the per-hospital routing index below)
        Gpu::HostVector<Real> unit_beds_h(Nunit, 0.0_rt);
        Gpu::HostVector<int> unit_hi_h(Nunit, 0), unit_hj_h(Nunit, 0), unit_uh_h(Nunit, 0);
        for (int u = 0; u < Nunit; ++u) {
            int u_h = u; // default: self-route (tracts with no data route to themselves)
            auto it = tract_data.find({a_demo->FIPS[u], a_demo->Tract[u]});
            if (it != tract_data.end()) {
                unit_beds_h[u] = it->second.beds;
                auto jt = tract2unit.find({it->second.hosp_fips, it->second.hosp_tract});
                if (jt != tract2unit.end()) { u_h = jt->second; }
            }
            unit_uh_h[u] = u_h;
            IntVect hiv = domain.atOffset(static_cast<Long>(a_demo->Start[u_h]));
            unit_hi_h[u] = hiv[0];
            unit_hj_h[u] = hiv[1];
        }

        // Global hospital tables: index every tract with a positive bed supply, record its cell and
        // county, and group the hospitals by county for the patient-transfer decision. Built from the
        // global demographic data, so every rank holds identical tables. unit_hidx_h[u] is the index
        // of the hospital a unit's patients are routed to by default (-1 if none).
        std::vector<int> unit2hosp(Nunit, -1);
        m_nhosp = 0;
        m_hosp_i_h.clear();
        m_hosp_j_h.clear();
        m_hosp_county.clear();
        m_hosp_tract.clear();
        for (int u = 0; u < Nunit; ++u) {
            if (unit_beds_h[u] > 0.0_rt) {
                unit2hosp[u] = m_nhosp++;
                m_hosp_i_h.push_back(unit_hi_h[u]);
                m_hosp_j_h.push_back(unit_hj_h[u]);
                m_hosp_county.push_back(a_demo->FIPS[u]);
                m_hosp_tract.push_back(a_demo->Tract[u]);
            }
        }
        m_county_hosps.clear();
        m_hosp_county_slot.assign(m_nhosp, -1);
        std::map<int, int> county2slot;
        for (int h = 0; h < m_nhosp; ++h) {
            auto res = county2slot.emplace(m_hosp_county[h], static_cast<int>(m_county_hosps.size()));
            if (res.second) { m_county_hosps.emplace_back(); }
            m_hosp_county_slot[h] = res.first->second;
            m_county_hosps[res.first->second].push_back(h);
        }
        m_hosp_i_d.resize(m_nhosp);
        m_hosp_j_d.resize(m_nhosp);
        Gpu::copy(Gpu::hostToDevice, m_hosp_i_h.begin(), m_hosp_i_h.end(), m_hosp_i_d.begin());
        Gpu::copy(Gpu::hostToDevice, m_hosp_j_h.begin(), m_hosp_j_h.end(), m_hosp_j_d.begin());
        m_transfer_i_d.resize(m_nhosp);
        m_transfer_j_d.resize(m_nhosp);
        m_transfer_count_d.resize(m_nhosp);
        m_transfer_target.assign(m_nhosp, -1);

        Gpu::HostVector<int> unit_hidx_h(Nunit, -1);
        for (int u = 0; u < Nunit; ++u) { unit_hidx_h[u] = unit2hosp[unit_uh_h[u]]; }

        Gpu::HostVector<int> comm2unit_h(Ncommunity, 0);
        for (int u = 0; u < Nunit; ++u) {
            for (int c = a_demo->Start[u]; c < a_demo->Start[u + 1]; ++c) {
                if ((c >= 0) && (c < Ncommunity)) { comm2unit_h[c] = u; }
            }
        }

        Gpu::DeviceVector<Real> unit_beds_d(Nunit);
        Gpu::DeviceVector<int> unit_hi_d(Nunit), unit_hj_d(Nunit), unit_hidx_d(Nunit), comm2unit_d(Ncommunity);
        Gpu::copy(Gpu::hostToDevice, unit_beds_h.begin(), unit_beds_h.end(), unit_beds_d.begin());
        Gpu::copy(Gpu::hostToDevice, unit_hi_h.begin(), unit_hi_h.end(), unit_hi_d.begin());
        Gpu::copy(Gpu::hostToDevice, unit_hj_h.begin(), unit_hj_h.end(), unit_hj_d.begin());
        Gpu::copy(Gpu::hostToDevice, unit_hidx_h.begin(), unit_hidx_h.end(), unit_hidx_d.begin());
        Gpu::copy(Gpu::hostToDevice, comm2unit_h.begin(), comm2unit_h.end(), comm2unit_d.begin());
        const Real* ub = unit_beds_d.data();
        const int* uhi = unit_hi_d.data();
        const int* uhj = unit_hj_d.data();
        const int* uhidx = unit_hidx_d.data();
        const int* c2u = comm2unit_d.data();
        const int* start_d = a_demo->Start_d.data();

        m_assignment.define(a_hosp_data.boxArray(), a_hosp_data.DistributionMap(), 3, 0);
        const Box dom = domain;
        for (MFIter mfi(a_hosp_data); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            auto hd = a_hosp_data.array(mfi);
            auto asg = m_assignment.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                IntVect iv{AMREX_D_DECL(i, j, k)};
                const int community = static_cast<int>(dom.index(iv));
                if ((community >= 0) && (community < Ncommunity)) {
                    const int unit = c2u[community];
                    hd(i, j, k, HospMod::bed_supply) = (community == start_d[unit]) ? ub[unit] : 0.0_rt;
                    asg(i, j, k, 0) = uhi[unit];
                    asg(i, j, k, 1) = uhj[unit];
                    asg(i, j, k, 2) = uhidx[unit];
                } else {
                    hd(i, j, k, HospMod::bed_supply) = 0.0_rt;
                    asg(i, j, k, 0) = i;
                    asg(i, j, k, 1) = j;
                    asg(i, j, k, 2) = -1;
                }
            });
        }
        Gpu::synchronize();
        m_tract_routing = true;

        // Patient transfer (on by default): a just-admitted patient whose nearest hospital is over
        // capacity is sent to the lowest-load hospital in that hospital's county. Inert until a
        // hospital exceeds capacity, so it leaves the ample-bed verification runs unchanged.
        m_patient_transfer = true;
        {
            ParmParse pp("hospital_model");
            pp.query("patient_transfer", m_patient_transfer);
            pp.query("transfer_output_file", m_transfer_file); // empty => no transfer log
        }
        m_write_transfers = m_patient_transfer && !m_transfer_file.empty();

        // Medical workers staff the nearest hospital: retarget each med_sca worker's work_i/work_j
        // to its assigned hospital community so they commute to (and are counted for capacity at)
        // the real hospital, aligning staff with the beds and routed patients. The worker-flow data
        // only resolves home/work to census tracts, so the community-within-tract assignment is free
        // to set. While retargeting, tally each hospital's medical workforce (uhidx[unit] is the
        // worker's hospital index) so the staff can be split into hospital workgroups below.
        Gpu::DeviceVector<int> mw_count_d(m_nhosp, 0);
        int* mwc = mw_count_d.data();
        for (int alev = 0; alev <= a_pc.finestLevel(); ++alev) {
            auto& plev = a_pc.GetParticles(alev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi = a_pc.MakeMFIter(alev); mfi.isValid(); ++mfi) {
                auto& ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
                const size_t np = ptile.GetArrayOfStructs().numParticles();
                auto& soa = ptile.GetStructOfArrays();
                auto naics_ptr = soa.GetIntData(IntIdx::naics).data();
                auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
                auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
                    if ((naics_ptr[ip] == NAICSCodes::NAICS::med_sca) && (work_i_ptr[ip] >= 0)) {
                        IntVect wiv{AMREX_D_DECL(work_i_ptr[ip], work_j_ptr[ip], 0)};
                        const int c = static_cast<int>(dom.index(wiv));
                        if ((c >= 0) && (c < Ncommunity)) {
                            const int unit = c2u[c];
                            work_i_ptr[ip] = uhi[unit];
                            work_j_ptr[ip] = uhj[unit];
                            const int h = uhidx[unit];
                            if (h >= 0) { Gpu::Atomic::AddNoRet(&mwc[h], 1); }
                        }
                    }
                });
            }
        }
        Gpu::synchronize();

        // Hospital workgroups: split each hospital's medical workforce into groups of ~workgroup_size,
        // the same way the workplace model groups co-workers. Without this every worker carries
        // workgroup 1 (one facility per community), so with tract routing a hospital's entire pooled
        // staff would mix as a single group; here the staff are divided into workgroups so worker-to-
        // worker (d2d) transmission is confined to realistic teams (med/surg nursing units are ~30
        // beds, ~20 frontline staff, comparable to the default workgroup_size). The number of groups
        // is fixed from the hospital's global workforce; each worker draws a group uniformly.
        int wg_size = 20;
        {
            ParmParse pp_agent("agent");
            pp_agent.query("workgroup_size", wg_size);
            ParmParse pp_hosp("hospital_model");
            pp_hosp.query("workgroup_size", wg_size); // hospital-specific override of the workplace size
        }
        if (wg_size < 1) { wg_size = 1; }

        std::vector<int> mw_count_h(m_nhosp, 0);
        Gpu::copy(Gpu::deviceToHost, mw_count_d.begin(), mw_count_d.end(), mw_count_h.begin());
        ParallelDescriptor::ReduceIntSum(mw_count_h.data(), m_nhosp);
        std::vector<int> ngroups_h(m_nhosp, 1);
        Long total_groups = 0, total_mw = 0;
        for (int h = 0; h < m_nhosp; ++h) {
            ngroups_h[h] = std::max(1, static_cast<int>(std::lround(double(mw_count_h[h]) / double(wg_size))));
            total_groups += ngroups_h[h];
            total_mw += mw_count_h[h];
        }
        Gpu::DeviceVector<int> ngroups_d(m_nhosp);
        Gpu::copy(Gpu::hostToDevice, ngroups_h.begin(), ngroups_h.end(), ngroups_d.begin());
        const int* ngrp = ngroups_d.data();

        for (int alev = 0; alev <= a_pc.finestLevel(); ++alev) {
            auto& plev = a_pc.GetParticles(alev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi = a_pc.MakeMFIter(alev); mfi.isValid(); ++mfi) {
                auto& ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
                const size_t np = ptile.GetArrayOfStructs().numParticles();
                auto& soa = ptile.GetStructOfArrays();
                auto naics_ptr = soa.GetIntData(IntIdx::naics).data();
                auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
                auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
                auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
                amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int ip, RandomEngine const& engine) noexcept {
                    if ((naics_ptr[ip] == NAICSCodes::NAICS::med_sca) && (work_i_ptr[ip] >= 0)) {
                        // work cell is now the hospital cell, so c2u maps it to the hospital unit and
                        // uhidx to the hospital's own index
                        IntVect wiv{AMREX_D_DECL(work_i_ptr[ip], work_j_ptr[ip], 0)};
                        const int c = static_cast<int>(dom.index(wiv));
                        if ((c >= 0) && (c < Ncommunity)) {
                            const int h = uhidx[c2u[c]];
                            const int n = (h >= 0) ? ngrp[h] : 1;
                            workgroup_ptr[ip] = 1 + Random_int(n, engine);
                        }
                    }
                });
            }
        }
        Gpu::synchronize();

        amrex::Print() << "Hospital bed supply + patient routing set from tract-level HHS data (" << tract_data.size()
                       << " tracts, " << m_nhosp << " hospitals): " << a_hospital_data_file << "\n";
        amrex::Print() << "  Same-county patient transfer for over-capacity hospitals: "
                       << (m_patient_transfer ? "on" : "off") << "\n";
        amrex::Print() << "  Medical workforce split into hospital workgroups of ~" << wg_size << " staff ("
                       << total_groups << " workgroups for " << total_mw << " medical workers, mean "
                       << (total_groups > 0 ? double(total_mw) / double(total_groups) : 0.0) << " per group)\n";
        if (m_write_transfers) {
            amrex::Print() << "  Logging patient transfers to: " << m_transfer_file << "\n";
            if (ParallelDescriptor::IOProcessor()) {
                std::ofstream f(m_transfer_file, std::ios::out | std::ios::trunc);
                f << "# patient transfers from over-capacity hospitals to the lowest-load hospital in the same county\n";
                f << "# columns: day  from_FIPS  from_tract  to_FIPS  to_tract  n_patients\n";
            }
        }
    } else {
        // County-level: apportion each county's beds to its communities by population,
        // bed_supply(community) = (county beds / county population) x community population.
        auto county_beds = readHospitalBedData(a_hospital_data_file);
        const auto& county_pop = a_demo->CountyPop;

        int max_fips = 0;
        for (const auto& kv : county_beds) {
            max_fips = std::max(max_fips, kv.first);
        }
        for (const auto& kv : county_pop) {
            max_fips = std::max(max_fips, kv.first);
        }
        const int dsize = max_fips + 1;

        Gpu::HostVector<Real> density_h(dsize, 0.0_rt);
        for (const auto& kv : county_beds) {
            auto it = county_pop.find(kv.first);
            if ((it != county_pop.end()) && (it->second > 0)) { density_h[kv.first] = kv.second / Real(it->second); }
        }
        Gpu::DeviceVector<Real> density_d(dsize);
        Gpu::copy(Gpu::hostToDevice, density_h.begin(), density_h.end(), density_d.begin());
        const Real* density_ptr = density_d.data();

        for (MFIter mfi(a_hosp_data); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            auto hd = a_hosp_data.array(mfi);
            const auto pop = population.const_array(mfi);
            const auto fips = a_fips_mf->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                const int f = fips(i, j, k, 0);
                const Real dens = ((f >= 0) && (f < dsize)) ? density_ptr[f] : 0.0_rt;
                hd(i, j, k, HospMod::bed_supply) = dens * pop(i, j, k, 0);
            });
        }
        Gpu::synchronize();
        amrex::Print() << "Hospital bed supply set from county-level HHS data (" << county_beds.size()
                       << " counties): " << a_hospital_data_file << "\n";
    }
}

void HospitalData::computeTransferTargets (const MultiFab& a_hosp_data) {
    BL_PROFILE("HospitalData::computeTransferTargets");
    if (m_nhosp == 0) { return; }

    // Gather the current per-hospital load. Each hospital cell lies in exactly one rank's valid box,
    // so a rank reads the load of the hospitals it owns into a device array (others stay 0) and a
    // global sum-reduction broadcasts the full vector to every rank.
    Gpu::DeviceVector<Real> load_d(m_nhosp, 0.0_rt);
    Real* load_ptr = load_d.data();
    const int* hci = m_hosp_i_d.data();
    const int* hcj = m_hosp_j_d.data();
    const int nhosp = m_nhosp;
    for (MFIter mfi(a_hosp_data); mfi.isValid(); ++mfi) {
        const Box vbx = mfi.validbox();
        const auto arr = a_hosp_data.const_array(mfi);
        amrex::ParallelFor(nhosp, [=] AMREX_GPU_DEVICE (int h) noexcept {
            const IntVect iv{AMREX_D_DECL(hci[h], hcj[h], 0)};
            if (vbx.contains(iv)) { load_ptr[h] = arr(iv, HospMod::load); }
        });
    }
    Gpu::synchronize();

    std::vector<Real> load_h(m_nhosp);
    Gpu::copy(Gpu::deviceToHost, load_d.begin(), load_d.end(), load_h.begin());
    ParallelDescriptor::ReduceRealSum(load_h.data(), m_nhosp);

    // Lowest-load hospital in each county.
    std::vector<int> county_min(m_county_hosps.size(), -1);
    for (size_t s = 0; s < m_county_hosps.size(); ++s) {
        int best = -1;
        for (int h : m_county_hosps[s]) {
            if ((best < 0) || (load_h[h] < load_h[best])) { best = h; }
        }
        county_min[s] = best;
    }

    // Per-hospital routing target: the nearest hospital itself if it is within capacity (load <= 1) or
    // is the only hospital in its county, else the lowest-load hospital in that county.
    std::vector<int> tgt_i(m_nhosp), tgt_j(m_nhosp);
    for (int h = 0; h < m_nhosp; ++h) {
        const int slot = m_hosp_county_slot[h];
        int target = h;
        if ((m_county_hosps[slot].size() > 1) && (load_h[h] > 1.0_rt)) { target = county_min[slot]; }
        m_transfer_target[h] = target;
        tgt_i[h] = m_hosp_i_h[target];
        tgt_j[h] = m_hosp_j_h[target];
    }
    Gpu::copy(Gpu::hostToDevice, tgt_i.begin(), tgt_i.end(), m_transfer_i_d.begin());
    Gpu::copy(Gpu::hostToDevice, tgt_j.begin(), tgt_j.end(), m_transfer_j_d.begin());
    if (m_write_transfers) {
        amrex::ParallelFor(m_nhosp, [c = m_transfer_count_d.data()] AMREX_GPU_DEVICE (int h) noexcept { c[h] = 0; });
        Gpu::synchronize();
    }
}

void HospitalData::rerouteHospitalized (AgentContainer& a_pc, const MultiFab& a_hosp_data, int a_iter) {
    BL_PROFILE("HospitalData::rerouteHospitalized");

    // When patient transfer is active, refresh the per-hospital routing targets from the current
    // (one-day-lagged) load before routing the day's new admissions.
    const bool transfer = m_patient_transfer && (m_nhosp > 0);
    if (transfer) { computeTransferTargets(a_hosp_data); }
    const int* tgt_i = transfer ? m_transfer_i_d.data() : nullptr;
    const int* tgt_j = transfer ? m_transfer_j_d.data() : nullptr;
    const bool record = m_write_transfers && transfer;
    int* tcount = record ? m_transfer_count_d.data() : nullptr;

    for (int alev = 0; alev <= a_pc.finestLevel(); ++alev) {
        auto& plev = a_pc.GetParticles(alev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = a_pc.MakeMFIter(alev); mfi.isValid(); ++mfi) {
            auto& ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            const auto& ptd = ptile.getParticleTileData();
            const size_t np = ptile.GetArrayOfStructs().numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
            auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();
            auto assign = m_assignment.const_array(mfi);
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
                // Route only agents just admitted: assignHospital() sets hosp == home on
                // admission, and the daily step runs this while agents are at home, so the
                // home-indexed assignment lookup is in the current tile. An already-routed
                // agent (hosp != home) has been moved to its hospital cell, whose tile no
                // longer contains its home cell; re-reading the assignment for it would be
                // an out-of-bounds access. Its hospital cell is fixed by its home, so it is
                // simply left in place.
                if (inHospital(ip, ptd)
                    && hosp_i_ptr[ip] == home_i_ptr[ip] && hosp_j_ptr[ip] == home_j_ptr[ip]) {
                    // hospital index of this community's nearest hospital (component 2); with
                    // transfer on, an over-capacity nearest hospital sends the patient to the
                    // lowest-load hospital in its county (precomputed routing target). Falls back
                    // to the static nearest-hospital cell when transfer is off or no hospital is
                    // assigned (h0 < 0).
                    const int h0 = assign(home_i_ptr[ip], home_j_ptr[ip], 0, 2);
                    if (transfer && (h0 >= 0)) {
                        const int near_i = assign(home_i_ptr[ip], home_j_ptr[ip], 0, 0);
                        const int near_j = assign(home_i_ptr[ip], home_j_ptr[ip], 0, 1);
                        hosp_i_ptr[ip] = tgt_i[h0];
                        hosp_j_ptr[ip] = tgt_j[h0];
                        // count the patient if the target differs from the nearest hospital (transferred)
                        if (tcount && ((tgt_i[h0] != near_i) || (tgt_j[h0] != near_j))) {
                            Gpu::Atomic::AddNoRet(&tcount[h0], 1);
                        }
                    } else {
                        hosp_i_ptr[ip] = assign(home_i_ptr[ip], home_j_ptr[ip], 0, 0);
                        hosp_j_ptr[ip] = assign(home_i_ptr[ip], home_j_ptr[ip], 0, 1);
                    }
                }
            });
        }
    }
    if (record) { writeTransferLog(a_iter); }
}

void HospitalData::writeTransferLog (int a_iter) {
    BL_PROFILE("HospitalData::writeTransferLog");
    // Gather the per-source-hospital transfer counts (each source counted on the ranks holding its
    // patients) onto the I/O rank, then append one line per source hospital that sent patients away.
    std::vector<int> count_h(m_nhosp, 0);
    Gpu::copy(Gpu::deviceToHost, m_transfer_count_d.begin(), m_transfer_count_d.end(), count_h.begin());
    ParallelDescriptor::ReduceIntSum(count_h.data(), m_nhosp, ParallelDescriptor::IOProcessorNumber());

    if (!ParallelDescriptor::IOProcessor()) { return; }
    Long day_total = 0;
    std::ofstream f(m_transfer_file, std::ios::out | std::ios::app);
    for (int h = 0; h < m_nhosp; ++h) {
        if (count_h[h] <= 0) { continue; }
        const int to = m_transfer_target[h];
        f << std::setw(6) << a_iter << std::setw(10) << m_hosp_county[h] << std::setw(10) << m_hosp_tract[h]
          << std::setw(10) << m_hosp_county[to] << std::setw(10) << m_hosp_tract[to] << std::setw(12) << count_h[h]
          << "\n";
        day_total += count_h[h];
    }
    if (day_total > 0) { amrex::Print() << "Day " << a_iter << ": " << day_total << " patients transferred\n"; }
}

// ---------------------------------------------------------------------------------------------
// AgentContainer glue. Defined here (not in AgentContainer.cpp) so that the HospitalData type --
// and all of its device code -- stays out of the AgentContainer translation unit. The out-of-line
// destructor is required for the std::unique_ptr<HospitalData> member to use an incomplete type in
// the header.
// ---------------------------------------------------------------------------------------------

AgentContainer::~AgentContainer() = default;

/*! No-op when the medical-workers model is off. */
void AgentContainer::initHospitalCapacityModel (const iMultiFab* a_fips_mf, const DemographicData* a_demo) {
    BL_PROFILE("AgentContainer::initHospitalCapacityModel");
    if (!m_model_medical_workers) { return; }
    if (!m_hosp_obj) { m_hosp_obj = std::make_unique<HospitalData>(); }
    m_hosp_obj->initialize(*this, m_hosp_data, a_fips_mf, a_demo, m_hospital->bedsPerThousand(), m_hospital->useHHSData(),
                           m_hospital->hospitalDataFile());
}

void AgentContainer::rerouteHospitalizedToHospital (int a_iter) {
    if (m_hosp_obj && m_hosp_obj->tractRouting()) { m_hosp_obj->rerouteHospitalized(*this, m_hosp_data, a_iter); }
}

/*! Per-day medical-worker vs other-worker infection counts for disease a_d (rank-summed), for
 *  calibrating the in-hospital transmission parameters against the observed HCW infection risk.
 *  Returns {mw_total, mw_susceptible, mw_newly_infected, ow_total, ow_susceptible, ow_newly_infected}.
 *  A medical worker is an employed agent (work_i >= 0) with NAICS med_sca; an other worker is any
 *  other employed agent. Kept in this translation unit with the rest of the medical-worker device
 *  code. */
std::array<Long, 8> AgentContainer::getMedicalWorkerCounts (const int a_d) {
    BL_PROFILE("AgentContainer::getMedicalWorkerCounts");
    ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum>
            reduce_ops;
    auto r = ParticleReduce<ReduceData<int, int, int, int, int, int, int, int>>(
            *this,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                                 const int i) noexcept -> GpuTuple<int, int, int, int, int, int, int, int> {
                const int naics = ptd.m_idata[IntIdx::naics][i];
                const int work_i = ptd.m_idata[IntIdx::work_i][i];
                const int status = ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][i];
                const bool worker = (work_i >= 0);
                const bool mw = worker && (naics == NAICSCodes::NAICS::med_sca);
                const bool ow = worker && (naics != NAICSCodes::NAICS::med_sca);
                const bool at_risk = (status == Status::never) || (status == Status::susceptible);
                const bool new_inf = isNewlyInfected(i, ptd, a_d);
                const bool dead = (status == Status::dead); // cumulative deaths (status stays dead)
                return {mw ? 1 : 0, (mw && at_risk) ? 1 : 0, (mw && new_inf) ? 1 : 0,
                        ow ? 1 : 0, (ow && at_risk) ? 1 : 0, (ow && new_inf) ? 1 : 0,
                        (mw && dead) ? 1 : 0, (ow && dead) ? 1 : 0};
            },
            reduce_ops);
    std::array<Long, 8> counts;
    counts[0] = static_cast<Long>(amrex::get<0>(r));
    counts[1] = static_cast<Long>(amrex::get<1>(r));
    counts[2] = static_cast<Long>(amrex::get<2>(r));
    counts[3] = static_cast<Long>(amrex::get<3>(r));
    counts[4] = static_cast<Long>(amrex::get<4>(r));
    counts[5] = static_cast<Long>(amrex::get<5>(r));
    counts[6] = static_cast<Long>(amrex::get<6>(r));
    counts[7] = static_cast<Long>(amrex::get<7>(r));
    ParallelDescriptor::ReduceLongSum(&counts[0], 8, ParallelDescriptor::IOProcessorNumber());
    return counts;
}
