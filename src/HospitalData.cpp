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
#include <map>
#include <sstream>
#include <string>
#include <utility>

using namespace amrex;
using namespace ExaEpi::Utils;

namespace {

/*! Bed supply and assigned (nearest) hospital tract for one census tract. */
struct TractHospInfo {
    Real beds = 0.0_rt;  /*!< staffed beds at this tract's hospital (0 if none) */
    int hosp_fips = -1;  /*!< FIPS of the tract this tract's patients are routed to */
    int hosp_tract = -1; /*!< census tract this tract's patients are routed to */
};

/*! Returns "tract" or "county" by scanning the data-file header for "level=tract". */
std::string readHospitalDataLevel (const std::string& a_fname) {
    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(a_fname, fileCharPtr);
    std::string s(fileCharPtr.dataPtr());
    return (s.find("level=tract") != std::string::npos) ? std::string("tract") : std::string("county");
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

        // per-unit bed supply and assigned hospital cell = first community cell of the hospital unit
        Gpu::HostVector<Real> unit_beds_h(Nunit, 0.0_rt);
        Gpu::HostVector<int> unit_hi_h(Nunit, 0), unit_hj_h(Nunit, 0);
        for (int u = 0; u < Nunit; ++u) {
            int u_h = u; // default: self-route (tracts with no data route to themselves)
            auto it = tract_data.find({a_demo->FIPS[u], a_demo->Tract[u]});
            if (it != tract_data.end()) {
                unit_beds_h[u] = it->second.beds;
                auto jt = tract2unit.find({it->second.hosp_fips, it->second.hosp_tract});
                if (jt != tract2unit.end()) { u_h = jt->second; }
            }
            IntVect hiv = domain.atOffset(static_cast<Long>(a_demo->Start[u_h]));
            unit_hi_h[u] = hiv[0];
            unit_hj_h[u] = hiv[1];
        }

        Gpu::HostVector<int> comm2unit_h(Ncommunity, 0);
        for (int u = 0; u < Nunit; ++u) {
            for (int c = a_demo->Start[u]; c < a_demo->Start[u + 1]; ++c) {
                if ((c >= 0) && (c < Ncommunity)) { comm2unit_h[c] = u; }
            }
        }

        Gpu::DeviceVector<Real> unit_beds_d(Nunit);
        Gpu::DeviceVector<int> unit_hi_d(Nunit), unit_hj_d(Nunit), comm2unit_d(Ncommunity);
        Gpu::copy(Gpu::hostToDevice, unit_beds_h.begin(), unit_beds_h.end(), unit_beds_d.begin());
        Gpu::copy(Gpu::hostToDevice, unit_hi_h.begin(), unit_hi_h.end(), unit_hi_d.begin());
        Gpu::copy(Gpu::hostToDevice, unit_hj_h.begin(), unit_hj_h.end(), unit_hj_d.begin());
        Gpu::copy(Gpu::hostToDevice, comm2unit_h.begin(), comm2unit_h.end(), comm2unit_d.begin());
        const Real* ub = unit_beds_d.data();
        const int* uhi = unit_hi_d.data();
        const int* uhj = unit_hj_d.data();
        const int* c2u = comm2unit_d.data();
        const int* start_d = a_demo->Start_d.data();

        m_assignment.define(a_hosp_data.boxArray(), a_hosp_data.DistributionMap(), 2, 0);
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
                } else {
                    hd(i, j, k, HospMod::bed_supply) = 0.0_rt;
                    asg(i, j, k, 0) = i;
                    asg(i, j, k, 1) = j;
                }
            });
        }
        Gpu::synchronize();
        m_tract_routing = true;

        // Medical workers staff the nearest hospital: retarget each med_sca worker's work_i/work_j
        // to its assigned hospital community so they commute to (and are counted for capacity at)
        // the real hospital, aligning staff with the beds and routed patients. The worker-flow data
        // only resolves home/work to census tracts, so the community-within-tract assignment is free
        // to set.
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
                        }
                    }
                });
            }
        }
        Gpu::synchronize();

        amrex::Print() << "Hospital bed supply + patient routing set from tract-level HHS data (" << tract_data.size()
                       << " tracts): " << a_hospital_data_file << "\n";
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

void HospitalData::rerouteHospitalized (AgentContainer& a_pc) const {
    BL_PROFILE("HospitalData::rerouteHospitalized");
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
                if (inHospital(ip, ptd)) {
                    hosp_i_ptr[ip] = assign(home_i_ptr[ip], home_j_ptr[ip], 0, 0);
                    hosp_j_ptr[ip] = assign(home_i_ptr[ip], home_j_ptr[ip], 0, 1);
                }
            });
        }
    }
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

void AgentContainer::rerouteHospitalizedToHospital () {
    if (m_hosp_obj && m_hosp_obj->tractRouting()) { m_hosp_obj->rerouteHospitalized(*this); }
}
