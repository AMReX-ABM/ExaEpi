/*! @file UrbanPopData.cpp
    \brief Implementation of #UrbanPopData class
*/

#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_BLassert.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Particles.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include "AgentContainer.H"
#include "UrbanPopData.H"

using namespace amrex;
using namespace UrbanPop;

using std::ifstream;
using std::istringstream;
using std::ostringstream;
using std::runtime_error;
using std::string;
using std::to_string;
using std::unordered_map;
using std::unordered_set;

using ParallelDescriptor::MyProc;
using ParallelDescriptor::NProcs;

template <typename T>
void copyToDeviceAsync (const Vector<T>& h_vec, Gpu::DeviceVector<T>& d_vec) {
    d_vec.resize(0);
    d_vec.resize(h_vec.size());
    Gpu::copyAsync(Gpu::hostToDevice, h_vec.begin(), h_vec.end(), d_vec.begin());
}

bool BlockGroup::readAgents (ifstream& f, Vector<UrbanPopAgent>& agents, amrex::Vector<AgentExtras>& agents_extras,
                             const std::map<int64_t, int>& geoid_to_block_groups, const Vector<BlockGroup>& block_groups) {
    BL_PROFILE("BlockGroup::readAgents");
    string buf;
    num_households = 0;
    num_employed = 0;
    num_students = 0;
    num_educators = 0;
    int start_i = agents.size();
    agents.resize(start_i + home_population);
    agents_extras.resize(start_i + home_population);
    // used for counting up the number of unique households
    unordered_set<int> households;
    f.seekg(file_offset);
    for (int i = start_i; i < agents.size(); i++) {
        auto& agent = agents[i];
        if (!agent.readBinary(f)) {
            Abort("File is corrupted: end of file before read for offset " + to_string(file_offset) + " geoid " +
                  to_string(geoid) + "\n");
        }
        if (agent.id == -1) { Abort("File is corrupted: couldn't read agent p_id at offset " + to_string(file_offset) + "\n"); }
        if (agent.home_geoid != geoid) {
            Abort("File is corrupted: wrong geoid, read " + to_string(agent.home_geoid) + " expected " + to_string(geoid) +
                  " file offset " + to_string(file_offset) + " home pop " + to_string(home_population) + "\n");
        }
        households.insert(agent.household_id);
        agents_extras[i].home_xy = IntVect(x, y);
        auto it = geoid_to_block_groups.find(agent.work_geoid);
        if (it == geoid_to_block_groups.end()) { Abort("Cannot find block group for work location"); }
        auto& work_block_group = block_groups[it->second];
        agents_extras[i].work_xy = IntVect(work_block_group.x, work_block_group.y);
        if (agent.naics != -1) {
            num_employed++;
            agents_extras[i].naics_population = work_block_group.work_populations[agent.naics + 1];
            AMREX_ASSERT(agents_extras[i].naics_population > 0 && agents_extras[i].naics_population < 100000);
            agents_extras[i].work_population = work_block_group.work_populations[0];
            if (agents_extras[i].work_population <= 0 || agents_extras[i].work_population >= 260000) {
                Print() << "work pop " << agents_extras[i].work_population << "\n";
            }
            AMREX_ASSERT(agents_extras[i].work_population > 0 && agents_extras[i].work_population < 260000);
            if (agent.school_id != 0) { num_educators++; }
        } else {
            agents_extras[i].naics_population = 0;
            agents_extras[i].work_population = 0;
            if (agent.naics == -1 && agent.school_id == 0) { AMREX_ASSERT(agent.home_geoid == agent.work_geoid); }
            if (agent.naics == -1 && agent.school_id != 0) { num_students++; }
        }
        agents_extras[i].home_population = home_population;
        // Print() << "Agent " << i << " home " << agents_extras[i].home_xy << " work " << agents_extras[i].work_xy << "\n";
    }
    num_households = households.size();

    return true;
}

bool BlockGroup::read (std::istream& f) {
    BL_PROFILE("BlockGroup::read");
    // Read binary format:
    // - geoid: uint64 (8 bytes)
    // - foff: uint64 (8 bytes)
    // - h_pop: uint32 (4 bytes)
    // - w_pop: uint32 (4 bytes)
    // - naics counts: uint32 * NAICS_COUNT (4 bytes each)
    if (!f.read(reinterpret_cast<char*>(&geoid), sizeof(uint64_t))) { return false; }
    if (!f.read(reinterpret_cast<char*>(&file_offset), sizeof(uint64_t))) { return false; }
    if (!f.read(reinterpret_cast<char*>(&home_population), sizeof(uint32_t))) { return false; }
    uint32_t total_work_pop;
    if (!f.read(reinterpret_cast<char*>(&total_work_pop), sizeof(uint32_t))) { return false; }
    // Read NAICS counts (NAICS_COUNT values)
    work_populations.clear();
    work_populations.push_back(total_work_pop); // First element is total

    for (int i = 0; i < NAICS_COUNT; i++) {
        uint32_t naics_count;
        if (!f.read(reinterpret_cast<char*>(&naics_count), sizeof(uint32_t))) { return false; }
        work_populations.push_back(naics_count);
    }
    AMREX_ASSERT(home_population > 0 || work_populations[0] > 0);
    AMREX_ASSERT(work_populations.size() == NAICS_COUNT + 1);
    return true;
}

static void readBlockGroupsFile (std::ifstream& urbanpop_file, Vector<BlockGroup>& block_groups) {
    BL_PROFILE("readBlockGroupsFile");
    // Each process opens the file separately
    // Read file header
    uint32_t magic_number, version, num_naics, num_geoids, agent_record_size;
    uint64_t num_agents, index_end_offset;
    urbanpop_file.read(reinterpret_cast<char*>(&magic_number), sizeof(uint32_t));
    urbanpop_file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    urbanpop_file.read(reinterpret_cast<char*>(&num_naics), sizeof(uint32_t));
    urbanpop_file.read(reinterpret_cast<char*>(&num_geoids), sizeof(uint32_t));
    urbanpop_file.read(reinterpret_cast<char*>(&num_agents), sizeof(uint64_t));
    urbanpop_file.read(reinterpret_cast<char*>(&agent_record_size), sizeof(uint32_t));
    urbanpop_file.read(reinterpret_cast<char*>(&index_end_offset), sizeof(uint64_t));
    if (!urbanpop_file) { Abort("Failed to read UrbanPop header"); }
    // Validate magic number
    if (magic_number != 0x55504F50) { Abort("Invalid index file format: magic number mismatch"); }
    // Verify NAICS count matches expected
    if (num_naics != NAICS_COUNT) {
        Abort("NAICS count mismatch: file has " + to_string(num_naics) + " but code expects " + to_string(NAICS_COUNT));
    }

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "Reading combined binary file version " << version << "\n";
        Print() << "  Index section: " << index_end_offset << " bytes\n";
        Print() << "  GEOIDs: " << num_geoids << "\n";
        Print() << "  Agents: " << num_agents << "\n";
        Print() << "  Agent record size: " << agent_record_size << " bytes\n";
    }
    block_groups.reserve(num_geoids);
    // Read each block group entry
    for (uint32_t block_i = 0; block_i < num_geoids; block_i++) {
        BlockGroup block_group;
        if (!block_group.read(urbanpop_file)) { Abort("Failed to read block group " + to_string(block_i)); }
        block_group.block_i = block_i;
        block_groups.push_back(block_group);
    }
}

static std::pair<int, double> getAllLoadBalance (const long num) {
    int all = num;
    ParallelDescriptor::ReduceIntSum(all);
    int max_num = num;
    ParallelDescriptor::ReduceIntMax(max_num);
    double load_balance = (double)all / (double)NProcs() / max_num;
    return {all, load_balance};
}

/*! \brief Read in UrbanPop data from given file
 */
void UrbanPopData::init (ExaEpi::TestParams& params, Geometry& geom, BoxArray& ba, DistributionMapping& dm) {
    BL_PROFILE("UrbanPopData::init");
    std::string fname = params.urbanpop_filename;

    urbanpop_file.open(fname, std::ios::binary);
    if (!urbanpop_file) { Abort("Failed to open file: " + fname); }

    // every rank reads all the block groups from the index file
    readBlockGroupsFile(urbanpop_file, block_groups);
    // now sort block groups by geoid to make all FIPS units consecutively grouped
    std::sort(block_groups.begin(), block_groups.end(), [] (const BlockGroup& bg1, const BlockGroup& bg2) {
        return bg1.geoid < bg2.geoid;
    });

    // get FIPS codes and block group start indices from block group array. These vectors are used when initializing infections
    // Each community is a block group
    int current_FIPS = -1;
    int num_communities = 0;
    for (int i = 0; i < block_groups.size(); i++) {
        auto& block_group = block_groups[i];
        // FIPS is the first 5 digits of the GEOID, which is 12 digits
        int64_t fips = static_cast<int64_t>(block_group.geoid / 1e7);
        if (current_FIPS != fips) {
            FIPS_codes.push_back(fips);
            fips_community_start.push_back(num_communities);
            current_FIPS = fips;
        }
        num_communities++;
        if (geoid_to_block_groups.insert({block_group.geoid, i}).second == false) { Abort("Cannot insert new block group"); }
    }
    fips_community_start.push_back(num_communities);

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "Found " << FIPS_codes.size() << " FIPS demographic units\n";
        // for (int i = 0; i < FIPS_codes.size(); i++) {
        //     Print() << "    FIPS " << FIPS_codes[i] << " " << fips_community_start[i] << "\n";
        // }
    }

    AMREX_ALWAYS_ASSERT(block_groups.size() == num_communities);

    // add in a buffer to ensure we can fit them all in a 2D grid
    geom = getGeometry(num_communities);

    ba.define(geom.Domain());
    ba.maxSize(params.max_box_size);
    dm.define(ba);

    Print() << "Base domain: " << geom.Domain() << "\n";
    Print() << "Max box size: " << params.max_box_size << "\n";
    Print() << "Number of boxes: " << ba.size() << " over " << ParallelDescriptor::NProcs() << " ranks. \n";
    Print() << "Number of block groups (communities): " << block_groups.size() << "\n";

    geoid_mf.define(ba, dm, 2, 0);
    community_mf.define(ba, dm, 1, 0);

    geoid_mf.setVal(-1);
    community_mf.setVal(-1);

    std::ofstream geoid_coords_ofs;
    //  allocate block groups to x,y grid locations and use a map to keep track of them for later processing
    int max_x = geom.Domain().bigEnd()[0];
    int max_y = geom.Domain().bigEnd()[1];
    Print() << " max " << max_x << "," << max_y << "\n";
    int x = 0;
    int y = 0;
    for (int bi = 0; bi < block_groups.size(); bi++) {
        auto& block_group = block_groups[bi];
        block_group.x = x;
        block_group.y = y;
        auto xy = IntVect(x, y);
        if (xy_to_block_groups.insert({xy, bi}).second == false) { Abort("Duplicate xy location found for block groups"); }
        x++;
        if (x > max_x) {
            x = 0;
            y++;
            if (y > max_y) { Abort("Not enough grid points for all the block groups\n"); }
        }
        num_communities++;
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static int get_max_nborhood (int nborhood_size, int community_size) {
    int max_nborhood = static_cast<int>(Math::round(static_cast<Real>(community_size) / nborhood_size));
    return max_nborhood > 0 ? max_nborhood : 1;
}

void UrbanPopData::initAgents (AgentContainer& pc, const ExaEpi::TestParams& params) {
    BL_PROFILE("UrbanPopData::initAgents");

    int myproc = ParallelDescriptor::MyProc();
    auto dx = pc.ParticleGeom(0).CellSizeArray();

    int home_population = 0;
    int work_population = 0;
    int num_households = 0;
    int num_employed = 0;
    int num_students = 0;
    int num_educators = 0;
    int num_communities = 0;
    int nborhood_size = params.nborhood_size;
    int num_nborhoods = 0;

    if (!urbanpop_file) { Abort("File " + params.urbanpop_filename + " is not open\n"); }
    for (MFIter mfi = pc.MakeMFIter(0); mfi.isValid(); ++mfi) {
        Vector<UrbanPopAgent> agents;
        Vector<AgentExtras> agents_extras;

        auto geoid_arr = geoid_mf[mfi].array();
        auto community_indices_arr = community_mf[mfi].array();

        const Box& tilebox = mfi.tilebox();
        {
            int min_x = lbound(tilebox).x;
            int max_x = ubound(tilebox).x + 1;
            int min_y = lbound(tilebox).y;
            int max_y = ubound(tilebox).y + 1;

            for (int x = min_x; x < max_x; x++) {
                for (int y = min_y; y < max_y; y++) {
                    auto xy = IntVect(x, y);
                    auto it = xy_to_block_groups.find(xy);
                    if (it == xy_to_block_groups.end()) { continue; }
                    int bi = it->second;
                    AMREX_ALWAYS_ASSERT(bi >= 0 && bi < block_groups.size());
                    auto& block_group = block_groups[bi];
                    AMREX_ASSERT(block_group.x >= min_x && block_group.x < max_x && block_group.y >= min_y &&
                                 block_group.y < max_y);
                    home_population += block_group.home_population;
                    work_population += block_group.work_populations[0];
                    block_group.readAgents(urbanpop_file, agents, agents_extras, geoid_to_block_groups, block_groups);
                    num_households += block_group.num_households;
                    num_employed += block_group.num_employed;
                    num_students += block_group.num_students;
                    num_educators += block_group.num_educators;
                    num_communities++;
                    //  FIPS is the first 5 digits of the GEOID, which is 12 digits
                    int64_t fips = static_cast<int64_t>(block_group.geoid / 1e7);
                    geoid_arr(x, y, 0, 0) = fips;
                    // Census tract is the 7 remaining digits after the FIPS code
                    geoid_arr(x, y, 0, 1) = static_cast<int64_t>(block_group.geoid - fips * 1e7);
                    community_indices_arr(x, y, 0) = bi;
                    num_nborhoods += get_max_nborhood(nborhood_size, block_group.home_population);
                }
            }
        }

        if (num_communities == 0) { continue; }

        auto& ptile = pc.DefineAndReturnParticleTile(0, mfi);
        ptile.resize(agents.size());
        auto aos = &ptile.GetArrayOfStructs()[0];

        Gpu::DeviceVector<UrbanPopAgent> agents_d;
        Gpu::DeviceVector<AgentExtras> agents_extras_d;
        copyToDeviceAsync(agents, agents_d);
        copyToDeviceAsync(agents_extras, agents_extras_d);
        Gpu::streamSynchronize();

        auto agents_ptr = agents_d.data();
        auto agents_extras_ptr = agents_extras_d.data();

        auto& soa = ptile.GetStructOfArrays();
        auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
        auto family_ptr = soa.GetIntData(IntIdx::family).data();
        auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
        auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
        auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
        auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
        auto trav_i_ptr = soa.GetIntData(IntIdx::trav_i).data();
        auto trav_j_ptr = soa.GetIntData(IntIdx::trav_j).data();
        soa.GetIntData(IntIdx::hosp_i).assign(-1);
        soa.GetIntData(IntIdx::hosp_j).assign(-1);
        auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
        auto hh_cluster_ptr = soa.GetIntData(IntIdx::hh_cluster).data();
        auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
        auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
        auto school_closed_ptr = soa.GetIntData(IntIdx::school_closed).data();
        auto naics_ptr = soa.GetIntData(IntIdx::naics).data();
        auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
        auto work_nborhood_ptr = soa.GetIntData(IntIdx::work_nborhood).data();
        int workgroup_size = params.workgroup_size;
        soa.GetIntData(IntIdx::withdrawn).assign(0);
        soa.GetIntData(IntIdx::random_travel).assign(-1);
        soa.GetIntData(IntIdx::air_travel).assign(-1);

        int i_RT = IntIdx::nattribs;
        int r_RT = RealIdx::nattribs;
        int n_disease = pc.m_num_diseases;

        // Get disease parameters for GPU
        const DiseaseParm** disease_parms_d = new const DiseaseParm*[n_disease];
        for (int d = 0; d < n_disease; d++) {
            disease_parms_d[d] = pc.getDiseaseParameters_d(d);
        }

        for (int d = 0; d < n_disease; d++) {
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::treatment_timer).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::prob).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::latent_period).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::infectious_period).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::incubation_period).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::hospital_delay).assign(0.0_rt);
            soa.GetIntData(i_RT + i0(d) + IntIdxDisease::symptomatic).assign(0);
        }
        auto np = soa.numParticles();
        AMREX_ALWAYS_ASSERT(np == agents.size());

#ifdef CHECK_PARTICLE_LOCATIONS
        const auto& geom = pc.Geom(0);
        const auto domain = geom.Domain();
        const auto plo = geom.ProbLoArray();
        const auto dxi = geom.InvCellSizeArray();
#endif

        ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
            auto& p = aos[i];
            auto& agent = agents_ptr[i];
            // agent ID in amrex must be > 0
            p.id() = agent.id + 1;
            p.cpu() = myproc;
            AMREX_ASSERT(tilebox.contains(agents_extras_ptr[i].home_xy));
            home_i_ptr[i] = agents_extras_ptr[i].home_xy[0];
            home_j_ptr[i] = agents_extras_ptr[i].home_xy[1];
            p.pos(0) = static_cast<ParticleReal>((home_i_ptr[i] + 0.5_rt) * dx[0]);
            p.pos(1) = static_cast<ParticleReal>((home_j_ptr[i] + 0.5_rt) * dx[1]);
            work_i_ptr[i] = agents_extras_ptr[i].work_xy[0];
            work_j_ptr[i] = agents_extras_ptr[i].work_xy[1];
#ifdef CHECK_PARTICLE_LOCATIONS
            // this is the code for checking particle locations within boxes that is called by Ok()
            AgentContainer::CellAssignor assignor;
            IntVect iv2 = assignor(p, plo, dxi, domain);
            AMREX_ASSERT(tilebox.contains(iv2));
#endif
            // Age group (under 5, 5-17, 18-29, 30-64, 65+)
            if (agent.age < 5) {
                age_group_ptr[i] = AgeGroups::u5;
            } else if (agent.age < 18) {
                age_group_ptr[i] = AgeGroups::a5to17;
            } else if (agent.age < 30) {
                age_group_ptr[i] = AgeGroups::a18to29;
            } else if (agent.age < 50) {
                age_group_ptr[i] = AgeGroups::a30to49;
            } else if (agent.age < 65) {
                age_group_ptr[i] = AgeGroups::a50to64;
            } else {
                age_group_ptr[i] = AgeGroups::o65;
            }
            family_ptr[i] = agent.household_id;
            int max_nborhood = get_max_nborhood(nborhood_size, agents_extras_ptr[i].home_population);
            nborhood_ptr[i] = Random_int(max_nborhood, engine);
            hh_cluster_ptr[i] = agent.household_id / 4;
            school_grade_ptr[i] = agent.grade;
            school_id_ptr[i] = agent.school_id;
            school_closed_ptr[i] = 0;
            naics_ptr[i] = agent.naics;
            // set up workers
            if (agent.naics != -1) {
                if (agent.school_id == 0) {
                    // the group work population for this agent is for the NAICS category for the agent
                    int max_workgroup = agents_extras_ptr[i].naics_population / workgroup_size + 1;
                    // a workgroup of 0 indicates not working
                    workgroup_ptr[i] = Random_int(max_workgroup, engine) + 1;
                    AMREX_ASSERT(workgroup_ptr[i] > 0 && workgroup_ptr[i] < max_workgroup * (NAICS_COUNT + 1));
                    int max_work_nborhood = get_max_nborhood(nborhood_size, agents_extras_ptr[i].work_population);
                    work_nborhood_ptr[i] = Random_int(max_work_nborhood, engine);
                    AMREX_ASSERT(work_nborhood_ptr[i] < 5000);
                } else {
                    // educator, workgroup is school, as is nborhood
                    workgroup_ptr[i] = school_id_ptr[i];
                    work_nborhood_ptr[i] = school_id_ptr[i];
                }
                work_i_ptr[i] = home_i_ptr[i];
                work_j_ptr[i] = home_j_ptr[i];
            } else {
                workgroup_ptr[i] = 0;
                // everyone interacts in the work nborhood, even thoes that don't work (they interact during the day in their
                // home neighborhoods, effectively
                work_nborhood_ptr[i] = nborhood_ptr[i];
            }

            trav_i_ptr[i] = home_i_ptr[i];
            trav_j_ptr[i] = home_j_ptr[i];
        });
        Gpu::synchronize();

        // Set initial disease status and immunity
        auto status_ptrs = new int*[n_disease];
        auto disease_counter_ptrs = new ParticleReal*[n_disease];
        for (int d = 0; d < n_disease; d++) {
            status_ptrs[d] = soa.GetIntData(i_RT + i0(d) + IntIdxDisease::status).data();
            disease_counter_ptrs[d] = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::disease_counter).data();
        }

        ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
            for (int d = 0; d < n_disease; d++) {
                // Check if agent should be initially immune for this disease
                if (disease_parms_d[d]->initial_immunity_fraction > 0.0_prt
                    && amrex::Random(engine) < disease_parms_d[d]->initial_immunity_fraction) {
                    status_ptrs[d][i] = Status::immune;
                    // Set immune counter to random point in immunity period
                    // Sample full immune duration, then pick random point within it
                    ParticleReal full_immune_duration =
                            static_cast<ParticleReal>(amrex::RandomGamma(disease_parms_d[d]->immune_length_alpha,
                                                                         disease_parms_d[d]->immune_length_beta, engine));
                    disease_counter_ptrs[d][i] = amrex::Random(engine) * full_immune_duration;
                } else {
                    status_ptrs[d][i] = Status::never;
                    disease_counter_ptrs[d][i] = 0.0_prt;
                }
            }
        });
        Gpu::synchronize();

        delete[] status_ptrs;
        delete[] disease_counter_ptrs;
        delete[] disease_parms_d;

        // now ensure that all members of the same family have the same home nborhood
        // and ensure all members of the same hh cluster have the same home neighborhood
        ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
            // search forwards to find the last member of the family and use that agent's nborhood
            int nborhood = nborhood_ptr[i];
            for (int j = i + 1; j < np; j++) {
                if (home_i_ptr[i] != home_i_ptr[j] || home_j_ptr[i] != home_j_ptr[j]) { break; }
#define INTER_NH_HCS
#ifdef INTER_NH_HCS
                if (family_ptr[i] != family_ptr[j]) { break; }
#else
                // intra NH definition
                if (hh_cluster_ptr[i] != hh_cluster_ptr[j]) { break; }
#endif
                nborhood = nborhood_ptr[j];
            }
            nborhood_ptr[i] = nborhood;
        });
        Gpu::synchronize();
    }

    urbanpop_file.close();
    AMREX_ALWAYS_ASSERT(pc.OK());

    pc.comm_mf.define(community_mf.boxArray(), community_mf.DistributionMap(), 1, 0);
    iMultiFab::Copy(pc.comm_mf, community_mf, 0, 0, 1, 0);

    // AllPrint() << "Process " << MyProc() << ": population " << home_population << " in " << num_communities << "
    // communities\n";
    auto [all_num_communities, load_balance_communities] = getAllLoadBalance(num_communities);
    auto [all_num_agents, load_balance_agents] = getAllLoadBalance(home_population);
    ParallelContext::BarrierAll();

    ParallelDescriptor::ReduceIntSum(home_population);
    ParallelDescriptor::ReduceIntSum(work_population);
    ParallelDescriptor::ReduceIntSum(num_households);
    ParallelDescriptor::ReduceIntSum(num_employed);
    ParallelDescriptor::ReduceIntSum(num_students);
    ParallelDescriptor::ReduceIntSum(num_educators);
    ParallelDescriptor::ReduceIntSum(num_nborhoods);

    Print() << std::fixed << std::setprecision(2) << "Population:  " << all_num_agents << " (balance " << load_balance_agents
            << ")\n"
            << "Employed:     " << num_employed << "\n"
            << "Students:     " << num_students << "\n"
            << "Educators:    " << num_educators << "\n"
            << "Households:   " << num_households << "\n"
            << "Neigborhoods: " << num_nborhoods << " (avg " << (static_cast<Real>(all_num_agents) / num_nborhoods) << ")\n"
            << "Communities:  " << all_num_communities << " (balance " << load_balance_communities << ")\n";

    // Print() << "Work population " << work_population << " home population " << home_population << "\n";
    AMREX_ALWAYS_ASSERT(num_employed == work_population);

    num_communities = all_num_communities;
}
