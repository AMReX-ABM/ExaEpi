/*! @file UrbanPopData.cpp
    \brief Implementation of #UrbanPopData class
*/

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <AMReX.H>
#include <AMReX_Arena.H>
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
            // leading 2 digits of the 12-digit GEOID are the state FIPS (consistent with the
            // 5-digit county FIPS derived from the leading 5 digits in UrbanPopData::init)
            agents_extras[i].work_state_fips = static_cast<int>(work_block_group.geoid / 10000000000LL);
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

/*! \brief Read a per-(state, NAICS-code) work-group target size table (see
    utilities/UrbanPop-scripts/compute_workgroup_sizes.py) into a flat, MAX_STATE_FIPS *
    NAICS_COUNT-length vector, index-aligned with UrbanPop::naics_descriptions. Every entry
    defaults to default_size before the file is parsed, so any (state, NAICS) combination
    the file omits (or the whole vector, if fname is empty) silently falls back to the flat,
    historical behavior. Comment lines (leading '#') are skipped. Aborts if a NAICS code in
    the file is not found in UrbanPop::naics_descriptions, or a state FIPS is out of range
    -- catches a stale/mismatched table loudly rather than silently misassigning. */
static Vector<int> readWorkgroupSizeTable (const std::string& fname, int default_size) {
    Vector<int> sizes(UrbanPopData::MAX_STATE_FIPS * NAICS_COUNT, default_size);
    if (fname.empty()) { return sizes; }

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    istringstream is(fileCharPtrString, istringstream::in);

    // build a lookup once: naics code string -> index into naics_descriptions
    unordered_map<string, int> naics_index;
    for (int i = 0; i < NAICS_COUNT; i++) {
        naics_index[naics_descriptions[i]] = i;
    }

    string line;
    int nrows_declared = -1;
    int nrows_read = 0;
    while (std::getline(is, line)) {
        if (line.empty() || line[0] == '#') { continue; }
        istringstream lis(line);
        if (nrows_declared < 0) {
            lis >> nrows_declared;
            continue;
        } // first non-comment line: row count
        int state_fips;
        string code;
        int size;
        if (!(lis >> state_fips >> code >> size)) { continue; }
        if (state_fips < 0 || state_fips >= UrbanPopData::MAX_STATE_FIPS) {
            Abort("workgroup_size_filename '" + fname + "': state FIPS " + to_string(state_fips) +
                  " is out of range (>= MAX_STATE_FIPS=" + to_string(UrbanPopData::MAX_STATE_FIPS) + ")");
        }
        auto it = naics_index.find(code);
        if (it == naics_index.end()) {
            Abort("workgroup_size_filename '" + fname + "': unknown NAICS code '" + code +
                  "' (not in UrbanPop::naics_descriptions -- table may be stale or for a different NAICS encoding)");
        }
        if (size < 1) {
            Abort("workgroup_size_filename '" + fname + "': invalid work-group size " + to_string(size) + " for state " +
                  to_string(state_fips) + " NAICS " + code + " (must be >= 1)");
        }
        sizes[state_fips * NAICS_COUNT + it->second] = size;
        nrows_read++;
    }
    if (ParallelDescriptor::IOProcessor()) {
        Print() << "Read " << nrows_read << " per-(state, NAICS) work-group sizes from " << fname << " (declared "
                << nrows_declared << "); combinations not in the file use the flat "
                << "default (" << default_size << ").\n";
    }
    return sizes;
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
    County_on_proc.resize(FIPS_codes.size());
    for (int i = 0; i < County_on_proc.size(); i++) {
        County_on_proc[i] = 0;
    }

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

    workgroup_size_table = readWorkgroupSizeTable(params.workgroup_size_filename, params.workgroup_size);
    copyToDeviceAsync(workgroup_size_table, workgroup_size_table_d);
    Gpu::streamSynchronize();

    fillGridMetadataOnHost();

    // block_groups is fully populated identically on every rank (readBlockGroupsFile reads the
    // whole index everywhere), so these three histograms need no cross-rank gather -- build and
    // print directly on the IOProcessor.
    if (ParallelDescriptor::IOProcessor()) {
        std::map<Long, Long> home_size_hist, work_size_hist, naics_worker_hist;
        for (auto& bg : block_groups) {
            if (bg.home_population > 0) { home_size_hist[bg.home_population]++; }
            if (bg.work_populations[0] > 0) { work_size_hist[bg.work_populations[0]]++; }
            for (int n = 0; n < NAICS_COUNT; n++) {
                int count = bg.work_populations[n + 1];
                if (count > 0) { naics_worker_hist[count]++; }
            }
        }
        ExaEpi::Utils::printHistogram("Community home population", home_size_hist, 50, 60, 0, true);
        ExaEpi::Utils::printHistogram("Community work population", work_size_hist, 50, 60, 0, true);
        ExaEpi::Utils::printHistogram("Workers per (community, NAICS)", naics_worker_hist, 50, 60, 0, true);
    }
}

void UrbanPopData::fillGridMetadataOnHost () {
    BL_PROFILE("UrbanPopData::fillGridMetadataOnHost");

    iMultiFab geoid_mf_h(geoid_mf.boxArray(), geoid_mf.DistributionMap(), 2, 0, MFInfo().SetArena(The_Pinned_Arena()));
    iMultiFab community_mf_h(community_mf.boxArray(), community_mf.DistributionMap(), 1, 0,
                             MFInfo().SetArena(The_Pinned_Arena()));

    for (MFIter mfi(geoid_mf_h); mfi.isValid(); ++mfi) {
        auto geoid_arr = geoid_mf_h[mfi].array();
        auto community_arr = community_mf_h[mfi].array();

        const Box& box = mfi.validbox();
        const auto lo = lbound(box);
        const auto hi = ubound(box);

        for (int x = lo.x; x <= hi.x; ++x) {
            for (int y = lo.y; y <= hi.y; ++y) {
                geoid_arr(x, y, 0, 0) = -1;
                geoid_arr(x, y, 0, 1) = -1;
                community_arr(x, y, 0) = -1;

                auto it = xy_to_block_groups.find(IntVect(x, y));
                if (it == xy_to_block_groups.end()) { continue; }

                int bi = it->second;
                AMREX_ALWAYS_ASSERT(bi >= 0 && bi < block_groups.size());
                const auto& block_group = block_groups[bi];
                const int64_t fips = static_cast<int64_t>(block_group.geoid / 1e7);

                geoid_arr(x, y, 0, 0) = static_cast<int>(fips);
                geoid_arr(x, y, 0, 1) = static_cast<int>(block_group.geoid - fips * 1e7);
                community_arr(x, y, 0) = bi;
            }
        }

        auto& geoid_src = geoid_mf_h[mfi];
        auto& geoid_dst = geoid_mf[mfi];
        AMREX_ALWAYS_ASSERT(geoid_src.size() == geoid_dst.size());
        Gpu::copy(Gpu::hostToDevice, geoid_src.dataPtr(), geoid_src.dataPtr() + geoid_src.size(), geoid_dst.dataPtr());

        auto& community_src = community_mf_h[mfi];
        auto& community_dst = community_mf[mfi];
        AMREX_ALWAYS_ASSERT(community_src.size() == community_dst.size());
        Gpu::copy(Gpu::hostToDevice, community_src.dataPtr(), community_src.dataPtr() + community_src.size(),
                  community_dst.dataPtr());
    }

    Gpu::streamSynchronize();
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

    // rank-local/partial (this rank's tiles only) tallies for the household size, household-
    // cluster size, and workgroup target size histograms -- merged across ranks below via
    // ExaEpi::Utils::gatherHistogramCounts once the full loop has finished. household_id (and
    // therefore household_id/4) is only unique *within* a block group (BlockGroup::readAgents
    // counts them with a local unordered_set), so the occupant-tally keys below combine it with
    // the owning block group's geoid to get a globally-unique id -- keying on household_id alone
    // would silently merge unrelated households from different communities that happen to reuse
    // the same small local id.
    std::unordered_map<int64_t, int> household_occupants;
    std::unordered_map<int64_t, int> cluster_occupants;
    std::map<Long, Long> workgroup_target_hist;

    if (!urbanpop_file) { Abort("File " + params.urbanpop_filename + " is not open\n"); }
    for (MFIter mfi = pc.MakeMFIter(0); mfi.isValid(); ++mfi) {
        Vector<UrbanPopAgent> agents;
        Vector<AgentExtras> agents_extras;

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
                    int agents_start_i = agents.size();
                    block_group.readAgents(urbanpop_file, agents, agents_extras, geoid_to_block_groups, block_groups);
                    num_households += block_group.num_households;
                    num_employed += block_group.num_employed;
                    num_students += block_group.num_students;
                    num_educators += block_group.num_educators;
                    num_communities++;

                    // household size / cluster size / workgroup target size tallies -- host-side,
                    // right after this block group's agents are read. hh_cluster grouping uses
                    // household_id % num_clusters (see the assignment kernel below) rather than
                    // household_id / 4 -- consecutive household IDs in the UrbanPop data turn out
                    // to be strongly size-correlated (empirically, lag-1 Pearson r ~= 0.70), so
                    // dividing into consecutive blocks of 4 systematically clumps same-sized
                    // (often large) households together; striding by num_clusters spaces grouped
                    // households far enough apart in the original ordering to break that
                    // correlation (verified empirically to outperform even a true random shuffle,
                    // since it guarantees uniform spacing rather than relying on chance).
                    int num_clusters = std::max(1, (block_group.num_households + 3) / 4);
                    for (int i = agents_start_i; i < agents.size(); i++) {
                        auto& agent = agents[i];
                        agents_extras[i].home_num_households = block_group.num_households;
                        int64_t hh_key = (block_group.geoid << 16) | (uint16_t)agent.household_id;
                        household_occupants[hh_key]++;
                        int64_t cluster_key = (block_group.geoid << 16) | (uint16_t)(agent.household_id % num_clusters);
                        cluster_occupants[cluster_key]++;
                        if (agent.naics != -1) {
                            int state_fips = agents_extras[i].work_state_fips;
                            int target = workgroup_size_table[state_fips * NAICS_COUNT + agent.naics];
                            workgroup_target_hist[target]++;
                        }
                    }
                    //  FIPS is the first 5 digits of the GEOID, which is 12 digits
                    int64_t fips = static_cast<int64_t>(block_group.geoid / 1e7);
                    num_nborhoods += get_max_nborhood(nborhood_size, block_group.home_population);
                    for (int i = 0; i < FIPS_codes.size(); i++) {
                        if (FIPS_codes[i] == fips) {
                            County_on_proc[i] = 1;
                            break;
                        }
                    }
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
        auto workgroup_size_table_ptr = workgroup_size_table_d.data();
        soa.GetIntData(IntIdx::withdrawn).assign(0);
        soa.GetIntData(IntIdx::random_travel).assign(-1);
        soa.GetIntData(IntIdx::air_travel).assign(-1);

        int i_RT = IntIdx::nattribs;
        int r_RT = RealIdx::nattribs;
        int n_disease = pc.m_num_diseases;
        for (int d = 0; d < n_disease; d++) {
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::treatment_timer).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::disease_counter).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::prob).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::latent_period).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::infectious_period).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::incubation_period).assign(0.0_rt);
            soa.GetRealData(r_RT + r0(d) + RealIdxDisease::hospital_delay).assign(0.0_rt);
            soa.GetIntData(i_RT + i0(d) + IntIdxDisease::status).assign(0);
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
            // strided modulo, not consecutive division -- see the tally loop above for why
            int num_clusters = amrex::max(1, (agents_extras_ptr[i].home_num_households + 3) / 4);
            hh_cluster_ptr[i] = agent.household_id % num_clusters;
            school_grade_ptr[i] = agent.grade;
            school_id_ptr[i] = agent.school_id;
            school_closed_ptr[i] = 0;
            naics_ptr[i] = agent.naics;
            // set up workers
            if (agent.naics != -1) {
                if (agent.travel == TRAVEL::_wfh) {
                    // declared work-from-home: no real commute, and no physical collocation with
                    // real workplace colleagues, so treat like a non-worker for workgroup/
                    // work_nborhood purposes (naics_population/work_population describe the
                    // assigned-but-never-visited work_geoid's community, not home, so they're not
                    // a meaningful size for a group this agent never physically joins). Still
                    // nominally employed (naics_ptr set above) for other purposes.
                    workgroup_ptr[i] = 0;
                    work_nborhood_ptr[i] = nborhood_ptr[i];
                    work_i_ptr[i] = home_i_ptr[i];
                    work_j_ptr[i] = home_j_ptr[i];
                } else if (agent.school_id == 0) {
                    // the group work population for this agent is for the NAICS category for the agent,
                    // and the target work-group size is looked up for the agent's workplace state
                    int state_fips = agents_extras_ptr[i].work_state_fips;
                    AMREX_ASSERT(state_fips >= 0 && state_fips < UrbanPopData::MAX_STATE_FIPS);
                    int max_workgroup = agents_extras_ptr[i].naics_population /
                                                workgroup_size_table_ptr[state_fips * NAICS_COUNT + agent.naics] +
                                        1;
                    // a workgroup of 0 indicates not working
                    workgroup_ptr[i] = Random_int(max_workgroup, engine) + 1;
                    AMREX_ASSERT(workgroup_ptr[i] > 0 && workgroup_ptr[i] < max_workgroup * (NAICS_COUNT + 1));
                    int max_work_nborhood = get_max_nborhood(nborhood_size, agents_extras_ptr[i].work_population);
                    work_nborhood_ptr[i] = Random_int(max_work_nborhood, engine);
                    AMREX_ASSERT(work_nborhood_ptr[i] < 5000);
                } else {
                    // Educator: work-based mixing is already modeled via the school_class_group
                    // assigned in AgentContainer::assignSchoolClasses (see InteractionModSchool.H),
                    // which covers every school_id>0 agent -- students and educators alike -- in
                    // properly class-sized buckets. Routing educators through the workgroup/
                    // work_nborhood channels too on top of that (as raw school_id, unsplit by
                    // workgroup_size/nborhood_size) double-counts their contacts and, for a large
                    // school (e.g. a university with thousands of staff), collapses them into one
                    // giant undifferentiated transmission pool. Treat them like a non-worker for
                    // these two channels instead.
                    workgroup_ptr[i] = 0;
                    work_nborhood_ptr[i] = nborhood_ptr[i];
                }
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

    // household/cluster occupant tallies are two-stage: first "how many agents share this ID"
    // (many distinct keys, per-rank-partial), now converted to "how many IDs have this occupant
    // count" (the actual histogram, few distinct keys, a small payload to gather)
    {
        std::map<Long, Long> household_size_hist, cluster_size_hist;
        for (auto& kv : household_occupants) {
            household_size_hist[kv.second]++;
        }
        for (auto& kv : cluster_occupants) {
            cluster_size_hist[kv.second]++;
        }

        auto merged_household = ExaEpi::Utils::gatherHistogramCounts(household_size_hist);
        auto merged_cluster = ExaEpi::Utils::gatherHistogramCounts(cluster_size_hist);
        auto merged_workgroup_target = ExaEpi::Utils::gatherHistogramCounts(workgroup_target_hist);

        if (ParallelDescriptor::IOProcessor()) {
            ExaEpi::Utils::printHistogram("Household size", merged_household);
            // cluster sizes range far wider than household sizes (auto-sizing would pick a
            // bucket width of 1 or 2), so use a fixed width of 5 for a more legible histogram
            ExaEpi::Utils::printHistogram("Household-cluster size", merged_cluster, 50, 60, 5);
            ExaEpi::Utils::printHistogram("Workgroup target size", merged_workgroup_target);
        }
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

/*! \brief Compute a per-community population-size scale factor that corrects the community
 *  interaction model (InteractionModComm.H) from density-dependent to frequency-dependent
 *  transmission, decoupled from the overall calibrated magnitude (xmit_comm_scale). Neighborhood
 *  transmission (InteractionModNborhood.H) deliberately does NOT use this correction -- see that
 *  file's header for why:
 *    raw[c]   = 1 / population[c]                                   -- fixed correction, not tunable
 *    scale[c] = clip(xmit_comm_scale * raw[c]/mean(raw), min_scale, max_scale)
 *  where mean(raw) is the population-weighted mean of raw[] over all communities. InteractionModComm.H
 *  multiplies a susceptible's infection probability once per *raw count* of infectious agents in
 *  their entire community, so without correction the force of infection scales with the absolute
 *  size of the community (num_infected ~= population[c] * prevalence) rather than with
 *  local prevalence alone. Dividing by population[c] exactly cancels that out: num_infected *
 *  raw[c] ~= prevalence, independent of population[c]. This is a fixed correction for how the
 *  interaction code counts contacts, not an epidemiological hypothesis to calibrate per scenario --
 *  every fit sweep converged on this same 1/population form (previously exposed as a size_beta
 *  parameter that always landed on -1.0), so it's built in rather than left tunable. home_population
 *  is always available, so every community participates. */
namespace {
/*! Clip bounds on the per-community size-scale factor -- not exposed as user-tunable
    parameters since every fit sweep converged on the same 1/population form and these
    bounds only guard against extreme multipliers for unusually small/large communities. */
constexpr Real size_min_scale = 0.05_rt;
constexpr Real size_max_scale = 20.0_rt;
} // namespace

amrex::Vector<amrex::Real> computeCommunitySizeScale (const amrex::Vector<BlockGroup>& block_groups,
                                                      const ExaEpi::TestParams& params) {
    Vector<Real> scale(block_groups.size(), 1.0_rt);
    if (block_groups.empty()) { return scale; }

    Real weight_sum = 0.0_rt;
    for (const auto& bg : block_groups) {
        weight_sum += (Real)bg.home_population;
    }

    Vector<Real> raw(block_groups.size(), 1.0_rt);
    Real weighted_raw_sum = 0.0_rt;
    for (int c = 0; c < (int)block_groups.size(); ++c) {
        Real pop = (Real)block_groups[c].home_population;
        raw[c] = 1.0_rt / pop;
        weighted_raw_sum += pop * raw[c];
    }
    Real mean_raw = (weight_sum > 0.0_rt) ? (weighted_raw_sum / weight_sum) : 1.0_rt;

    for (int c = 0; c < (int)block_groups.size(); ++c) {
        Real s = params.xmit_comm_scale * raw[c] / mean_raw;
        scale[c] = std::max(size_min_scale, std::min(size_max_scale, s));
    }

    amrex::Print() << "SizeScale: " << block_groups.size() << " communities (xmit_comm_scale=" << params.xmit_comm_scale << ")\n";

    return scale;
}

/*! \brief Compute a per-community work-population scale factor, exactly mirroring
 *  computeCommunitySizeScale but keyed on work_populations[0] (total workers whose workplace is
 *  this community) instead of home_population:
 *    raw[c]   = 1 / work_population[c]                           -- fixed correction, not tunable
 *    scale[c] = clip(xmit_comm_scale * raw[c]/mean(raw), min_scale, max_scale)
 *  where mean(raw) is the work-population-weighted mean of raw[] over all communities. Communities
 *  with zero work population (no one's workplace is there) get scale=1.0, unaffected by
 *  xmit_comm_scale, and don't contribute to the weighted mean. Shares xmit_comm_scale/min_scale/
 *  max_scale with computeCommunitySizeScale (a decoupled sweep found no benefit to tuning them
 *  separately from the home/night values). */
amrex::Vector<amrex::Real> computeCommunityWorkSizeScale (const amrex::Vector<BlockGroup>& block_groups,
                                                          const ExaEpi::TestParams& params) {
    Vector<Real> scale(block_groups.size(), 1.0_rt);
    if (block_groups.empty()) { return scale; }

    Real weight_sum = 0.0_rt;
    for (const auto& bg : block_groups) {
        Real pop = (Real)bg.work_populations[0];
        if (pop > 0.0_rt) { weight_sum += pop; }
    }

    Vector<Real> raw(block_groups.size(), 1.0_rt);
    Real weighted_raw_sum = 0.0_rt;
    for (int c = 0; c < (int)block_groups.size(); ++c) {
        Real pop = (Real)block_groups[c].work_populations[0];
        if (pop <= 0.0_rt) { continue; }
        raw[c] = 1.0_rt / pop;
        weighted_raw_sum += pop * raw[c];
    }
    Real mean_raw = (weight_sum > 0.0_rt) ? (weighted_raw_sum / weight_sum) : 1.0_rt;

    for (int c = 0; c < (int)block_groups.size(); ++c) {
        if (block_groups[c].work_populations[0] <= 0) { continue; }
        Real s = params.xmit_comm_scale * raw[c] / mean_raw;
        scale[c] = std::max(size_min_scale, std::min(size_max_scale, s));
    }

    amrex::Print() << "WorkSizeScale: " << block_groups.size() << " communities (xmit_comm_scale=" << params.xmit_comm_scale
                   << ")\n";

    return scale;
}
