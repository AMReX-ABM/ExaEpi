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

#include <zlib.h>

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
using std::ostringstream;
using std::runtime_error;
using std::string;
using std::to_string;
using std::unordered_set;

using ParallelDescriptor::MyProc;
using ParallelDescriptor::NProcs;

template <typename T>
void copyToDeviceAsync (const Vector<T>& h_vec, Gpu::DeviceVector<T>& d_vec) {
    d_vec.resize(0);
    d_vec.resize(h_vec.size());
    Gpu::copyAsync(Gpu::hostToDevice, h_vec.begin(), h_vec.end(), d_vec.begin());
}

/*! \brief Read one block group's agent frame and inflate it into `scratch.raw`.

    Each block group's agents are stored as an independently-compressed frame (see the v3 notes in
    UrbanPop-scripts/upop_to_exaepi.py), which is what keeps a compressed file seekable: a rank
    inflates exactly the block groups its tiles own and never touches the rest, just as it used to
    seek straight to them. The inflated size is not stored in the index because it is always
    home_population * record_size(), and requiring zlib to produce exactly that many bytes doubles
    as a corruption check. */
static void readFrame (ifstream& f, const BlockGroup& block_group, uint32_t codec, FrameScratch& scratch) {
    BL_PROFILE("readFrame");
    const uLongf raw_nbytes = (uLongf)block_group.home_population * UrbanPopAgent::record_size();
    scratch.raw.resize(raw_nbytes);
    f.seekg(block_group.frame_offset);
    if (codec == CODEC_NONE) {
        if (!f.read(scratch.raw.data(), raw_nbytes)) {
            Abort("File is corrupted: end of file reading frame for geoid " + to_string(block_group.geoid) + "\n");
        }
        return;
    }
    scratch.compressed.resize(block_group.frame_nbytes);
    if (!f.read(scratch.compressed.data(), block_group.frame_nbytes)) {
        Abort("File is corrupted: end of file reading frame for geoid " + to_string(block_group.geoid) + "\n");
    }
    uLongf out_nbytes = raw_nbytes;
    int rc = uncompress(reinterpret_cast<Bytef*>(scratch.raw.data()), &out_nbytes,
                        reinterpret_cast<const Bytef*>(scratch.compressed.data()), block_group.frame_nbytes);
    if (rc != Z_OK) {
        Abort("File is corrupted: zlib error " + to_string(rc) + " inflating frame for geoid " + to_string(block_group.geoid) +
              "\n");
    }
    if (out_nbytes != raw_nbytes) {
        Abort("File is corrupted: frame for geoid " + to_string(block_group.geoid) + " inflated to " + to_string(out_nbytes) +
              " bytes, expected " + to_string(raw_nbytes) + "\n");
    }
}

bool BlockGroup::readAgents (ifstream& f, uint32_t codec, FrameScratch& scratch, Vector<UrbanPopAgent>& agents,
                             amrex::Vector<AgentExtras>& agents_extras, const std::map<int64_t, int>& geoid_to_block_groups,
                             const Vector<BlockGroup>& block_groups) {
    BL_PROFILE("BlockGroup::readAgents");
    num_households = 0;
    num_employed = 0;
    num_students = 0;
    num_educators = 0;
    int start_i = agents.size();
    agents.resize(start_i + home_population);
    agents_extras.resize(start_i + home_population);
    // used for counting up the number of unique households
    unordered_set<int> households;
    readFrame(f, *this, codec, scratch);
    const UrbanPop::AgentFrame frame(scratch.raw.data(), home_population);
    for (int i = start_i; i < agents.size(); i++) {
        auto& agent = agents[i];
        frame.get(i - start_i, agent);
        if (agent.id == -1) { Abort("File is corrupted: couldn't read agent p_id at offset " + to_string(frame_offset) + "\n"); }
        if (agent.home_geoid != geoid) {
            Abort("File is corrupted: wrong geoid, read " + to_string(agent.home_geoid) + " expected " + to_string(geoid) +
                  " frame offset " + to_string(frame_offset) + " home pop " + to_string(home_population) + "\n");
        }
        households.insert(agent.household_id);
        agents_extras[i].home_xy = IntVect(x, y);
        auto it = geoid_to_block_groups.find(agent.work_geoid);
        if (it == geoid_to_block_groups.end()) { Abort("Cannot find block group for work location"); }
        auto& work_block_group = block_groups[it->second];
        agents_extras[i].work_xy = IntVect(work_block_group.x, work_block_group.y);
        if (agent.naics != -1) {
            num_employed++;
            AMREX_ASSERT(work_block_group.work_populations[agent.naics + 1] > 0 &&
                         work_block_group.work_populations[agent.naics + 1] < 100000);
            AMREX_ASSERT(work_block_group.work_populations[0] > 0 && work_block_group.work_populations[0] < 260000);
            if (agent.school_id != 0) { num_educators++; }
        } else {
            if (agent.school_id == 0) { AMREX_ASSERT(agent.home_geoid == agent.work_geoid); }
            if (agent.school_id != 0) { num_students++; }
        }
        // Print() << "Agent " << i << " home " << agents_extras[i].home_xy << " work " << agents_extras[i].work_xy << "\n";
    }
    num_households = households.size();

    return true;
}

bool BlockGroup::read (std::istream& f) {
    BL_PROFILE("BlockGroup::read");
    // Read binary format:
    // - geoid: uint64 (8 bytes)
    // - frame_offset: uint64 (8 bytes)
    // - frame_nbytes: uint32 (4 bytes) -- compressed size of the agent frame
    // - h_pop: uint32 (4 bytes)
    // - w_pop: uint32 (4 bytes)
    // - naics counts: uint32 * NAICS_COUNT (4 bytes each)
    if (!f.read(reinterpret_cast<char*>(&geoid), sizeof(uint64_t))) { return false; }
    if (!f.read(reinterpret_cast<char*>(&frame_offset), sizeof(uint64_t))) { return false; }
    if (!f.read(reinterpret_cast<char*>(&frame_nbytes), sizeof(uint32_t))) { return false; }
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

static void readBlockGroupsFile (std::ifstream& urbanpop_file, Vector<BlockGroup>& block_groups, uint32_t& codec,
                                 const bool verbose) {
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
    urbanpop_file.read(reinterpret_cast<char*>(&codec), sizeof(uint32_t));
    urbanpop_file.read(reinterpret_cast<char*>(&index_end_offset), sizeof(uint64_t));
    if (!urbanpop_file) { Abort("Failed to read UrbanPop header"); }
    // Validate magic number
    if (magic_number != 0x55504F50) { Abort("Invalid index file format: magic number mismatch"); }
    // Verify the format version. v2 added the group-structure fields (nborhood, hh_cluster,
    // work_nborhood, workgroup, school_class, school_class_group) that ExaEpi used to draw
    // itself at init and now reads straight from the file -- a v1 file simply does not have
    // them, so there is nothing to fall back to.
    if (version != FORMAT_VERSION) {
        Abort("UrbanPop file format version " + to_string(version) + " but this build requires version " +
              to_string(FORMAT_VERSION) + " -- regenerate the .bin with UrbanPop-scripts/upop_to_exaepi.py");
    }
    // The format carries a codec field so a future codec doesn't need another version bump; this
    // build implements deflate and uncompressed only.
    if (codec != CODEC_NONE && codec != CODEC_DEFLATE) {
        Abort("UrbanPop file uses codec " + to_string(codec) + ", which this build cannot decompress");
    }
    // Verify NAICS count matches expected
    if (num_naics != NAICS_COUNT) {
        Abort("NAICS count mismatch: file has " + to_string(num_naics) + " but code expects " + to_string(NAICS_COUNT));
    }
    // Belt and braces alongside the version check: the version is bumped by hand, the record
    // size is not, so this catches a field being added, widened or reordered in
    // UrbanPopAgentStruct.H without the version being bumped to match. Without it the mismatch
    // shows up as agents silently read at the wrong offsets.
    if (agent_record_size != UrbanPopAgent::record_size()) {
        Abort("UrbanPop agent record size mismatch: file has " + to_string(agent_record_size) + " bytes but this build reads " +
              to_string(UrbanPopAgent::record_size()) + " -- the .bin and UrbanPopAgentStruct.H are out of sync");
    }

    if (verbose && ParallelDescriptor::IOProcessor()) {
        Print() << "Reading combined binary file version " << version << "\n";
        Print() << "  Index section: " << index_end_offset << " bytes\n";
        Print() << "  GEOIDs: " << num_geoids << "\n";
        Print() << "  Agents: " << num_agents << "\n";
        Print() << "  Agent record size: " << agent_record_size << " bytes\n";
        Print() << "  Agent frames: " << (codec == CODEC_DEFLATE ? "deflate" : "uncompressed") << "\n";
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

/*! \brief Tally the true daytime headcount per community (see UrbanPopData::day_population's doc
    comment) via a full, independent scan of the raw agent file, done identically/redundantly on
    every rank -- deliberately NOT folded into initAgents()'s per-rank-partial tile loop, since
    that only runs on a fresh start and would leave day_population empty on a restarted run. Pure
    host-side counting (no particle/GPU work), so the redundant per-rank scan is cheap relative to
    the one-time cost of reading the file at all. */
static Vector<Real> computeDayPopulation (ifstream& f, uint32_t codec, const Vector<BlockGroup>& block_groups,
                                          const std::map<int64_t, int>& geoid_to_block_groups) {
    BL_PROFILE("computeDayPopulation");
    Vector<Real> day_population(block_groups.size(), 0.0_rt);
    UrbanPopAgent agent;
    FrameScratch scratch;
    for (int bi = 0; bi < (int)block_groups.size(); ++bi) {
        const auto& block_group = block_groups[bi];
        if (block_group.home_population == 0) { continue; }
        readFrame(f, block_group, codec, scratch);
        const UrbanPop::AgentFrame frame(scratch.raw.data(), block_group.home_population);
        for (int i = 0; i < block_group.home_population; ++i) {
            frame.get(i, agent);
            // non-workers and work-from-home agents stay at their home block group during the
            // day; everyone else (including school employees/students, whose work_geoid is the
            // school's block group) physically goes to their work block group -- mirrors the
            // work_i/work_j assignment in initAgents' agent-initialization kernel exactly
            if (agent.naics == -1 || agent.travel == TRAVEL::_wfh) {
                day_population[bi] += 1.0_rt;
            } else {
                auto it = geoid_to_block_groups.find(agent.work_geoid);
                if (it == geoid_to_block_groups.end()) { Abort("Cannot find block group for work location"); }
                day_population[it->second] += 1.0_rt;
            }
        }
    }
    return day_population;
}

/*! \brief Read in UrbanPop data from given file
 */
void UrbanPopData::init (ExaEpi::TestParams& params, Geometry& geom, BoxArray& ba, DistributionMapping& dm) {
    BL_PROFILE("UrbanPopData::init");
    std::string fname = params.urbanpop_filename;

    urbanpop_file.open(fname, std::ios::binary);
    if (!urbanpop_file) { Abort("Failed to open file: " + fname); }

    // every rank reads all the block groups from the index file
    readBlockGroupsFile(urbanpop_file, block_groups, codec, params.verbose);
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
            Population.push_back(0);
            current_FIPS = fips;
        }
        num_communities++;
        if (geoid_to_block_groups.insert({block_group.geoid, i}).second == false) { Abort("Cannot insert new block group"); }
        CountyPop[fips] += block_group.home_population;
        Population.back() += block_group.home_population;
    }
    fips_community_start.push_back(num_communities);
    County_on_proc.resize(FIPS_codes.size());
    for (int i = 0; i < County_on_proc.size(); i++) {
        County_on_proc[i] = 0;
    }

    // device copy of fips_community_start, for AgentContainer::setAirTravel (see AirTravelDemoData)
    fips_community_start_d.resize(fips_community_start.size());
    Gpu::copyAsync(Gpu::hostToDevice, fips_community_start.begin(), fips_community_start.end(), fips_community_start_d.begin());
    Gpu::streamSynchronize();

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "Found " << FIPS_codes.size() << " FIPS demographic units\n";
        // for (int i = 0; i < FIPS_codes.size(); i++) {
        //     Print() << "    FIPS " << FIPS_codes[i] << " " << fips_community_start[i] << "\n";
        // }
    }

    AMREX_ALWAYS_ASSERT(block_groups.size() == num_communities);

    // add in a buffer to ensure we can fit them all in a 2D grid
    geom = getGeometry(num_communities);

    //  allocate block groups to x,y grid locations and use a map to keep track of them for later
    //  processing. Also accumulate each cell's home population so the boxes built below can be
    //  distributed across ranks by agent count (see box_population below) instead of just box
    //  count -- a plain SFC/round-robin map otherwise leaves whichever rank owns a population-dense
    //  cluster of communities (e.g. an urban county) with far more agents than the rest.
    int max_x = geom.Domain().bigEnd()[0];
    int max_y = geom.Domain().bigEnd()[1];
    Print() << " max " << max_x << "," << max_y << "\n";
    Vector<Long> cell_population((max_x + 1) * (max_y + 1), 0);
    int x = 0;
    int y = 0;
    for (int bi = 0; bi < block_groups.size(); bi++) {
        auto& block_group = block_groups[bi];
        block_group.x = x;
        block_group.y = y;
        auto xy = IntVect(x, y);
        if (xy_to_block_groups.insert({xy, bi}).second == false) { Abort("Duplicate xy location found for block groups"); }
        cell_population[y * (max_x + 1) + x] += block_group.home_population;
        x++;
        if (x > max_x) {
            x = 0;
            y++;
            if (y > max_y) { Abort("Not enough grid points for all the block groups\n"); }
        }
        num_communities++;
    }

    ba.define(geom.Domain());
    ba.maxSize(params.max_box_size);

    // Sum each box's population from the per-cell populations above, then assign ranks with the
    // knapsack algorithm so agent count -- not just box count -- is balanced across ranks.
    std::vector<Long> box_population(ba.size(), 0);
    for (int k = 0; k < ba.size(); k++) {
        const Box& bx = ba[k];
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); j++) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); i++) {
                box_population[k] += cell_population[j * (max_x + 1) + i];
            }
        }
    }
    dm.KnapSackProcessorMap(box_population, ParallelDescriptor::NProcs());

    Print() << "Base domain: " << geom.Domain() << "\n";
    Print() << "Max box size: " << params.max_box_size << "\n";
    Print() << "Number of boxes: " << ba.size() << " over " << ParallelDescriptor::NProcs() << " ranks. \n";
    Print() << "Number of block groups (communities): " << block_groups.size() << "\n";

    geoid_mf.define(ba, dm, 2, 0);
    community_mf.define(ba, dm, 1, 0);
    unit_mf.define(ba, dm, 1, 0);

    geoid_mf.setVal(-1);
    community_mf.setVal(-1);
    unit_mf.setVal(-1);

    std::ofstream geoid_coords_ofs;

    day_population = computeDayPopulation(urbanpop_file, codec, block_groups, geoid_to_block_groups);

    fillGridMetadataOnHost();

    // block_groups is fully populated identically on every rank (readBlockGroupsFile reads the
    // whole index everywhere), so these three histograms need no cross-rank gather -- build and
    // print directly on the IOProcessor.
    if (params.verbose && ParallelDescriptor::IOProcessor()) {
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
    iMultiFab unit_mf_h(unit_mf.boxArray(), unit_mf.DistributionMap(), 1, 0, MFInfo().SetArena(The_Pinned_Arena()));

    for (MFIter mfi(geoid_mf_h); mfi.isValid(); ++mfi) {
        auto geoid_arr = geoid_mf_h[mfi].array();
        auto community_arr = community_mf_h[mfi].array();
        auto unit_arr = unit_mf_h[mfi].array();

        const Box& box = mfi.validbox();
        const auto lo = lbound(box);
        const auto hi = ubound(box);

        for (int x = lo.x; x <= hi.x; ++x) {
            for (int y = lo.y; y <= hi.y; ++y) {
                geoid_arr(x, y, 0, 0) = -1;
                geoid_arr(x, y, 0, 1) = -1;
                community_arr(x, y, 0) = -1;
                unit_arr(x, y, 0) = -1;

                auto it = xy_to_block_groups.find(IntVect(x, y));
                if (it == xy_to_block_groups.end()) { continue; }

                int bi = it->second;
                AMREX_ALWAYS_ASSERT(bi >= 0 && bi < block_groups.size());
                const auto& block_group = block_groups[bi];
                const int64_t fips = static_cast<int64_t>(block_group.geoid / 1e7);

                geoid_arr(x, y, 0, 0) = static_cast<int>(fips);
                geoid_arr(x, y, 0, 1) = static_cast<int>(block_group.geoid - fips * 1e7);
                community_arr(x, y, 0) = bi;
                // fips_community_start[u] <= bi < fips_community_start[u+1] identifies the FIPS
                // unit u that community bi belongs to (see AirTravelDemoData/FIPS_codes).
                unit_arr(x, y, 0) =
                        static_cast<int>(std::upper_bound(fips_community_start.begin(), fips_community_start.end(), bi) -
                                         fips_community_start.begin()) -
                        1;
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

        auto& unit_src = unit_mf_h[mfi];
        auto& unit_dst = unit_mf[mfi];
        AMREX_ALWAYS_ASSERT(unit_src.size() == unit_dst.size());
        Gpu::copy(Gpu::hostToDevice, unit_src.dataPtr(), unit_src.dataPtr() + unit_src.size(), unit_dst.dataPtr());
    }

    Gpu::streamSynchronize();
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

    // rank-local/partial (this rank's tiles only) tallies for the household size and household-
    // cluster size histograms -- merged across ranks below via
    // ExaEpi::Utils::gatherHistogramCounts once the full loop has finished. household_id,
    // hh_cluster and nborhood are only unique *within* a block group, so the keys below combine
    // them with the owning block group's geoid to get a globally-unique id -- keying on the raw
    // id alone would silently merge unrelated groups from different communities that happen to
    // reuse the same small local id.
    std::unordered_map<int64_t, int> household_occupants;
    std::unordered_map<int64_t, int> cluster_occupants;
    std::unordered_set<int64_t> nborhoods_seen;

    if (!urbanpop_file) { Abort("File " + params.urbanpop_filename + " is not open\n"); }
    // hoisted out of the loop so the inflate buffers are reused across every block group this
    // rank reads instead of being reallocated per tile (this loop is not OpenMP-parallel)
    FrameScratch scratch;
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
                    block_group.readAgents(urbanpop_file, codec, scratch, agents, agents_extras, geoid_to_block_groups,
                                           block_groups);
                    num_households += block_group.num_households;
                    num_employed += block_group.num_employed;
                    num_students += block_group.num_students;
                    num_educators += block_group.num_educators;
                    num_communities++;

                    // household size / cluster size tallies -- host-side, right after this block
                    // group's agents are read. Both ids come from the file (see
                    // UrbanPop-scripts/group_assignment.py); this only counts occupants of each.
                    // household_id and hh_cluster are only unique within a block group, so the
                    // keys combine them with the owning geoid.
                    for (int i = agents_start_i; i < agents.size(); i++) {
                        auto& agent = agents[i];
                        household_occupants[(block_group.geoid << 16) | (uint16_t)agent.household_id]++;
                        cluster_occupants[(block_group.geoid << 16) | (uint16_t)agent.hh_cluster]++;
                        nborhoods_seen.insert((block_group.geoid << 16) | (uint16_t)agent.nborhood);
                    }
                    //  FIPS is the first 5 digits of the GEOID, which is 12 digits
                    int64_t fips = static_cast<int64_t>(block_group.geoid / 1e7);
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
        auto school_class_ptr = soa.GetIntData(IntIdx::school_class).data();
        auto school_class_group_ptr = soa.GetIntData(IntIdx::school_class_group).data();
        soa.GetIntData(IntIdx::withdrawn).assign(0);
        soa.GetIntData(IntIdx::random_travel).assign(-1);
        soa.GetIntData(IntIdx::air_travel).assign(-1);
        // -1 is the "no weather unit for this agent" sentinel that
        // AgentContainer::initializeWeatherIndex_UrbanPop assigns for a community whose FIPS is not
        // in the active weather set. That function is the only thing that ever writes this field,
        // and it only runs when agent.weather_filename is given -- so without this, a run with no
        // weather file leaves weatherLookup as whatever was in the heap, which
        // AgentContainer::advanceWeatherIndex then increments weekly and every plot file dumps.
        soa.GetIntData(IntIdx::weatherLookup).assign(-1);

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

        // No RNG here: every structural group an agent belongs to is drawn in preprocessing and
        // read straight out of the file (see UrbanPop-scripts/group_assignment.py). That is what
        // makes the initial population identical regardless of MPI rank count, and independent of
        // agent.seed.
        ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
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
            school_grade_ptr[i] = agent.grade;
            school_id_ptr[i] = agent.school_id;
            school_closed_ptr[i] = 0;
            naics_ptr[i] = agent.naics;

            // --- group structure, all straight from the file ---
            // Whole households share a home neighborhood, every establishment sits in one work
            // neighborhood and is split into work-groups from there, and every school_id > 0
            // agent has a class. See UrbanPop-scripts/group_assignment.py for how each is built
            // and for the exemptions baked into them (educators and declared work-from-home
            // agents get workgroup 0 and their home neighborhood as their work neighborhood,
            // like the unemployed -- nobody who is not physically at a workplace joins one).
            nborhood_ptr[i] = agent.nborhood;
            hh_cluster_ptr[i] = agent.hh_cluster;
            workgroup_ptr[i] = agent.workgroup;
            work_nborhood_ptr[i] = agent.work_nborhood;
            school_class_ptr[i] = agent.school_class;
            school_class_group_ptr[i] = agent.school_class_group;
            AMREX_ASSERT(nborhood_ptr[i] >= 0 && work_nborhood_ptr[i] >= 0 && workgroup_ptr[i] >= 0 && hh_cluster_ptr[i] >= 0);
            AMREX_ASSERT((agent.school_id != 0) == (school_class_group_ptr[i] >= 0));

            if (agent.naics != -1 && agent.school_id == 0 && agent.travel == TRAVEL::_wfh) {
                // Declared work-from-home: no real commute, so spend the day at home. Still
                // nominally employed (naics_ptr is set above) for other purposes. Note this is
                // checked *after* the educator case, because an educator's work_i/work_j must
                // stay at the real school's location regardless of commute mode -- school_id is
                // only unique within a location, so moving an educator home silently reassigns
                // them to whatever unrelated school happens to be numbered the same there.
                work_i_ptr[i] = home_i_ptr[i];
                work_j_ptr[i] = home_j_ptr[i];
            }

            trav_i_ptr[i] = home_i_ptr[i];
            trav_j_ptr[i] = home_j_ptr[i];
        });
        Gpu::synchronize();
    }

    // household/cluster occupant tallies are two-stage: first "how many agents share this ID"
    // (many distinct keys, per-rank-partial), now converted to "how many IDs have this occupant
    // count" (the actual histogram, few distinct keys, a small payload to gather)
    if (params.verbose) {
        std::map<Long, Long> household_size_hist, cluster_size_hist;
        for (auto& kv : household_occupants) {
            household_size_hist[kv.second]++;
        }
        for (auto& kv : cluster_occupants) {
            cluster_size_hist[kv.second]++;
        }

        auto merged_household = ExaEpi::Utils::gatherHistogramCounts(household_size_hist);
        auto merged_cluster = ExaEpi::Utils::gatherHistogramCounts(cluster_size_hist);

        if (ParallelDescriptor::IOProcessor()) {
            ExaEpi::Utils::printHistogram("Household size", merged_household);
            // cluster sizes range far wider than household sizes (auto-sizing would pick a
            // bucket width of 1 or 2), so use a fixed width of 5 for a more legible histogram
            ExaEpi::Utils::printHistogram("Household-cluster size", merged_cluster, 50, 60, 5);
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
    int num_nborhoods = static_cast<int>(nborhoods_seen.size());
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
 *  transmission, decoupled from the overall calibrated magnitude (DiseaseParm::xmit_comm_scale,
 *  applied separately, per-disease, since this function has no disease to key on). Neighborhood
 *  transmission (InteractionModNborhood.H) deliberately does NOT use this correction -- see that
 *  file's header for why:
 *    raw[c]   = 1 / population[c]                                   -- fixed correction, not tunable
 *    scale[c] = clip(raw[c]/mean(raw), min_scale, max_scale)
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

amrex::Vector<amrex::Real> computeCommunitySizeScale (const amrex::Vector<BlockGroup>& block_groups, const bool verbose) {
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
        Real s = raw[c] / mean_raw;
        scale[c] = std::max(size_min_scale, std::min(size_max_scale, s));
    }

    if (verbose) { amrex::Print() << "SizeScale: " << block_groups.size() << " communities\n"; }

    return scale;
}

/*! \brief Compute a per-community work-population scale factor, exactly mirroring
 *  computeCommunitySizeScale but keyed on day_population (true daytime headcount -- see its doc
 *  comment for why this is used instead of BlockGroup::work_populations[0]) instead of
 *  home_population:
 *    raw[c]   = 1 / day_population[c]                             -- fixed correction, not tunable
 *    scale[c] = clip(raw[c]/mean(raw), min_scale, max_scale)
 *  where mean(raw) is the day-population-weighted mean of raw[] over all communities. Communities
 *  with zero daytime population get scale=1.0 and don't contribute to the weighted mean. Shares
 *  min_scale/max_scale with computeCommunitySizeScale (a decoupled sweep found no benefit to
 *  tuning them separately from the home/night values). The disease-specific overall magnitude
 *  (DiseaseParm::xmit_comm_scale) is applied separately, per-disease. */
amrex::Vector<amrex::Real> computeCommunityWorkSizeScale (const amrex::Vector<amrex::Real>& day_population, const bool verbose) {
    Vector<Real> scale(day_population.size(), 1.0_rt);
    if (day_population.empty()) { return scale; }

    Real weight_sum = 0.0_rt;
    for (Real pop : day_population) {
        if (pop > 0.0_rt) { weight_sum += pop; }
    }

    Vector<Real> raw(day_population.size(), 1.0_rt);
    Real weighted_raw_sum = 0.0_rt;
    for (int c = 0; c < (int)day_population.size(); ++c) {
        Real pop = day_population[c];
        if (pop <= 0.0_rt) { continue; }
        raw[c] = 1.0_rt / pop;
        weighted_raw_sum += pop * raw[c];
    }
    Real mean_raw = (weight_sum > 0.0_rt) ? (weighted_raw_sum / weight_sum) : 1.0_rt;

    for (int c = 0; c < (int)day_population.size(); ++c) {
        if (day_population[c] <= 0.0_rt) { continue; }
        Real s = raw[c] / mean_raw;
        scale[c] = std::max(size_min_scale, std::min(size_max_scale, s));
    }

    if (verbose) { amrex::Print() << "WorkSizeScale: " << day_population.size() << " communities\n"; }

    return scale;
}
