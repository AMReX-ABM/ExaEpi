/*! @file GeneratePopulation.cpp
    \brief In-process population generation; see GeneratePopulation.H.
*/
#include "GeneratePopulation.H"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include <zlib.h>

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_OpenMP.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include "PmedmProblem.H"
#include "PopulationBundle.H"
#include "PopulationGenerator.H"
#include "Sha256.H"

namespace PopGen {

namespace {

using Clock = std::chrono::steady_clock;

double since (Clock::time_point& t) {
    const auto now = Clock::now();
    const double s = std::chrono::duration<double>(now - t).count();
    t = now;
    return s;
}

std::map<std::string, std::vector<double>> readAllocations (const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open allocations file " + path); }
    std::map<std::string, std::vector<double>> out;
    std::int32_t n;
    while (f.read(reinterpret_cast<char*>(&n), 4)) {
        std::string puma(static_cast<std::size_t>(n), '\0');
        f.read(&puma[0], n);
        std::int32_t dg[2];
        f.read(reinterpret_cast<char*>(dg), 8);
        std::vector<double> al(static_cast<std::size_t>(dg[0]) * static_cast<std::size_t>(dg[1]));
        f.read(reinterpret_cast<char*>(al.data()), static_cast<std::streamsize>(al.size() * sizeof(double)));
        if (!f) { throw std::runtime_error("truncated allocations file " + path); }
        out[puma] = std::move(al);
    }
    return out;
}

//! A PUMA's solve size, donors x block groups: the load estimate for splitting work.
std::int64_t pumaCells (const PopulationBundle& b, int p) {
    const auto bgo = b.get<std::int64_t>("solve.bg_offset");
    const auto dno = b.get<std::int64_t>("solve.donor_offset");
    return (bgo[p + 1] - bgo[p]) * (dno[p + 1] - dno[p]);
}

/*! Owner rank of each PUMA: longest processing time first on donors x block groups, ties to the
    lower PUMA index and the lower rank, so every rank computes the same split. */
std::vector<int> splitPumas (const PopulationBundle& b, int nranks) {
    const int np = pumaCount(b);
    std::vector<std::int64_t> cells(np);
    for (int p = 0; p < np; ++p) {
        cells[p] = pumaCells(b, p);
    }
    std::vector<int> order(np);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&] (int x, int y) {
        return cells[x] > cells[y];
    });
    std::vector<std::int64_t> load(nranks, 0);
    std::vector<int> owner(np);
    for (int p : order) {
        const int r = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
        owner[p] = r;
        load[r] += cells[p];
    }
    return owner;
}

/*! Concatenate every rank's int64 buffer, in rank order, on every rank. */
std::vector<std::int64_t> allGather (const std::vector<std::int64_t>& mine) {
#ifdef AMREX_USE_MPI
    const int nranks = amrex::ParallelDescriptor::NProcs();
    if (nranks > 1) {
        const auto comm = amrex::ParallelDescriptor::Communicator();
        int n = static_cast<int>(mine.size());
        std::vector<int> counts(nranks), displs(nranks, 0);
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
        for (int r = 1; r < nranks; ++r) {
            displs[r] = displs[r - 1] + counts[r - 1];
        }
        std::vector<std::int64_t> all(static_cast<std::size_t>(displs.back()) + static_cast<std::size_t>(counts.back()));
        MPI_Allgatherv(mine.data(), n, MPI_INT64_T, all.data(), counts.data(), displs.data(), MPI_INT64_T, comm);
        return all;
    }
#endif
    return mine;
}

template <class T, class S>
void putColumn (std::vector<char>& frame, const std::vector<S>& src, std::size_t lo, std::size_t hi) {
    const std::size_t at = frame.size();
    frame.resize(at + (hi - lo) * sizeof(T));
    for (std::size_t i = lo; i < hi; ++i) {
        const T v = static_cast<T>(src[i]);
        std::memcpy(frame.data() + at + (i - lo) * sizeof(T), &v, sizeof(T));
    }
}

//! Abort if any value of a column does not fit the .bin field it is written to.
template <class T, class S>
void checkFits (const std::vector<S>& v, const char* name) {
    for (const auto x : v) {
        if (static_cast<std::int64_t>(x) < std::numeric_limits<T>::min() ||
            static_cast<std::int64_t>(x) > std::numeric_limits<T>::max()) {
            throw std::runtime_error(std::string("generated ") + name + " value " + std::to_string(static_cast<std::int64_t>(x)) +
                                     " does not fit its .bin field");
        }
    }
}

} // namespace

GeneratedPopulation generatePopulation (const GenerationSettings& settings, int naics_count) {
    GeneratedPopulation out;
    auto t0 = Clock::now(), t = t0;
    const PopulationBundle b(settings.bundle);
    const auto cols = b.strings("solve.constraints");
    const int n_naics = static_cast<int>(b.strings("naics.codes").size());
    if (n_naics != naics_count) {
        throw std::runtime_error("population bundle has " + std::to_string(n_naics) + " NAICS codes, this build expects " +
                                 std::to_string(naics_count));
    }
    const int np = pumaCount(b);
    const int me = amrex::ParallelDescriptor::MyProc(), nranks = amrex::ParallelDescriptor::NProcs();
    const auto owner = splitPumas(b, nranks);
    std::map<std::string, std::vector<double>> injected;
    if (!settings.inject_allocations.empty()) { injected = readAllocations(settings.inject_allocations); }

    // Perturb, solve and place this rank's PUMAs, largest first; gather everyone's placements.
    // Each PUMA is solved on its own at its own shape, so on CPU they run on separate OpenMP
    // threads (each solve single-threaded) without changing any result; they are packed in PUMA
    // order afterwards.
    std::vector<int> todo;
    for (int p = 0; p < np; ++p) {
        if (owner[p] == me) { todo.push_back(p); }
    }
    std::stable_sort(todo.begin(), todo.end(), [&] (int x, int y) {
        return pumaCells(b, x) > pumaCells(b, y);
    });
    const PmedmSolver solver(settings.solver);
    std::vector<Placements> placed(np);
    std::vector<int> iters(np, 0);
    std::vector<std::string> errors(np);
    auto solveAndPlace = [&] (int p) {
        const PmedmProblem prob(b, p);
        std::vector<double> al;
        if (!injected.empty()) {
            const auto it = injected.find(prob.puma);
            if (it == injected.end()) { throw std::runtime_error("no injected allocation for PUMA " + prob.puma); }
            al = it->second;
        } else {
            const auto Y = perturbedTargets(prob, settings.seed, settings.rep);
            const auto logq = perturbedLogPrior(prob, settings.seed, settings.rep);
            auto res = solver.solve(prob, Y, logq);
            iters[p] = res.iterations;
            al = std::move(res.allocation);
        }
        placed[p] = placePuma(b, prob, al, cols, settings.seed, settings.rep);
    };
    const int n_todo = static_cast<int>(todo.size());
#if defined(AMREX_USE_OMP) && defined(AMREX_USE_GPU)
    // One OpenMP thread per stream; AMReX keeps a stream index per thread, so each thread's
    // kernels, copies and cuBLAS calls go to its own stream. (Its per-thread table is sized by
    // the OpenMP thread count at startup, hence that cap too.)
    const int n_streams =
            std::max(1, std::min({settings.gpu_streams, amrex::Gpu::numGpuStreams(), amrex::OpenMP::get_max_threads(), n_todo}));
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_streams) if (n_streams > 1)
#elif defined(AMREX_USE_OMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int q = 0; q < n_todo; ++q) {
#if defined(AMREX_USE_OMP) && defined(AMREX_USE_GPU)
        amrex::Gpu::Device::setStreamIndex(amrex::OpenMP::get_thread_num());
#endif
        try {
            solveAndPlace(todo[q]);
        } catch (const std::exception& e) { errors[todo[q]] = e.what(); }
#if defined(AMREX_USE_OMP) && defined(AMREX_USE_GPU)
        amrex::Gpu::Device::resetStreamIndex();
#endif
    }
    for (const auto& e : errors) {
        if (!e.empty()) { throw std::runtime_error(e); }
    }
    std::vector<std::int64_t> mine;
    int iterations = 0;
    for (int p = 0; p < np; ++p) {
        if (owner[p] != me) { continue; }
        const auto& pl = placed[p];
        iterations += iters[p];
        mine.push_back(p);
        mine.push_back(static_cast<std::int64_t>(pl.size()));
        mine.insert(mine.end(), pl.bg.begin(), pl.bg.end());
        mine.insert(mine.end(), pl.donor.begin(), pl.donor.end());
        mine.insert(mine.end(), pl.count.begin(), pl.count.end());
    }
    amrex::ParallelDescriptor::ReduceIntSum(iterations);
    const auto all = allGather(mine);
    const double t_solve = since(t); // through the gather: the slowest rank's solves
    std::vector<Placements> per_puma(np);
    for (std::size_t q = 0; q < all.size();) {
        const auto p = all[q], n = all[q + 1];
        auto& pl = per_puma[p];
        const auto* base = all.data() + q + 2;
        pl.bg.assign(base, base + n);
        pl.donor.assign(base + n, base + 2 * n);
        pl.count.assign(base + 2 * n, base + 3 * n);
        q += 2 + 3 * static_cast<std::size_t>(n);
    }
    Placements pl;
    for (const auto& x : per_puma) {
        pl.append(x);
    }
    // Per-stage digests (generate_exaepi.py's, for bit-for-bit checks) cost seconds of hashing at
    // state scale, so only on request.
    const bool sd = settings.stage_digests;
    if (sd) { out.digests["placements"] = placementDigest(pl); }

    // Stages S0-S10, identically on every rank (S3's industry fills split over the ranks). Each
    // stage's time stops before its digest, if any, is taken.
    std::vector<std::pair<const char*, double>> stage_times;
    auto ts = Clock::now();
    auto lap = [&] (const char* name) {
        stage_times.emplace_back(name, since(ts));
    };
    auto P = buildPersons(b, pl, settings.seed, settings.rep);
    lap("S0-S2");
    pl = Placements();
    if (sd) { out.digests["S0-S2 persons"] = personsDigest(P); }
    const SizeTables tables(b);
    Partition part;
    part.rank = me;
    part.nranks = nranks;
    part.allgather = allGather;
    ts = Clock::now();
    auto work = allocateWorkers(b, P, tables, settings.seed, settings.rep, nullptr, &part);
    lap("S3");
    if (sd) { out.digests["S3 workers"] = Digest().add(work).hex16(); }
    ts = Clock::now();
    auto school = allocateStudents(b, P, work, settings.seed, settings.rep);
    lap("S4");
    if (sd) { out.digests["S4 students"] = Digest().add(school).add(work).add(P.grade).hex16(); }
    ts = Clock::now();
    allocateTeachers(b, P, work, school, settings.seed, settings.rep);
    lap("S5");
    if (sd) { out.digests["S5 teachers"] = Digest().add(school).add(work).add(P.grade).hex16(); }
    ts = Clock::now();
    const auto G = assignGroups(b, P, work, school, tables, settings.seed, settings.rep, sd ? &out.digests : nullptr);
    lap(sd ? "S6-S10 (with digests)" : "S6-S10");
    const double t_stages = since(t);

    // The .bin's view of it: block-group index and columnar agent frames.
    const std::size_t n = P.size();
    out.num_agents = static_cast<std::int64_t>(n);
    checkFits<std::int32_t>(G.school_class_group, "school_class_group");
    checkFits<std::int32_t>(G.work_group, "work_group");
    checkFits<std::int16_t>(P.h, "household_id");
    checkFits<std::int16_t>(G.school_id, "school_id");
    checkFits<std::int16_t>(G.nborhood, "nborhood");
    checkFits<std::int16_t>(G.work_nborhood, "work_nborhood");
    checkFits<std::int16_t>(G.workgroup, "workgroup");
    checkFits<std::int16_t>(G.hh_cluster, "hh_cluster");
    checkFits<std::int16_t>(G.school_class, "school_class");
    checkFits<std::int8_t>(P.age, "age");
    checkFits<std::int8_t>(P.grade, "grade");

    // Home block groups: the runs of the (sorted) block-group column.
    std::vector<std::size_t> home_lo;
    for (std::size_t i = 0; i < n; ++i) {
        if (i == 0 || P.bg[i] != P.bg[i - 1]) { home_lo.push_back(i); }
    }
    const std::size_t n_homes = home_lo.size();
    home_lo.push_back(n);
    auto isHome = [&] (std::int64_t geoid) {
        const auto end = home_lo.begin() + static_cast<std::ptrdiff_t>(n_homes);
        const auto h = std::lower_bound(home_lo.begin(), end, geoid, [&] (std::size_t lo, std::int64_t g) {
            return P.bg[lo] < g;
        });
        return h != end && P.bg[*h] == geoid;
    };
    // Workers per destination and NAICS, destinations in ascending geoid order.
    const std::size_t nw = static_cast<std::size_t>(n_naics) + 1;
    std::unordered_map<std::int64_t, std::size_t> dest_ix;
    std::vector<std::int64_t> dests;
    for (std::size_t i = 0; i < n; ++i) {
        if (dest_ix.emplace(work[i], dests.size()).second) { dests.push_back(work[i]); }
    }
    std::vector<std::int64_t> sorted_dests(dests);
    std::sort(sorted_dests.begin(), sorted_dests.end());
    for (std::size_t k = 0; k < sorted_dests.size(); ++k) {
        dest_ix[sorted_dests[k]] = k; // now the rank in geoid order
    }
    std::vector<int> work_pops(sorted_dests.size() * nw, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (P.naics[i] == -1) { continue; }
        int* wp = &work_pops[dest_ix.at(work[i]) * nw];
        ++wp[0];
        ++wp[static_cast<std::size_t>(P.naics[i]) + 1];
    }
    auto workPops = [&] (std::int64_t geoid) {
        const auto it = std::lower_bound(sorted_dests.begin(), sorted_dests.end(), geoid);
        if (it == sorted_dests.end() || *it != geoid) { return std::vector<int>(nw, 0); }
        const auto* wp = &work_pops[static_cast<std::size_t>(it - sorted_dests.begin()) * nw];
        return std::vector<int>(wp, wp + nw);
    };
    // Work-only block groups first (ascending geoid), then the home ones.
    for (const auto geoid : sorted_dests) {
        if (isHome(geoid)) { continue; }
        GeneratedBlockGroup bg;
        bg.geoid = geoid;
        bg.work_populations = workPops(geoid);
        out.block_groups.push_back(std::move(bg));
    }
    const std::size_t first_home = out.block_groups.size();
    out.block_groups.resize(first_home + n_homes);
    std::vector<std::string> frame_hashes(n_homes);
    // Frames are independent, so built (and hashed) in parallel.
    const auto nh64 = static_cast<std::int64_t>(n_homes);
#ifdef AMREX_USE_OMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
    for (std::int64_t k = 0; k < nh64; ++k) {
        const std::size_t lo = home_lo[k], hi = home_lo[k + 1];
        auto& bg = out.block_groups[first_home + static_cast<std::size_t>(k)];
        bg.geoid = P.bg[lo];
        bg.home_population = static_cast<int>(hi - lo);
        bg.work_populations = workPops(bg.geoid);
        auto& f = bg.frame;
        std::vector<std::int64_t> id(hi - lo);
        std::iota(id.begin(), id.end(), static_cast<std::int64_t>(lo));
        putColumn<std::int64_t>(f, id, 0, hi - lo);
        putColumn<std::int64_t>(f, P.bg, lo, hi);
        putColumn<std::int64_t>(f, work, lo, hi);
        putColumn<std::int32_t>(f, G.school_class_group, lo, hi);
        putColumn<std::int32_t>(f, G.work_group, lo, hi);
        putColumn<std::int16_t>(f, P.naics, lo, hi);
        putColumn<std::int16_t>(f, P.h, lo, hi);
        putColumn<std::int16_t>(f, G.school_id, lo, hi);
        putColumn<std::int16_t>(f, G.nborhood, lo, hi);
        putColumn<std::int16_t>(f, G.work_nborhood, lo, hi);
        putColumn<std::int16_t>(f, G.workgroup, lo, hi);
        putColumn<std::int16_t>(f, G.hh_cluster, lo, hi);
        putColumn<std::int16_t>(f, G.school_class, lo, hi);
        putColumn<std::int8_t>(f, P.age, lo, hi);
        putColumn<std::int8_t>(f, P.sex, lo, hi);
        putColumn<std::int8_t>(f, P.race, lo, hi);
        putColumn<std::int8_t>(f, P.travel, lo, hi);
        putColumn<std::int8_t>(f, P.veh_occ, lo, hi);
        putColumn<std::int8_t>(f, P.grade, lo, hi);
        frame_hashes[static_cast<std::size_t>(k)] = frameHash(f.data(), f.size());
    }
    out.digest = populationDigest(frame_hashes);
    lap("frames");

    amrex::Print() << "Generated population from " << settings.bundle << " (seed " << settings.seed << ", rep " << settings.rep
                   << "): " << n << " agents in " << n_homes << " block groups; "
                   << (injected.empty() ? "solve+place " : "injected allocations, place ") << t_solve << " s"
                   << (injected.empty() ? " (" + std::to_string(iterations) + " iterations over " + std::to_string(np) + " PUMAs)"
                                        : "")
                   << ", stages " << t_stages << " s, total " << since(t0) << " s; digest " << out.digest << "\n";
    if (settings.verbose >= 1) {
        amrex::Print() << "  stage times:";
        for (const auto& [name, secs] : stage_times) {
            amrex::Print() << " " << name << " " << secs << " s";
        }
        amrex::Print() << "\n";
    }
    if (sd) {
        for (const auto& [stage, d] : out.digests) {
            amrex::Print() << "  " << stage << ": " << d << "\n";
        }
    }
    return out;
}

void writePopulationBin (const GeneratedPopulation& pop, const std::string& path, std::uint32_t format_version,
                         std::uint32_t record_size) {
    constexpr std::uint32_t MAGIC = 0x55504F50, CODEC_DEFLATE = 1;
    constexpr int LEVEL = 6; // upop_to_exaepi.py COMPRESS_LEVEL
    std::ofstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot write " + path); }
    auto put = [&] (const auto& v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    const auto num_naics = static_cast<std::uint32_t>(pop.block_groups.front().work_populations.size() - 1);
    const auto num_geoids = static_cast<std::uint32_t>(pop.block_groups.size());
    const std::uint64_t header = 40, entry = 8 + 8 + 4 + 4 + 4 + 4 * static_cast<std::uint64_t>(num_naics);
    const std::uint64_t index_end = header + entry * num_geoids;
    // Frames first (in memory), so the index can carry their offsets.
    // (Each frame is deflated on its own, so in parallel; the bytes do not depend on it.)
    std::vector<std::vector<unsigned char>> blobs(pop.block_groups.size());
    const auto nbg = static_cast<std::int64_t>(pop.block_groups.size());
    int failed = 0;
#ifdef AMREX_USE_OMP
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : failed)
#endif
    for (std::int64_t k = 0; k < nbg; ++k) {
        const auto& fr = pop.block_groups[k].frame;
        if (fr.empty()) { continue; }
        uLongf len = compressBound(static_cast<uLong>(fr.size()));
        blobs[k].resize(len);
        if (compress2(blobs[k].data(), &len, reinterpret_cast<const Bytef*>(fr.data()), static_cast<uLong>(fr.size()), LEVEL) !=
            Z_OK) {
            ++failed;
        }
        blobs[k].resize(len);
    }
    if (failed) { throw std::runtime_error("deflate failed writing " + path); }
    put(MAGIC);
    put(format_version);
    put(num_naics);
    put(num_geoids);
    put(static_cast<std::uint64_t>(pop.num_agents));
    put(record_size);
    put(CODEC_DEFLATE);
    put(index_end);
    std::uint64_t offset = index_end;
    for (std::size_t k = 0; k < pop.block_groups.size(); ++k) {
        const auto& bg = pop.block_groups[k];
        const bool home = bg.home_population > 0;
        put(static_cast<std::uint64_t>(bg.geoid));
        put(home ? offset : std::uint64_t(0));
        put(static_cast<std::uint32_t>(blobs[k].size()));
        put(static_cast<std::uint32_t>(bg.home_population));
        for (int w : bg.work_populations) {
            put(static_cast<std::uint32_t>(w));
        }
        offset += blobs[k].size();
    }
    for (const auto& blob : blobs) {
        f.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
    }
    if (!f) { throw std::runtime_error("error writing " + path); }
}

} // namespace PopGen
