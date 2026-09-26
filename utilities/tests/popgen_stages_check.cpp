// Check src/PopulationGenerator bit for bit against the Python reference generator.
//
// Make the reference first, from data/UrbanPop:
//   python generate_exaepi.py --bundle B --seed S --save-alloc X.alloc --bin X
// which writes the solved allocations (X.alloc) and the per-stage digests (X.digests.json). Then,
// from the repository root:
//   g++ -std=c++17 -O2 -ffp-contract=off -Isrc utilities/tests/popgen_stages_check.cpp
//       src/PopulationGenerator.cpp src/PmedmProblem.cpp src/PopulationBundle.cpp -lz -o /tmp/psc
//   /tmp/psc B S REP X.alloc X.digests.json
//
// The C++ stages run on the injected allocations (the solver is checked separately, by
// pmedm_check), and each stage's digest must equal the reference's. Exits non-zero on a mismatch.

#include "PmedmProblem.H"
#include "PopulationBundle.H"
#include "PopulationGenerator.H"
#include "Sha256.H"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::map<std::string, std::vector<double>> readAllocations (const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open " + path); }
    std::map<std::string, std::vector<double>> out;
    std::int32_t n;
    while (f.read(reinterpret_cast<char*>(&n), 4)) {
        std::string puma(n, '\0');
        f.read(&puma[0], n);
        std::int32_t dg[2];
        f.read(reinterpret_cast<char*>(dg), 8);
        std::vector<double> al(static_cast<std::size_t>(dg[0]) * dg[1]);
        f.read(reinterpret_cast<char*>(al.data()), static_cast<std::streamsize>(al.size() * 8));
        out[puma] = std::move(al);
    }
    return out;
}

//! The string values of a flat JSON object ("key": "value" pairs; numbers are skipped).
std::map<std::string, std::string> readDigests (const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    std::map<std::string, std::string> out;
    std::size_t i = 0;
    while ((i = s.find('"', i)) != std::string::npos) {
        const std::size_t ke = s.find('"', i + 1);
        const std::string key = s.substr(i + 1, ke - i - 1);
        std::size_t c = s.find_first_not_of(" :\n", ke + 1);
        if (c != std::string::npos && s[c] == '"') {
            const std::size_t ve = s.find('"', c + 1);
            out[key] = s.substr(c + 1, ve - c - 1);
            i = ve + 1;
        } else {
            i = ke + 1;
        }
    }
    return out;
}

} // namespace

int main (int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: popgen_stages_check BUNDLE SEED REP ALLOC DIGESTS.json\n";
        return 2;
    }
    try {
        using clock = std::chrono::steady_clock;
        const PopGen::PopulationBundle b(argv[1]);
        const std::int64_t seed = std::stoll(argv[2]), rep = std::stoll(argv[3]);
        const auto allocs = readAllocations(argv[4]);
        const auto ref = readDigests(argv[5]);
        const auto cols = b.strings("solve.constraints");
        bool ok = true;
        auto check = [&] (const std::string& stage, const std::string& got, double secs) {
            const auto it = ref.find(stage);
            const bool same = it != ref.end() && it->second == got;
            ok &= same;
            std::cout << (same ? "  ok   " : "  FAIL ") << stage << ": " << got
                      << (same ? "" : " (reference " + (it == ref.end() ? std::string("missing") : it->second) + ")") << "  "
                      << secs << " s\n";
        };

        auto t0 = clock::now();
        PopGen::Placements pl;
        for (int p = 0; p < PopGen::pumaCount(b); ++p) {
            const PopGen::PmedmProblem prob(b, p);
            const auto it = allocs.find(prob.puma);
            if (it == allocs.end()) { continue; }
            pl.append(PopGen::placePuma(b, prob, it->second, cols, seed, rep));
        }
        auto secs = [&] (clock::time_point& t) {
            const auto now = clock::now();
            const double s = std::chrono::duration<double>(now - t).count();
            t = now;
            return s;
        };
        check("placements", PopGen::placementDigest(pl), secs(t0));
        const auto P = PopGen::buildPersons(b, pl, seed, rep);
        check("S0-S2 persons", PopGen::personsDigest(P), secs(t0));
        const PopGen::SizeTables tables(b);
        PopGen::WorkerStats ws;
        auto work = PopGen::allocateWorkers(b, P, tables, seed, rep, &ws);
        check("S3 workers", PopGen::Digest().add(work).hex16(), secs(t0));
        std::cout << "    workers " << ws.workers << ", fallback " << ws.fallback << ", unplaceable " << ws.unplaceable
                  << ", one-worker cells " << ws.one_worker_cells << " of " << ws.cells << "\n";
        std::cout << P.size() << " persons; popgen_stages_check " << (ok ? "passed" : "FAILED") << "\n";
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
