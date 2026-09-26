// Solve every PUMA of a bundle with src/PmedmSolver for one (seed, rep) and write the results for
// utilities/tests/pmedm_check.py to compare against the Python oracle (popgen/solver.py).
//
// Built by CMake with -DExaEpi_POPGEN_TESTS=ON as bin/popgen_pmedm_check:
//   popgen_pmedm_check BUNDLE SEED REP OUT.bin [PUMA ...]
//
// OUT.bin holds, per PUMA: int32 name length, name bytes, int32 K T G D iterations, float64
// grad_norm and seconds, then float64 Y (nDual), log q (D) and allocation (D x G).

#include "PmedmSolver.H"
#include "PopulationBundle.H"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_Print.H>

namespace {
template <class T>
void put (std::ofstream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
void putVec (std::ofstream& f, const std::vector<double>& v) {
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
}
} // namespace

int main (int argc, char** argv) {
    // AMReX would read ParmParse inputs from argv; hand it only the program name.
    int one = 1;
    amrex::Initialize(one, argv);
    int rc = 0;
    if (argc < 5) {
        amrex::Print() << "usage: popgen_pmedm_check BUNDLE SEED REP OUT.bin [PUMA ...]\n";
        rc = 2;
    } else {
        const PopGen::PopulationBundle b(argv[1]);
        const std::int64_t seed = std::stoll(argv[2]), rep = std::stoll(argv[3]);
        std::vector<std::string> only(argv + 5, argv + argc);
        std::ofstream out(argv[4], std::ios::binary);
        const PopGen::PmedmSolver solver;
        double total = 0;
        for (int p = 0; p < PopGen::pumaCount(b); ++p) {
            const PopGen::PmedmProblem prob(b, p);
            if (!only.empty() && std::find(only.begin(), only.end(), prob.puma) == only.end()) { continue; }
            const auto t0 = std::chrono::steady_clock::now();
            const auto Y = PopGen::perturbedTargets(prob, seed, rep);
            const auto logq = PopGen::perturbedLogPrior(prob, seed, rep);
            const auto res = solver.solve(prob, Y, logq);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            total += secs;
            amrex::Print() << "  PUMA " << prob.puma << ": " << prob.D << " donors x " << prob.G << " bgs, " << res.iterations
                           << " iterations, |grad| " << res.grad_norm << ", " << secs << " s\n";
            const auto len = static_cast<std::int32_t>(prob.puma.size());
            put(out, len);
            out.write(prob.puma.data(), len);
            for (int v : {prob.K, prob.T, prob.G, prob.D, res.iterations}) {
                put(out, static_cast<std::int32_t>(v));
            }
            put(out, res.grad_norm);
            put(out, secs);
            putVec(out, Y);
            putVec(out, logq);
            putVec(out, res.allocation);
        }
        amrex::Print() << "solve total " << total << " s\n";
    }
    amrex::Finalize();
    return rc;
}
