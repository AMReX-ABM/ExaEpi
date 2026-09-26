// Check src/DenseLinAlg: the backend GEMM (cuBLAS / rocBLAS / reference kernel) against a host
// long-double reference, bitwise run-to-run reproducibility, and the portable kernel.
//
// Built by CMake with -DExaEpi_POPGEN_TESTS=ON as bin/popgen_gemm_check; run it with no arguments.
// Shapes are the solver's two products for a large NM PUMA (5,000 donors, 123 constraints, 96 block
// groups) -- D = C L (NN) and C^T W (TN) -- plus small odd shapes and beta != 0. Exits non-zero on
// failure.

#include "DenseLinAlg.H"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_Print.H>

using PopGen::Op;

namespace {

// Deterministic values in [-1, 1) from a 64-bit LCG, so failures reproduce.
template <class T>
std::vector<T> fill (std::size_t n, std::uint64_t seed) {
    std::vector<T> v(n);
    std::uint64_t s = seed * 0x9E3779B97F4A7C15ULL + 1;
    for (auto& x : v) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        x = T(double(s >> 11) * 0x1.0p-53 * 2.0 - 1.0);
    }
    return v;
}

template <class T>
struct Dev {
    amrex::Gpu::DeviceVector<T> d;
    explicit Dev (const std::vector<T>& h) : d(h.size()) {
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h.begin(), h.end(), d.begin());
        amrex::Gpu::streamSynchronize();
    }
    std::vector<T> host () const {
        std::vector<T> h(d.size());
        amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, d.begin(), d.end(), h.begin());
        amrex::Gpu::streamSynchronize();
        return h;
    }
};

template <class T>
bool checkShape (PopGen::DenseLinAlg& la, Op ta, Op tb, int M, int N, int K, T beta, const char* label) {
    const int lda = ta == Op::N ? K : M, ldb = tb == Op::N ? N : K, ldc = N;
    const auto hA = fill<T>(std::size_t(M) * K, 1), hB = fill<T>(std::size_t(K) * N, 2), hC = fill<T>(std::size_t(M) * N, 3);
    // Host reference in long double, and the scale sum |a||b| each element's rounding is judged by.
    std::vector<long double> ref(std::size_t(M) * N), mag(std::size_t(M) * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            long double s = 0, m = 0;
            for (int k = 0; k < K; ++k) {
                const long double a = ta == Op::N ? hA[i * lda + k] : hA[k * lda + i];
                const long double b = tb == Op::N ? hB[k * ldb + j] : hB[j * ldb + k];
                s += a * b;
                m += std::fabs(a * b);
            }
            ref[i * ldc + j] = s + (long double)beta * hC[i * ldc + j];
            mag[i * ldc + j] = m + std::fabs((long double)beta * hC[i * ldc + j]);
        }
    }
    const Dev<T> A(hA), B(hB);
    Dev<T> C1(hC), C2(hC), C3(hC);
    la.gemm(ta, tb, M, N, K, T(1), A.d.data(), lda, B.d.data(), ldb, beta, C1.d.data(), ldc);
    la.gemm(ta, tb, M, N, K, T(1), A.d.data(), lda, B.d.data(), ldb, beta, C2.d.data(), ldc);
    PopGen::gemmReference(ta, tb, M, N, K, T(1), A.d.data(), lda, B.d.data(), ldb, beta, C3.d.data(), ldc);
    amrex::Gpu::streamSynchronize();
    const auto r1 = C1.host(), r2 = C2.host(), r3 = C3.host();

    const bool same = std::memcmp(r1.data(), r2.data(), r1.size() * sizeof(T)) == 0;
    const double eps = std::is_same<T, float>::value ? 0x1.0p-24 : 0x1.0p-53;
    double worst = 0, worst_ref = 0; // error in units of eps * K * sum|a||b|
    for (std::size_t e = 0; e < r1.size(); ++e) {
        const double scale = double(mag[e]) * eps * K + 1e-300;
        worst = std::fmax(worst, double(std::fabs(r1[e] - ref[e])) / scale);
        worst_ref = std::fmax(worst_ref, double(std::fabs(r3[e] - ref[e])) / scale);
    }
    const bool ok = same && worst <= 1.0 && worst_ref <= 1.0;
    amrex::Print() << (ok ? "  ok   " : "  FAIL ") << label << (std::is_same<T, float>::value ? " f32 " : " f64 ") << M << "x"
                   << N << "x" << K << " beta " << double(beta) << ": run-to-run " << (same ? "identical" : "DIFFERS")
                   << ", error/bound backend " << worst << ", reference " << worst_ref << "\n";
    return ok;
}

template <class T>
bool checkAll (PopGen::DenseLinAlg& la) {
    bool ok = true;
    ok &= checkShape<T>(la, Op::N, Op::N, 5000, 96, 123, T(0), "D = C L   ");
    ok &= checkShape<T>(la, Op::T, Op::N, 123, 96, 5000, T(0), "C^T W     ");
    ok &= checkShape<T>(la, Op::N, Op::T, 37, 11, 29, T(0.5), "NT odd    ");
    ok &= checkShape<T>(la, Op::T, Op::T, 7, 13, 5, T(-1), "TT odd    ");
    ok &= checkShape<T>(la, Op::N, Op::N, 1, 1, 1, T(2), "1x1x1     ");
    return ok;
}

} // namespace

int main (int argc, char** argv) {
    amrex::Initialize(argc, argv);
    bool ok = true;
    {
        PopGen::DenseLinAlg la;
        amrex::Print() << "DenseLinAlg backend: " << PopGen::DenseLinAlg::backend() << "\n";
        ok &= checkAll<float>(la);
        ok &= checkAll<double>(la);
        amrex::Print() << (ok ? "gemm_check: all passed\n" : "gemm_check: FAILED\n");
    }
    amrex::Finalize();
    return ok ? 0 : 1;
}
