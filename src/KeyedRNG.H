/*! @file KeyedRNG.H
    \brief KR64: keyed, counter-based random numbers for rank-invariant population generation.

    Every random decision in population generation is a pure function of a key tuple of global,
    stable identifiers -- (seed, replicate, stage, geoid, household, person, ...) -- never a
    position in a shared stream such as amrex::Random. So a generated population does not depend on
    processing order, thread layout, GPU launch configuration or MPI rank count.

    This is a bit-exact port of utilities/UrbanPop-scripts/popgen/kr64.py, which is normative;
    utilities/tests/kr64_kat.cpp checks both against popgen/kr64_vectors.txt. All arithmetic is
    modulo 2^64 and key words are int64 reinterpreted as uint64.

        mix64(z):     z = (z ^ (z >> 30)) * M1;  z = (z ^ (z >> 27)) * M2;  return z ^ (z >> 31)
        absorb(h, w): mix64(h ^ mix64(w + G))
        key(w1..wn):  fold absorb over the words from h = IV
        draw(K, j):   absorb(K, j)

    Stage words come from PopGenStages.H, generated from popgen/stages.py.
*/
#ifndef KEYED_RNG_H_
#define KEYED_RNG_H_

#include <cmath>
#include <cstdint>

#if __has_include(<AMReX_GpuQualifiers.H>)
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#endif
#ifndef AMREX_GPU_HOST_DEVICE
#define AMREX_GPU_HOST_DEVICE
#endif
#ifndef AMREX_FORCE_INLINE
#define AMREX_FORCE_INLINE inline
#endif

namespace PopGen {

namespace KR64Const {
constexpr std::uint64_t G = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t M1 = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t M2 = 0x94D049BB133111EBULL;
constexpr std::uint64_t IV = 0x243F6A8885A308D3ULL;
} // namespace KR64Const

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::uint64_t mix64 (std::uint64_t z) noexcept {
    z = (z ^ (z >> 30)) * KR64Const::M1;
    z = (z ^ (z >> 27)) * KR64Const::M2;
    return z ^ (z >> 31);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::uint64_t absorb (std::uint64_t h, std::int64_t w) noexcept {
    return mix64(h ^ mix64(static_cast<std::uint64_t>(w) + KR64Const::G));
}

/*! \brief A key under construction: start from (seed, rep, stage), then append identifiers.

    KR64 k = KR64(seed, rep, Stage::CHILDCARE).with(geoid).with(h).with(p);
    double u = k.u01(0);
*/
struct KR64 {
    std::uint64_t h = KR64Const::IV;

    AMREX_GPU_HOST_DEVICE KR64() noexcept = default;
    AMREX_GPU_HOST_DEVICE KR64 (std::int64_t seed, std::int64_t rep, std::int64_t stage) noexcept {
        h = absorb(absorb(absorb(h, seed), rep), stage);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    KR64 with (std::int64_t w) const noexcept {
        KR64 k;
        k.h = absorb(h, w);
        return k;
    }

    //! The j-th 64-bit output for this key.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    std::uint64_t u64 (std::int64_t j = 0) const noexcept { return absorb(h, j); }

    //! Uniform double in [0, 1): (x >> 11) * 2^-53, exact.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    double u01 (std::int64_t j = 0) const noexcept { return static_cast<double>(u64(j) >> 11) * 0x1p-53; }

    //! Uniform integer in [0, n), 1 <= n < 2^53: min(n - 1, floor(u01 * n)).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    std::int64_t index (std::int64_t j, std::int64_t n) const noexcept {
        auto i = static_cast<std::int64_t>(std::floor(u01(j) * static_cast<double>(n)));
        return i < n - 1 ? i : n - 1;
    }
};

//! Uniform double in [0, 1) from a raw 64-bit draw.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double u01 (std::uint64_t x) noexcept {
    return static_cast<double>(x >> 11) * 0x1p-53;
}

//! Uniform integer in [0, n) from a raw 64-bit draw.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::int64_t index (std::uint64_t x, std::int64_t n) noexcept {
    auto i = static_cast<std::int64_t>(std::floor(u01(x) * static_cast<double>(n)));
    return i < n - 1 ? i : n - 1;
}

/*! \brief Draw an index with probability proportional to integer weights.

    `cum` holds the inclusive running sum of the weights (cum[m-1] > 0). Returns the smallest i
    with cum[i] > index(x, cum[m-1]); zero-weight entries are never chosen.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::int64_t intCdf (const std::int64_t* cum, std::int64_t m, std::uint64_t x) noexcept {
    std::int64_t r = index(x, cum[m - 1]);
    std::int64_t lo = 0, hi = m - 1;
    while (lo < hi) {
        std::int64_t mid = lo + (hi - lo) / 2;
        if (cum[mid] > r) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

/*! \brief Draw an index with probability proportional to float weights.

    `cum` holds the sequential inclusive running sum of the weights, accumulated in canonical order
    (as numpy.cumsum does). Returns the smallest i with cum[i] > u01(x) * cum[m-1], clamped to m-1.
*/
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
std::int64_t floatCdf (const double* cum, std::int64_t m, std::uint64_t x) noexcept {
    double t = u01(x) * cum[m - 1];
    std::int64_t lo = 0, hi = m;
    while (lo < hi) {
        std::int64_t mid = lo + (hi - lo) / 2;
        if (cum[mid] > t) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo < m - 1 ? lo : m - 1;
}

} // namespace PopGen

#endif
