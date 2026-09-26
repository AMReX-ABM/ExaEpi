/*! @file PmedmSolver.cpp
    \brief Per-run P-MEDM re-solve; see PmedmSolver.H.
*/
#include "PmedmSolver.H"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>

#include "DenseLinAlg.H"
#include "KeyedRNG.H"
#include "PopGenStages.H"

namespace PopGen {

// ---------------------------------------------------------------------------------------------
// Problem assembly and perturbation (problem.py, perturb.py)
// ---------------------------------------------------------------------------------------------

namespace {
constexpr double CG0_SE_FACTOR = 0.1;
constexpr double TWO_PI = 6.283185307179586;
} // namespace

int pumaCount (const PopulationBundle& b) {
    return static_cast<int>(b.get<std::int64_t>("solve.bg_offset").size()) - 1;
}

PmedmProblem::PmedmProblem (const PopulationBundle& b, int p) {
    puma = b.strings("solve.puma").at(p);
    fips = std::stoll(puma);
    const auto bgo = b.get<std::int64_t>("solve.bg_offset");
    const auto tro = b.get<std::int64_t>("solve.tract_offset");
    const auto dno = b.get<std::int64_t>("solve.donor_offset");
    const std::int64_t g0 = bgo[p], g1 = bgo[p + 1], t0 = tro[p], t1 = tro[p + 1], d0 = dno[p], d1 = dno[p + 1];
    const auto est_bg_all = b.get<double>("solve.est_bg");
    K = static_cast<int>(est_bg_all.shape.at(1));
    G = static_cast<int>(g1 - g0);
    T = static_cast<int>(t1 - t0);
    D = static_cast<int>(d1 - d0);

    const auto geo = b.get<std::int64_t>("solve.bg_geoid");
    const auto tract = b.get<std::int32_t>("solve.bg_tract");
    bg_geoid.assign(geo.begin() + g0, geo.begin() + g1);
    bg_tract.assign(tract.begin() + g0, tract.begin() + g1);
    const auto di = b.get<std::int32_t>("solve.donor_index");
    const auto hs = b.get<std::int16_t>("solve.donor_hh_size");
    donor_index.assign(di.begin() + d0, di.begin() + d1);
    donor_hh_size.assign(hs.begin() + d0, hs.begin() + d1);
    const auto pw = b.get<double>("solve.prior_weight");
    wt.assign(pw.begin() + d0, pw.begin() + d1);

    const auto se_bg_all = b.get<double>("solve.se_bg");
    const auto est_tr_all = b.get<double>("solve.est_tract");
    const auto se_tr_all = b.get<double>("solve.se_tract");
    est_bg.assign(est_bg_all.begin() + g0 * K, est_bg_all.begin() + g1 * K);
    se_bg.assign(se_bg_all.begin() + g0 * K, se_bg_all.begin() + g1 * K);
    est_tract.assign(est_tr_all.begin() + t0 * K, est_tr_all.begin() + t1 * K);
    se_tract.assign(se_tr_all.begin() + t0 * K, se_tr_all.begin() + t1 * K);

    // Donor constraint matrix from the global CSR rows d0 .. d1.
    const auto ip = b.get<std::int64_t>("solve.c_indptr");
    const auto ix = b.get<std::uint8_t>("solve.c_indices");
    const auto cv = b.get<double>("solve.c_values");
    C.assign(static_cast<std::size_t>(D) * K, 0.0);
    for (int d = 0; d < D; ++d) {
        for (std::int64_t j = ip[d0 + d]; j < ip[d0 + d + 1]; ++j) {
            C[static_cast<std::size_t>(d) * K + ix[j]] = cv[j];
        }
    }

    N = 0;
    for (double w : wt) {
        N += w;
    }
    const double scale = static_cast<double>(D) / (N * N);
    V.assign(nDual(), 0.0);
    for (int k = 0; k < K; ++k) {
        double s = 0;
        for (int g = 0; g < G; ++g) {
            s += se_bg[g * K + k] * se_bg[g * K + k];
        }
        const double sg0 = CG0_SE_FACTOR * std::sqrt(s);
        V[k] = sg0 * sg0 * scale;
        for (int t = 0; t < T; ++t) {
            V[K + k * T + t] = se_tract[t * K + k] * se_tract[t * K + k] * scale;
        }
        for (int g = 0; g < G; ++g) {
            V[K + K * T + k * G + g] = se_bg[g * K + k] * se_bg[g * K + k] * scale;
        }
    }
}

std::vector<double> pmedmTargets (const PmedmProblem& prob, const std::vector<double>& est_bg,
                                  const std::vector<double>& est_tract) {
    const int K = prob.K, T = prob.T, G = prob.G;
    std::vector<double> Y(prob.nDual(), 0.0);
    for (int k = 0; k < K; ++k) {
        double s = 0;
        for (int g = 0; g < G; ++g) {
            s += est_bg[g * K + k];
        }
        Y[k] = s / prob.N;
        for (int t = 0; t < T; ++t) {
            Y[K + k * T + t] = est_tract[t * K + k] / prob.N;
        }
        for (int g = 0; g < G; ++g) {
            Y[K + K * T + k * G + g] = est_bg[g * K + k] / prob.N;
        }
    }
    return Y;
}

std::vector<double> perturbedTargets (const PmedmProblem& prob, std::int64_t seed, std::int64_t rep,
                                      std::vector<double>* est_bg_out) {
    const int K = prob.K, T = prob.T, G = prob.G;
    const KR64 base = KR64(seed, rep, Stage::PERTURB_TARGET).with(prob.fips);
    std::vector<double> y2(static_cast<std::size_t>(G) * K);
    for (int g = 0; g < G; ++g) {
        for (int k = 0; k < K; ++k) {
            const std::size_t i = static_cast<std::size_t>(g) * K + k;
            const double est = prob.est_bg[i], se = prob.se_bg[i];
            if (!(est > 0)) {
                y2[i] = 0.0; // a zero estimate stays zero
                continue;
            }
            // Box-Muller on draws 0 and 1, then a log-normal with mean est and sd se.
            const KR64 key = base.with(k).with(prob.bg_geoid[g]);
            const double z = std::sqrt(-2.0 * std::log1p(-key.u01(0))) * std::cos(TWO_PI * key.u01(1));
            const double r = se / est;
            const double s2 = std::log1p(r * r);
            y2[i] = std::exp(std::log(est) - 0.5 * s2 + std::sqrt(s2) * z);
        }
    }
    // Tracts rebuilt as published tract + (perturbed - published) summed over their block groups.
    std::vector<double> y1(prob.est_tract);
    std::vector<double> delta(static_cast<std::size_t>(T) * K, 0.0);
    for (int g = 0; g < G; ++g) {
        for (int k = 0; k < K; ++k) {
            delta[prob.bg_tract[g] * K + k] += y2[g * K + k] - prob.est_bg[g * K + k];
        }
    }
    for (std::size_t i = 0; i < y1.size(); ++i) {
        y1[i] = std::max(y1[i] + delta[i], 0.0);
    }
    if (est_bg_out) { *est_bg_out = y2; }
    return pmedmTargets(prob, y2, y1);
}

std::vector<double> perturbedLogPrior (const PmedmProblem& prob, std::int64_t seed, std::int64_t rep) {
    const KR64 base = KR64(seed, rep, Stage::PERTURB_PRIOR).with(prob.fips);
    std::vector<double> w(prob.D);
    double total = 0;
    for (int d = 0; d < prob.D; ++d) {
        w[d] = prob.wt[d] * -std::log1p(-base.with(d).u01(0)); // weight x Exp(1)
        total += w[d];
    }
    for (auto& x : w) {
        x = std::log(x / total / prob.G);
    }
    return w;
}

// ---------------------------------------------------------------------------------------------
// Solver (solver.py)
// ---------------------------------------------------------------------------------------------

namespace {

constexpr int HIST = 10;
constexpr int REFRESH = 100;
constexpr double C1 = 1e-4;
constexpr int NSTEPS = 12;
constexpr int MT = 256;
constexpr int MAX_BLOCKS = 1024;
constexpr int MAX_NV = 16;

// Device scalars (doubles) and integer state.
enum Scal { S_GD, S_LIN, S_QUAD, S_USE, S_T, S_GAMMA, S_COUNT };
enum IState { I_N, I_HEAD, I_SLOT, I_KEEP, I_COUNT };

/*! Deterministic sums of NV per-element terms: f(i, acc) adds element i's terms into acc[0 .. NV).
    Two passes with a block count set by n alone, and fixed-order block reductions, so the result
    is the same on every run of a given device and size. out is device memory. */
template <int NV, class F>
void deviceSums (int n, F const& f, double* out, double* partials) {
    static_assert(NV <= MAX_NV);
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    const int nb = std::max(1, std::min((n + MT - 1) / MT, MAX_BLOCKS));
    const auto stream = amrex::Gpu::gpuStream();
    amrex::launch<MT>(nb, stream, [=] AMREX_GPU_DEVICE () noexcept {
        double acc[NV];
        for (int v = 0; v < NV; ++v) {
            acc[v] = 0.0;
        }
        for (int i = static_cast<int>(blockIdx.x) * MT + static_cast<int>(threadIdx.x); i < n; i += nb * MT) {
            f(i, acc);
        }
        for (int v = 0; v < NV; ++v) {
            const double s = amrex::Gpu::blockReduceSum<MT>(acc[v]);
            if (threadIdx.x == 0) { partials[blockIdx.x * NV + v] = s; }
        }
    });
    amrex::launch<MT>(1, stream, [=] AMREX_GPU_DEVICE () noexcept {
        for (int v = 0; v < NV; ++v) {
            double a = 0.0;
            for (int b = static_cast<int>(threadIdx.x); b < nb; b += MT) {
                a += partials[b * NV + v];
            }
            const double s = amrex::Gpu::blockReduceSum<MT>(a);
            if (threadIdx.x == 0) { out[v] = s; }
        }
    });
#elif defined(AMREX_USE_GPU)
    amrex::ignore_unused(n, f, out, partials);
    amrex::Abort("PmedmSolver: deterministic reductions are implemented for CUDA, HIP and CPU only");
#else
    amrex::ignore_unused(partials);
    double acc[NV] = {};
    for (int i = 0; i < n; ++i) {
        f(i, acc);
    }
    for (int v = 0; v < NV; ++v) {
        out[v] = acc[v];
    }
#endif
}

/*! Deterministic maximum of v(i) over i < n, into out[0] (device memory). */
template <class F>
void deviceMax (int n, F const& v, double* out, double* partials) {
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    const int nb = std::max(1, std::min((n + MT - 1) / MT, MAX_BLOCKS));
    const auto stream = amrex::Gpu::gpuStream();
    amrex::launch<MT>(nb, stream, [=] AMREX_GPU_DEVICE () noexcept {
        double m = -std::numeric_limits<double>::infinity();
        for (int i = static_cast<int>(blockIdx.x) * MT + static_cast<int>(threadIdx.x); i < n; i += nb * MT) {
            m = amrex::max(m, v(i));
        }
        const double r = amrex::Gpu::blockReduceMax<MT>(m);
        if (threadIdx.x == 0) { partials[blockIdx.x] = r; }
    });
    amrex::launch<MT>(1, stream, [=] AMREX_GPU_DEVICE () noexcept {
        double m = -std::numeric_limits<double>::infinity();
        for (int b = static_cast<int>(threadIdx.x); b < nb; b += MT) {
            m = amrex::max(m, partials[b]);
        }
        const double r = amrex::Gpu::blockReduceMax<MT>(m);
        if (threadIdx.x == 0) { out[0] = r; }
    });
#elif defined(AMREX_USE_GPU)
    amrex::ignore_unused(n, v, out, partials);
    amrex::Abort("PmedmSolver: deterministic reductions are implemented for CUDA, HIP and CPU only");
#else
    amrex::ignore_unused(partials);
    double m = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        m = std::max(m, v(i));
    }
    out[0] = m;
#endif
}

/*! gradient = Y + V l - [row sums ; tract sums ; all entries] of dL = C^T w (K x G). */
template <class TL>
void assembleGradient (int nd, int K, int T, int G, const TL* dL, const int* tptr, const int* tbg, const double* Y,
                       const double* V, const double* l, double* gout) {
    const int KT = K * T;
    amrex::ParallelFor(nd, [=] AMREX_GPU_DEVICE (int i) noexcept {
        double s = 0.0;
        if (i < K) {
            for (int g = 0; g < G; ++g) {
                s += static_cast<double>(dL[i * G + g]);
            }
        } else if (i < K + KT) {
            const int k = (i - K) / T, t = (i - K) % T;
            for (int q = tptr[t]; q < tptr[t + 1]; ++q) {
                s += static_cast<double>(dL[k * G + tbg[q]]);
            }
        } else {
            s = static_cast<double>(dL[i - K - KT]);
        }
        gout[i] = Y[i] + V[i] * l[i] - s;
    });
}

template <class T>
void toDevice (const std::vector<T>& h, amrex::Gpu::DeviceVector<T>& d) {
    d.resize(h.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h.begin(), h.end(), d.begin());
}

template <class T>
std::vector<T> toHost (const amrex::Gpu::DeviceVector<T>& d) {
    std::vector<T> h(d.size());
    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, d.begin(), d.end(), h.begin());
    amrex::Gpu::streamSynchronize();
    return h;
}

} // namespace

PmedmResult PmedmSolver::solve (const PmedmProblem& prob, const std::vector<double>& Y_h,
                                const std::vector<double>& logq_h) const {
    using amrex::Gpu::DeviceVector;
    const int K = prob.K, T = prob.T, G = prob.G, D = prob.D, nd = prob.nDual(), KG = K * G, KT = K * T;
    const int DG = D * G;
    if (static_cast<int>(Y_h.size()) != nd || static_cast<int>(logq_h.size()) != D) {
        throw std::invalid_argument("PmedmSolver::solve: Y or logq has the wrong length for PUMA " + prob.puma);
    }
    DenseLinAlg la(amrex::Gpu::gpuStream());

    // Tract membership as CSR (block groups of each tract, ascending) for the gradient's tract sums.
    std::vector<int> tptr_h(T + 1, 0), tbg_h(G);
    for (int g = 0; g < G; ++g) {
        ++tptr_h[prob.bg_tract[g] + 1];
    }
    for (int t = 0; t < T; ++t) {
        tptr_h[t + 1] += tptr_h[t];
    }
    {
        std::vector<int> fill(tptr_h.begin(), tptr_h.end() - 1);
        for (int g = 0; g < G; ++g) {
            tbg_h[fill[prob.bg_tract[g]]++] = g;
        }
    }
    std::vector<float> C32_h(prob.C.begin(), prob.C.end());

    DeviceVector<double> C64, Yv, Vv, logq, l(nd, 0.0), g(nd), gprev(nd), d(nd), E(DG), tmp64(DG), L64(KG);
    DeviceVector<double> W(static_cast<std::size_t>(2 * HIST) * nd, 0.0), gram(4 * HIST * HIST), Wg(2 * HIST), coef(2 * HIST),
            red(MAX_NV), scal(S_COUNT, 0.0), partials(MAX_BLOCKS * MAX_NV);
    DeviceVector<float> C32, w(DG), wprev(DG), Dm(DG), L32(KG), dL(KG);
    DeviceVector<int> tract, tptr, tbg, ist(I_COUNT, 0);
    toDevice(prob.C, C64);
    toDevice(C32_h, C32);
    toDevice(Y_h, Yv);
    toDevice(prob.V, Vv);
    toDevice(logq_h, logq);
    toDevice(prob.bg_tract, tract);
    toDevice(tptr_h, tptr);
    toDevice(tbg_h, tbg);
    amrex::Gpu::streamSynchronize();

    double* pl = l.data();
    double* pd = d.data();
    double* pE = E.data();
    double* pW = W.data();
    double* pred = red.data();
    double* pscal = scal.data();
    double* ppart = partials.data();
    double* pcoef = coef.data();
    float* pw = w.data();
    float* pDm = Dm.data();
    int* pist = ist.data();
    const double* pY = Yv.data();
    const double* pV = Vv.data();
    const double* plogq = logq.data();
    const int* ptract = tract.data();
    const int* ptptr = tptr.data();
    const int* ptbg = tbg.data();
    double* pg = g.data();
    double* pgprev = gprev.data();

    // Leff(v) for a dual vector v, as float64 or float32.
    auto leff64 = [&] (const double* v, double* L) {
        amrex::ParallelFor(KG, [=] AMREX_GPU_DEVICE (int i) noexcept {
            const int k = i / G, gg = i % G;
            L[i] = v[k] + v[K + k * T + ptract[gg]] + v[K + KT + i];
        });
    };
    auto leff32 = [&] (const double* v, float* L) {
        amrex::ParallelFor(KG, [=] AMREX_GPU_DEVICE (int i) noexcept {
            const int k = i / G, gg = i % G;
            L[i] = static_cast<float>(v[k] + v[K + k * T + ptract[gg]] + v[K + KT + i]);
        });
    };
    // E = log q - C Leff(l), exactly, in float64.
    auto exactE = [&] () {
        leff64(pl, L64.data());
        la.gemm(Op::N, Op::N, D, G, K, 1.0, C64.data(), K, L64.data(), G, 0.0, tmp64.data(), G);
        const double* ptmp = tmp64.data();
        amrex::ParallelFor(DG, [=] AMREX_GPU_DEVICE (int i) noexcept {
            pE[i] = plogq[i / G] - ptmp[i];
        });
    };
    // w = softmax(E) in float32: exp of (E - max E) cast to float32, divided by its float64 sum.
    auto weights = [&] () {
        deviceMax(
                DG,
                [=] AMREX_GPU_DEVICE (int i) noexcept {
                    return pE[i];
                },
                pred, ppart);
        amrex::ParallelFor(DG, [=] AMREX_GPU_DEVICE (int i) noexcept {
            pw[i] = std::exp(static_cast<float>(pE[i] - pred[0]));
        });
        deviceSums<1>(
                DG,
                [=] AMREX_GPU_DEVICE (int i, double* acc) noexcept {
                    acc[0] += pw[i];
                },
                pred + 1, ppart);
        amrex::ParallelFor(DG, [=] AMREX_GPU_DEVICE (int i) noexcept {
            pw[i] = pw[i] / static_cast<float>(pred[1]);
        });
    };
    auto grad = [&] (double* gout) {
        la.gemm(Op::T, Op::N, K, G, D, 1.0f, C32.data(), K, pw, G, 0.0f, dL.data(), G);
        assembleGradient(nd, K, T, G, static_cast<const float*>(dL.data()), ptptr, ptbg, pY, pV, pl, gout);
    };

    // -H g by the compact L-BFGS representation. W holds S (rows 0 .. HIST-1) and Y (rows HIST ..
    // 2 HIST-1) as a ring: logical row i (oldest first) is physical row (head + i) % HIST, and rows
    // not yet written are zero, exactly as the unused history rows in solver.py.
    auto direction = [&] () {
        la.gemm(Op::N, Op::T, 2 * HIST, 2 * HIST, nd, 1.0, pW, nd, pW, nd, 0.0, gram.data(), 2 * HIST);
        la.gemm(Op::N, Op::N, 2 * HIST, 1, nd, 1.0, pW, nd, pg, 1, 0.0, Wg.data(), 1);
        const double* pgram = gram.data();
        const double* pWg = Wg.data();
        amrex::single_task([=] AMREX_GPU_DEVICE () noexcept {
            constexpr int H = HIST, H2 = 2 * HIST;
            const int n = pist[I_N], head = pist[I_HEAD];
            int ph[H];
            for (int i = 0; i < H; ++i) {
                ph[i] = (head + i) % H;
            }
            double SY[H][H], YY[H][H], R[H][H], Dg[H], a[H], b[H], u[H], top[H];
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < H; ++j) {
                    SY[i][j] = pgram[ph[i] * H2 + H + ph[j]];
                    YY[i][j] = pgram[(H + ph[i]) * H2 + H + ph[j]];
                }
            }
            for (int i = 0; i < H; ++i) {
                const bool vi = i >= H - n;
                for (int j = 0; j < H; ++j) {
                    const bool vj = j >= H - n;
                    R[i][j] = (vi && vj && j >= i) ? SY[i][j] : 0.0;
                }
                if (!vi) { R[i][i] = 1.0; }
                Dg[i] = vi ? SY[i][i] : 0.0;
            }
            const double sy = SY[H - 1][H - 1], yy = YY[H - 1][H - 1];
            const double gamma = (n > 0 && yy > 0) ? sy / yy : 1e-3;
            for (int i = 0; i < H; ++i) {
                a[i] = pWg[ph[i]];
                b[i] = gamma * pWg[H + ph[i]];
            }
            for (int i = H - 1; i >= 0; --i) { // R u = a
                double s = a[i];
                for (int j = i + 1; j < H; ++j) {
                    s -= R[i][j] * u[j];
                }
                u[i] = s / R[i][i];
            }
            for (int i = 0; i < H; ++i) { // R^T top = Dg u + gamma YY u - b
                double s = Dg[i] * u[i] - b[i];
                double yyu = 0.0;
                for (int j = 0; j < H; ++j) {
                    yyu += YY[i][j] * u[j];
                }
                s += gamma * yyu;
                for (int j = 0; j < i; ++j) {
                    s -= R[j][i] * top[j];
                }
                top[i] = s / R[i][i];
            }
            // d = -(gamma g + S^T top - gamma Y^T u) = -gamma g - W^T coef
            for (int i = 0; i < H; ++i) {
                pcoef[ph[i]] = top[i];
                pcoef[H + ph[i]] = -gamma * u[i];
            }
            pscal[S_GAMMA] = gamma;
        });
        amrex::ParallelFor(nd, [=] AMREX_GPU_DEVICE (int i) noexcept {
            pd[i] = -pscal[S_GAMMA] * pg[i];
        });
        la.gemm(Op::T, Op::N, nd, 1, 2 * HIST, -1.0, pW, nd, pcoef, 1, 1.0, pd, 1);
    };

    auto step = [&] (int it) {
        direction();
        // Descent check (fall back to steepest descent) and the line search's linear and
        // quadratic coefficients, for both d and -g in one pass.
        deviceSums<6>(
                nd,
                [=] AMREX_GPU_DEVICE (int i, double* acc) noexcept {
                    const double yl = pY[i] + pV[i] * pl[i];
                    acc[0] += pg[i] * pd[i];
                    acc[1] += pg[i] * pg[i];
                    acc[2] += yl * pd[i];
                    acc[3] += yl * pg[i];
                    acc[4] += pd[i] * pV[i] * pd[i];
                    acc[5] += pg[i] * pV[i] * pg[i];
                },
                pred, ppart);
        amrex::single_task([=] AMREX_GPU_DEVICE () noexcept {
            const bool use = pred[0] < 0;
            pscal[S_USE] = use ? 1.0 : 0.0;
            pscal[S_GD] = use ? pred[0] : -pred[1];
            pscal[S_LIN] = use ? pred[2] : -pred[3];
            pscal[S_QUAD] = 0.5 * (use ? pred[4] : pred[5]);
        });
        amrex::ParallelFor(nd, [=] AMREX_GPU_DEVICE (int i) noexcept {
            if (pscal[S_USE] == 0.0) { pd[i] = -pg[i]; }
        });
        leff32(pd, L32.data());
        la.gemm(Op::N, Op::N, D, G, K, 1.0f, C32.data(), K, L32.data(), G, 0.0f, pDm, G);

        // All twelve step lengths' log1p arguments in one pass; the choice is made on the device.
        deviceSums<NSTEPS>(
                DG,
                [=] AMREX_GPU_DEVICE (int i, double* acc) noexcept {
                    const float wi = pw[i], Di = pDm[i];
                    float t = 1.0f;
                    for (int j = 0; j < NSTEPS; ++j) {
                        acc[j] += static_cast<double>(wi * std::expm1(-t * Di));
                        t *= 0.5f;
                    }
                },
                pred, ppart);
        amrex::single_task([=] AMREX_GPU_DEVICE () noexcept {
            const double gd = pscal[S_GD], lin = pscal[S_LIN], quad = pscal[S_QUAD];
            double chosen = std::ldexp(1.0, -(NSTEPS - 1));
            double t = 1.0;
            for (int j = 0; j < NSTEPS; ++j) {
                const double dec = t * lin + t * t * quad + std::log1p(pred[j]);
                if (dec <= C1 * t * gd) {
                    chosen = t;
                    break;
                }
                t *= 0.5;
            }
            pscal[S_T] = chosen;
        });

        // s = t d (kept in d), l += s; E updated along D, or recomputed exactly.
        amrex::ParallelFor(nd, [=] AMREX_GPU_DEVICE (int i) noexcept {
            pd[i] = pscal[S_T] * pd[i];
            pl[i] += pd[i];
        });
        if ((it + 1) % REFRESH == 0) {
            exactE();
        } else {
            amrex::ParallelFor(DG, [=] AMREX_GPU_DEVICE (int i) noexcept {
                pE[i] -= pscal[S_T] * static_cast<double>(pDm[i]);
            });
        }
        weights();
        std::swap(pg, pgprev);
        grad(pg);

        // History: keep (s, y) when s.y is safely positive.
        {
            const double* gnew = pg;
            const double* gold = pgprev;
            deviceSums<3>(
                    nd,
                    [=] AMREX_GPU_DEVICE (int i, double* acc) noexcept {
                        const double s = pd[i], y = gnew[i] - gold[i];
                        acc[0] += s * y;
                        acc[1] += s * s;
                        acc[2] += y * y;
                    },
                    pred, ppart);
            amrex::single_task([=] AMREX_GPU_DEVICE () noexcept {
                const bool keep = pred[0] > 1e-12 * std::sqrt(pred[1] * pred[2]);
                pist[I_KEEP] = keep ? 1 : 0;
                if (keep) {
                    pist[I_SLOT] = pist[I_HEAD];
                    pist[I_HEAD] = (pist[I_HEAD] + 1) % HIST;
                    if (pist[I_N] < HIST) { ++pist[I_N]; }
                }
            });
            amrex::ParallelFor(nd, [=] AMREX_GPU_DEVICE (int i) noexcept {
                if (pist[I_KEEP]) {
                    const int slot = pist[I_SLOT];
                    pW[static_cast<std::size_t>(slot) * nd + i] = pd[i];
                    pW[static_cast<std::size_t>(HIST + slot) * nd + i] = gnew[i] - gold[i];
                }
            });
        }
    };

    exactE();
    weights();
    grad(pg);
    int it = 0;
    double moved = std::numeric_limits<double>::infinity();
    const int check = std::max(1, m_settings.check_every);
    while (moved > m_settings.tol_moved && it < m_settings.max_iter) {
        amrex::Gpu::copyAsync(amrex::Gpu::deviceToDevice, w.begin(), w.end(), wprev.begin());
        for (int c = 0; c < check; ++c, ++it) {
            step(it);
        }
        // w is normalised, so half its L1 change is the share of households that moved.
        const float* pwp = wprev.data();
        deviceSums<1>(
                DG,
                [=] AMREX_GPU_DEVICE (int i, double* acc) noexcept {
                    acc[0] += 0.5 * std::abs(static_cast<double>(pw[i]) - static_cast<double>(pwp[i]));
                },
                pred, ppart);
        double m = 0;
        amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost, pred, pred + 1, &m);
        amrex::Gpu::streamSynchronize();
        moved = m;
    }

    // Final allocation and gradient norm in float64 at the returned prices.
    exactE();
    deviceMax(
            DG,
            [=] AMREX_GPU_DEVICE (int i) noexcept {
                return pE[i];
            },
            pred, ppart);
    double* pa = tmp64.data();
    amrex::ParallelFor(DG, [=] AMREX_GPU_DEVICE (int i) noexcept {
        pa[i] = std::exp(pE[i] - pred[0]);
    });
    deviceSums<1>(
            DG,
            [=] AMREX_GPU_DEVICE (int i, double* acc) noexcept {
                acc[0] += pa[i];
            },
            pred + 1, ppart);
    amrex::ParallelFor(DG, [=] AMREX_GPU_DEVICE (int i) noexcept {
        pa[i] = pa[i] / pred[1];
    });
    DeviceVector<double> dL64(KG);
    la.gemm(Op::T, Op::N, K, G, D, 1.0, C64.data(), K, pa, G, 0.0, dL64.data(), G);
    assembleGradient(nd, K, T, G, static_cast<const double*>(dL64.data()), ptptr, ptbg, pY, pV, pl, pg);
    deviceSums<1>(
            nd,
            [=] AMREX_GPU_DEVICE (int i, double* acc) noexcept {
                acc[0] += pg[i] * pg[i];
            },
            pred + 2, ppart);
    const double N = prob.N;
    amrex::ParallelFor(DG, [=] AMREX_GPU_DEVICE (int i) noexcept {
        pa[i] = pa[i] * N;
    });

    PmedmResult res;
    res.allocation = toHost(tmp64);
    const auto r = toHost(red);
    res.iterations = it;
    res.grad_norm = std::sqrt(r[2]);
    return res;
}

} // namespace PopGen
