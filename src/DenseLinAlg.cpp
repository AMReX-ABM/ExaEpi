/*! @file DenseLinAlg.cpp
    \brief Dense matrix products for the P-MEDM solver; see DenseLinAlg.H.
*/
#include "DenseLinAlg.H"

#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_GpuLaunch.H>

#if defined(AMREX_USE_CUDA)
#include <cublas_v2.h>
#elif defined(AMREX_USE_HIP)
#if __has_include(<rocblas/rocblas.h>)
#include <rocblas/rocblas.h>
#else
#include <rocblas.h>
#endif
#elif defined(EXAEPI_POPGEN_OPENBLAS)
#include <cblas.h>
#endif

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

namespace PopGen {

namespace {

template <class T>
void referenceImpl (Op ta, Op tb, int M, int N, int K, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc) {
    const bool at = ta == Op::T, bt = tb == Op::T;
#ifdef AMREX_USE_GPU
    // One thread per output element, a sequential sum over k.
    amrex::ParallelFor(M * N, [=] AMREX_GPU_DEVICE (int e) noexcept {
        const int i = e / N, j = e % N;
        T acc = 0;
        for (int k = 0; k < K; ++k) {
            const T a = at ? A[k * lda + i] : A[i * lda + k];
            const T b = bt ? B[j * ldb + k] : B[k * ldb + j];
            acc += a * b;
        }
        C[i * ldc + j] = beta == T(0) ? alpha * acc : alpha * acc + beta * C[i * ldc + j];
    });
#else
    // Row by row, accumulating the row over ascending k: the same per-element summation order as
    // the device kernel, but streaming through B's rows so the inner loop vectorises.
#ifdef AMREX_USE_OMP
#pragma omp parallel for if (!omp_in_parallel())
#endif
    for (int i = 0; i < M; ++i) {
        std::vector<T> acc(N, T(0));
        for (int k = 0; k < K; ++k) {
            const T a = at ? A[k * lda + i] : A[i * lda + k];
            if (bt) {
                for (int j = 0; j < N; ++j) {
                    acc[j] += a * B[j * ldb + k];
                }
            } else {
                const T* b = B + k * ldb;
                for (int j = 0; j < N; ++j) {
                    acc[j] += a * b[j];
                }
            }
        }
        T* c = C + i * ldc;
        for (int j = 0; j < N; ++j) {
            c[j] = beta == T(0) ? alpha * acc[j] : alpha * acc[j] + beta * c[j];
        }
    }
#endif
}

#if defined(AMREX_USE_CUDA)
constexpr std::size_t CUBLAS_WORKSPACE_BYTES = std::size_t(32) << 20;

void check (cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) { amrex::Abort(std::string("cuBLAS ") + what + " failed, status " + std::to_string(int(s))); }
}
cublasOperation_t op (Op o) {
    return o == Op::T ? CUBLAS_OP_T : CUBLAS_OP_N;
}
#elif defined(AMREX_USE_HIP)
void check (rocblas_status s, const char* what) {
    if (s != rocblas_status_success) {
        amrex::Abort(std::string("rocBLAS ") + what + " failed, status " + std::to_string(int(s)));
    }
}
rocblas_operation op (Op o) {
    return o == Op::T ? rocblas_operation_transpose : rocblas_operation_none;
}
#elif defined(EXAEPI_POPGEN_OPENBLAS)
CBLAS_TRANSPOSE op (Op o) {
    return o == Op::T ? CblasTrans : CblasNoTrans;
}
#endif

} // namespace

void gemmReference (Op ta, Op tb, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta,
                    float* C, int ldc) {
    referenceImpl(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemmReference (Op ta, Op tb, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb,
                    double beta, double* C, int ldc) {
    referenceImpl(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

std::mutex& deviceSetupMutex () {
    static std::mutex m;
    return m;
}

DenseLinAlg::DenseLinAlg (Stream stream) : m_stream(stream) {
#if defined(AMREX_USE_CUDA)
    std::lock_guard<std::mutex> lock(deviceSetupMutex());
    cublasHandle_t h;
    check(cublasCreate(&h), "create");
    check(cublasSetStream(h, m_stream), "set stream");
    // Full-precision float32: no TF32 or other reduced-precision tensor-core paths.
    check(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH), "set math mode");
    // A fixed workspace per handle, so algorithm choice cannot depend on what memory is free.
    m_workspace = amrex::The_Arena()->alloc(CUBLAS_WORKSPACE_BYTES);
    check(cublasSetWorkspace(h, m_workspace, CUBLAS_WORKSPACE_BYTES), "set workspace");
    m_handle = h;
#elif defined(AMREX_USE_HIP)
    std::lock_guard<std::mutex> lock(deviceSetupMutex());
    rocblas_handle h;
    check(rocblas_create_handle(&h), "create");
    check(rocblas_set_stream(h, m_stream), "set stream");
    // Some rocBLAS kernels accumulate with atomics by default, which makes results vary from run
    // to run; the solver needs the same product every time.
    check(rocblas_set_atomics_mode(h, rocblas_atomics_not_allowed), "set atomics mode");
    m_handle = h;
#elif defined(EXAEPI_POPGEN_OPENBLAS)
    // Threaded OpenBLAS may partition a product differently with the thread count, and so round
    // differently; one thread per call keeps results independent of it. (Process-wide setting:
    // nothing else in ExaEpi calls BLAS; set once, as solves may construct handles concurrently.)
    static const bool one_thread = (openblas_set_num_threads(1), true);
    amrex::ignore_unused(one_thread);
#endif
}

DenseLinAlg::~DenseLinAlg () {
#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)
    std::lock_guard<std::mutex> lock(deviceSetupMutex());
#endif
#if defined(AMREX_USE_CUDA)
    if (m_handle) { cublasDestroy(static_cast<cublasHandle_t>(m_handle)); }
    if (m_workspace) {
        AMREX_CUDA_SAFE_CALL(cudaStreamSynchronize(m_stream));
        amrex::The_Arena()->free(m_workspace);
    }
#elif defined(AMREX_USE_HIP)
    if (m_handle) { rocblas_destroy_handle(static_cast<rocblas_handle>(m_handle)); }
#endif
}

// Row-major C = op(A) op(B) is column-major C^T = op(B)^T op(A)^T, and a row-major matrix read as
// column-major is its transpose -- so the column-major GPU calls swap the operands and M with N,
// keeping each operand's own transpose flag. CBLAS takes row-major directly.
void DenseLinAlg::gemm (Op ta, Op tb, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb,
                        float beta, float* C, int ldc) {
#if defined(AMREX_USE_CUDA)
    check(cublasSgemm(static_cast<cublasHandle_t>(m_handle), op(tb), op(ta), N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc),
          "sgemm");
#elif defined(AMREX_USE_HIP)
    check(rocblas_sgemm(static_cast<rocblas_handle>(m_handle), op(tb), op(ta), N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc),
          "sgemm");
#elif defined(EXAEPI_POPGEN_OPENBLAS)
    cblas_sgemm(CblasRowMajor, op(ta), op(tb), M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    gemmReference(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

void DenseLinAlg::gemm (Op ta, Op tb, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb,
                        double beta, double* C, int ldc) {
#if defined(AMREX_USE_CUDA)
    check(cublasDgemm(static_cast<cublasHandle_t>(m_handle), op(tb), op(ta), N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc),
          "dgemm");
#elif defined(AMREX_USE_HIP)
    check(rocblas_dgemm(static_cast<rocblas_handle>(m_handle), op(tb), op(ta), N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc),
          "dgemm");
#elif defined(EXAEPI_POPGEN_OPENBLAS)
    cblas_dgemm(CblasRowMajor, op(ta), op(tb), M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    gemmReference(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

const char* DenseLinAlg::backend () {
#if defined(AMREX_USE_CUDA)
    return "cuBLAS (pedantic math)";
#elif defined(AMREX_USE_HIP)
    return "rocBLAS";
#elif defined(EXAEPI_POPGEN_OPENBLAS)
    return "OpenBLAS (one thread per call)";
#else
    return "reference kernel";
#endif
}

} // namespace PopGen
