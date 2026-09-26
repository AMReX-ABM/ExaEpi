/*! @file DenseLinAlg.H
    \brief Dense matrix products for the P-MEDM solver, on the vendor BLAS of each backend.

    The population-generation solver (PmedmSolver) spends almost all its time in two products per
    iteration, D = C Leff(d) and C^T w, with C the donors x constraints matrix. This wrapper gives
    them one row-major interface over:

        CUDA        cuBLAS, CUBLAS_PEDANTIC_MATH (full-precision float32, never TF32), one handle
                    per stream with a fixed workspace
        HIP         rocBLAS
        otherwise   gemmReference, a portable kernel (CPU builds for now: a CPU BLAS is a later
                    addition, and must then be held to one thread per call -- threaded BLAS may
                    split the inner dimension differently with the thread count)

    Determinism: a generated population must not depend on the run's layout, so every product has
    to be bitwise reproducible for a fixed device type and shape. cuBLAS and rocBLAS are, given a
    fixed math mode and workspace (GEMM uses no split-K atomics then). gemmReference sums each
    output element sequentially in ascending k, whatever the thread count.

    All matrices are row-major. Pointers are device memory on GPU builds and host memory otherwise.
*/
#ifndef DENSE_LIN_ALG_H_
#define DENSE_LIN_ALG_H_

#include <AMReX_Gpu.H>

namespace PopGen {

enum class Op { N, T };

#ifdef AMREX_USE_GPU
using Stream = amrex::gpuStream_t;
inline Stream currentStream () {
    return amrex::Gpu::gpuStream();
}
#else
struct Stream {}; //!< CPU builds have no streams
inline Stream currentStream () {
    return {};
}
#endif

/*! Row-major C (M x N) = alpha op(A) op(B) + beta C, with op(A) M x K and op(B) K x N, computed
    by the portable kernel: each C element is one sequential sum over k = 0 .. K-1. */
void gemmReference(Op ta, Op tb, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta,
                   float* C, int ldc);
void gemmReference(Op ta, Op tb, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb,
                   double beta, double* C, int ldc);

class DenseLinAlg {
  public:
    /*! Bind to a stream (the current AMReX stream by default); owns one BLAS handle for it. */
    explicit DenseLinAlg(Stream stream = currentStream());
    ~DenseLinAlg();
    DenseLinAlg(const DenseLinAlg&) = delete;
    DenseLinAlg& operator=(const DenseLinAlg&) = delete;

    /*! Row-major C (M x N) = alpha op(A) op(B) + beta C, asynchronous on this object's stream. */
    void gemm(Op ta, Op tb, int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta,
              float* C, int ldc);
    void gemm(Op ta, Op tb, int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta,
              double* C, int ldc);

    Stream stream () const noexcept { return m_stream; }

    /*! Name of the backend in use, for the log. */
    static const char* backend();

  private:
    Stream m_stream;
    // Unused in CPU builds; [[maybe_unused]] keeps clang's -Wunused-private-field quiet there.
    [[maybe_unused]] void* m_handle = nullptr;    /*!< cublasHandle_t / rocblas_handle, opaque here */
    [[maybe_unused]] void* m_workspace = nullptr; /*!< fixed cuBLAS workspace */
};

} // namespace PopGen

#endif
