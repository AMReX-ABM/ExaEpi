/*! @file DiseaseCoupling.H
 *  \brief Contains data and functions for coupling multiple diseases */

#ifndef DISEASE_COUPLING_H
#define DISEASE_COUPLING_H

#include "AgentDefinitions.H"
#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <string>

using namespace amrex;

/*! \brief unwrapped 1D index corresponding to a 2D index */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static int idx1D (const int a_i, /*!< i */
                  const int a_j, /*!< j */
                  const int a_n /*!< size of square 2D array */) {
    return a_n * a_i + a_j;
}

/*! \brief Disease coupling parameters
 *
 * Structure containing disease coupling parameters */
template <typename PTDType>
struct DiseaseCouplingParm {
    /*! Number of diseases */
    int m_n;
    /*! Coimmunity matrix */
    Vector<Real> m_coimmune_mat;
    /*! Co-susceptibility matrix */
    Vector<Real> m_cosuscept_mat;

    /*! Coimmunity matrix (GPU) */
    Gpu::DeviceVector<Real> m_coimmune_mat_d;
    /*! Co-susceptibility matrix (GPU) */
    Gpu::DeviceVector<Real> m_cosuscept_mat_d;

    /*! Constructor */
    DiseaseCouplingParm (const int a_n) {
        AMREX_ALWAYS_ASSERT(a_n > 0);
        m_n = a_n;

        m_coimmune_mat.resize(m_n * m_n);
        m_cosuscept_mat.resize(m_n * m_n);
        m_coimmune_mat_d.resize(m_n * m_n);
        m_cosuscept_mat_d.resize(m_n * m_n);

        /* default - no cross-immunity */
        for (int i = 0; i < m_n; i++) {
            for (int j = 0; j < m_n; j++) {
                if (i == j) {
                    m_coimmune_mat[idx1D(i, j, m_n)] = 1.0;
                } else {
                    m_coimmune_mat[idx1D(i, j, m_n)] = 0.0;
                }
            }
        }
        /* default - no co-susceptibility */
        for (int i = 0; i < m_n; i++) {
            for (int j = 0; j < m_n; j++) {
                m_cosuscept_mat[idx1D(i, j, m_n)] = 1.0;
            }
        }
    }

    /*! Read inputs from file */
    inline void ReadInputs (const std::string& a_prefix /*!< prefix */) {
        ParmParse pp(a_prefix);
        {
            auto key_str = "coimmunity_matrix";
            if (pp.contains(key_str)) {
                AMREX_ALWAYS_ASSERT(pp.countval(key_str) == m_n * m_n);
                pp.getarr(key_str, m_coimmune_mat);
            }
        }
        {
            auto key_str = "cosusceptibility_matrix";
            if (pp.contains(key_str)) {
                AMREX_ALWAYS_ASSERT(pp.countval(key_str) == m_n * m_n);
                pp.getarr(key_str, m_cosuscept_mat);
            }
        }

        Print() << "Co-immunity matrix:\n";
        for (int i = 0; i < m_n; i++) {
            Print() << "    ";
            for (int j = 0; j < m_n; j++) {
                Print() << m_coimmune_mat[idx1D(i, j, m_n)] << "    ";
            }
            Print() << "\n";
        }
        Print() << "Co-susceptibility matrix:\n";
        for (int i = 0; i < m_n; i++) {
            Print() << "    ";
            for (int j = 0; j < m_n; j++) {
                Print() << m_cosuscept_mat[idx1D(i, j, m_n)] << "    ";
            }
            Print() << "\n";
        }

        // copy to GPU
        Gpu::copy(Gpu::hostToDevice, m_coimmune_mat.begin(), m_coimmune_mat.end(), m_coimmune_mat_d.begin());
        Gpu::copy(Gpu::hostToDevice, m_cosuscept_mat.begin(), m_cosuscept_mat.end(), m_cosuscept_mat_d.begin());
    }

    /*! \brief Compute the net immunity for each agent given past infections */
    inline void getCoimmunity (Gpu::DeviceVector<ParticleReal>& a_coimmunity, /*!< Coimmunity array */
                               const int a_d,                                 /*!< Disease index */
                               const PTDType& a_ptd /*!< Particle tile data */) {
        auto np = a_coimmunity.size();
        auto ci_arr = a_coimmunity.data();
        auto coimm_mat_arr = m_coimmune_mat_d.data();
        auto nd = m_n;

        ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
            ci_arr[i] = 0.0;
            for (int d = 0; d < nd; d++) {
                auto status = a_ptd.m_runtime_idata[i0(d) + IntIdxDisease::status][i];
                auto flag_disease = static_cast<ParticleReal>(status == Status::immune);
                auto imm_fac = coimm_mat_arr[idx1D(a_d, d, nd)] * flag_disease;
                ci_arr[i] = std::max((ParticleReal)ci_arr[i], (ParticleReal)imm_fac);
            }
        });
    }

    /*! \brief Compute the net susceptibility for each agent given current infections */
    inline void getCosusceptibility (Gpu::DeviceVector<ParticleReal>& a_cosusceptibility, /*!< Cosusceptibility array */
                                     const int a_d,                                       /*!< Disease index */
                                     const PTDType& a_ptd /*!< Particle tile data */) {
        auto np = a_cosusceptibility.size();
        auto cs_arr = a_cosusceptibility.data();
        auto cosus_mat_arr = m_cosuscept_mat_d.data();
        auto nd = m_n;

        ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
            cs_arr[i] = 1.0;
            for (int d = 0; d < nd; d++) {
                if (d == a_d) { continue; }
                auto status = a_ptd.m_runtime_idata[i0(d) + IntIdxDisease::status][i];
                auto counter = a_ptd.m_runtime_rdata[r0(d) + RealIdxDisease::disease_counter][i];
                auto incub = a_ptd.m_runtime_rdata[r0(d) + RealIdxDisease::incubation_period][i];
                if ((status == Status::infected) && (counter >= incub)) { cs_arr[i] *= cosus_mat_arr[idx1D(a_d, d, nd)]; }
            }
        });
    }
};

#endif
