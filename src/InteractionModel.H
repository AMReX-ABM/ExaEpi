/*! @file InteractionModel.H
    \brief Interaction model utilities shared by all interaction model classes */

#ifndef _INTERACTION_MODEL_H_
#define _INTERACTION_MODEL_H_

#include "AgentDefinitions.H"
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>

using namespace amrex;

template <typename PTDType>
struct GetCommunityIndex {
    // AMREX_GPU_HOST_DEVICE
    void init (const Geometry& geom, const Box& _valid_box, Array4<int> const& comm_arr) {
        valid_box = _valid_box;
        dxi = geom.InvCellSizeArray();
        plo = geom.ProbLoArray();
        domain = geom.Domain();
        bin_size = {AMREX_D_DECL(1, 1, 1)};

        int max_communities = numTilesInBox(valid_box, true, bin_size);
        comm_to_local_index_d.resize(max_communities, -1);
        auto d_ptr = comm_to_local_index_d.data();

        num_comms = 0;
        Gpu::DeviceScalar<int> num_comms_d(num_comms);
        int* num_comms_ptr = num_comms_d.dataPtr();
        auto bx = valid_box;

        auto bsize = bin_size;
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (comm_arr(i, j, k) != -1) {
                Box tbx;
                auto ti = getTileIndex(IntVect(i, j), bx, true, bsize, tbx);
                d_ptr[ti] = Gpu::Atomic::Add(num_comms_ptr, 1);
            }
        });
        Gpu::synchronize();
        num_comms = num_comms_d.dataValue();
    }

    AMREX_GPU_HOST_DEVICE
    int getIndex (int* d_ptr, const PTDType& ptd, int i) const noexcept {
        Box tbx;
        auto iv = getParticleCell(ptd, i, plo, dxi, domain);

        auto index = d_ptr[getTileIndex(iv, valid_box, true, bin_size, tbx)];

        AMREX_ASSERT(index >= 0);
        AMREX_ASSERT(index < num_comms);
        return index;
    }

    AMREX_GPU_HOST_DEVICE
    int max () { return num_comms; }

    GpuArray<Real, AMREX_SPACEDIM> plo;
    GpuArray<Real, AMREX_SPACEDIM> dxi;
    Box domain;
    IntVect bin_size;
    Box valid_box;
    Gpu::DeviceVector<int> comm_to_local_index_d;
    int num_comms;
};

#ifdef AMREX_USE_CUDA
#define devMemset cudaMemset
#else
#ifdef AMREX_USE_SYCL
#define devMemset(p, v, s) (Gpu::Device::streamQueue().memset((p), (v), (s)).wait())
#else
#ifdef AMREX_USE_HIP
#define devMemset(p, v, s) ((void)hipMemset((p), (v), (s)))
#else
#define devMemset memset
#endif
#endif
#endif

#ifdef AMREX_USE_OMP
#define OMP_MAX_THREADS omp_get_max_threads()
#define OMP_THREAD_NUM omp_get_thread_num()
#else
#define OMP_MAX_THREADS 1
#define OMP_THREAD_NUM 0
#endif

#endif
