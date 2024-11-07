/*! @file InteractionModel.H
    \brief Defines interaction model base class */

#ifndef _INTERACTION_MODEL_H_
#define _INTERACTION_MODEL_H_

#include <map>
#include <string>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>
#include "AgentDefinitions.H"

using namespace amrex;

// Count infected in each group and use the number as the exponent for calculating the probability.
// Slow interactions involve binning each group and doing an n^2 evaluation - might be needed for contact tracing
#define FAST_INTERACTIONS

namespace ExaEpi
{
    /*! \brief Name of models */
    namespace InteractionNames
    {
        // The generic model is never used.
        //const std::string generic = "generic";
        const std::string home = "home";
        const std::string work = "work";
        const std::string school = "school";
        const std::string home_nborhood = "home_neighborhood";
        const std::string work_nborhood = "work_neighborhood";
        const std::string transit = "transit";
        const std::string random = "random";
        const std::string airTravel = "airTravel";
    }
}

/*! \brief Base class for defining interaction models
 *
 *  Contains things that are common to all interaction model classes.
 *  + The member data pointers are used to temporarily point to the SoA attributes of
 *    the agents for a given level and MFIter; for the various interaction models
 *    inheriting from this class, InteractionModel::getAgentAttribs() is called
 *    to set these pointers before using them.
 */
template <typename PCType /*!< agent container type */,
          typename PTDType /*!< agent container tile data type */,
          typename PType /*!< agent type */>
class InteractionModel
{
    public:

        /*! \brief null constructor */
        InteractionModel<PCType, PTDType, PType> (bool _fast_bin) : fast_bin(_fast_bin) {}

        /*! \brief default destructor */
        virtual ~InteractionModel<PCType, PTDType, PType> () = default;

        /*! \brief Interact agents for a model */
        virtual void interactAgents(PCType&, MultiFab&) = 0;

        std::pair<amrex::DenseBins<PTDType>*, bool> getBins(const std::pair<int, int> &pair_idx) {
            bool found = false;
            if (m_bins.find(pair_idx) != m_bins.end()) found = true;
            return {&m_bins[pair_idx], found};
        }

        bool fast_bin;

    private:

        std::map<std::pair<int, int>, amrex::DenseBins<PTDType> > m_bins;
};

template <typename PTDType>
struct Binner
{
        Binner(const Geometry &geom, const IntVect &_bin_size, const Box &_box, const int _max_group, const int _bin_idx) :
            bin_size(_bin_size), box(_box), max_group(_max_group), bin_idx(_bin_idx) {
            dxi = geom.InvCellSizeArray();
            plo = geom.ProbLoArray();
            domain = geom.Domain();
        }

        AMREX_GPU_HOST_DEVICE
        unsigned int operator() (const PTDType& ptd, int i) const noexcept {
            Box tbx;
            auto iv = getParticleCell(ptd, i, plo, dxi, domain);
            auto tid = getTileIndex(iv, box, true, bin_size, tbx);
            if (bin_idx != -1) {
                auto group = ptd.m_idata[bin_idx][i];
                return static_cast<unsigned int>(tid * max_group + group);
            } else {
                return static_cast<unsigned int>(tid);
            }
        }

    private:

        GpuArray<Real,AMREX_SPACEDIM> plo;
        GpuArray<Real,AMREX_SPACEDIM> dxi;
        Box domain;
        IntVect bin_size;
        Box box;
        int max_group;
        int bin_idx;
};


/*! Simulate the interactions between agents at home and compute
    the infection probability for each agent:

    + Create bins of agents if not already created (see
      #amrex::GetParticleBin, #amrex::DenseBins):
      + The bin size is 1 cell
      + #amrex::GetParticleBin maps a particle to its bin index
      + amrex::DenseBins::build() creates the bin-sorted array of particle indices and
        the offset array for each bin (where the offset of a bin is its starting location
        in the bin-sorted array of particle indices).

    + For each agent *i* in the bin-sorted array of agents:
      + Find its bin and the range of indices in the bin-sorted array for agents in its bin
      + If the agent is #Status::immune, do nothing.
      + If the agent is #Status::infected with the number of days infected (RealIdxDisease::disease_counter)
        less than the incubation length, do nothing.
      + Else, for each agent *j* in the same bin:
        + If the agent is #Status::immune, do nothing.
        + If the agent is #Status::infected with the number of days infected (RealIdxDisease::disease_counter)
          less than the incubation length, do nothing.
        + Else if *i* is not infected and *j* is infected, compute probability of *i* getting infected
          from *j* (see below).

    Summary of how the probability of agent A getting infected from agent B is computed:
    + Compute infection probability reduction factor from vaccine efficacy (#DiseaseParm::vac_eff)
    + Within family - if their IntIdx::nborhood and IntIdx::family indices are same,
      and the agents are at home:
      + If B is a child, use the appropriate transmission probability (#DiseaseParm::xmit_child_SC or
        #DiseaseParm::xmit_child) depending on whether B goes to school or not (#IntIdx::school)
      + If B is an adult, use the appropriate transmission probability (#DiseaseParm::xmit_adult_SC or
        #DiseaseParm::xmit_adult) depending on whether B works at a school or not (#IntIdx::school)
*/
template <typename IModel, typename AgentContainer, typename PTDType, typename CandidateFunc, typename BinaryInteractionFunc>
void interactAgentsImpl(IModel &interaction_model, AgentContainer& agents, int bin_idx)
{
    BL_PROFILE("interactAgentsimpl");
    int n_disease = agents.numDiseases();
    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    CandidateFunc isCandidate;
    BinaryInteractionFunc binaryInteraction;

    for (int lev = 0; lev < agents.numLevels(); ++lev)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto& ptile = agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto num_tiles = numTilesInBox(mfi.validbox(), true, bin_size);

            int max_group = 0;
            if (bin_idx != -1) {
                max_group = agents.getMaxGroup(bin_idx);
#ifdef DEBUG
                auto bingroup_ptr = soa.GetIntData(bin_idx).data();
                ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                    AMREX_ASSERT(bingroup_ptr[i] <= max_group);
                });
#endif
            }
            max_group++;
            // max group should never be larger than the total number of particles
            AMREX_ALWAYS_ASSERT(max_group <= np);

            // agents are binned by community + another group. These values can change from one time step to the next so they
            // have to be recomputed (rebuild the bins) every time step
            Binner<PTDType> binner(agents.Geom(lev), bin_size, mfi.validbox(), max_group, bin_idx);
            auto [bins_ptr, found] = interaction_model.getBins({mfi.index(), mfi.LocalTileIndex()});
            // If Redistribute() changes the order of particles (the default), then we need to rebuild each time step
            //if (!found) {
                // FIXME: there should be a command-line option to switch between the GPU and serial versions
                // The GPU bin policy is faster, but non-deterministic.
                if (interaction_model.fast_bin)
                    bins_ptr->build(BinPolicy::GPU, np, ptd, num_tiles * max_group, binner);
                else
                    bins_ptr->build(BinPolicy::Serial, np, ptd, num_tiles * max_group, binner);
            //}

            AMREX_ALWAYS_ASSERT(bins_ptr->numBins() >= 0);
            auto inds = bins_ptr->permutationPtr();
            auto offsets = bins_ptr->offsetsPtr();

            for (int d = 0; d < n_disease; d++) {
                auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                auto lparm = agents.getDiseaseParameters_d(d);
                auto lparm_h = agents.getDiseaseParameters_h(d);
                Real scale = 1.0_prt;  // TODO this should vary based on cell
                Real infect = lparm_h->infect * (1.0_rt - lparm_h->vac_eff);

                ParallelFor(bins_ptr->numItems(), [=] AMREX_GPU_DEVICE (int ii) noexcept {
                    auto infectious_i = inds[ii];
                    int i_cell = binner(ptd, infectious_i);
                    auto cell_start = offsets[i_cell];
                    auto cell_stop = offsets[i_cell + 1];
                    AMREX_ALWAYS_ASSERT((Long)infectious_i < np);
                    // Doing infectious first requires an atomic operation for GPUs, but generally requires far fewer operations
                    // because there are usually more susceptible agents than infectious. This can be a large performance
                    // difference for CPU only runs.
                    if (isInfectious(infectious_i, ptd, d) && isCandidate(infectious_i, ptd)) {
                        //Real i_mask = mask_arr(home_i_ptr[i], home_j_ptr[i], 0);
                        for (auto jj = cell_start; jj < cell_stop; ++jj) {
                            auto susceptible_i = inds[jj];
                            AMREX_ALWAYS_ASSERT((Long)susceptible_i < np);
                            if (infectious_i != susceptible_i &&
                                isSusceptible(susceptible_i, ptd, d) &&
                                isCandidate(susceptible_i, ptd)) {
                                ParticleReal prob = 1.0_prt - infect * binaryInteraction(infectious_i, susceptible_i, ptd, lparm, scale);
                                // The atomic operation is needed because we find all the susceptible for each infectious in turn.
                                // It can be eliminated by switching the order of infectious and susceptible.
                                Gpu::Atomic::Multiply(&prob_ptr[susceptible_i], prob);
                            }
                        }
                    }
                });
                Gpu::synchronize();
            }
        }
    }

}


template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real infectProb(const PTDType &ptd, int infectious_i, int susceptible_i, const Real* xmit_SC, const Real* xmit) {
    // school < 0 means a child normally attends school, but not today
    if (ptd.m_idata[IntIdx::school][infectious_i] < 0) { // not attending school, use _SC contacts
        return xmit_SC[ptd.m_idata[IntIdx::age_group][susceptible_i]];
    } else {
        return xmit[ptd.m_idata[IntIdx::age_group][susceptible_i]];
    }
}


template <typename PTDType>
struct GetCommunityIndex
{
        GetCommunityIndex (const Geometry &geom, const Box &_valid_box) :  valid_box(_valid_box) {
            dxi = geom.InvCellSizeArray();
            plo = geom.ProbLoArray();
            domain = geom.Domain();
            bin_size = {AMREX_D_DECL(1, 1, 1)};
        }

        AMREX_GPU_HOST_DEVICE
        int operator() (const PTDType& ptd, int i) const noexcept {
            Box tbx;
            auto iv = getParticleCell(ptd, i, plo, dxi, domain);
            return getTileIndex(iv, valid_box, true, bin_size, tbx);
        }

        AMREX_GPU_HOST_DEVICE
        int max() {
            return numTilesInBox(valid_box, true, bin_size);
        }


    private:

        GpuArray<Real,AMREX_SPACEDIM> plo;
        GpuArray<Real,AMREX_SPACEDIM> dxi;
        Box domain;
        IntVect bin_size;
        Box valid_box;
};



#endif
