/*! @file InteractionModAirTravel.H
 * \brief Contains the class describing agent interactions for air travel
 */

#ifndef _INTERACTION_MOD_AIRTRAVEL_H_
#define _INTERACTION_MOD_AIRTRAVEL_H_

#include "InteractionModel.H"
#include "DiseaseParm.H"
#include "AgentDefinitions.H"

using namespace amrex;

/*! \brief One-on-one interaction between an infectious agent and a susceptible agent.
 *
 * This function defines the one-on-one interaction between an infectious agent and a
 * susceptible agent on air travel. */
template <typename PTDType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
static void binaryInteractionAirTravel ( const int a_i, /*!< Index of infectious agent */
                                      const int a_j, /*!< Index of susceptible agent */
                                      const PTDType& a_ptd_i, /*!< Particle tile data for infectious agent */
                                      const PTDType& a_ptd_j, /*!< Particle tile data for susceptible agent*/
                                      const DiseaseParm* const a_lparm,  /*!< disease paramters */
                                      const Real a_social_scale, /*!< Social scale */
                                      ParticleReal* const a_prob_ptr /*!< infection probability */)
{
    Real infect = a_lparm->infect;
    infect *= a_lparm->vac_eff;

    auto age_group_ptr_j = a_ptd_j.m_idata[IntIdx::age_group];
    auto nborhood_ptr_i = a_ptd_i.m_idata[IntIdx::nborhood];
    auto nborhood_ptr_j = a_ptd_j.m_idata[IntIdx::nborhood];
    auto school_ptr_i = a_ptd_i.m_idata[IntIdx::school];
    auto withdrawn_ptr_i = a_ptd_i.m_idata[IntIdx::withdrawn];
    auto withdrawn_ptr_j = a_ptd_j.m_idata[IntIdx::withdrawn];
    if (withdrawn_ptr_i[a_i] || withdrawn_ptr_j[a_j]) { return; }
    auto random_travel_ptr_i = a_ptd_i.m_idata[IntIdx::random_travel];
    auto random_travel_ptr_j = a_ptd_j.m_idata[IntIdx::random_travel];
    if (random_travel_ptr_i[a_i] || random_travel_ptr_j[a_j]) {return;}
    auto air_travel_ptr_i = a_ptd_i.m_idata[IntIdx::air_travel];
    auto air_travel_ptr_j = a_ptd_j.m_idata[IntIdx::air_travel];
    if (air_travel_ptr_i[a_i] || air_travel_ptr_j[a_j]) {return;}

    //infect *= i_mask;
    //infect *= j_mask;
    ParticleReal prob = 1.0_prt;

    // school < 0 means a child normally attends school, but not today
    /* Should always be in the same community = same cell */
    if (school_ptr_i[a_i] < 0) {  // not attending school, use _SC contacts
        prob *= 1.0_prt - infect * a_lparm->xmit_comm_SC[age_group_ptr_j[a_j]] * a_social_scale;
    } else {
        prob *= 1.0_prt - infect * a_lparm->xmit_comm[age_group_ptr_j[a_j]] * a_social_scale;
    }
    // /* Neighborhood? */
    if (nborhood_ptr_i[a_i] == nborhood_ptr_j[a_j]) {
        if (school_ptr_i[a_i] < 0)  {
            // not attending school, use _SC contacts
            prob *= 1.0_prt - infect * a_lparm->xmit_hood_SC[age_group_ptr_j[a_j]] * a_social_scale;
        } else {
            prob *= 1.0_prt - infect * a_lparm->xmit_hood[age_group_ptr_j[a_j]] * a_social_scale;
        }
    }

    Gpu::Atomic::Multiply(&a_prob_ptr[a_j], prob);
}

/*! \brief Class describing agent interactions for air travel */
template <typename AC, typename ACT, typename ACTD, typename A>
class InteractionModAirTravel : public InteractionModel<AC,ACT,ACTD,A>
{
    public:

        /*! \brief null constructor */
        InteractionModAirTravel() { }

        /*! \brief default destructor */
        virtual ~InteractionModAirTravel() = default;

        /*! \brief Simulate agent interaction for air travel */
        void interactAgents( AC&, MultiFab&) override {
            amrex::Abort("Do not use this interface for this interaction model");
        }

        /*! \brief Simulate agent interaction for air travel */
        void interactAgents( AC&, MultiFab&, AC&) override ;

    protected:

    private:
};

/*! Simulate the interactions for agents on air travel
*/
template <typename AC, typename ACT, typename ACTD, typename A>
void InteractionModAirTravel<AC,ACT,ACTD,A>::interactAgents(AC& a_agents, /*!< Agent container */
                                                         MultiFab& /*a_mask*/, /*!< Masking behavior */
                                                         AC& a_on_air_travel)
{
    BL_PROFILE("InteractionModAirTravel::interactAgents");
    int n_disease = a_agents.numDiseases();

    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    for (int lev = 0; lev < a_agents.numLevels(); ++lev)
    {
        const Geometry& geom = a_agents.Geom(lev);
        const auto dxi = geom.InvCellSizeArray();
        const auto plo = geom.ProbLoArray();
        const auto domain = geom.Domain();

        this->makeBins( a_agents, bin_size, lev, ExaEpi::InteractionNames::home );
        this->makeBins( a_on_air_travel, bin_size, lev, ExaEpi::InteractionNames::airTravel );

        // we make two passes here, in the first case, agents on air travel infect local agents
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto pair_ind = std::make_pair(mfi.index(), mfi.LocalTileIndex());
            auto bins_ptr = a_agents.getBins(pair_ind, ExaEpi::InteractionNames::home);
            auto travel_bins_ptr = a_on_air_travel.getBins(pair_ind, ExaEpi::InteractionNames::airTravel);

            auto& ptile = a_agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            auto& travel_ptile = a_on_air_travel.ParticlesAt(lev, mfi);
            const auto& travel_ptd = travel_ptile.getParticleTileData();

            auto& aos   = ptile.GetArrayOfStructs();
            const auto np = aos.numParticles();
            auto pstruct_ptr = aos().dataPtr();

            auto binner = GetParticleBin{plo, dxi, domain, bin_size, mfi.validbox()};
            AMREX_ALWAYS_ASSERT(bins_ptr->numBins() >= 0);
            AMREX_ALWAYS_ASSERT(travel_bins_ptr->numBins() >= 0);
            auto inds = bins_ptr->permutationPtr();
            auto travel_inds = travel_bins_ptr->permutationPtr();
            auto travel_offsets = travel_bins_ptr->offsetsPtr();

            for (int d = 0; d < n_disease; d++) {

                auto prob_ptr = this->getAgentProbPtr(a_agents,lev,mfi,d);
                //auto mask_arr = a_mask[mfi].array();
                auto lparm = a_agents.getDiseaseParameters_d(d);

                ParallelFor( bins_ptr->numItems(), [=] AMREX_GPU_DEVICE (int ii) noexcept
                {
                    auto i = inds[ii];
                    int i_cell = binner(pstruct_ptr[i]);
                    auto cell_start = travel_offsets[i_cell];
                    auto cell_stop  = travel_offsets[i_cell+1];

                    AMREX_ALWAYS_ASSERT( (Long) i < np);

                    if ( notSusceptible<ACTD>(i, ptd, d) )  { return; }

                    //Real i_mask = mask_arr(home_i_ptr[i], home_j_ptr[i], 0);
                    for (auto jj = cell_start; jj < cell_stop; ++jj) {

                        auto j = travel_inds[jj];
                        AMREX_ALWAYS_ASSERT( (Long) j < np);

                        //Real j_mask = mask_arr(home_i_ptr[j], home_j_ptr[j], 0);
                        //if (i == j) continue;

                        if ( isInfectious<ACTD>(j, travel_ptd, d) ) {
                            Real social_scale = 1.0_prt;  // TODO this should vary based on cell
                            binaryInteractionAirTravel<ACTD>( j, i, travel_ptd, ptd, lparm, social_scale, prob_ptr );
                        }
                    }
                });
                Gpu::synchronize();
            }
        }

        // now, the local agents infect the ones on travel
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = a_agents.MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto pair_ind = std::make_pair(mfi.index(), mfi.LocalTileIndex());
            auto bins_ptr = a_agents.getBins(pair_ind, ExaEpi::InteractionNames::home);
            auto travel_bins_ptr = a_on_air_travel.getBins(pair_ind, ExaEpi::InteractionNames::airTravel);

            auto& ptile = a_agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            auto& travel_ptile = a_on_air_travel.ParticlesAt(lev, mfi);
            const auto& travel_ptd = travel_ptile.getParticleTileData();

            auto& aos   = ptile.GetArrayOfStructs();
            const auto np = aos.numParticles();
            auto pstruct_ptr = aos().dataPtr();

            auto binner = GetParticleBin{plo, dxi, domain, bin_size, mfi.validbox()};
            AMREX_ALWAYS_ASSERT(bins_ptr->numBins() >= 0);
            AMREX_ALWAYS_ASSERT(travel_bins_ptr->numBins() >= 0);
            auto inds = bins_ptr->permutationPtr();
            auto offsets = bins_ptr->offsetsPtr();
            auto travel_inds = travel_bins_ptr->permutationPtr();

            for (int d = 0; d < n_disease; d++) {

                auto prob_ptr = this->getAgentProbPtr(a_on_air_travel,lev,mfi,d);
                //auto mask_arr = a_mask[mfi].array();
                auto lparm = a_agents.getDiseaseParameters_d(d);

                ParallelFor( travel_bins_ptr->numItems(), [=] AMREX_GPU_DEVICE (int ii) noexcept
                {
                    auto i = travel_inds[ii];
                    int i_cell = binner(pstruct_ptr[i]);
                    auto cell_start = offsets[i_cell];
                    auto cell_stop  = offsets[i_cell+1];

                    AMREX_ALWAYS_ASSERT( (Long) i < np);

                    if ( notSusceptible<ACTD>(i, ptd, d) )  { return; }

                    //Real i_mask = mask_arr(home_i_ptr[i], home_j_ptr[i], 0);
                    for (auto jj = cell_start; jj < cell_stop; ++jj) {

                        auto j = inds[jj];
                        AMREX_ALWAYS_ASSERT( (Long) j < np);

                        //Real j_mask = mask_arr(home_i_ptr[j], home_j_ptr[j], 0);
                        //if (i == j) continue;

                        if ( isInfectious<ACTD>(j, ptd, d) ) {
                            Real social_scale = 1.0_prt;  // TODO this should vary based on cell
                            binaryInteractionAirTravel<ACTD>( j, i, ptd, travel_ptd, lparm, social_scale, prob_ptr );
                        }
                    }
                });
                Gpu::synchronize();
            }
        }


    }
}

#endif
