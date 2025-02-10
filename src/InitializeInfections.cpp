/*! @file InitializeInfections.cpp
*/

#include <AMReX_ParticleUtil.H>

#include "InitializeInfections.H"


using namespace amrex;
using namespace ExaEpi;


/*! \brief Infect agents in a random community in a given unit and return the total
    number of agents infected

    + Choose a random community in the given unit.
    + For each box on each processor:
        + Create bins of agents if not already created (see #amrex::GetParticleBin, #amrex::DenseBins):
        + The bin size is 1 cell.
        + #amrex::GetParticleBin maps a particle to its bin index.
        + amrex::DenseBins::build() creates the bin-sorted array of particle indices and
            the offset array for each bin (where the offset of a bin is its starting location.
        + For each grid cell: if the community at this cell is the randomly chosen community,
        + Get bin index and the agent (particle) indices in this bin.
        + Choose a random agent in the bin; if the agent is already infected, move on, else
            infect the agent. Increment the counter variables for number of infections.
            (See the code for nuances in this step.)
    + Sum up number of infected agents over all processors and return that value.
*/
static int infect_random_community (AgentContainer& pc, /*!< Agent container (particle container)*/
                                    const Vector<int> &unit_community_start, /*!< Start community number for each unit */
                                    iMultiFab &comm_mf,
                                    std::map<std::pair<int, int>,
                                    DenseBins<AgentContainer::ParticleType> >& bin_map, /*!< Map of dense bins with agents */
                                    int unit, /*!< Unit number to infect */
                                    const int d_idx, /*!< Disease index */
                                    int ninfect, /*!< Target number of agents to infect */
                                    const bool fast_bin /*!< Use GPU binning - fast but non-deterministic */  ) {
    // chose random community
    int ncomms = unit_community_start[unit + 1] - unit_community_start[unit];
    int comm_offset = unit_community_start[unit];

    int random_comm = -1;
    if (ParallelDescriptor::IOProcessor()) random_comm = Random_int(ncomms) + comm_offset;
    ParallelDescriptor::Bcast(&random_comm, 1);

    const Geometry& geom = pc.Geom(0);
    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    const auto dxi = geom.InvCellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto domain = geom.Domain();

    int num_infected = 0;
    for (MFIter mfi = pc.MakeMFIter(0); mfi.isValid(); ++mfi) {
        DenseBins<AgentContainer::ParticleType>& bins = bin_map[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto& agents_tile = pc.GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto& aos = agents_tile.GetArrayOfStructs();
        auto& soa = agents_tile.GetStructOfArrays();
        const size_t np = aos.numParticles();

        if (np == 0) continue;
        auto pstruct_ptr = aos().dataPtr();
        const Box& box = mfi.tilebox();

        int ntiles = numTilesInBox(box, true, bin_size);

        auto binner = GetParticleBin{plo, dxi, domain, bin_size, box};
        if (bins.numBins() < 0) {
            if (fast_bin) bins.build(BinPolicy::GPU, np, pstruct_ptr, ntiles, binner);
            else bins.build(BinPolicy::Serial, np, pstruct_ptr, ntiles, binner);
        }
        auto inds = bins.permutationPtr();
        auto offsets = bins.offsetsPtr();

        int i_RT = IntIdx::nattribs;
        int r_RT = RealIdx::nattribs;

        auto status_ptr = soa.GetIntData(i_RT+i0(d_idx)+IntIdxDisease::status).data();

        auto counter_ptr           = soa.GetRealData(r_RT+r0(d_idx)+RealIdxDisease::disease_counter).data();
        auto latent_period_ptr = soa.GetRealData(r_RT+r0(d_idx)+RealIdxDisease::latent_period).data();
        auto infectious_period_ptr = soa.GetRealData(r_RT+r0(d_idx)+RealIdxDisease::infectious_period).data();
        auto incubation_period_ptr = soa.GetRealData(r_RT+r0(d_idx)+RealIdxDisease::incubation_period).data();

        auto comm_arr = comm_mf[mfi].array();

        const auto lparm = pc.getDiseaseParameters_d(d_idx);

        Gpu::DeviceScalar<int> num_infected_d(num_infected);
        int* num_infected_p = num_infected_d.dataPtr();
        ParallelForRNG(box, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            if (comm_arr(i, j, k) != random_comm) return;
            Box tbx;
            int i_cell = getTileIndex({AMREX_D_DECL(i, j, k)}, box, true, bin_size, tbx);
            auto cell_start = offsets[i_cell];
            auto cell_stop  = offsets[i_cell + 1];
            int num_this_community = cell_stop - cell_start;
            AMREX_ASSERT(num_this_community > 0 && cell_stop <= (int)np);

            int ntry = 0;
            int ni = 0;
            int stop = std::min(cell_start + ninfect, cell_stop);
            for (int ip = cell_start; ip < stop; ++ip) {
                int ind = cell_start + amrex::Random_int(num_this_community, engine);
                auto pindex = inds[ind];
                if (status_ptr[pindex] == Status::infected
                    || status_ptr[pindex] == Status::immune) {
                    if (++ntry < 100) {
                        --ip;
                    } else {
                        ip += ninfect;
                    }
                } else {
                    setInfected(&(status_ptr[pindex]), &(counter_ptr[pindex]), &(latent_period_ptr[pindex]),
                                &(infectious_period_ptr[pindex]), &(incubation_period_ptr[pindex]), engine, lparm);
                    ++ni;
                }
            }
            *num_infected_p = ni;
        });

        Gpu::Device::streamSynchronize();
        num_infected += num_infected_d.dataValue();
        if (num_infected >= ninfect) break;
    }

    ParallelDescriptor::ReduceIntSum(num_infected);
    //Print() << "Infecting unit " << unit << " out of " << unit_community_start.size() - 1 << " units, "
    //        << "random community " << random_comm << " out of " << ncomms << " comms, ranging from "
    //        << comm_offset << " to " << unit_community_start[unit + 1] << " and infected " << num_infected << "\n";
    return num_infected;
}

/*! \brief Set initial cases for the simulation

    Set the initial cases of infection for the simulation based on the #CaseData:
    For each infection hub (where #CaseData::N_hubs is the number of hubs):
    + Get the FIPS code of that hub (#CaseData::FIPS_hubs)
    + Create a vector of unit numbers corresponding to that FIPS code
    + Get the number of cases for that FIPS code (#CaseData::Size_hubs)
    + Randomly infect that many agents in the units corresponding to the FIPS code, i.e.,
        cycle through units and infect agents in random communities in that unit till the
        number of infected agents is equal or greater than the number of infections for this
        FIPS code. See #ExaEpi::Initialization::infect_random_community().
*/
void setInitialCasesFromFile (AgentContainer& pc, /*!< Agent container (particle container) */
                              CaseData& cases, /*!< Case data */
                              const std::string& d_name, /*!< Disease name */
                              int d_idx,        /*!< Disease index */
                              const Vector<int> &FIPS_codes,
                              const Vector<int> &unit_community_start, /*!< Start community number for each unit */
                              iMultiFab &comm_mf,
                              const bool fast_bin)
{
    BL_PROFILE("setInitialCasesFromFile");

    std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType> > bin_map;

    Print() << "Initializing infections for " << d_name << "\n";
    int ntry = 5;
    int ninf = 0;
    for (int ihub = 0; ihub < cases.N_hubs; ++ihub) {
        if (cases.Size_hubs[ihub] > 0) {
            int FIPS = cases.FIPS_hubs[ihub];
            std::vector<int> units;
            units.resize(0);
            for (int i = 0; i < FIPS_codes.size(); ++i) {
                if (FIPS_codes[i] == FIPS) units.push_back(i);
            }
            //int unit = FIPS_code_to_i[FIPS];
            if (units.size() > 0) {
                Print() << "    Attempting to infect: " << cases.Size_hubs[ihub] << " people in FIPS " << FIPS << "... ";
                int u = 0;
                int i = 0;
                while (i < cases.Size_hubs[ihub]) {
                    int nSuccesses = infect_random_community(pc, unit_community_start, comm_mf, bin_map, units[u],
                                                             d_idx, ntry, fast_bin);
                    ninf += nSuccesses;
                    i += nSuccesses;
                    u = (u + 1) % units.size(); //sometimes we infect fewer than ntry, but switch to next unit anyway
                }
                Print() << "infected " << i<< " (total " << ninf << ") after processing. \n";
            }
        }
    }
    amrex::ignore_unused(ninf);
}

void setInitialCasesRandom (AgentContainer& pc, /*!< Agent container (particle container) */
                            int num_cases, /*!< Number of initial cases */
                            const std::string& d_name, /*!< Disease name */
                            int d_idx,        /*!< Disease index */
                            const Vector<int> &unit_community_start, /*!< Start community number for each unit */
                            iMultiFab &comm_mf,
                            const bool fast_bin)
{
    BL_PROFILE("setInitialCasesRandom");

    std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType> > bin_map;

    Print() << "Initializing infections for " << d_name << "\n";

    int ninf = 0;
    for (int ihub = 0; ihub < num_cases; ++ihub) {
        int i = 0;
        while (i < 1) {
            int unit = 0;
            if (ParallelDescriptor::IOProcessor()) unit = Random_int(unit_community_start.size() - 1);
            ParallelDescriptor::Bcast(&unit, 1);
            int nSuccesses = infect_random_community(pc, unit_community_start, comm_mf, bin_map, unit, d_idx, 1, fast_bin);
            ninf += nSuccesses;
            i += nSuccesses;
        }
    }
    amrex::ignore_unused(ninf);
}




