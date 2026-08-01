/*! @file InitializeInfections.cpp
 */

#include <AMReX_ParticleUtil.H>

#include <fstream>
#include <map>

#include "InitializeInfections.H"

using namespace amrex;
using namespace ExaEpi;

typedef std::map<std::pair<int, int>, DenseBins<AgentContainer::ParticleType>> BinMap;

/*! Maximum number of index cases handled per call to infectRandomCommunity (matches NTRY below). */
static constexpr int NTRY_MAX = 10;

/*! \brief Build a per-unit cumulative population distribution over its communities. See header
    for details. */
Vector<float> ExaEpi::Initialization::buildCommunityCumProb (const Vector<int>& unit_community_start,
                                                              const Vector<int>& community_population) {
    int ncomm_total = unit_community_start.back();
    AMREX_ALWAYS_ASSERT((int)community_population.size() == ncomm_total);

    Vector<float> cum_prob(ncomm_total, 0.0f);
    int nunits = (int)unit_community_start.size() - 1;
    for (int u = 0; u < nunits; ++u) {
        int lo = unit_community_start[u];
        int hi = unit_community_start[u + 1];
        double total_pop = 0.0;
        for (int c = lo; c < hi; ++c) { total_pop += community_population[c]; }
        if (total_pop <= 0.0) {
            // leave this unit's slice at 0.0 -- sentinel for "fall back to uniform draw"
            continue;
        }
        double running = 0.0;
        for (int c = lo; c < hi; ++c) {
            running += community_population[c];
            cum_prob[c] = static_cast<float>(running / total_pop);
        }
        cum_prob[hi - 1] = 1.0f; // clamp to avoid float-rounding gaps at the top of the slice
    }
    return cum_prob;
}

/*! \brief Draw `n` independent, population-weighted community indices within a unit (matching
    Epicast's scheme of drawing each index case independently and uniformly across every
    susceptible agent in the county -- equivalent to drawing a community proportional to its
    population, once per case). Falls back to a uniform draw if the unit has zero total
    population (see #buildCommunityCumProb). Drawn on the IOProcessor and broadcast, mirroring
    the existing single-draw pattern used elsewhere in this file (and in AirTravelFlow). */
static void drawWeightedCommunities (const Vector<int>& unit_community_start, const Vector<float>& community_cum_prob,
                                     int unit, int n, Vector<int>& out /*!< resized to n */) {
    out.resize(n);
    int lo = unit_community_start[unit];
    int hi = unit_community_start[unit + 1];
    if (ParallelDescriptor::IOProcessor()) {
        bool zero_pop = (community_cum_prob[hi - 1] <= 0.0f);
        for (int k = 0; k < n; ++k) {
            if (zero_pop) {
                out[k] = lo + Random_int(hi - lo);
                continue;
            }
            Real r = amrex::Random();
            int idx;
            if (hi - lo <= 16) {
                for (idx = lo; idx < hi - 1 && r > community_cum_prob[idx]; ++idx) {}
            } else {
                int a = lo, b = hi - 1;
                while (a < b) {
                    int mid = a + (b - a) / 2;
                    if (community_cum_prob[mid] < r) { a = mid + 1; } else { b = mid; }
                }
                idx = a;
            }
            out[k] = idx;
        }
    }
    ParallelDescriptor::Bcast(out.data(), n);
}

/*! \brief Infect agents in a random community in a given unit and return the total
    number of agents infected

    + Draw `ninfect` (<=10) independent, population-weighted community indices in the given unit
      (see #drawWeightedCommunities) -- one draw per index case, rather than one shared community
      for the whole batch, so that co-located seeding clusters (and the resulting artificially
      fast local outbreaks) are avoided.
    + For each box on each processor:
        + Create bins of agents if not already created (see #amrex::GetParticleBin, #amrex::DenseBins):
        + The bin size is 1 cell.
        + #amrex::GetParticleBin maps a particle to its bin index.
        + amrex::DenseBins::build() creates the bin-sorted array of particle indices and
            the offset array for each bin (where the offset of a bin is its starting location.
        + For each grid cell: count how many of the drawn communities match this cell's community
          (usually 0 or 1, occasionally more if two draws land on the same community by chance).
        + Get bin index and the agent (particle) indices in this bin.
        + Choose a random agent in the bin; if the agent is already infected, move on, else
            infect the agent. Increment the counter variables for number of infections.
            (See the code for nuances in this step.)
    + Sum up number of infected agents over all processors and return that value.
*/
static int infectRandomCommunity (AgentContainer& pc,                      /*!< Agent container (particle container)*/
                                  const Vector<int>& unit_community_start, /*!< Start community number for each unit */
                                  const Vector<float>& community_cum_prob, /*!< Cumulative population distribution per unit */
                                  iMultiFab& comm_mf,                      /*!< Community numbers */
                                  BinMap& bin_map,                         /*!< Map of dense bins with agents */
                                  int unit,                                /*!< Unit number to infect */
                                  const int d_idx,                         /*!< Disease index */
                                  int ninfect,                             /*!< Target number of agents to infect */
                                  const bool fast_bin /*!< Use GPU binning - fast but non-deterministic */) {
    AMREX_ALWAYS_ASSERT(ninfect <= NTRY_MAX);

    Vector<int> batch_h;
    drawWeightedCommunities(unit_community_start, community_cum_prob, unit, ninfect, batch_h);
    GpuArray<int, NTRY_MAX> batch{};
    for (int b = 0; b < ninfect; ++b) { batch[b] = batch_h[b]; }
    const int n_batch = ninfect;

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

        if (np == 0) { continue; }
        auto pstruct_ptr = aos().dataPtr();
        const Box& box = mfi.tilebox();

        int ntiles = numTilesInBox(box, true, bin_size);

        auto binner = GetParticleBin{plo, dxi, domain, bin_size, box};

        if (bins.numBins() < 0) {
            if (fast_bin) {
                bins.build(BinPolicy::GPU, np, pstruct_ptr, ntiles, binner);
            } else {
                bins.build(BinPolicy::Serial, np, pstruct_ptr, ntiles, binner);
            }
        }

        auto inds = bins.permutationPtr();
        auto offsets = bins.offsetsPtr();

        int i_RT = IntIdx::nattribs;
        int r_RT = RealIdx::nattribs;

        auto status_ptr = soa.GetIntData(i_RT + i0(d_idx) + IntIdxDisease::status).data();
        auto symptomatic_ptr = soa.GetIntData(i_RT + i0(d_idx) + IntIdxDisease::symptomatic).data();

        auto counter_ptr = soa.GetRealData(r_RT + r0(d_idx) + RealIdxDisease::disease_counter).data();
        auto latent_period_ptr = soa.GetRealData(r_RT + r0(d_idx) + RealIdxDisease::latent_period).data();
        auto infectious_period_ptr = soa.GetRealData(r_RT + r0(d_idx) + RealIdxDisease::infectious_period).data();
        auto incubation_period_ptr = soa.GetRealData(r_RT + r0(d_idx) + RealIdxDisease::incubation_period).data();
        auto hospital_delay_ptr = soa.GetRealData(r_RT + r0(d_idx) + RealIdxDisease::hospital_delay).data();
        auto hospital_random_ptr = soa.GetRealData(r_RT + r0(d_idx) + RealIdxDisease::hospital_random).data();
        auto comm_arr = comm_mf[mfi].array();

        const auto lparm = pc.getDiseaseParameters_d(d_idx);

        Gpu::DeviceScalar<int> num_infected_d(0);
        int* num_infected_p = num_infected_d.dataPtr();
        ParallelForRNG(box, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept {
            int this_comm = comm_arr(i, j, k);
            int n_here = 0;
            for (int b = 0; b < n_batch; ++b) { if (batch[b] == this_comm) { ++n_here; } }
            if (n_here == 0) { return; }

            Box tbx;
            int i_cell = getTileIndex({AMREX_D_DECL(i, j, k)}, box, true, bin_size, tbx);
            auto cell_start = offsets[i_cell];
            auto cell_stop = offsets[i_cell + 1];
            int num_this_community = cell_stop - cell_start;
            AMREX_ASSERT(num_this_community > 0 && cell_stop <= (int)np);

            int ntry = 0;
            int ni = 0;
            int stop = std::min(cell_start + n_here, cell_stop);
            for (int ip = cell_start; ip < stop; ++ip) {
                int ind = cell_start + amrex::Random_int(num_this_community, engine);
                auto pindex = inds[ind];
                if (status_ptr[pindex] == Status::infected || status_ptr[pindex] == Status::immune) {
                    if (++ntry < 100) {
                        --ip;
                    } else {
                        ip += n_here;
                    }
                } else {
                    setInfected(&(status_ptr[pindex]), &(symptomatic_ptr[pindex]), &(counter_ptr[pindex]),
                                &(latent_period_ptr[pindex]), &(infectious_period_ptr[pindex]), &(incubation_period_ptr[pindex]),
                                &(hospital_delay_ptr[pindex]), &(hospital_random_ptr[pindex]), engine, lparm);
                    ++ni;
                }
            }
            Gpu::Atomic::AddNoRet(num_infected_p, ni);
        });

        Gpu::Device::streamSynchronize();
        num_infected += num_infected_d.dataValue();
        // Note: no early-exit here -- the batch's distinct (population-weighted) communities may
        // be scattered across multiple boxes on this rank, so every box must be scanned.
    }

    ParallelDescriptor::ReduceIntSum(num_infected);
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
        FIPS code. See #ExaEpi::Initialization::infectRandomCommunity().
*/
void setInitialCasesFromFile (AgentContainer& pc,                      /*!< Agent container (particle container) */
                              CaseData& cases,                         /*!< Case data */
                              const std::string& d_name,               /*!< Disease name */
                              int d_idx,                               /*!< Disease index */
                              const Vector<int>& FIPS_codes,
                              const Vector<int>& unit_community_start, /*!< Start community number for each unit */
                              const Vector<float>& community_cum_prob, /*!< Cumulative population distribution per unit */
                              iMultiFab& comm_mf, const bool fast_bin) {
    BL_PROFILE("setInitialCasesFromFile");

    std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType>> bin_map;

    Print() << "Initializing infections for " << d_name << "\n";
#ifdef COMPARE_TO_EPICAST
    Print() << "WARNINNG: limited version for comparing to Epicast\n";
    const int NTRY = 1;
#else
    const int NTRY = 10;
#endif

    int ntry = NTRY;
    int ninf = 0;
    for (int ihub = 0; ihub < cases.N_hubs; ++ihub) {
        if (cases.Size_hubs[ihub] > 0) {
            int FIPS = cases.FIPS_hubs[ihub];
            std::vector<int> units;
            units.resize(0);
            for (int i = 0; i < FIPS_codes.size(); ++i) {
                if (FIPS_codes[i] == FIPS) { units.push_back(i); }
            }
            // int unit = FIPS_code_to_i[FIPS];
            if (units.size() > 0) {
                Print() << "    Attempting to infect: " << cases.Size_hubs[ihub] << " people in FIPS " << FIPS << "... ";
                int u = 0;
                int i = 0;
                while (i < cases.Size_hubs[ihub]) {
                    int diff = cases.Size_hubs[ihub] - i;
                    ntry = diff > NTRY ? NTRY : diff;
                    int nSuccesses = infectRandomCommunity(pc, unit_community_start, community_cum_prob, comm_mf, bin_map,
                                                           units[u], d_idx, ntry, fast_bin);
                    ninf += nSuccesses;
                    i += nSuccesses;
                    u = (u + 1) % units.size(); // sometimes we infect fewer than ntry, but switch to next unit anyway
                }
                Print() << "infected " << i << " (total " << ninf << ") after processing. \n";
            }
        }
    }
    amrex::ignore_unused(ninf);
}

void setInitialCasesRandom (AgentContainer& pc,                      /*!< Agent container (particle container) */
                            int num_cases,                           /*!< Number of initial cases */
                            const std::string& d_name,               /*!< Disease name */
                            int d_idx,                               /*!< Disease index */
                            const Vector<int>& FIPS_codes,           /*!< FIPS code for each unit */
                            const Vector<int>& unit_community_start, /*!< Start community number for each unit */
                            const Vector<float>& community_cum_prob, /*!< Cumulative population distribution per unit */
                            iMultiFab& comm_mf, const bool fast_bin) {
    BL_PROFILE("setInitialCasesRandom");

    std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType>> bin_map;

    Print() << "Initializing infections for " << d_name << "\n";

    std::map<int, int> fips_infection_counts;

    int ninf = 0;
    for (int ihub = 0; ihub < num_cases; ++ihub) {
        int i = 0;
        while (i < 1) {
            int unit = 0;
            if (ParallelDescriptor::IOProcessor()) { unit = Random_int(unit_community_start.size() - 1); }
            ParallelDescriptor::Bcast(&unit, 1);
            int nSuccesses =
                    infectRandomCommunity(pc, unit_community_start, community_cum_prob, comm_mf, bin_map, unit, d_idx, 1, fast_bin);
            if (nSuccesses > 0) { fips_infection_counts[FIPS_codes[unit]] += nSuccesses; }
            ninf += nSuccesses;
            i += nSuccesses;
        }
    }

    if (ParallelDescriptor::IOProcessor()) {
        std::string out_filename = d_name + "_random_initial.cases";
        std::ofstream ofs(out_filename);
        for (auto& [fips, count] : fips_infection_counts) {
            ofs << fips << " " << count << " " << count << "\n";
        }
        Print() << "Wrote random initial case locations to " << out_filename << " (use with disease.initial_case_type = file)\n";
    }

    amrex::ignore_unused(ninf);
}
