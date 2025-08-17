/*! @file AgentContainer.cpp
    \brief Function implementations for #AgentContainer class
*/

#include "AgentContainer.H"
#include "AgentDefinitions.H"

using namespace amrex;
using namespace ExaEpi::Utils;

/*! Add runtime SoA attributes */
void AgentContainer::addAttributes () {
    const bool communicate_this_comp = true;
    {
        int count(0);
        for (int i = 0; i < m_num_diseases * RealIdxDisease::nattribs; i++) {
            AddRealComp(communicate_this_comp);
            count++;
        }
        Print() << "Added " << count << " real-type run-time SoA attibute(s).\n";
    }
    {
        int count(0);
        for (int i = 0; i < m_num_diseases * IntIdxDisease::nattribs; i++) {
            AddIntComp(communicate_this_comp);
            count++;
        }
        Print() << "Added " << count << " integer-type run-time SoA attibute(s).\n";
    }
}

/*! Constructor:
 *  + Initializes particle container for agents
 *  + Read in contact probabilities from command line input file
 *  + Read in disease parameters from command line input file
 */
AgentContainer::AgentContainer (const amrex::Geometry& a_geom,                   /*!< Physical domain */
                                const amrex::DistributionMapping& a_dmap,        /*!< Distribution mapping */
                                const amrex::BoxArray& a_ba,                     /*!< Box array */
                                const int& a_num_diseases,                       /*!< Number of diseases */
                                const std::vector<std::string>& a_disease_names, /*!< names of the diseases */
                                const std::vector<int>& a_fips_codes,            /*!< FIPS codes */
                                const bool fast,                                 /*!< faster but non-deterministic computation*/
                                const short a_ic_type /*!< type of initialization */)
    : amrex::ParticleContainer<0, 0, RealIdx::nattribs, IntIdx::nattribs>(a_geom, a_dmap, a_ba),
      m_fips_codes(a_fips_codes),
      m_student_counts(a_ba, a_dmap, (a_ic_type == ExaEpi::ICType::Census) ?  static_cast<int>(SchoolCensusIDType::total) :  static_cast<int>(SchoolType::total), 0),
      m_school_stats(a_ba, a_dmap, ((a_ic_type == ExaEpi::ICType::Census) ?  static_cast<int>(SchoolCensusIDType::total) :  static_cast<int>(SchoolType::total)) *  static_cast<int>(SchoolPolicy::SchoolStats::nattribs), 0) {
    BL_PROFILE("AgentContainer::AgentContainer");

    ic_type = a_ic_type;

    m_num_diseases = a_num_diseases;
    AMREX_ASSERT(m_num_diseases < ExaEpi::max_num_diseases);

    m_student_counts.setVal(0); // Initialize the MultiFab to zero

    // Variables for School Closure Policy
    m_school_stats.setVal(0);
    const int num_fips = m_fips_codes.size();
    m_fips_student_stats.resize(num_fips * SchoolPolicy::SchoolStats::nattribs, 0); 

    addAttributes();

    {
        amrex::ParmParse pp("agent");
        pp.query("shelter_compliance", m_shelter_compliance);
        pp.query("symptomatic_withdraw_compliance", m_symptomatic_withdraw_compliance);
        int stratio[SchoolType::total];
        for (unsigned int i = 0; i < SchoolType::total; i++) {
            stratio[i] = m_student_teacher_ratio[i];
        }

        queryArray(pp, "student_teacher_ratio", stratio, SchoolType::total);
        for (unsigned int i = 0; i < SchoolType::total; ++i) {
            m_student_teacher_ratio[i] = stratio[i];
        }
    }

    {
        using namespace ExaEpi;

        /* Create the interaction model objects and push to container */
        m_interactions.clear();
        m_interactions[InteractionNames::home] = new InteractionModHome<PCType, PTDType, PType>(fast);
        m_interactions[InteractionNames::work] = new InteractionModWork<PCType, PTDType, PType>(fast);
        m_interactions[InteractionNames::school] = new InteractionModSchool<PCType, PTDType, PType>(fast);
        m_interactions[InteractionNames::home_nborhood] = new InteractionModHomeNborhood<PCType, PTDType, PType>(fast);
        m_interactions[InteractionNames::work_nborhood] = new InteractionModWorkNborhood<PCType, PTDType, PType>(fast);

        m_hospital = std::make_unique<HospitalModel<PCType, PTDType, PType>>(fast);
    }

    m_h_parm.resize(m_num_diseases);
    m_d_parm.resize(m_num_diseases);

    for (int d = 0; d < m_num_diseases; d++) {
        m_h_parm[d] = new DiseaseParm{a_disease_names[d]};
        m_d_parm[d] = (DiseaseParm*)amrex::The_Arena()->alloc(sizeof(DiseaseParm));

        // first read inputs common to all diseases
        m_h_parm[d]->readInputs("disease");
        // now read any disease-specific input, if available
        m_h_parm[d]->readInputs(std::string("disease_" + a_disease_names[d]));
        m_h_parm[d]->initialize();

#ifdef AMREX_USE_GPU
        amrex::Gpu::htod_memcpy(m_d_parm[d], m_h_parm[d], sizeof(DiseaseParm));
#else
        std::memcpy(m_d_parm[d], m_h_parm[d], sizeof(DiseaseParm));
#endif
    }

    m_disease_coupling = std::make_unique<DiseaseCouplingParm<PTDType>>(m_num_diseases);
    m_disease_coupling->ReadInputs("disease_coupling");

    max_attribute_values.fill(-1);
}

/*! \brief Send agents on a random walk around the neighborhood

    For each agent, set its position to a random one near its current position
*/
void AgentContainer::moveAgentsRandomWalk () {
    BL_PROFILE("AgentContainer::moveAgentsRandomWalk");

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& aos = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
                ParticleType& p = pstruct[i];
                p.pos(0) += static_cast<ParticleReal>((2 * amrex::Random(engine) - 1) * dx[0]);
                p.pos(1) += static_cast<ParticleReal>((2 * amrex::Random(engine) - 1) * dx[1]);
            });
        }
    }
}

/*! \brief Move agents to work

    For each agent, set its position to the work community (IntIdx::work_i, IntIdx::work_j)
*/
void AgentContainer::moveAgentsToWork () {
    BL_PROFILE("AgentContainer::moveAgentsToWork");

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev = GetParticles(lev);

        bool is_census = (ic_type == ExaEpi::ICType::Census);
        auto grid_to_lnglat_ptr = &grid_to_lnglat;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            auto& soa = ptile.GetStructOfArrays();
            auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
            auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
                if (!inHospital(ip, ptd)) {
                    ParticleType& p = pstruct[ip];
                    if (is_census) { // using census data
                        p.pos(0) = static_cast<ParticleReal>((work_i_ptr[ip] + 0.5_rt) * dx[0]);
                        p.pos(1) = static_cast<ParticleReal>((work_j_ptr[ip] + 0.5_rt) * dx[1]);
                    } else {
                        Real lng, lat;
                        (*grid_to_lnglat_ptr)(work_i_ptr[ip], work_j_ptr[ip], lng, lat);
                        p.pos(0) = static_cast<ParticleReal>(lng);
                        p.pos(1) = static_cast<ParticleReal>(lat);
                    }
                }
            });
        }
    }

    m_at_work = true;

    Redistribute();
    AMREX_ASSERT(OK());
}

/*! \brief Move agents to home

    For each agent, set its position to the home community (IntIdx::home_i, IntIdx::home_j)
*/
void AgentContainer::moveAgentsToHome () {
    BL_PROFILE("AgentContainer::moveAgentsToHome");

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev = GetParticles(lev);

        bool is_census = (ic_type == ExaEpi::ICType::Census);
        auto grid_to_lnglat_ptr = &grid_to_lnglat;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            auto& soa = ptile.GetStructOfArrays();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
                if (!inHospital(ip, ptd)) {
                    ParticleType& p = pstruct[ip];
                    if (is_census) { // using census data
                        p.pos(0) = static_cast<ParticleReal>((home_i_ptr[ip] + 0.5_rt) * dx[0]);
                        p.pos(1) = static_cast<ParticleReal>((home_j_ptr[ip] + 0.5_rt) * dx[1]);
                    } else {
                        Real lng, lat;
                        (*grid_to_lnglat_ptr)(home_i_ptr[ip], home_j_ptr[ip], lng, lat);
                        p.pos(0) = static_cast<ParticleReal>(lng);
                        p.pos(1) = static_cast<ParticleReal>(lat);
                    }
                }
            });
        }
    }

    m_at_work = false;

    Redistribute();
    AMREX_ASSERT(OK());
}

/*! \brief Move agents randomly

    For each agent, set its position to a random location with a probabilty of 0.01%
*/
void AgentContainer::moveRandomTravel (const amrex::Real random_travel_prob) {
    BL_PROFILE("AgentContainer::moveRandomTravel");

    const Box& domain = Geom(0).Domain();
    int i_max = domain.length(0);
    int j_max = domain.length(1);
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = plev[{mfi.index(), mfi.LocalTileIndex()}];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
                if (!inHospital(i, ptd) && !withdrawn_ptr[i]) {
                    ParticleType& p = pstruct[i];
                    if (amrex::Random(engine) < random_travel_prob) {
                        random_travel_ptr[i] = i;
                        int i_random = int(amrex::Real(i_max) * amrex::Random(engine));
                        int j_random = int(amrex::Real(j_max) * amrex::Random(engine));
                        p.pos(0) = i_random;
                        p.pos(1) = j_random;
                    }
                }
            });
        }
    }

    // Don't need to redistribute here because it happens after agents move to work
    // Redistribute();
}

/*! \brief Select agents to travel by air

*/
void AgentContainer::moveAirTravel (const iMultiFab& unit_mf, AirTravelFlow& air, DemographicData& demo) {
    BL_PROFILE("AgentContainer::moveAirTravel");
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            const auto unit_arr = unit_mf[mfi].array();
            auto& ptile = plev[{mfi.index(), mfi.LocalTileIndex()}];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto air_travel_ptr = soa.GetIntData(IntIdx::air_travel).data();
            auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto trav_i_ptr = soa.GetIntData(IntIdx::trav_i).data();
            auto trav_j_ptr = soa.GetIntData(IntIdx::trav_j).data();
            auto air_travel_prob_ptr = air.air_travel_prob_d.data();

            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
                int unit = unit_arr(home_i_ptr[i], home_j_ptr[i], 0);
                if (!inHospital(i, ptd) && random_travel_ptr[i] < 0 && air_travel_ptr[i] < 0) {
                    if (withdrawn_ptr[i] == 1) { return; }
                    if (amrex::Random(engine) < air_travel_prob_ptr[unit]) {
                        ParticleType& p = pstruct[i];
                        p.pos(0) = trav_i_ptr[i];
                        p.pos(1) = trav_j_ptr[i];
                        air_travel_ptr[i] = i;
                    }
                }
            });
        }
    }
}

void AgentContainer::setAirTravel (const iMultiFab& unit_mf, AirTravelFlow& air, DemographicData& demo) {
    BL_PROFILE("AgentContainer::setAirTravel");

    amrex::Print() << "Compute air travel statistics" << "\n";
    const Box& domain = Geom(0).Domain();
    int i_max = domain.length(0);
    int j_max = domain.length(1);
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            const auto unit_arr = unit_mf[mfi].array();
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& aos = ptile.GetArrayOfStructs();
            const size_t np = aos.numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto trav_i_ptr = soa.GetIntData(IntIdx::trav_i).data();
            auto trav_j_ptr = soa.GetIntData(IntIdx::trav_j).data();
            auto Start = demo.Start_d.data();
            auto dest_airports_ptr = air.dest_airports_d.data();
            auto dest_airports_offset_ptr = air.dest_airports_offset_d.data();
            auto dest_airports_prob_ptr = air.dest_airports_prob_d.data();
            auto arrivalUnits_ptr = air.arrivalUnits_d.data();
            auto arrivalUnits_offset_ptr = air.arrivalUnits_offset_d.data();
            auto arrivalUnits_prob_ptr = air.arrivalUnits_prob_d.data();
            auto assigned_airport_ptr = air.assigned_airport_d.data();

            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
                trav_i_ptr[i] = -1;
                trav_j_ptr[i] = -1;
                int unit = unit_arr(home_i_ptr[i], home_j_ptr[i], 0);
                int orgAirport = assigned_airport_ptr[unit];
                int destAirport = -1;
                Real lowProb = 0.0_rt;
                Real random = amrex::Random(engine);
                // choose a destination airport for the agent (number of airports is often small, so let's visit in sequential
                // order)
                for (int idx = dest_airports_offset_ptr[orgAirport]; idx < dest_airports_offset_ptr[orgAirport + 1]; idx++) {
                    float hiProb = dest_airports_prob_ptr[idx];
                    if (random > lowProb && random < hiProb) {
                        destAirport = dest_airports_ptr[idx];
                        break;
                    }
                    lowProb = dest_airports_ptr[idx];
                }
                if (destAirport >= 0) {
                    int destUnit = -1;
                    Real random1 = amrex::Random(engine);
                    int low = arrivalUnits_offset_ptr[destAirport], high = arrivalUnits_offset_ptr[destAirport + 1];
                    if (high - low <= 16) {
                        // this sequential algo. is very slow when we have to go through hundreds or thoudsands of units to select
                        // a destination
                        float lProb = 0.0;
                        for (int idx = low; idx < high; idx++) {
                            if (random1 > lProb && random1 < arrivalUnits_prob_ptr[idx]) {
                                destUnit = arrivalUnits_ptr[idx];
                                break;
                            }
                            lProb = arrivalUnits_prob_ptr[idx];
                        }
                    } else {           // binary search algorithm
                        while (low < high) {
                            if (random1 < low) {
                                break; // low is the found airport index
                            }
                            // if random1 falls within (low, high), half the range
                            int mid = low + (high - low) / 2;
                            if (arrivalUnits_prob_ptr[mid] < random1) {
                                low = mid + 1;
                            } else {
                                high = mid - 1;
                            }
                        }
                        destUnit = arrivalUnits_ptr[low];
                    }
                    if (destUnit >= 0) {
                        // randomly select a community in the dest unit
                        int comm_to = Start[destUnit] + amrex::Random_int(Start[destUnit + 1] - Start[destUnit], engine);
                        int new_i = comm_to % i_max;
                        int new_j = comm_to / i_max;
                        if (new_i >= 0 && new_j >= 0 && new_i < i_max && new_j < j_max) {
                            trav_i_ptr[i] = new_i;
                            trav_j_ptr[i] = new_j;
                        }
                    }
                }
            });
        }
    }
}

/*! \brief Return agents from random travel
 */
void AgentContainer::returnRandomTravel () {
    BL_PROFILE("AgentContainer::returnRandomTravel");

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);
        const auto dx = Geom(lev).CellSizeArray();

        bool is_census = (ic_type == ExaEpi::ICType::Census);
        auto grid_to_lnglat_ptr = &grid_to_lnglat;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = plev[{mfi.index(), mfi.LocalTileIndex()}];
            auto& aos = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                if (random_travel_ptr[i] >= 0) {
                    ParticleType& p = pstruct[i];
                    random_travel_ptr[i] = -1;
                    if (is_census) {
                        p.pos(0) = static_cast<ParticleReal>((home_i_ptr[i] + 0.5_rt) * dx[0]);
                        p.pos(1) = static_cast<ParticleReal>((home_j_ptr[i] + 0.5_rt) * dx[1]);
                    } else {
                        Real lng, lat;
                        (*grid_to_lnglat_ptr)(home_i_ptr[i], home_j_ptr[i], lng, lat);
                        p.pos(0) = static_cast<ParticleReal>(lng);
                        p.pos(1) = static_cast<ParticleReal>(lat);
                    }
                }
            });
        }
    }
    Redistribute();
    AMREX_ALWAYS_ASSERT(OK());
}

/*! \brief Return agents from air travel
 */
void AgentContainer::returnAirTravel () {
    BL_PROFILE("AgentContainer::returnAirTravel");

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);
        const auto dx = Geom(lev).CellSizeArray();

        bool is_census = (ic_type == ExaEpi::ICType::Census);
        auto grid_to_lnglat_ptr = &grid_to_lnglat;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = plev[{mfi.index(), mfi.LocalTileIndex()}];
            auto& aos = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto air_travel_ptr = soa.GetIntData(IntIdx::air_travel).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                if (air_travel_ptr[i] >= 0) {
                    ParticleType& p = pstruct[i];
                    air_travel_ptr[i] = -1;
                    if (is_census) { // using census data
                        p.pos(0) = static_cast<ParticleReal>((home_i_ptr[i] + 0.5_rt) * dx[0]);
                        p.pos(1) = static_cast<ParticleReal>((home_j_ptr[i] + 0.5_rt) * dx[1]);
                    } else {
                        Real lng, lat;
                        (*grid_to_lnglat_ptr)(home_i_ptr[i], home_j_ptr[i], lng, lat);
                        p.pos(0) = static_cast<ParticleReal>(lng);
                        p.pos(1) = static_cast<ParticleReal>(lat);
                    }
                }
            });
        }
    }
    Redistribute();
    AMREX_ALWAYS_ASSERT(OK());
}

/*! \brief Updates disease status of each agent */
void AgentContainer::updateStatus (MFPtrVec& a_disease_stats /*!< Community-wise disease stats tracker */) {
    BL_PROFILE("AgentContainer::updateStatus");

    m_disease_status.updateAgents(*this, a_disease_stats);
    m_hospital->treatAgents(*this, a_disease_stats);

    // move hospitalized agents to their hospital location
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev = GetParticles(lev);

        bool is_census = (ic_type == ExaEpi::ICType::Census);
        auto grid_to_lnglat_ptr = &grid_to_lnglat;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            auto& soa = ptile.GetStructOfArrays();
            auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
            auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
                if (inHospital(ip, ptd)) {
                    ParticleType& p = pstruct[ip];
                    if (is_census) {
                        p.pos(0) = static_cast<ParticleReal>((hosp_i_ptr[ip] + 0.5_prt) * dx[0]);
                        p.pos(1) = static_cast<ParticleReal>((hosp_j_ptr[ip] + 0.5_prt) * dx[1]);
                    } else {
                        Real lng, lat;
                        (*grid_to_lnglat_ptr)(hosp_i_ptr[ip], hosp_j_ptr[ip], lng, lat);
                        p.pos(0) = static_cast<ParticleReal>(lng);
                        p.pos(1) = static_cast<ParticleReal>(lat);
                    }
                }
            });
        }
    }
}

/*! \brief Start shelter-in-place */
void AgentContainer::shelterStart () {
    BL_PROFILE("AgentContainer::shelterStart");

    amrex::Print() << "Starting shelter in place order \n";

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            if (np == 0) { continue; }

            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto withdrawn_date_ptr = soa.GetIntData(IntIdx::withdrawn_date).data();

            auto shelter_compliance = m_shelter_compliance;
            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept {
                if (amrex::Random(engine) < shelter_compliance) {
                    withdrawn_ptr[i] = 1;
                    withdrawn_date_ptr[i] = m_current_day;
                }
            });
        }
    }
}

/*! \brief Stop shelter-in-place */
void AgentContainer::shelterStop () {
    BL_PROFILE("AgentContainer::shelterStop");

    amrex::Print() << "Stopping shelter in place order \n";

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            if (np == 0) { continue; }

            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto withdrawn_date_ptr = soa.GetIntData(IntIdx::withdrawn_date).data();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                withdrawn_ptr[i] = 0;
                withdrawn_date_ptr[i] = 0;

            });
        }
    }
}

/*! \brief Infect agents based on their current status and the computed probability of infection.
    The infection probability is computed in AgentContainer::interactAgentsHomeWork() or
    AgentContainer::interactAgents() */
void AgentContainer::infectAgents (MFPtrVec& a_disease_stats /*!< Community-wise disease stats tracker */) {
    BL_PROFILE("AgentContainer::infectAgents");

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& soa = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            if (np == 0) { continue; }

            int i_RT = IntIdx::nattribs;
            int r_RT = RealIdx::nattribs;
            int n_disease = m_num_diseases;

            for (int d = 0; d < n_disease; d++) {
                (*a_disease_stats[d])[mfi].setVal<RunOn::Gpu>(0, mfi.tilebox(), DiseaseStats::new_cases, 1);
                auto ds_arr = (*a_disease_stats[d])[mfi].array();

                auto status_ptr = soa.GetIntData(i_RT + i0(d) + IntIdxDisease::status).data();

                auto prob_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::prob).data();
                auto counter_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::disease_counter).data();
                auto latent_period_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::latent_period).data();
                auto infectious_period_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::infectious_period).data();
                auto incubation_period_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::incubation_period).data();
                auto hospital_delay_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::hospital_delay).data();
                auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
                auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();

                const auto lparm = m_d_parm[d];

                Gpu::DeviceVector<ParticleReal> coimmunity, cosusceptibility;
                coimmunity.resize(np);
                cosusceptibility.resize(np);

                m_disease_coupling->getCoimmunity(coimmunity, d, ptd);
                m_disease_coupling->getCosusceptibility(cosusceptibility, d, ptd);

                auto ci_arr = coimmunity.data();
                auto cs_arr = cosusceptibility.data();

                amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept {
                    if (prob_ptr[i] == 1.0_prt) {
                        prob_ptr[i] = 0.0_prt;
                    } else {
                        prob_ptr[i] = 1.0_prt - prob_ptr[i] / cs_arr[i];
                    }
                    prob_ptr[i] *= (1.0_prt - ci_arr[i]);
                    if (status_ptr[i] == Status::never || status_ptr[i] == Status::susceptible) {
                        if (amrex::Random(engine) < prob_ptr[i]) {
                            setInfected(&(status_ptr[i]), &(counter_ptr[i]), &(latent_period_ptr[i]), &(infectious_period_ptr[i]),
                                        &(incubation_period_ptr[i]), &(hospital_delay_ptr[i]), engine, lparm);
                            Gpu::Atomic::AddNoRet(&ds_arr(home_i_ptr[i], home_j_ptr[i], 0, DiseaseStats::new_cases), 1.0_rt);

                            return;
                        }
                    }
                });
            }
        }
    }
}

/*! \brief Computes the number of agents with various #Status in each grid cell of the
    computational domain.

    Given a MultiFab with at least 5 x (number of diseases) components that is defined with
    the same box array and distribution mapping as this #AgentContainer, the MultiFab will
    contain (at the end of this function) the following *in each cell*:
    For each disease (d being the disease index):
    + component 5*d+0: total number of agents in this grid cell.
    + component 5*d+1: number of agents that have never been infected (#Status::never)
    + component 5*d+2: number of agents that are infected (#Status::infected)
    + component 5*d+3: number of agents that are immune (#Status::immune)
    + component 5*d+4: number of agents that are susceptible infected (#Status::susceptible)
*/
void AgentContainer::generateCellData (MultiFab& mf, /*!< MultiFab with at least a_ncomp*m_num_diseases components */
                                       const int a_ncomp /*!< Number of components per disease */) const {
    BL_PROFILE("AgentContainer::generateCellData");

    const int lev = 0;

    AMREX_ASSERT(OK());
    AMREX_ASSERT(numParticlesOutOfRange(*this, 0) == 0);
    AMREX_ASSERT(a_ncomp == (Status::dead + 1));

    const auto& geom = Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const auto domain = geom.Domain();
    int n_disease = m_num_diseases;

    ParticleToMesh(
            *this, mf, lev,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd, int i,
                                 Array4<Real> const& count) {
                auto p = ptd.m_aos[i];
                auto iv = getParticleCell(p, plo, dxi, domain);

                for (int d = 0; d < n_disease; d++) {
                    int status = ptd.m_runtime_idata[i0(d) + IntIdxDisease::status][i];
                    Gpu::Atomic::AddNoRet(&count(iv, a_ncomp * d + 0), 1.0_rt);
                    if (status != Status::dead) { Gpu::Atomic::AddNoRet(&count(iv, a_ncomp * d + status + 1), 1.0_rt); }
                }
            },
            false);
}

/*! \brief Computes the total number of agents with each #Status

    Returns a vector with 5 components corresponding to each value of #Status; each element is
    the total number of agents at a step with the corresponding #Status (in that order).

    Status list: 0 - never, 1 - infected, 2 - immune, 3 - susceptible, 4 - dead, 5 - exposed, 6 - asymptomatic,
                 7 - presymptomatic, 8 - symptomatic
*/
std::array<Long, 9> AgentContainer::getTotals (const int a_d /*!< disease index */) {
    BL_PROFILE("getTotals");
    amrex::ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum,
                     ReduceOpSum>
            reduce_ops;
    auto r = amrex::ParticleReduce<ReduceData<int, int, int, int, int, int, int, int, int>>(
            *this,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                                 const int i) noexcept -> amrex::GpuTuple<int, int, int, int, int, int, int, int, int> {
                int s[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                auto status = ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][i];

                AMREX_ALWAYS_ASSERT(status >= 0);
                AMREX_ALWAYS_ASSERT(status <= 4);

                s[status] = 1;

                if (status == Status::infected) { // exposed
                    if (notInfectiousButInfected(i, ptd, a_d)) {
                        s[5] = 1;                 // exposed, but not infectious
                    } else {                      // infectious
                        if (ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::symptomatic][i] == SymptomStatus::asymptomatic) {
                            s[6] = 1;             // asymptomatic and will remain so
                        } else if (ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::symptomatic][i] ==
                                   SymptomStatus::presymptomatic) {
                            s[7] = 1;             // asymptomatic but will develop symptoms
                        } else if (ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::symptomatic][i] == SymptomStatus::symptomatic) {
                            s[8] = 1;             // Infectious and symptomatic
                        } else {
                            amrex::Abort("how did I get here?");
                        }
                    }
                }
                return {s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]};
            },
            reduce_ops);

    std::array<Long, 9> counts = {amrex::get<0>(r), amrex::get<1>(r), amrex::get<2>(r), amrex::get<3>(r), amrex::get<4>(r),
                                  amrex::get<5>(r), amrex::get<6>(r), amrex::get<7>(r), amrex::get<8>(r)};
    ParallelDescriptor::ReduceLongSum(&counts[0], 9, ParallelDescriptor::IOProcessorNumber());
    return counts;
}

int AgentContainer::getMaxGroup (const int group_idx) {
    BL_PROFILE("getMaxGroup");
    if (max_attribute_values[group_idx] == -1) {
        ReduceOps<ReduceOpMax> reduce_ops;
        auto r = ParticleReduce<ReduceData<int>>(
                *this,
                [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                                     const int i) noexcept -> GpuTuple<int> {
                    return {ptd.m_idata[group_idx][i]};
                },
                reduce_ops);
        max_attribute_values[group_idx] = amrex::get<0>(r);
    }
    return max_attribute_values[group_idx];
}

/*! \brief Interaction and movement of agents during morning commute
 *
 * + Move agents to work
 * + Simulate interactions during morning commute (public transit/carpool/etc ?)
 */
void AgentContainer::morningCommute (MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */) {
    BL_PROFILE("AgentContainer::morningCommute");
    // if (haveInteractionModel(ExaEpi::InteractionNames::transit)) {
    //     m_interactions[ExaEpi::InteractionNames::transit]->interactAgents( *this, a_mask_behavior );
    // }
    moveAgentsToWork();
}

/*! \brief Interaction and movement of agents during evening commute
 *
 * + Simulate interactions during evening commute (public transit/carpool/etc ?)
 * + Simulate interactions at locations agents may stop by on their way home
 * + Move agents to home
 */
void AgentContainer::eveningCommute (MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */) {
    BL_PROFILE("AgentContainer::eveningCommute");
    // if (haveInteractionModel(ExaEpi::InteractionNames::transit)) {
    //     m_interactions[ExaEpi::InteractionNames::transit]->interactAgents( *this, a_mask_behavior );
    // }
    // if (haveInteractionModel(ExaEpi::InteractionNames::grocery_store)) {
    //     m_interactions[ExaEpi::InteractionNames::grocery_store]->interactAgents( *this, a_mask_behavior );
    // }
    moveAgentsToHome();
}

/*! \brief Interaction of agents during day time - work and school */
void AgentContainer::interactDay (MultiFab& a_mask_behavior /*!< Masking behavior */) {
    BL_PROFILE("AgentContainer::interactDay");
    if (haveInteractionModel(ExaEpi::InteractionNames::work)) {
        m_interactions[ExaEpi::InteractionNames::work]->interactAgents(*this, a_mask_behavior);
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::school)) {
        m_interactions[ExaEpi::InteractionNames::school]->interactAgents(*this, a_mask_behavior);
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::work_nborhood)) {
        m_interactions[ExaEpi::InteractionNames::work_nborhood]->interactAgents(*this, a_mask_behavior);
    }
    m_hospital->interactAgents(*this, a_mask_behavior);
}

/*! \brief Interaction of agents during evening (after work) - social stuff */
void AgentContainer::interactEvening (MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */) {
    BL_PROFILE("AgentContainer::interactEvening");
}

/*! \brief Interaction of agents during nighttime time - at home */
void AgentContainer::interactNight (MultiFab& a_mask_behavior /*!< Masking behavior */) {
    BL_PROFILE("AgentContainer::interactNight");
    if (haveInteractionModel(ExaEpi::InteractionNames::home)) {
        m_interactions[ExaEpi::InteractionNames::home]->interactAgents(*this, a_mask_behavior);
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::home_nborhood)) {
        m_interactions[ExaEpi::InteractionNames::home_nborhood]->interactAgents(*this, a_mask_behavior);
    }
}

void AgentContainer::setCurrentDay(int day) { m_current_day = day; }

void AgentContainer::printStudentTeacherCounts () const {
    ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum,
              ReduceOpSum>
            reduce_ops;
    auto r = ParticleReduce<ReduceData<int, int, int, int, int, int, int, int, int, int>>(
            *this,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                                 const int i) noexcept -> GpuTuple<int, int, int, int, int, int, int, int, int, int> {
                int counts[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                if (ptd.m_idata[IntIdx::school_id][i] > 0) {
                    int pos = (ptd.m_idata[IntIdx::workgroup][i] > 0 ? 0 : 5);
                    int grade = ptd.m_idata[IntIdx::school_grade][i];
                    counts[pos + getSchoolType(grade) - SchoolType::college] = 1;
                }
                return {counts[0], counts[1], counts[2], counts[3], counts[4],
                        counts[5], counts[6], counts[7], counts[8], counts[9]};
            },
            reduce_ops);

    std::array<Long, 10> counts = {amrex::get<0>(r), amrex::get<1>(r), amrex::get<2>(r), amrex::get<3>(r), amrex::get<4>(r),
                                   amrex::get<5>(r), amrex::get<6>(r), amrex::get<7>(r), amrex::get<8>(r), amrex::get<9>(r)};
    ParallelDescriptor::ReduceLongSum(&counts[0], 10, ParallelDescriptor::IOProcessorNumber());
    if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
        int total_educators = 0;
        int total_students = 0;
        for (int i = 0; i < 5; i++) {
            total_educators += counts[i];
            total_students += counts[i + 5];
        }
        Print() << std::fixed << std::setprecision(1) << "School counts: (educators, students, ratio)\n"
                << "  College    " << counts[0] << " " << counts[5] << " " << ((Real)counts[5] / counts[0]) << "\n"
                << "  High       " << counts[1] << " " << counts[6] << " " << ((Real)counts[6] / counts[1]) << "\n"
                << "  Middle     " << counts[2] << " " << counts[7] << " " << ((Real)counts[7] / counts[2]) << "\n"
                << "  Elementary " << counts[3] << " " << counts[8] << " " << ((Real)counts[8] / counts[3]) << "\n"
                << "  Childcare  " << counts[4] << " " << counts[9] << " " << ((Real)counts[9] / counts[4]) << "\n"
                << "  Total      " << total_educators << " " << total_students << " " << ((Real)total_students / total_educators)
                << "\n";
    }
}

void AgentContainer::printAgeGroupCounts () const {
    ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
    auto r = ParticleReduce<ReduceData<int, int, int, int, int, int>>(
            *this,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                                 const int i) noexcept -> GpuTuple<int, int, int, int, int, int> {
                int counts[6] = {0, 0, 0, 0, 0, 0};
                int age_group = ptd.m_idata[IntIdx::age_group][i];
                counts[age_group] = 1;
                return {counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]};
            },
            reduce_ops);

    std::array<Long, 6> counts = {amrex::get<0>(r), amrex::get<1>(r), amrex::get<2>(r),
                                  amrex::get<3>(r), amrex::get<4>(r), amrex::get<5>(r)};
    ParallelDescriptor::ReduceLongSum(&counts[0], 6, ParallelDescriptor::IOProcessorNumber());
    if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
        int total_agents = 0;
        for (int i = 0; i < 6; i++) {
            total_agents += counts[i];
        }
        Print() << std::fixed << std::setprecision(1) << "Age group counts (percentage):\n"
                << "  under 5   " << counts[0] << " " << 100.0 * (Real)counts[0] / total_agents << "\n"
                << "  5 to 17    " << counts[1] << " " << 100.0 * (Real)counts[1] / total_agents << "\n"
                << "  18 to 29   " << counts[2] << " " << 100.0 * (Real)counts[2] / total_agents << "\n"
                << "  30 to 49   " << counts[3] << " " << 100.0 * (Real)counts[3] / total_agents << "\n"
                << "  50 to 64   " << counts[4] << " " << 100.0 * (Real)counts[4] / total_agents << "\n"
                << "  over 64    " << counts[5] << " " << 100.0 * (Real)counts[5] / total_agents << "\n"
                << "  Total      " << total_agents << "\n";
    }
}

void AgentContainer::printStudentTeacherCountsPerCommunity() const {
    const int n_school_types = 5; // College, High, Middle, Elementary, Childcare
    // +2 for total teachers/students, +2 for infected/withdrawn students
    const int ncomp = 2 * n_school_types + 2 + 2;
    const int idx_infected_students = 2 * n_school_types + 2;
    const int idx_withdrawn_students = 2 * n_school_types + 3;
    const char* school_names[5] = {"College", "High", "Middle", "Elementary", "Childcare"};

    // Prepare a domain-wide accumulator for all levels
    std::array<Real, ncomp> domain_totals = {0.0};

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        const auto& geom = Geom(lev);
        Box domain = geom.Domain();
        int nx = domain.length(0);
        int ny = domain.length(1);
        BoxArray ba = this->ParticleBoxArray(lev);
        const auto& dmap = this->ParticleDistributionMap(lev);
        MultiFab mf(ba, dmap, ncomp, 0);
        mf.setVal(0);

        const auto plo = geom.ProbLoArray();
        const auto dxi = geom.InvCellSizeArray();

        ParticleToMesh(
            *this, mf, lev,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd, int i,
                                  Array4<Real> const& count) {
                auto p = ptd.m_aos[i];
                auto iv = getParticleCell(p, plo, dxi, domain);
                int school_id = ptd.m_idata[IntIdx::school_id][i];
                int grade = ptd.m_idata[IntIdx::school_grade][i];
                int workgroup = ptd.m_idata[IntIdx::workgroup][i];
                int home_i = ptd.m_idata[IntIdx::home_i][i];
                int home_j = ptd.m_idata[IntIdx::home_j][i];
                int comm_to = home_i + home_j * domain.length(0); // linear index for community
                IntVect iv_from_comm = domain.atOffset(comm_to);

                if (!domain.contains(iv)) {
                    printf("Warning: Particle at (%g,%g) binned to out-of-bounds cell (%d,%d)\n", p.pos(0), p.pos(1), iv[0], iv[1]);
                }
                if (iv != iv_from_comm) {
                    printf("Warning: iv (%d,%d) != iv_from_comm (%d,%d)\n", iv[0], iv[1], iv_from_comm[0], iv_from_comm[1]);
                }

                if (school_id > 0) {
                    int school_type = getSchoolType(grade) - SchoolType::college;
                    if (school_type >= 0 && school_type < n_school_types) {
                        if (workgroup > 0) {
                            Gpu::Atomic::AddNoRet(&count(iv_from_comm, school_type), 1.0_rt);
                            Gpu::Atomic::AddNoRet(&count(iv_from_comm, 2 * n_school_types), 1.0_rt);
                        } else {
                            Gpu::Atomic::AddNoRet(&count(iv_from_comm, n_school_types + school_type), 1.0_rt);
                            Gpu::Atomic::AddNoRet(&count(iv_from_comm, 2 * n_school_types + 1), 1.0_rt);
                            // Count withdrawn students
                            int withdrawn = ptd.m_idata[IntIdx::withdrawn][i];
                            if (withdrawn) {
                                Gpu::Atomic::AddNoRet(&count(iv_from_comm, idx_withdrawn_students), 1.0_rt);
                            }
                            // Count infected students (assume single disease, index 0)
                            int status = ptd.m_runtime_idata[i0(0) + IntIdxDisease::status][i];
                            if (status == Status::infected) {
                                Gpu::Atomic::AddNoRet(&count(iv_from_comm, idx_infected_students), 1.0_rt);
                            }
                        }
                    }
                }
            },
            false
        );

        // Flatten MultiFab data into a 1D array (row-major: i, j, component)
        std::vector<Real> flat_data(nx * ny * ncomp, 0.0);
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const auto& arr = mf[mfi].array();
            const Box& bx = mfi.validbox();
            for (IntVect iv = bx.smallEnd(); iv <= bx.bigEnd(); bx.next(iv)) {
                int i = iv[0] - domain.smallEnd(0);
                int j = iv[1] - domain.smallEnd(1);
                for (int c = 0; c < ncomp; ++c) {
                    flat_data[(i * ny + j) * ncomp + c] += arr(iv, c);
                }
            }
        }

        // Reduce the flat array across all ranks
        ParallelDescriptor::ReduceRealSum(flat_data.data(), flat_data.size(), ParallelDescriptor::IOProcessorNumber());

        // On IOProc, print per-community counts and accumulate domain-wide totals
        if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    Real total_teachers = flat_data[(i * ny + j) * ncomp + 2 * n_school_types];
                    Real total_students = flat_data[(i * ny + j) * ncomp + 2 * n_school_types + 1];
                    Real infected_students = flat_data[(i * ny + j) * ncomp + idx_infected_students];
                    Real withdrawn_students = flat_data[(i * ny + j) * ncomp + idx_withdrawn_students];
                    if (total_teachers > 0 || total_students > 0) {
                        Print() << "Level " << lev << " Community (" << (i + domain.smallEnd(0)) << "," << (j + domain.smallEnd(1)) << "):\n";
                        Print() << "  Teachers: ";
                        for (int t = 0; t < n_school_types; ++t)
                            Print() << flat_data[(i * ny + j) * ncomp + t] << " ";
                        Print() << " | Total: " << total_teachers << "\n";
                        Print() << "  Students: ";
                        for (int t = 0; t < n_school_types; ++t)
                            Print() << flat_data[(i * ny + j) * ncomp + n_school_types + t] << " ";
                        Print() << " | Total: " << total_students << "\n";
                        Print() << "  Infected students: " << infected_students << "\n";
                        Print() << "  Withdrawn students: " << withdrawn_students << "\n";
                    }
                    // Accumulate domain-wide totals
                    for (int c = 0; c < ncomp; ++c) {
                        domain_totals[c] += flat_data[(i * ny + j) * ncomp + c];
                    }
                }
            }
        }
    }

    // Reduce domain-wide totals across all ranks
    ParallelDescriptor::ReduceRealSum(domain_totals.data(), ncomp, ParallelDescriptor::IOProcessorNumber());

    // Print domain-wide totals (only on IOProc)
    if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
        Print() << "Domain-wide School counts: (educators, students, ratio)\n";
        for (int t = 0; t < n_school_types; ++t) {
            Real teachers = domain_totals[t];
            Real students = domain_totals[n_school_types + t];
            Real ratio = (teachers > 0) ? (students / teachers) : 0.0;
            Print() << "  " << school_names[t] << "    " << teachers << " " << students << " " << ratio << "\n";
        }
        Real total_teachers = domain_totals[2 * n_school_types];
        Real total_students = domain_totals[2 * n_school_types + 1];
        Real total_ratio = (total_teachers > 0) ? (total_students / total_teachers) : 0.0;
        Print() << "  Total      " << total_teachers << " " << total_students << " " << total_ratio << "\n";
        Print() << "  Infected students: " << domain_totals[idx_infected_students] << "\n";
        Print() << "  Withdrawn students: " << domain_totals[idx_withdrawn_students] << "\n";
    }
    // Note: With the above reductions, both per-community and domain-wide counts are correct regardless of number of ranks.
}

void AgentContainer::printStudentTeacherCountsPerUnit(const amrex::iMultiFab& FIPS_mf, const amrex::iMultiFab& geoid_mf, bool use_census, int num_units, const std::vector<int>& FIPS_codes) const {
    const int n_school_types = 5; // College, High, Middle, Elementary, Childcare
    // +2 for total teachers/students, +2 for infected/withdrawn students
    const int ncomp = 2 * n_school_types + 2 + 2;
    const int idx_infected_students = 2 * n_school_types + 2;
    const int idx_withdrawn_students = 2 * n_school_types + 3;
    const char* school_names[5] = {"College", "High", "Middle", "Elementary", "Childcare"};

    // Build FIPS code to 0-based index map
    std::unordered_map<int, int> fips_to_index;
    for (int i = 0; i < FIPS_codes.size(); ++i) {
        fips_to_index[FIPS_codes[i]] = i;
    }

    // Prepare a domain-wide accumulator for all units
    std::vector<Real> domain_totals(ncomp, 0.0);
    // Per-unit counts: flat array [unit * ncomp + c]
    std::vector<Real> flat_data(num_units * ncomp, 0.0);

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        const auto& geom = Geom(lev);
        BoxArray ba = this->ParticleBoxArray(lev);
        const auto& dmap = this->ParticleDistributionMap(lev);
        // For each tile, get the relevant FIPS array
        const auto& plev = GetParticles(lev);
        for (MFIter mfi(ba, dmap); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto ptile_it = plev.find(std::make_pair(gid, tid));
            if (ptile_it == plev.end()) continue;
            const auto& ptile = ptile_it->second;
            const auto& soa = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
            auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
            auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            // For infected status
            auto status_ptr = soa.GetIntData(IntIdxDisease::status).data(); // assuming single disease, adjust if needed
            // Set up FIPS_arr and geoid_arr as Array4<const int>
            Array4<const int> FIPS_arr;
            Array4<const int> geoid_arr;
            if (use_census) {
                FIPS_arr = FIPS_mf[mfi].array();
            } else {
                geoid_arr = geoid_mf[mfi].array();
            }

            for (size_t i = 0; i < np; ++i) {
                int unit = -1;
                if (use_census) {
                    int fips_code = FIPS_arr(home_i_ptr[i], home_j_ptr[i], 0);
                    auto it = fips_to_index.find(fips_code);
                    if (it == fips_to_index.end()) continue;
                    unit = it->second;
                } else {
                    int geoid_code = geoid_arr(home_i_ptr[i], home_j_ptr[i], 0);
                    auto it = fips_to_index.find(geoid_code);
                    if (it == fips_to_index.end()) continue;
                    unit = it->second;
                }
                if (unit < 0 || unit >= num_units) continue;
                int school_id = school_id_ptr[i];
                int grade = school_grade_ptr[i];
                int workgroup = workgroup_ptr[i];
                if (school_id > 0) {
                    int school_type = getSchoolType(grade) - SchoolType::college;
                    if (school_type >= 0 && school_type < n_school_types) {
                        if (workgroup > 0) {
                            flat_data[unit * ncomp + school_type] += 1.0;
                            flat_data[unit * ncomp + 2 * n_school_types] += 1.0;
                        } else {
                            flat_data[unit * ncomp + n_school_types + school_type] += 1.0;
                            flat_data[unit * ncomp + 2 * n_school_types + 1] += 1.0;
                            // Withdrawn
                            if (withdrawn_ptr[i]) {
                                flat_data[unit * ncomp + idx_withdrawn_students] += 1.0;
                            }
                            // Infected (assume single disease, index 0)
                            int status = status_ptr[i];
                            if (status == Status::infected) {
                                flat_data[unit * ncomp + idx_infected_students] += 1.0;
                            }
                        }
                    }
                }
            }
        }
    }

    // Reduce the flat array across all ranks
    ParallelDescriptor::ReduceRealSum(flat_data.data(), flat_data.size(), ParallelDescriptor::IOProcessorNumber());

    // On IOProc, print per-unit counts and accumulate domain-wide totals
    if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
        for (int unit = 0; unit < num_units; ++unit) {
            Real total_teachers = flat_data[unit * ncomp + 2 * n_school_types];
            Real total_students = flat_data[unit * ncomp + 2 * n_school_types + 1];
            Real infected_students = flat_data[unit * ncomp + idx_infected_students];
            Real withdrawn_students = flat_data[unit * ncomp + idx_withdrawn_students];
            if (total_teachers > 0 || total_students > 0) {
                Print() << (use_census ? "Census" : "UrbanPop") << " Unit " << FIPS_codes[unit] << ":\n";
                Print() << "  Teachers: ";
                for (int t = 0; t < n_school_types; ++t)
                    Print() << flat_data[unit * ncomp + t] << " ";
                Print() << " | Total: " << total_teachers << "\n";
                Print() << "  Students: ";
                for (int t = 0; t < n_school_types; ++t)
                    Print() << flat_data[unit * ncomp + n_school_types + t] << " ";
                Print() << " | Total: " << total_students << "\n";
                Print() << "  Infected students: " << infected_students << "\n";
                Print() << "  Withdrawn students: " << withdrawn_students << "\n";
            }
            // Accumulate domain-wide totals
            for (int c = 0; c < ncomp; ++c) {
                domain_totals[c] += flat_data[unit * ncomp + c];
            }
        }
    }

    // Reduce domain-wide totals across all ranks
    ParallelDescriptor::ReduceRealSum(domain_totals.data(), ncomp, ParallelDescriptor::IOProcessorNumber());

    // Print domain-wide totals (only on IOProc)
    if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
        Print() << (use_census ? "Census" : "UrbanPop") << " Domain-wide School counts: (educators, students, ratio)\n";
        for (int t = 0; t < n_school_types; ++t) {
            Real teachers = domain_totals[t];
            Real students = domain_totals[n_school_types + t];
            Real ratio = (teachers > 0) ? (students / teachers) : 0.0;
            Print() << "  " << school_names[t] << "    " << teachers << " " << students << " " << ratio << "\n";
        }
        Real total_teachers = domain_totals[2 * n_school_types];
        Real total_students = domain_totals[2 * n_school_types + 1];
        Real total_ratio = (total_teachers > 0) ? (total_students / total_teachers) : 0.0;
        Print() << "  Total      " << total_teachers << " " << total_students << " " << total_ratio << "\n";
        Print() << "  Infected students: " << domain_totals[idx_infected_students] << "\n";
        Print() << "  Withdrawn students: " << domain_totals[idx_withdrawn_students] << "\n";
    }
    // Note: With the above reductions, both per-unit and domain-wide counts are correct regardless of number of ranks.
}

/*! \brief Community-wise school infection stats and status tracker */
void AgentContainer::updateSchoolInfection(iMultiFab& a_school_stats) /*!< Community-wise school infection stats and status tracker */
{
    BL_PROFILE("AgentContainer::updateSchoolInfo");

    int school_closure_flag = 0;
    int debug_print = 1;
    std::string closure_option = "by_community";
    double sc_infection_threshold = 0.20;
    int sc_length = 10;

    amrex::ParmParse pp("agent");
    pp.query("set_school_closure", school_closure_flag);
    pp.query("school_closure_option", closure_option);
    pp.query("school_closure_threshold", sc_infection_threshold);
    pp.query("school_closure_length",  sc_length);
    pp.query("school_closure_debug_print", debug_print);

    // if (!school_closure_flag) { return; }

    AMREX_ASSERT(sc_infection_threshold >= 0.0 && sc_infection_threshold <= 1.0);

    int school_closure_option = SchoolPolicy::SchoolDismissalType::ByCommunity;
    if (closure_option == "by_school"){school_closure_option = SchoolPolicy::SchoolDismissalType::BySchool; }
    else if (closure_option == "by_fips"){school_closure_option = SchoolPolicy::SchoolDismissalType::ByFIPS; }

    bool is_census = (ic_type == ExaEpi::ICType::Census);

    /*! \brief By FIPS */
    if (school_closure_option == SchoolPolicy::SchoolDismissalType::ByFIPS) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_fips_codes.empty(), "FIPS data must be initialized for ByFIPS closures.");
        const int num_fips = m_fips_codes.size();
        if (num_fips == 0) return;

        // Create a GPU-friendly lookup table to map FIPS code -> 0-based index
        int min_fips = m_fips_codes[0];
        int max_fips = m_fips_codes[0];
        for (int code : m_fips_codes) {
            if (code < min_fips) min_fips = code;
            if (code > max_fips) max_fips = code;
        }
        amrex::Gpu::DeviceVector<int> gpu_fips_map(max_fips - min_fips + 1, -1);

        {
            amrex::Vector<int> host_map(gpu_fips_map.size(), -1);
            for (int i = 0; i < num_fips; ++i) {
                host_map[m_fips_codes[i] - min_fips] = i;
            }
            // Copy to device asynchronously
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host_map.begin(), host_map.end(), gpu_fips_map.begin());
        }
        amrex::Gpu::streamSynchronize(); // Required !

        auto const* map_ptr = gpu_fips_map.data();
        auto stats_ptr = m_fips_student_stats.data();
        amrex::Gpu::DeviceVector<int> device_student_total(num_fips, 0);
        auto* total_ptr = device_student_total.data();

        // 1. Reset infection counts each day
        amrex::ParallelFor(num_fips, [=] AMREX_GPU_DEVICE (int i) noexcept {
            stats_ptr[i * SchoolPolicy::SchoolStats::nattribs + SchoolPolicy::SchoolStats::SchoolInfectionCount] = 0;
        });


        // 2.Count students and infections per FIPS
        for (int lev = 0; lev <= finestLevel(); ++lev) {
            auto& plev = GetParticles(lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto& ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
                auto& soa = ptile.GetStructOfArrays();
                const auto np = ptile.numParticles();
                if (np == 0) continue;
    
                auto fips_ptr = soa.GetIntData(IntIdx::fips).data();
                auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
                auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
                auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
                auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
                auto withdrawn_date_ptr = soa.GetIntData(IntIdx::withdrawn_date).data();
                auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
                auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();
    
                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int p) noexcept {
                    int school_id = (is_census) ? school_id_ptr[p] : getSchoolType(school_grade_ptr[p]);
                    bool at_school = (is_census) ? school_id > SchoolCensusIDType::none : school_id > SchoolType::none;
                    bool is_daycare = (is_census) ? school_id >= SchoolCensusIDType::daycare_5 : school_id == SchoolType::daycare;
                    if (age_group_ptr[p] == 1 && at_school && !is_daycare) {
                        int fips_idx = map_ptr[fips_ptr[p] - min_fips];
                        if (fips_idx != -1) {
                            amrex::Gpu::Atomic::AddNoRet(&total_ptr[fips_idx], 1);

                            int reopen_day_offset = fips_idx * SchoolPolicy::SchoolStats::nattribs + SchoolPolicy::SchoolStats::SchoolReopenDay;
                            bool is_withdrawn = (withdrawn_ptr[p] == 1 || (hosp_i_ptr[p] > -1 && hosp_j_ptr[p] > -1))
                                                && (withdrawn_date_ptr[p] >= stats_ptr[reopen_day_offset]);
                            if (is_withdrawn) {
                                int offset = fips_idx * SchoolPolicy::SchoolStats::nattribs + SchoolPolicy::SchoolStats::SchoolInfectionCount;
                                amrex::Gpu::Atomic::AddNoRet(&stats_ptr[offset], 1);
                            }
                        }
                    }
                });
            }
        }

        // 3. Update school status based on FIPS-wide infection counts
        amrex::ParallelFor(num_fips, [=] AMREX_GPU_DEVICE (int i) noexcept {
            int dismissal_offset = i * SchoolPolicy::SchoolStats::nattribs + SchoolPolicy::SchoolStats::SchoolDismissal;
            int infection_offset = i * SchoolPolicy::SchoolStats::nattribs + SchoolPolicy::SchoolStats::SchoolInfectionCount;
            int status_day_offset = i * SchoolPolicy::SchoolStats::nattribs + SchoolPolicy::SchoolStats::SchoolStatusDayCount;
            int reopen_day_offset = i * SchoolPolicy::SchoolStats::nattribs + SchoolPolicy::SchoolStats::SchoolReopenDay;
    
            int old_status = stats_ptr[dismissal_offset];
            int new_status;

            int school_closure_threshold = sc_infection_threshold * total_ptr[i];
            
            amrex::Gpu::Atomic::AddNoRet(&stats_ptr[status_day_offset], 1);
            if (stats_ptr[infection_offset] > school_closure_threshold) {
                stats_ptr[dismissal_offset] = 1; // 1 = closed
                stats_ptr[status_day_offset] = 0;
            } else {
                stats_ptr[dismissal_offset] = 0; // 0 = open
                stats_ptr[reopen_day_offset] = m_current_day;
                stats_ptr[status_day_offset] = 0;

            }
        });
    
        // 4. Update agent status based on FIPS-wide school status
        for (int lev = 0; lev <= finestLevel(); ++lev) {
            auto& plev = GetParticles(lev);
            for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto& ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
                auto& soa = ptile.GetStructOfArrays();
                const auto np = ptile.numParticles();
                if (np == 0) continue;

                auto fips_ptr = soa.GetIntData(IntIdx::fips).data();
                auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
                auto school_closed_ptr = soa.GetIntData(IntIdx::school_closed).data();
                auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();

                amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int p) noexcept {
                    int school_id = (is_census) ? school_id_ptr[p] : getSchoolType(school_grade_ptr[p]);
                    bool at_school = (is_census) ? school_id > SchoolCensusIDType::none : school_id > SchoolType::none;
                    bool is_daycare = (is_census) ? school_id >= SchoolCensusIDType::daycare_5 : school_id == SchoolType::daycare;

                    // Update both students and teachers based on their FIPS
                    if (at_school && !is_daycare) {
                        int fips_code = fips_ptr[p];
                        if (fips_code >= min_fips && fips_code < min_fips + gpu_fips_map.size()){
                            int fips_idx = map_ptr[fips_code - min_fips];
                            if (fips_idx != -1) {
                                int dismissal_status = stats_ptr[fips_idx * SchoolPolicy::SchoolStats::nattribs + SchoolPolicy::SchoolStats::SchoolDismissal];
                                school_closed_ptr[p] = (dismissal_status == 1) ? 1 : 0; 
                            }
                        }
                    }
                });
            }
        }
        amrex::Gpu::streamSynchronize();
    } else { // By Community or By School Type

        int offset = is_census ? static_cast<int>(SchoolCensusIDType::total) : static_cast<int>(SchoolType::total);
        AMREX_ALWAYS_ASSERT(a_school_stats.nComp() == offset * SchoolPolicy::SchoolStats::nattribs);

        for (int lev = 0; lev <= finestLevel(); ++lev)
        {
            auto& plev = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                int gid = mfi.index();
                int tid = mfi.LocalTileIndex();
                auto& ptile = plev[std::make_pair(gid, tid)];
                auto& soa = ptile.GetStructOfArrays();
                const auto np = ptile.numParticles();
                if (np == 0) { continue; }

                auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
                auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
                auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
                auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
                auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
                auto school_closed_ptr = soa.GetIntData(IntIdx::school_closed).data();
                auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
                auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();
                auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
                auto withdrawn_date_ptr = soa.GetIntData(IntIdx::withdrawn_date).data();

                auto status_arr = a_school_stats[mfi].array();
                const auto& counts_arr = m_student_counts[mfi].array();

                const Box& bx = mfi.tilebox();

                //DEBUGGING PURPOSES
                // check that when cur_time == 0, everything is zero
                if (m_current_day == 0) {
                    // Assert that all particle data and grid data are zero
                    bool allZero = true;

                    // Check grid data if all particle data was zero
                    if (allZero) {
                        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                                for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
                                    for (int comp = 0; comp < status_arr.nComp(); ++comp) {
                                        if (status_arr(i, j, k, comp) != 0) {
                                            allZero = false;
                                            break;
                                        }
                                    }
                                    for (int comp = 0; comp < counts_arr.nComp(); ++comp) {
                                        if (counts_arr(i, j, k, comp) != 0) {
                                            allZero = false;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    AMREX_ALWAYS_ASSERT(allZero);
                }

                //Reset counts each day
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        for (int ii = 0; ii <  offset; ii ++){
                            status_arr(i, j, k, ii + offset * SchoolPolicy::SchoolStats::SchoolInfectionCount) = 0;
                        }

                    });
                Gpu::synchronize();

                // Infection Counts at a given day
                amrex::ParallelFor(np,
                    [=] AMREX_GPU_DEVICE(int p) noexcept
                    {
                        int home_i = home_i_ptr[p];
                        int home_j = home_j_ptr[p];
                        // int school_id = getSchoolType(school_grade_ptr[p]);
                        int school_id = school_id_ptr[p];
                        if (age_group_ptr[p] == 1) { // Exclude DayCare/Playgroud
                            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(school_id >= 0, "school_ptr can't be negative");
                            if (school_id > 0 && (withdrawn_ptr[p] == 1 || hosp_i_ptr[p] > -1 || hosp_j_ptr[p] > -1)) {
                                // Add to the community-level counter (index 0)
                                if (withdrawn_date_ptr[p] >= status_arr(home_i, home_j, 0, offset * SchoolPolicy::SchoolStats::SchoolReopenDay)) {
                                    amrex::Gpu::Atomic::AddNoRet(&status_arr(home_i, home_j, 0, offset * SchoolPolicy::SchoolStats::SchoolInfectionCount), 1);
                                }
                                // Also add to the per-school counter (index = school_id)
                                if (withdrawn_date_ptr[p] >= status_arr(home_i, home_j, 0, school_id + offset * SchoolPolicy::SchoolStats::SchoolReopenDay)) {
                                    amrex::Gpu::Atomic::AddNoRet(&status_arr(home_i, home_j, 0, school_id + offset * SchoolPolicy::SchoolStats::SchoolInfectionCount), 1);
                                }
                            }
                        }
                    });
                Gpu::synchronize();

                // close community/school based on infection count
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        int start_dismiss = (school_closure_option == SchoolPolicy::SchoolDismissalType::BySchool) ? 1 : 0;
                        int stop_dismiss = 1;
                        if (is_census) {
                            stop_dismiss = (school_closure_option == SchoolPolicy::SchoolDismissalType::BySchool) ? SchoolCensusIDType::daycare_5 : 1; // exclude daycare otherwise use SchoolCensusIDType::total
                        } else {
                            stop_dismiss = (school_closure_option == SchoolPolicy::SchoolDismissalType::BySchool) ? SchoolType::daycare : 1; // exclude daycare otherwise use SchoolType::total
                        }

                        for (int ii = start_dismiss; ii < stop_dismiss; ii++) //exclude DayCare
                        {
                            int student_total = 0;
                            if (school_closure_option == SchoolPolicy::SchoolDismissalType::ByCommunity) {
                                if (is_census) {
                                    student_total = counts_arr(i, j, k, 0) - counts_arr(i, j, k, SchoolCensusIDType::daycare_5);
                                } else {
                                    student_total = counts_arr(i, j, k, 0) - counts_arr(i, j, k, SchoolType::daycare);
                                }
                            } else { // BySchool
                                student_total = counts_arr(i, j, k, ii);
                            }

                            if (student_total == 0) continue;

                            int& dismissal_status = status_arr(i, j, k, ii + offset * SchoolPolicy::SchoolStats::SchoolDismissal);
                            int& day_count = status_arr(i, j, k, ii + offset * SchoolPolicy::SchoolStats::SchoolStatusDayCount);                
                            int infection_count = status_arr(i, j, k, ii + offset * SchoolPolicy::SchoolStats::SchoolInfectionCount);

                            amrex::Gpu::Atomic::AddNoRet(&day_count, 1); // increment day counter when open as well?
                    
                            if (dismissal_status == 0) {        
                                int thresh = static_cast<int>(sc_infection_threshold * student_total) + 1;
                                if (infection_count >= thresh)
                                {
                                    dismissal_status = 1; // Close the school
                                    if ( i == 14 && j == 2 && debug_print) {
                                        printf("School %d (%d, %d, %d) is now closed at day %d (dissmissal = %d). Infection number: MultiFab = %d, Day = %d\n",
                                            ii, i, j, k, m_current_day,
                                            dismissal_status,
                                            infection_count,
                                            day_count);
                                    }
                                    day_count = 0; // Reset the closure day counter
                                }
                                else {
                                    if ( i == 14 && j == 2 && debug_print) {
                                        printf("School %d (%d, %d, %d) is currenly opened %d. Infection number: MultiFab = %d, Day = %d\n",
                                            ii, i, j, k,
                                            dismissal_status,
                                            infection_count,
                                            day_count);
                                    }
                                }
                            } else { // School is closed
                                if (day_count >= sc_length) {
                                    dismissal_status = 0;
                                    status_arr(i, j, k, ii + offset * SchoolPolicy::SchoolStats::SchoolReopenDay) = m_current_day;

                                    if ( i == 14 && j == 2 && debug_print) {
                                        printf("School %d (%d, %d, %d) has now opened at day %d (dissmissal = %d). Infection number: MultiFab = %d, Day = %d\n",
                                            ii, i, j, k, m_current_day,
                                            dismissal_status,
                                            infection_count,
                                            day_count);
                                    }
                                    day_count = 0;

                                } else {
                                    if (i == 14 && j == 2 && debug_print) {
                                        printf("School %d (%d, %d, %d) is currently closed %d. Infection number: MultiFab = %d, Day = %d\n",
                                            ii, i, j, k,
                                            dismissal_status,
                                            infection_count,
                                            day_count);
                                    }
                                }
                            }
                        }

                    });
                Gpu::synchronize();

                // for debugging purposes
                amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                    {
                        if (i == 14 && j == 2 && debug_print) {
                            int start_dismiss = (school_closure_option == SchoolPolicy::SchoolDismissalType::BySchool) ? 1 : 0;
                            int stop_dismiss = 1;
                            if (is_census) {
                                stop_dismiss = (school_closure_option == SchoolPolicy::SchoolDismissalType::BySchool) ? SchoolCensusIDType::daycare_5 : 1; // exclude daycare otherwise use SchoolCensusIDType::total
                            } else {
                                stop_dismiss = (school_closure_option == SchoolPolicy::SchoolDismissalType::BySchool) ? SchoolType::daycare : 1; // exclude daycare otherwise use SchoolType::total
                            }
        
                            for (int ii = start_dismiss; ii < stop_dismiss; ii++) //exclude DayCare
                            {
                                int student_total = 0;
                                if (school_closure_option == SchoolPolicy::SchoolDismissalType::ByCommunity) {
                                    if (is_census) {
                                        student_total = counts_arr(i, j, k, 0) - counts_arr(i, j, k, SchoolCensusIDType::daycare_5);
                                    } else {
                                        student_total = counts_arr(i, j, k, 0) - counts_arr(i, j, k, SchoolType::daycare);
                                    }
                                } else { // BySchool
                                    student_total = counts_arr(i, j, k, ii);
                                }

                                printf("School %d (%d, %d, %d) has Infection number:  MultiFab = %d, Day = %d, with total students %d\n",
                                    ii, i, j, k,
                                    status_arr(i, j, k,  ii + offset * SchoolPolicy::SchoolStats::SchoolInfectionCount),
                                    status_arr(i, j, k,  ii + offset * SchoolPolicy::SchoolStats::SchoolStatusDayCount),
                                    student_total
                                    );
                            }
                        }
                    });

                Gpu::synchronize();

                // update school_closed_ptr for both teachers and children based on school closure status
                if (school_closure_flag){
                    amrex::ParallelFor( np,
                    [=] AMREX_GPU_DEVICE (int p) noexcept
                    {
                        bool update_this_agent = true;
                        if (is_census) {
                            int agent_school_id = school_id_ptr[p];
                            update_this_agent = (agent_school_id > 0 && agent_school_id != SchoolCensusIDType::daycare_5);
                        } else {
                            int agent_school_id = getSchoolType(school_grade_ptr[p]);
                            update_this_agent = (agent_school_id >= 0 && agent_school_id != SchoolType::daycare);
                        }

                        if (update_this_agent) {
                            int school_type = 0; // default to community-level school status
                            if (school_closure_option == SchoolPolicy::SchoolDismissalType::BySchool) {
                                if (is_census) {
                                    school_type = school_id_ptr[p];
                                } else {
                                    school_type = getSchoolType(school_grade_ptr[p]);
                                }
                            }
                            int school_status = status_arr(home_i_ptr[p], home_j_ptr[p], 0, school_type + offset * SchoolPolicy::SchoolStats::SchoolDismissal);
                            school_closed_ptr[p] = (school_status == 1) ? 1 : 0; 

                            AMREX_ASSERT(school_closed_ptr[p] == ((school_status == 1) ? 1 : 0));
                        }
                    });
                    Gpu::synchronize();
                }
            }
        }
    }

}

/*! \brief Sets a default 32-bit school attendance schedule for all student agents.

   For students assigned to a school, the schedule is set to active (MSB=1), and
   the daily bits are set to 'attend' (1) for the specified number of steps.
   For all other agents, the schedule is set to inactive (mask = 0).    
 */
void AgentContainer::initializeSchoolAttendanceForAll(int window_size)
{
    BL_PROFILE("AgentContainer::initializeSchoolAttendanceForAll");
    if (window_size > 30) {
        amrex::Abort("ERROR: Number of schedule days (window_size) cannot exceed 30 for a 32-bit mask.");
    }
    // MSB=0 (inactive) and all daily bits=0 (don't attend).
    const uint32_t inactive_mask = 0;
    // (1U << 31) sets the active flag (MSB = 1)& ((1U << window_size) - 1) sets the daily attendance bits.
    const uint32_t attendance_mask = (1U << 31) | ((1U << window_size) - 1);
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto mask_ptr = soa.GetIntData(IntIdx::school_attendance_mask).data();
            auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                if (age_group_ptr[i] < AgeGroups::a18to29 && school_id_ptr[i] > SchoolType::none) {
                    mask_ptr[i] = attendance_mask;
                } else {
                    mask_ptr[i] = inactive_mask;
                }
            });
            Gpu::synchronize();
        }
    }
    Print() << "[INFO] Set initial 32-bit school attendance for " << window_size << " days. (Day " << m_current_day << " to Day " << m_current_day + window_size << ").\n";
}

/*! \brief Applies user-defined attendance policies to modify student schedules for a specific time window.  
   It is time-aware, calculating schedules only for policies that are active within the 30-day window starting at time_offset.
  + First, pre-calculates 32-bit masks for each policy group.
  + Second,students are randomly assigned to a group and their schedule is updated.
 */
void AgentContainer::assignAttendanceFromPolicies(const std::vector<ExaEpi::SchoolPolicy>& policies, int time_offset)
{
    BL_PROFILE("AgentContainer::assignAttendanceFromPolicies");
    if (policies.empty()) { 
        amrex::Print() << "[INFO] No school policies defined. Skipping attendance assignment.\n";
        return; 
    }

    std::vector<SchoolPolicy::PolicyGPUData> host_policy_data;
    const uint32_t MASK_ACTIVE_AND_ALWAYS_ATTEND = 0xFFFFFFFFU & ~(1U << 30);

    for (const auto& policy : policies) {
        // Only consider policies that are active in the new time 30 day window
        if (policy.end_day <= time_offset || policy.start_day >= time_offset + 30) {
            continue; 
        }

        SchoolPolicy::PolicyGPUData data;
        data.school_type = policy.school_type;

        try {
            data.fips = std::stoi(policy.fips);
        } catch (const std::invalid_argument& e) {
            amrex::Abort("Invalid FIPS code in policy: " + policy.fips);
        }

        std::string pattern = policy.schedule_pattern;
        std::string unique_chars = pattern;
        unique_chars.erase(std::remove(unique_chars.begin(), unique_chars.end(), '_'), unique_chars.end());
        std::sort(unique_chars.begin(), unique_chars.end());
        unique_chars.erase(std::unique(unique_chars.begin(), unique_chars.end()), unique_chars.end());

        data.num_groups = unique_chars.length();
        
        // Debugging
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- POLICY DEBUGGER ---\n";
            amrex::Print() << "  - Raw Pattern String: '" << pattern << "'\n";
            amrex::Print() << "  - Unique Chars Found: '" << unique_chars << "'\n";
            amrex::Print() << "  - Calculated num_groups: " << data.num_groups << "\n";
            amrex::Print() << "  - FIPS Code (as int): " << data.fips << " (0 means ALL)\n";
            amrex::Print() << "-----------------------\n\n";
        }
        if (data.num_groups == 0 && pattern.find_first_not_of('_') == std::string::npos) {
            data.num_groups = 1;
        }

        if (data.num_groups > 4) { amrex::Abort("Policies with more than 4 unique groups are not supported."); }
        if (data.num_groups == 0) continue;

        // Pre-calculate the 32-bit mask for each group for the window [time_offset, time_offset + 30)
        for (int i = 0; i < data.num_groups; ++i) {
            char group_char = unique_chars.empty() ? 'A' : unique_chars[i]; // in case user put all '_____'
            uint32_t group_mask = MASK_ACTIVE_AND_ALWAYS_ATTEND;

            for (int d = 0; d < 30; ++d) {
                int day_in_sim = time_offset + d; 

                if (day_in_sim >= policy.start_day && day_in_sim < policy.end_day) {
                    // Determine the character for this day from the repeating pattern
                    char day_char = pattern[(day_in_sim - policy.start_day) % pattern.length()];
                    if (day_char == '_' || day_char != group_char) {
                        // The bit position 'd' is relative to the window's start.
                        // Set the bit to 0 to mark as absent.
                        group_mask &= ~(1U << d);
                    }
                }
            }
            data.group_masks[i] = group_mask;
        }
        host_policy_data.push_back(data);
    }

    if (host_policy_data.empty()) { return; }

    amrex::Gpu::DeviceVector<SchoolPolicy::PolicyGPUData> device_policy_data(host_policy_data.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_policy_data.begin(), host_policy_data.end(), device_policy_data.begin());
    auto* policies_ptr = device_policy_data.data();
    int num_policies = device_policy_data.size();

    // Apply the pre-calculated masks to each student
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
            auto mask_ptr = soa.GetIntData(IntIdx::school_attendance_mask).data();
            auto fips_ptr = soa.GetIntData(IntIdx::fips).data();

            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept {
                // Check if the student's schedule is active (MSB is 1)
                if ((mask_ptr[i] & (1U << 31)) != 0) {
                    for (int p_idx = 0; p_idx < num_policies; ++p_idx) {
                        const auto& policy = policies_ptr[p_idx];
                        bool school_type_matches = (policy.school_type == -1 || policy.school_type == getSchoolType(school_grade_ptr[i]));
                        bool fips_matches = (policy.fips == 0 || policy.fips == fips_ptr[i]);
                        if (school_type_matches && fips_matches) {
                            int group_idx = amrex::Random_int(policy.num_groups, engine);
                            mask_ptr[i] &= policy.group_masks[group_idx];
                        }
                    }
                }
            });
            Gpu::synchronize();
        }
    }
}

/*! \brief Update of all attendance schedules for a new 30-day window. */
void AgentContainer::updateAttendanceSchedules(int current_day, const std::vector<ExaEpi::SchoolPolicy>& policies)
{
    Print() << "[INFO] Updating attendance schedules for window starting on day " << current_day << ".\n";

    // 1. Set a baseline "always attend" schedule for the next 30 days.
    initializeSchoolAttendanceForAll(30);

    // 2. Apply policies relative to the start of this new window.
    assignAttendanceFromPolicies(policies, current_day);

    // 3. Record that the update has been performed for this window.
    m_last_mask_update_day = current_day;
}

/*! \brief Counts and prints the number of students attending school on a given day. */
void AgentContainer::countAttendingStudents(int current_day) const
{
    BL_PROFILE("AgentContainer::countAttendingStudents");

    // --- Setup based on the baseline function ---
    const int n_school_types = 5; // College, High, Middle, Elementary, Childcare
    const int ncomp = n_school_types; // We will count each of the 5 school types
    const char* school_names[5] = {"College", "High", "Middle", "Elementary", "Childcare"};
    const int last_mask_update_day = getLastMaskUpdateDay();

    // This array will accumulate the final counts from all AMR levels
    std::array<Real, ncomp> domain_totals = {0.0};

    // Loop over each AMR level, just like in the baseline
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        const auto& geom = Geom(lev);
        Box domain = geom.Domain();
        int nx = domain.length(0);
        int ny = domain.length(1);
        BoxArray ba = this->ParticleBoxArray(lev);
        const auto& dmap = this->ParticleDistributionMap(lev);
        
        // Create a MultiFab to serve as a distributed counter grid
        MultiFab mf(ba, dmap, ncomp, 0);
        mf.setVal(0.0);

        const auto plo = geom.ProbLoArray();
        const auto dxi = geom.InvCellSizeArray();

        // Use ParticleToMesh to iterate over particles and deposit counts into the MultiFab
        ParticleToMesh(
            *this, mf, lev,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd, int i,
                                  Array4<Real> const& count) {
                
                // 1. Check if the agent is a student
                bool is_student = (ptd.m_idata[IntIdx::workgroup][i] <= 0) &&
                                  (ptd.m_idata[IntIdx::school_id][i] > SchoolType::none);
                if (!is_student) return;

                // 2. Check if the student is scheduled to attend today
                bool attends = false;
                const uint32_t mask = static_cast<uint32_t>(ptd.m_idata[IntIdx::school_attendance_mask][i]);
                if ((mask & (1U << 31)) != 0) { // Check if schedule is active
                    const int relative_day = current_day - last_mask_update_day;
                    if (relative_day < 0 || relative_day >= 30) {
                        attends = true; // Default to attend if outside the 30-day window
                    } else {
                        attends = ((mask >> relative_day) & 1U) == 1U;
                    }
                }
                
                if (attends) {
                    // 3. If attending, identify school type and add to the counter
                    const int grade = ptd.m_idata[IntIdx::school_grade][i];
                    int school_type_idx = getSchoolType(grade) - SchoolType::college;

                    if (school_type_idx >= 0 && school_type_idx < n_school_types) {
                        // Get the particle's home cell to deposit the count
                        int home_i = ptd.m_idata[IntIdx::home_i][i];
                        int home_j = ptd.m_idata[IntIdx::home_j][i];
                        int comm_to = home_i + home_j * domain.length(0);
                        IntVect iv_from_comm = domain.atOffset(comm_to);
                        
                        // Atomically add 1 to the count for the correct school type
                        Gpu::Atomic::AddNoRet(&count(iv_from_comm, school_type_idx), 1.0_rt);
                    }
                }
            },
            false
        );

        // --- Data Reduction, exactly matching the baseline ---

        // Flatten the MultiFab into a 1D vector on each rank
        std::vector<Real> flat_data(nx * ny * ncomp, 0.0);
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            const auto& arr = mf[mfi].array();
            const Box& bx = mfi.validbox();
            for (IntVect iv = bx.smallEnd(); iv <= bx.bigEnd(); bx.next(iv)) {
                int i_local = iv[0] - domain.smallEnd(0);
                int j_local = iv[1] - domain.smallEnd(1);
                for (int c = 0; c < ncomp; ++c) {
                    flat_data[(i_local * ny + j_local) * ncomp + c] += arr(iv, c);
                }
            }
        }

        // Perform the global sum reduction on the flat vector.
        // After this, only the IOProcessor has the correct, complete data.
        ParallelDescriptor::ReduceRealSum(flat_data.data(), flat_data.size(), ParallelDescriptor::IOProcessorNumber());

        // On the IOProcessor, sum the grid counts into the domain_totals accumulator
        if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    for (int c = 0; c < ncomp; ++c) {
                        domain_totals[c] += flat_data[(i * ny + j) * ncomp + c];
                    }
                }
            }
        }
    } // End of AMR level loop

    // A final reduction on the domain_totals array (matches baseline pattern)
    ParallelDescriptor::ReduceRealSum(domain_totals.data(), ncomp, ParallelDescriptor::IOProcessorNumber());

    // On the IOProcessor, print the final report
    if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
        Long total_attending = 0;
        amrex::Print() << "\n--- School Attendance Report for Day " << current_day << " ---\n";
        
        for (int i = 0; i < n_school_types; ++i) {
            Long count = static_cast<Long>(domain_totals[i]);
            total_attending += count;
            // Spacing formatted to align names
            amrex::Print() << "  - " << std::left << std::setw(11) << school_names[i]
                           << ": " << count << " students attending\n";
        }

        amrex::Print() << "  ----------------------------------------\n";
        amrex::Print() << "  - TOTAL      : " << total_attending << " students attending\n"
                       << "------------------------------------------\n\n";
    
    }
}
