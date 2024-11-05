/*! @file AgentContainer.cpp
    \brief Function implementations for #AgentContainer class
*/

#include "AgentContainer.H"

using namespace amrex;


/*! Add runtime SoA attributes */
void AgentContainer::add_attributes()
{
    const bool communicate_this_comp = true;
    {
        int count(0);
        for (int i = 0; i < m_num_diseases*RealIdxDisease::nattribs; i++) {
            AddRealComp(communicate_this_comp);
            count++;
        }
        Print() << "Added " << count << " real-type run-time SoA attibute(s).\n";
    }
    {
        int count(0);
        for (int i = 0; i < m_num_diseases*IntIdxDisease::nattribs; i++) {
            AddIntComp(communicate_this_comp);
            count++;
        }
        Print() << "Added " << count << " integer-type run-time SoA attibute(s).\n";
    }
    return;
}

/*! Constructor:
    *  + Initializes particle container for agents
    *  + Read in contact probabilities from command line input file
    *  + Read in disease parameters from command line input file
*/
AgentContainer::AgentContainer (const amrex::Geometry            & a_geom,  /*!< Physical domain */
                                const amrex::DistributionMapping & a_dmap,  /*!< Distribution mapping */
                                const amrex::BoxArray            & a_ba,    /*!< Box array */
                                const int                        & a_num_diseases, /*!< Number of diseases */
                                const std::vector<std::string>   & a_disease_names, /*!< names of the diseases */
                                const bool                       fast  /*!< faster but non-deterministic computation*/)
    : amrex::ParticleContainer< 0,
                                0,
                                RealIdx::nattribs,
                                IntIdx::nattribs> (a_geom, a_dmap, a_ba),
        m_student_counts(a_ba, a_dmap, SchoolType::total_school_type, 0)
{
    BL_PROFILE("AgentContainer::AgentContainer");

    m_num_diseases = a_num_diseases;
    AMREX_ASSERT(m_num_diseases < ExaEpi::max_num_diseases);
    m_disease_names = a_disease_names;

    m_student_counts.setVal(0);  // Initialize the MultiFab to zero

    add_attributes();

    {
        amrex::ParmParse pp("agent");
        pp.query("symptomatic_withdraw", m_symptomatic_withdraw);
        pp.query("shelter_compliance", m_shelter_compliance);
        pp.query("symptomatic_withdraw_compliance", m_symptomatic_withdraw_compliance);
        pp.queryarr("student_teacher_ratios", m_student_teacher_ratios);

    }

    {
        using namespace ExaEpi;

        /* Create the interaction model objects and push to container */
        m_interactions.clear();
        //m_interactions[InteractionNames::generic] = new InteractionModGeneric<PCType,PTileType,PTDType,PType>;
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
        m_h_parm[d] = new DiseaseParm{};
        m_d_parm[d] = (DiseaseParm*)amrex::The_Arena()->alloc(sizeof(DiseaseParm));

        m_h_parm[d]->readContact();
        // first read inputs common to all diseases
        m_h_parm[d]->readInputs("disease");
        // now read any disease-specific input, if available
        m_h_parm[d]->readInputs(std::string("disease_"+m_disease_names[d]));
        m_h_parm[d]->Initialize();

#ifdef AMREX_USE_GPU
        amrex::Gpu::htod_memcpy(m_d_parm[d], m_h_parm[d], sizeof(DiseaseParm));
#else
        std::memcpy(m_d_parm[d], m_h_parm[d], sizeof(DiseaseParm));
#endif
    }

    max_attribute_values.fill(0);
}

/*! \brief Send agents on a random walk around the neighborhood

    For each agent, set its position to a random one near its current position
*/
void AgentContainer::moveAgentsRandomWalk ()
{
    BL_PROFILE("AgentContainer::moveAgentsRandomWalk");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
            {
                ParticleType& p = pstruct[i];
                p.pos(0) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[0]);
                p.pos(1) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*dx[1]);
            });
        }
    }
}

/*! \brief Move agents to work

    For each agent, set its position to the work community (IntIdx::work_i, IntIdx::work_j)
*/
void AgentContainer::moveAgentsToWork ()
{
    BL_PROFILE("AgentContainer::moveAgentsToWork");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            auto& soa = ptile.GetStructOfArrays();
            auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
            auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();

            amrex::ParallelFor( np,
            [=] AMREX_GPU_DEVICE (int ip) noexcept
            {
                if (!inHospital(ip, ptd)) {
                    ParticleType& p = pstruct[ip];
                    p.pos(0) = (work_i_ptr[ip] + 0.5_prt)*dx[0];
                    p.pos(1) = (work_j_ptr[ip] + 0.5_prt)*dx[1];
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
void AgentContainer::moveAgentsToHome ()
{
    BL_PROFILE("AgentContainer::moveAgentsToHome");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            auto& soa = ptile.GetStructOfArrays();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();

            amrex::ParallelFor( np,
            [=] AMREX_GPU_DEVICE (int ip) noexcept
            {
                if (!inHospital(ip, ptd)) {
                    ParticleType& p = pstruct[ip];
                    p.pos(0) = (home_i_ptr[ip] + 0.5_prt) * dx[0];
                    p.pos(1) = (home_j_ptr[ip] + 0.5_prt) * dx[1];
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
void AgentContainer::moveRandomTravel (const amrex::Real random_travel_prob)
{
    BL_PROFILE("AgentContainer::moveRandomTravel");

    const Box& domain = Geom(0).Domain();
    int i_max = domain.length(0);
    int j_max = domain.length(1);
    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto& ptile = plev[{mfi.index(), mfi.LocalTileIndex()}];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();
            auto& soa   = ptile.GetStructOfArrays();
            auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
            {
                if (!inHospital(i, ptd) && !withdrawn_ptr[i]) {
                    ParticleType& p = pstruct[i];
                    if (amrex::Random(engine) < random_travel_prob) {
                        random_travel_ptr[i] = i;
                        int i_random = int( amrex::Real(i_max)*amrex::Random(engine));
                        int j_random = int( amrex::Real(j_max)*amrex::Random(engine));
                        p.pos(0) = i_random;
                        p.pos(1) = j_random;
                    }
                }
            });
        }
    }

    // Don't need to redistribute here because it happens after agents move to work
    //Redistribute();
}

/*! \brief Select agents to travel by air

*/
void AgentContainer::moveAirTravel (const iMultiFab& unit_mf, AirTravelFlow& air, DemographicData& demo)
{
    BL_PROFILE("AgentContainer::moveAirTravel");
    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto unit_arr = unit_mf[mfi].array();
            auto& ptile = plev[{mfi.index(), mfi.LocalTileIndex()}];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();
            auto& soa   = ptile.GetStructOfArrays();
            auto air_travel_ptr = soa.GetIntData(IntIdx::air_travel).data();
            auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto trav_i_ptr = soa.GetIntData(IntIdx::trav_i).data();
            auto trav_j_ptr = soa.GetIntData(IntIdx::trav_j).data();
            auto air_travel_prob_ptr= air.air_travel_prob_d.data();

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
            {
                int unit = unit_arr(home_i_ptr[i], home_j_ptr[i], 0);
                if (!inHospital(i, ptd) && random_travel_ptr[i] <0 && air_travel_ptr[i] <0) {
                    if (withdrawn_ptr[i] == 1) {return ;}
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

void AgentContainer::setAirTravel (const iMultiFab& unit_mf, AirTravelFlow& air, DemographicData& demo)
{
    BL_PROFILE("AgentContainer::setAirTravel");

    amrex::Print()<<"Compute air travel statistics"<<"\n";
    const Box& domain = Geom(0).Domain();
    int i_max = domain.length(0);
    int j_max = domain.length(1);
    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const auto unit_arr = unit_mf[mfi].array();
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& aos   = ptile.GetArrayOfStructs();
            const size_t np = aos.numParticles();
            auto& soa   = ptile.GetStructOfArrays();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto trav_i_ptr = soa.GetIntData(IntIdx::trav_i).data();
            auto trav_j_ptr = soa.GetIntData(IntIdx::trav_j).data();
            auto Start = demo.Start_d.data();
            auto dest_airports_ptr= air.dest_airports_d.data();
            auto dest_airports_offset_ptr= air.dest_airports_offset_d.data();
            auto dest_airports_prob_ptr= air.dest_airports_prob_d.data();
            auto arrivalUnits_ptr= air.arrivalUnits_d.data();
            auto arrivalUnits_offset_ptr= air.arrivalUnits_offset_d.data();
            auto arrivalUnits_prob_ptr= air.arrivalUnits_prob_d.data();
            auto assigned_airport_ptr= air.assigned_airport_d.data();

            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept
            {
                trav_i_ptr[i] = -1;
                trav_j_ptr[i] = -1;
                int unit = unit_arr(home_i_ptr[i], home_j_ptr[i], 0);
                int orgAirport= assigned_airport_ptr[unit];
                int destAirport=-1;
                float lowProb=0.0;
                float random= amrex::Random(engine);
                //choose a destination airport for the agent (number of airports is often small, so let's visit in sequential order)
                for(int idx= dest_airports_offset_ptr[orgAirport]; idx<dest_airports_offset_ptr[orgAirport+1]; idx++){
                        float hiProb= dest_airports_prob_ptr[idx];
                        if(random>lowProb && random < hiProb) {
                                destAirport=dest_airports_ptr[idx];
                                break;
                        }
                        lowProb= dest_airports_ptr[idx];
                }
                if(destAirport >=0){
                  int destUnit=-1;
                  float random1= amrex::Random(engine);
                  int low=arrivalUnits_offset_ptr[destAirport], high=arrivalUnits_offset_ptr[destAirport+1];
                  if(high-low<=16){
                          //this sequential algo. is very slow when we have to go through hundreds or thoudsands of units to select a destination
                          float lProb=0.0;
                          for(int idx= low; idx<high; idx++){
                                  if(random1>lProb && random1 < arrivalUnits_prob_ptr[idx]) {
                                          destUnit=arrivalUnits_ptr[idx];
                                          break;
                                  }
                                  lProb= arrivalUnits_prob_ptr[idx];
                          }
                  }else{  //binary search algorithm
                          while(low<high){
                                  if(random1<low) break; //low is the found airport index
                                  //if random1 falls within (low, high), half the range
                                  int mid= low+ (high-low)/2;
                                  if(arrivalUnits_prob_ptr[mid]<random1) low=mid+1;
                                  else high=mid-1;
                          }
                          destUnit=arrivalUnits_ptr[low];
                  }
                  if(destUnit >=0){
                          //randomly select a community in the dest unit
                                int comm_to = Start[destUnit] + amrex::Random_int(Start[destUnit+1] - Start[destUnit], engine);
                          int new_i= comm_to%i_max;
                          int new_j= comm_to/i_max;
                          if(new_i>=0 && new_j>=0 && new_i<i_max && new_j<j_max){
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
void AgentContainer::returnRandomTravel ()
{
    BL_PROFILE("AgentContainer::returnRandomTravel");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);
        const auto dx = Geom(lev).CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto& ptile = plev[{mfi.index(), mfi.LocalTileIndex()}];
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();
            auto& soa   = ptile.GetStructOfArrays();
            auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();

            amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                if (random_travel_ptr[i] >= 0) {
                    ParticleType& p = pstruct[i];
                    random_travel_ptr[i] = -1;
                    p.pos(0) = (home_i_ptr[i] + 0.5_prt) * dx[0];
                    p.pos(1) = (home_j_ptr[i] + 0.5_prt) * dx[1];
                }
            });
        }
    }
    Redistribute();
    AMREX_ALWAYS_ASSERT(OK());
}


/*! \brief Return agents from air travel
*/
void AgentContainer::returnAirTravel ()
{
    BL_PROFILE("AgentContainer::returnAirTravel");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);
        const auto dx = Geom(lev).CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto& ptile = plev[{mfi.index(), mfi.LocalTileIndex()}];
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();
            auto& soa   = ptile.GetStructOfArrays();
            auto air_travel_ptr = soa.GetIntData(IntIdx::air_travel).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();

            amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                if (air_travel_ptr[i] >= 0) {
                    ParticleType& p = pstruct[i];
                    air_travel_ptr[i] = -1;
                    p.pos(0) = (home_i_ptr[i] + 0.5_prt) * dx[0];
                    p.pos(1) = (home_j_ptr[i] + 0.5_prt) * dx[1];
                }
            });
        }
    }
    Redistribute();
    AMREX_ALWAYS_ASSERT(OK());
}


/*! \brief Updates disease status of each agent */
void AgentContainer::updateStatus ( MFPtrVec& a_disease_stats /*!< Community-wise disease stats tracker */)
{
    BL_PROFILE("AgentContainer::updateStatus");

    m_disease_status.updateAgents(*this, a_disease_stats);
    m_hospital->treatAgents(*this, a_disease_stats);

    // move hospitalized agents to their hospital location
    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            const auto& ptd = ptile.getParticleTileData();
            auto& aos   = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            auto& soa = ptile.GetStructOfArrays();
            auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
            auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();

            amrex::ParallelFor( np,
            [=] AMREX_GPU_DEVICE (int ip) noexcept
            {
                if (inHospital(ip, ptd)) {
                    ParticleType& p = pstruct[ip];
                    p.pos(0) = (hosp_i_ptr[ip] + 0.5_prt)*dx[0];
                    p.pos(1) = (hosp_j_ptr[ip] + 0.5_prt)*dx[1];
                }
            });
        }
    }
}

/*! \brief Start shelter-in-place */
void AgentContainer::shelterStart ()
{
    BL_PROFILE("AgentContainer::shelterStart");

    amrex::Print() << "Starting shelter in place order \n";

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            auto shelter_compliance = m_shelter_compliance;
            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
            {
                if (amrex::Random(engine) < shelter_compliance) {
                    withdrawn_ptr[i] = 1;
                }
            });
        }
    }
}

/*! \brief Stop shelter-in-place */
void AgentContainer::shelterStop ()
{
    BL_PROFILE("AgentContainer::shelterStop");

    amrex::Print() << "Stopping shelter in place order \n";

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                withdrawn_ptr[i] = 0;
            });
        }
    }
}

/*! \brief Infect agents based on their current status and the computed probability of infection.
    The infection probability is computed in AgentContainer::interactAgentsHomeWork() or
    AgentContainer::interactAgents() */
void AgentContainer::infectAgents ()
{
    BL_PROFILE("AgentContainer::infectAgents");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();

            int i_RT = IntIdx::nattribs;
            int r_RT = RealIdx::nattribs;
            int n_disease = m_num_diseases;

            for (int d = 0; d < n_disease; d++) {

                auto status_ptr = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::status).data();

                auto counter_ptr           = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::disease_counter).data();
                auto prob_ptr              = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::prob).data();
                auto latent_period_ptr     = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::latent_period).data();
                auto infectious_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::infectious_period).data();
                auto incubation_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::incubation_period).data();

                auto* lparm = m_d_parm[d];

                amrex::ParallelForRNG( np,
                [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
                {
                    prob_ptr[i] = 1.0_rt - prob_ptr[i];
                    if ( status_ptr[i] == Status::never ||
                         status_ptr[i] == Status::susceptible ) {
                        if (amrex::Random(engine) < prob_ptr[i]) {
                            status_ptr[i] = Status::infected;
                            counter_ptr[i] = 0.0_rt;
                            latent_period_ptr[i] = amrex::RandomNormal(lparm->latent_length_mean, lparm->latent_length_std, engine);
                            infectious_period_ptr[i] = amrex::RandomNormal(lparm->infectious_length_mean, lparm->infectious_length_std, engine);
                            incubation_period_ptr[i] = amrex::RandomNormal(lparm->incubation_length_mean, lparm->incubation_length_std, engine);
                            if (latent_period_ptr[i] < 0) { latent_period_ptr[i] = 0.0_rt; }
                            if (infectious_period_ptr[i] < 0) { infectious_period_ptr[i] = 0.0_rt; }
                            if (incubation_period_ptr[i] < 0) { incubation_period_ptr[i] = 0.0_rt; }
                            if (incubation_period_ptr[i] > (infectious_period_ptr[i]+latent_period_ptr[i])) {
                                incubation_period_ptr[i] = std::floor(infectious_period_ptr[i]+latent_period_ptr[i]);
                            }
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
void AgentContainer::generateCellData (MultiFab& mf /*!< MultiFab with at least 5*m_num_diseases components */) const
{
    BL_PROFILE("AgentContainer::generateCellData");

    const int lev = 0;

    AMREX_ASSERT(OK());
    AMREX_ASSERT(numParticlesOutOfRange(*this, 0) == 0);

    const auto& geom = Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const auto domain = geom.Domain();
    int n_disease = m_num_diseases;

    ParticleToMesh(*this, mf, lev,
        [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                              int i,
                              Array4<Real> const& count)
        {
            auto p = ptd.m_aos[i];
            auto iv = getParticleCell(p, plo, dxi, domain);

            for (int d = 0; d < n_disease; d++) {
                int status = ptd.m_runtime_idata[i0(d)+IntIdxDisease::status][i];
                Gpu::Atomic::AddNoRet(&count(iv, 5*d+0), 1.0_rt);
                if (status != Status::dead) {
                    Gpu::Atomic::AddNoRet(&count(iv, 5*d+status+1), 1.0_rt);
                }
            }
        }, false);
}

/*! \brief Computes the total number of agents with each #Status

    Returns a vector with 5 components corresponding to each value of #Status; each element is
    the total number of agents at a step with the corresponding #Status (in that order).
*/
std::array<Long, 9> AgentContainer::getTotals (const int a_d /*!< disease index */) {
    BL_PROFILE("getTotals");
    amrex::ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
    auto r = amrex::ParticleReduce<ReduceData<int,int,int,int,int,int,int,int,int>> (
                  *this, [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd, const int i) noexcept
                  -> amrex::GpuTuple<int,int,int,int,int,int,int,int,int>
              {
                  int s[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                  auto status = ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::status][i];

                  AMREX_ALWAYS_ASSERT(status >= 0);
                  AMREX_ALWAYS_ASSERT(status <= 4);

                  s[status] = 1;

                  if (status == Status::infected) {  // exposed
                      if (notInfectiousButInfected(i, ptd, a_d)) {
                          s[5] = 1;  // exposed, but not infectious
                      } else { // infectious
                          if (ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::symptomatic][i] == SymptomStatus::asymptomatic) {
                              s[6] = 1;  // asymptomatic and will remain so
                          }
                          else if (ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::symptomatic][i] == SymptomStatus::presymptomatic) {
                              s[7] = 1;  // asymptomatic but will develop symptoms
                          }
                          else if (ptd.m_runtime_idata[i0(a_d)+IntIdxDisease::symptomatic][i] == SymptomStatus::symptomatic) {
                              s[8] = 1;  // Infectious and symptomatic
                          } else {
                              amrex::Abort("how did I get here?");
                          }
                      }
                  }
                  return {s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]};
              }, reduce_ops);

    std::array<Long, 9> counts = {amrex::get<0>(r), amrex::get<1>(r), amrex::get<2>(r), amrex::get<3>(r),
                                  amrex::get<4>(r), amrex::get<5>(r), amrex::get<6>(r), amrex::get<7>(r),
                                  amrex::get<8>(r)};
    ParallelDescriptor::ReduceLongSum(&counts[0], 9, ParallelDescriptor::IOProcessorNumber());
    return counts;
}

int AgentContainer::getMaxGroup (const int group_idx) {
    BL_PROFILE("getMaxGroup");
    if (max_attribute_values[group_idx] == 0) {
        amrex::ReduceOps<ReduceOpMax> reduce_ops;
        auto r = amrex::ParticleReduce<ReduceData<int>> (*this,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd, const int i) noexcept
            -> GpuTuple<int>
            {
                return {ptd.m_idata[group_idx][i]};
            }, reduce_ops);
        max_attribute_values[group_idx] = amrex::get<0>(r);
    }
    return max_attribute_values[group_idx];
}

/*! \brief Interaction and movement of agents during morning commute
 *
 * + Move agents to work
 * + Simulate interactions during morning commute (public transit/carpool/etc ?)
*/
void AgentContainer::morningCommute ( MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::morningCommute");
    //if (haveInteractionModel(ExaEpi::InteractionNames::transit)) {
    //    m_interactions[ExaEpi::InteractionNames::transit]->interactAgents( *this, a_mask_behavior );
    //}
    moveAgentsToWork();
}

/*! \brief Interaction and movement of agents during evening commute
 *
 * + Simulate interactions during evening commute (public transit/carpool/etc ?)
 * + Simulate interactions at locations agents may stop by on their way home
 * + Move agents to home
*/
void AgentContainer::eveningCommute ( MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */ )
{
    BL_PROFILE("AgentContainer::eveningCommute");
    //if (haveInteractionModel(ExaEpi::InteractionNames::transit)) {
    //    m_interactions[ExaEpi::InteractionNames::transit]->interactAgents( *this, a_mask_behavior );
    //}
    //if (haveInteractionModel(ExaEpi::InteractionNames::grocery_store)) {
    //    m_interactions[ExaEpi::InteractionNames::grocery_store]->interactAgents( *this, a_mask_behavior );
    //}
    moveAgentsToHome();
}

/*! \brief Interaction of agents during day time - work and school */
void AgentContainer::interactDay (MultiFab& a_mask_behavior /*!< Masking behavior */)
{
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
void AgentContainer::interactEvening (MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */)
{
    BL_PROFILE("AgentContainer::interactEvening");
}

/*! \brief Interaction of agents during nighttime time - at home */
void AgentContainer::interactNight (MultiFab& a_mask_behavior /*!< Masking behavior */)
{
    BL_PROFILE("AgentContainer::interactNight");
    if (haveInteractionModel(ExaEpi::InteractionNames::home)) {
        m_interactions[ExaEpi::InteractionNames::home]->interactAgents(*this, a_mask_behavior);
    }
    if (haveInteractionModel(ExaEpi::InteractionNames::home_nborhood)) {
        m_interactions[ExaEpi::InteractionNames::home_nborhood]->interactAgents(*this, a_mask_behavior);
    }
}
