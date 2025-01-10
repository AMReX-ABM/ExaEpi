/*! @file AgentContainer.cpp
    \brief Function implementations for #AgentContainer class
*/

#include "AgentContainer.H"

using namespace amrex;
using namespace ExaEpi::Utils;


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
                                const bool                       fast,  /*!< faster but non-deterministic computation*/
                                const short                      a_ic_type  /*!< type of initialization */)
    : amrex::ParticleContainer< 0,
                                0,
                                RealIdx::nattribs,
                                IntIdx::nattribs> (a_geom, a_dmap, a_ba),
        m_student_counts(a_ba, a_dmap, SchoolCensusIDType::total - 1, 0)
{
    BL_PROFILE("AgentContainer::AgentContainer");

    ic_type = a_ic_type;

    m_num_diseases = a_num_diseases;
    AMREX_ASSERT(m_num_diseases < ExaEpi::max_num_diseases);

    m_student_counts.setVal(0);  // Initialize the MultiFab to zero

    add_attributes();

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
        m_h_parm[d]->Initialize();

#ifdef AMREX_USE_GPU
        amrex::Gpu::htod_memcpy(m_d_parm[d], m_h_parm[d], sizeof(DiseaseParm));
#else
        std::memcpy(m_d_parm[d], m_h_parm[d], sizeof(DiseaseParm));
#endif
    }

    max_attribute_values.fill(-1);
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
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
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
void AgentContainer::moveAgentsToHome ()
{
    BL_PROFILE("AgentContainer::moveAgentsToHome");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

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
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
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
        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
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
        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
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
                Real lowProb = 0.0_rt;
                Real random = amrex::Random(engine);
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
                  Real random1= amrex::Random(engine);
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

        bool is_census = (ic_type == ExaEpi::ICType::Census);
        auto grid_to_lnglat_ptr = &grid_to_lnglat;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
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
void AgentContainer::returnAirTravel ()
{
    BL_PROFILE("AgentContainer::returnAirTravel");

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);
        const auto dx = Geom(lev).CellSizeArray();

        bool is_census = (ic_type == ExaEpi::ICType::Census);
        auto grid_to_lnglat_ptr = &grid_to_lnglat;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
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
void AgentContainer::updateStatus ( MFPtrVec& a_disease_stats, amrex::Real a_cur_time /*!< Community-wise disease stats tracker */)
{
    BL_PROFILE("AgentContainer::updateStatus");

    m_disease_status.updateAgents(*this, a_disease_stats, a_cur_time);
    m_hospital->treatAgents(*this, a_disease_stats);

    // move hospitalized agents to their hospital location
    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        const auto dx = Geom(lev).CellSizeArray();
        auto& plev  = GetParticles(lev);

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
void AgentContainer::shelterStart (amrex::Real a_cur_time)
{
    BL_PROFILE("AgentContainer::shelterStart");

    amrex::Print() << "Starting shelter in place order \n";

    for (int lev = 0; lev <= finestLevel(); ++lev)
    {
        auto& plev  = GetParticles(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            if (np == 0) continue;

            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto withdrawn_date_ptr = soa.GetIntData(IntIdx::withdrawn_date).data();

            auto shelter_compliance = m_shelter_compliance;
            amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
            {
                if (amrex::Random(engine) < shelter_compliance) {
                    withdrawn_ptr[i] = 1;
                    withdrawn_date_ptr[i] = static_cast<int>(a_cur_time);
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
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            if (np == 0) continue;

            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();
            auto withdrawn_date_ptr = soa.GetIntData(IntIdx::withdrawn_date).data();

            amrex::ParallelFor( np, [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                withdrawn_ptr[i] = 0;
                withdrawn_date_ptr[i] = 0;

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
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            int gid = mfi.index();
            int tid = mfi.LocalTileIndex();
            auto& ptile = plev[std::make_pair(gid, tid)];
            auto& soa   = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            if (np == 0) continue;

            int i_RT = IntIdx::nattribs;
            int r_RT = RealIdx::nattribs;
            int n_disease = m_num_diseases;

            for (int d = 0; d < n_disease; d++) {

                auto status_ptr = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::status).data();

                auto prob_ptr              = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::prob).data();
                auto counter_ptr           = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::disease_counter).data();
                auto latent_period_ptr     = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::latent_period).data();
                auto infectious_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::infectious_period).data();
                auto incubation_period_ptr = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::incubation_period).data();

                const auto lparm = m_d_parm[d];

                amrex::ParallelForRNG( np,
                [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept
                {
                    prob_ptr[i] = 1.0_prt - prob_ptr[i];
                    if ( status_ptr[i] == Status::never ||
                         status_ptr[i] == Status::susceptible ) {
                        if (amrex::Random(engine) < prob_ptr[i]) {
                            setInfected(&(status_ptr[i]), &(counter_ptr[i]), &(latent_period_ptr[i]), &(infectious_period_ptr[i]),
                                        &(incubation_period_ptr[i]), engine, lparm);
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

    Status list: 0 - never, 1 - infected, 2 - immune, 3 - susceptible, 4 - dead, 5 - exposed, 6 - asymptomatic,
                 7 - presymptomatic, 8 - symptomatic
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
    if (max_attribute_values[group_idx] == -1) {
        ReduceOps<ReduceOpMax> reduce_ops;
        auto r = ParticleReduce<ReduceData<int>> (*this,
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

void AgentContainer::printStudentTeacherCounts() const {
    ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum,
              ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
    auto r = ParticleReduce<ReduceData<int, int, int, int, int, int, int, int, int, int>> (
                *this, [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                                            const int i) noexcept
                -> GpuTuple<int, int, int, int, int, int, int, int, int, int>
            {
                int counts[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                if (ptd.m_idata[IntIdx::school_id][i] > 0) {
                    int pos = (ptd.m_idata[IntIdx::workgroup][i] > 0 ? 0 : 5);
                    int grade = ptd.m_idata[IntIdx::school_grade][i];
                    counts[pos + getSchoolType(grade) - SchoolType::college] = 1;
                }
                return {counts[0], counts[1], counts[2], counts[3], counts[4],
                        counts[5], counts[6], counts[7], counts[8], counts[9]};
            }, reduce_ops);

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
        Print() << "School counts: (educators, students, ratio)\n" << std::fixed << std::setprecision(1)
                << "  College    " << counts[0] << " " << counts[5] << " " << ((Real)counts[5] / counts[0]) << "\n"
                << "  High       " << counts[1] << " " << counts[6] << " " << ((Real)counts[6] / counts[1]) << "\n"
                << "  Middle     " << counts[2] << " " << counts[7] << " " << ((Real)counts[7] / counts[2]) << "\n"
                << "  Elementary " << counts[3] << " " << counts[8] << " " << ((Real)counts[8] / counts[3]) << "\n"
                << "  Childcare  " << counts[4] << " " << counts[9] << " " << ((Real)counts[9] / counts[4]) << "\n"
                << "  Total      " << total_educators << " " << total_students << " "
                << ((Real)total_students / total_educators) << "\n";
    }
}

void AgentContainer::printAgeGroupCounts() const {
    ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_ops;
    auto r = ParticleReduce<ReduceData<int, int, int, int, int, int>> (
                *this, [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                                            const int i) noexcept
                -> GpuTuple<int, int, int, int, int, int>
            {
                int counts[6] = {0, 0, 0, 0, 0, 0};
                int age_group = ptd.m_idata[IntIdx::age_group][i];
                counts[age_group] = 1;
                return {counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]};
            }, reduce_ops);

    std::array<Long, 6> counts = {amrex::get<0>(r), amrex::get<1>(r), amrex::get<2>(r), amrex::get<3>(r), amrex::get<4>(r),
                                  amrex::get<5>(r)};
    ParallelDescriptor::ReduceLongSum(&counts[0], 6, ParallelDescriptor::IOProcessorNumber());
    if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
        int total_agents = 0;
        for (int i = 0; i < 6; i++) {
            total_agents += counts[i];
        }
        Print() << "Age group counts (percentage):\n" << std::fixed << std::setprecision(1)
                << "  under 5   " << counts[0] << " " << 100.0 * (Real)counts[0] / total_agents << "\n"
                << "  5 to 17    " << counts[1] << " " << 100.0 * (Real)counts[1] / total_agents << "\n"
                << "  18 to 29   " << counts[2] << " " << 100.0 * (Real)counts[2] / total_agents << "\n"
                << "  30 to 49   " << counts[3] << " " << 100.0 * (Real)counts[3] / total_agents << "\n"
                << "  50 to 64   " << counts[4] << " " << 100.0 * (Real)counts[4] / total_agents << "\n"
                << "  over 64    " << counts[5] << " " << 100.0 * (Real)counts[5] / total_agents << "\n"
                << "  Total      " << total_agents << "\n";
    }
}

void AgentContainer::updateSchoolInfection(iMultiFab& a_school_stats, amrex::Real a_cur_time) /*!< Community-wise school infection stats and status tracker */
{
    BL_PROFILE("AgentContainer::updateSchoolInfo");

    struct SchoolDismissal
    {
        enum {
            ByCommunity = 0,   /*!< dismiss school with respect to total infection in a community */
            BySchool,   /*!< dismiss school with respect to total infection in specific school in a community */
            ByUnit  /*!< dismiss school with respect to total infection in a unit */
        };
    };

    int school_closure_flag = 0;
    std::string closure_option = "by_community";
    double sc_infection_threshold = 0.20;
    int  sc_length = 10;

    amrex::ParmParse pp("agent");
    pp.query("school_closure_flag", school_closure_flag);
    pp.query("school_closure_option", closure_option);
    pp.query("school_closure_threshold", sc_infection_threshold);
    pp.query("school_closure_length",  sc_length);

    AMREX_ASSERT(sc_infection_threshold >= 0.0 && sc_infection_threshold <= 1.0);

    int school_closure_option = SchoolDismissal::ByCommunity;
    if (closure_option == "by_community"){school_closure_option = SchoolDismissal::ByCommunity; }
    else if (closure_option == "by_school"){school_closure_option = SchoolDismissal::BySchool; }
    else if (closure_option == "by_unit"){school_closure_option = SchoolDismissal::ByUnit; }

    struct SchoolStats
    {
        enum {
            SchoolDismissal = 0,   /*!< whether school is open or closed */
            SchoolInfectionCount,   /*!< total infected students in community if school open */
            SchoolStatusDayCount,  /*!< day count of school being closed */
            SchoolReopenDay,
            nattribs
        };
    };

    int offset = SchoolCensusIDType::total;
    AMREX_ALWAYS_ASSERT(a_school_stats.nComp() == offset * SchoolStats::nattribs);

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

            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
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
            if (std::abs(a_cur_time) < 1e-9) {
                // Assert that all particle data and grid data are zero
                bool allZero = true;

                // Check grid data if all particle data was zero
                if (allZero) {
                    for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                            for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
                                if (status_arr(i, j, k) != 0 || counts_arr(i, j, k) != 0) {
                                    allZero = false;
                                    break;  // Early exit if any value is non-zero
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
                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolInfectionCount) = 0;
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
                            if (withdrawn_date_ptr[p] >= (status_arr(home_i, home_j, 0, offset * SchoolStats::SchoolReopenDay) - 1e-9)) {
                                amrex::Gpu::Atomic::Add(&status_arr(home_i, home_j, 0,  offset * SchoolStats::SchoolInfectionCount), 1);
                                if (school_id == SchoolCensusIDType::high_1) {
                                    amrex::Gpu::Atomic::Add(&status_arr(home_i, home_j, 0, 1 +  offset * SchoolStats::SchoolInfectionCount), 1);
                                }
                                else if (school_id == SchoolCensusIDType::middle_2) {
                                    amrex::Gpu::Atomic::Add(&status_arr(home_i, home_j, 0, 2 +  offset * SchoolStats::SchoolInfectionCount), 1);
                                }
                                else if (school_id == SchoolCensusIDType::elem_3) {
                                    amrex::Gpu::Atomic::Add(&status_arr(home_i, home_j, 0, 3 +  offset * SchoolStats::SchoolInfectionCount), 1);
                                }
                                else if (school_id == SchoolCensusIDType::elem_4) {
                                    amrex::Gpu::Atomic::Add(&status_arr(home_i, home_j, 0, 4 +  offset * SchoolStats::SchoolInfectionCount), 1);
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
                    if (i == 14 && j == 2) {
                        printf("\nCommunity (%d, %d, %d) has Infection number:  MultiFab = %d, Day = %d, with total students %d\n",
                               i, j, k,
                               status_arr(i, j, k,  offset * SchoolStats::SchoolInfectionCount),
                               status_arr(i, j, k,  offset * SchoolStats::SchoolStatusDayCount),
                               (counts_arr(i, j, k, SchoolCensusIDType::elem_3)
                                        + counts_arr(i, j, k, SchoolCensusIDType::elem_4)
                                        + counts_arr(i, j, k, SchoolCensusIDType::middle_2)
                                        + counts_arr(i, j, k, SchoolCensusIDType::high_1))
                            );
                    }
                });

            Gpu::synchronize();

            // close community/school based on infection count
            amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                {
                    int start_dismiss = 0;
                    int stop_dismiss  = 1;
                    if (school_closure_option == SchoolDismissal::BySchool){
                        start_dismiss = 1;
                        stop_dismiss  = 5;
                    }

                    for (int ii = start_dismiss; ii < stop_dismiss; ii++) //exclude DayCare
                    {
                        if (status_arr(i, j, k, ii +  offset * SchoolStats::SchoolDismissal) == 0) {
                            // Check if the school should be closed
                            int student_total;
                            if (ii == 0){
                                student_total = counts_arr(i, j, k, SchoolCensusIDType::elem_3)
                                        + counts_arr(i, j, k, SchoolCensusIDType::elem_4)
                                        + counts_arr(i, j, k, SchoolCensusIDType::middle_2)
                                        + counts_arr(i, j, k, SchoolCensusIDType::high_1); // handle daycare,playgroup later
                            }
                            else{ student_total = counts_arr(i, j, k, ii);}

                            long thresh = static_cast<int>(sc_infection_threshold * student_total);
                            if (status_arr(i, j, k, ii + offset * SchoolStats::SchoolInfectionCount) >= thresh)
                            {
                                status_arr(i, j, k, ii + offset * SchoolStats::SchoolDismissal) = 1; // close school
                                if ( i == 14 && j == 2 ) {
                                    printf("Community (%d, %d, %d) is now closed at day %f (dissmissal = %d). Infection number: MultiFab = %d, Day = %d\n",
                                        i, j, k, a_cur_time,
                                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolDismissal),
                                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolInfectionCount),
                                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolStatusDayCount));
                                }
                                status_arr(i, j, k, ii + offset * SchoolStats::SchoolStatusDayCount) = 0; //update school closure day to Day 0

                            }
                            else {
                                if ( i == 14 && j == 2 ) {
                                    printf("Community (%d, %d, %d) is currenly opened %d. Infection number: MultiFab = %d, Day = %d\n",
                                        i, j, k,
                                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolDismissal),
                                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolInfectionCount),
                                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolStatusDayCount));
                                }
                                amrex::Gpu::Atomic::Add(&status_arr(i, j, k, ii + offset * SchoolStats::SchoolStatusDayCount), 1);

                            }
                        } else { // School is closed
                            if (status_arr(i, j, k, ii + offset * SchoolStats::SchoolStatusDayCount) > sc_length - 1 ) {

                                status_arr(i, j, k, ii + offset * SchoolStats::SchoolDismissal) = 0;
                                status_arr(i, j, k, ii + offset * SchoolStats::SchoolReopenDay) = a_cur_time;

                                if ( i == 14 && j == 2) {
                                    printf("Community (%d, %d, %d) has now opened at day %f (dissmissal = %d). Infection number: MultiFab = %d, Day = %d\n",
                                        i, j, k, a_cur_time,
                                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolDismissal),
                                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolInfectionCount),
                                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolStatusDayCount));
                                }
                                status_arr(i, j, k, ii + offset * SchoolStats::SchoolStatusDayCount) = 0;

                            } else {
                                if (i == 14 && j == 2) {
                                    printf("Community (%d, %d, %d) is currently closed %d. Infection number: MultiFab = %d, Day = %d\n",
                                        i, j, k,
                                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolDismissal),
                                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolInfectionCount),
                                        status_arr(i, j, k, ii + offset * SchoolStats::SchoolStatusDayCount));
                                }
                                amrex::Gpu::Atomic::Add(&status_arr(i, j, k, ii + offset * SchoolStats::SchoolStatusDayCount), 1);

                            }
                        }
                    }
                });
            Gpu::synchronize();

            // update school_closed_ptr for both teachers and children based on school closure status
            if (school_closure_flag){
                amrex::ParallelFor( np,
                [=] AMREX_GPU_DEVICE (int p) noexcept
                {

                    if (age_group_ptr[p] >= 1 && school_id_ptr[p] > 0){ // teachers and student

                        int school_type = 0; // zero here means community-level
                        if (school_closure_option == SchoolDismissal::BySchool){
                            school_type = school_id_ptr[p];
                        }
                        int school_status = status_arr(home_i_ptr[p], home_j_ptr[p], 0, school_type + offset * SchoolStats::SchoolDismissal);

                        if (school_status == 0){ school_closed_ptr[p] = 0; }
                        else{
                            school_closed_ptr[p] = 1;
                            // Ensure that the school status is correctly marked as closed (1)
                            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(school_status == 1,
                                "School status should be 1 when the school is closed.");
                        }
                    }
                });
                Gpu::synchronize();
            }
        }
    }

}

//checking whether the previous function counts infection correctly
//Debugging purposes only
void AgentContainer::printSchoolInfection(iMultiFab& a_school_stats, amrex::Real a_cur_time) /*!< print function for debugging purposes */
{
    int count_infec = 0;  // Initialize the infection count

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

            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
            auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
            auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();
            auto withdrawn_ptr = soa.GetIntData(IntIdx::withdrawn).data();

            // Loop over all particles
            for (int i = 0; i < np; ++i)
            {
                int home_i = home_i_ptr[i];
                int home_j = home_j_ptr[i];
                int school_id = school_id_ptr[i];
                int age_group = age_group_ptr[i];
                int withdrawn = withdrawn_ptr[i];

                if (home_i == 14 && home_j == 2) {  // Filter for a specific community
                    if (age_group == 1) {  // Exclude DayCare
                        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(school_id >= 0, "school_ptr can't be negative");
                        if (school_id > 0 && (withdrawn == 1 || hosp_i_ptr[i] > -1 || hosp_j_ptr[i] < -1)) {
                            count_infec++;
                        }
                    }
                }
            }
        }
    }

    if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
        amrex::Print() << "Community via Manual (" << 14 << ", " << 2 << ", " << 0 << ") has Infection number = "
                       << count_infec << ", Day = " << a_cur_time << std::endl;
    }
}

