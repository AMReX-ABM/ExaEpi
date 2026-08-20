/*! @file AgentContainer.cpp
    \brief Function implementations for #AgentContainer class
*/

#include "AgentContainer.H"
#include "AgentDefinitions.H"

// repeat macro for repeating identical tokens
#define REPEAT_0(x)
#define REPEAT_1(x) x
#define REPEAT_2(x) REPEAT_1(x), x
#define REPEAT_3(x) REPEAT_2(x), x
#define REPEAT_4(x) REPEAT_3(x), x
#define REPEAT_5(x) REPEAT_4(x), x
#define REPEAT_6(x) REPEAT_5(x), x
#define REPEAT_7(x) REPEAT_6(x), x
#define REPEAT_8(x) REPEAT_7(x), x
#define REPEAT_9(x) REPEAT_8(x), x
#define REPEAT_10(x) REPEAT_9(x), x
#define REPEAT_11(x) REPEAT_10(x), x
#define REPEAT_12(x) REPEAT_11(x), x
#define REPEAT_13(x) REPEAT_12(x), x
#define REPEAT_14(x) REPEAT_13(x), x
#define REPEAT_15(x) REPEAT_14(x), x
#define REPEAT_16(x) REPEAT_15(x), x
#define REPEAT_17(x) REPEAT_16(x), x
#define REPEAT_18(x) REPEAT_17(x), x
#define REPEAT_19(x) REPEAT_18(x), x
#define REPEAT_20(x) REPEAT_19(x), x
#define REPEAT(n, x) REPEAT_##n(x)

// macro to create tuple from array elements
#define ARRAY_TO_TUPLE_1(arr) arr[0]
#define ARRAY_TO_TUPLE_2(arr) ARRAY_TO_TUPLE_1(arr), arr[1]
#define ARRAY_TO_TUPLE_3(arr) ARRAY_TO_TUPLE_2(arr), arr[2]
#define ARRAY_TO_TUPLE_4(arr) ARRAY_TO_TUPLE_3(arr), arr[3]
#define ARRAY_TO_TUPLE_5(arr) ARRAY_TO_TUPLE_4(arr), arr[4]
#define ARRAY_TO_TUPLE_6(arr) ARRAY_TO_TUPLE_5(arr), arr[5]
#define ARRAY_TO_TUPLE_7(arr) ARRAY_TO_TUPLE_6(arr), arr[6]
#define ARRAY_TO_TUPLE_8(arr) ARRAY_TO_TUPLE_7(arr), arr[7]
#define ARRAY_TO_TUPLE_9(arr) ARRAY_TO_TUPLE_8(arr), arr[8]
#define ARRAY_TO_TUPLE_10(arr) ARRAY_TO_TUPLE_9(arr), arr[9]
#define ARRAY_TO_TUPLE_11(arr) ARRAY_TO_TUPLE_10(arr), arr[10]
#define ARRAY_TO_TUPLE_12(arr) ARRAY_TO_TUPLE_11(arr), arr[11]
#define ARRAY_TO_TUPLE_13(arr) ARRAY_TO_TUPLE_12(arr), arr[12]
#define ARRAY_TO_TUPLE_14(arr) ARRAY_TO_TUPLE_13(arr), arr[13]
#define ARRAY_TO_TUPLE_15(arr) ARRAY_TO_TUPLE_14(arr), arr[14]
#define ARRAY_TO_TUPLE_16(arr) ARRAY_TO_TUPLE_15(arr), arr[15]
#define ARRAY_TO_TUPLE_17(arr) ARRAY_TO_TUPLE_16(arr), arr[16]
#define ARRAY_TO_TUPLE_18(arr) ARRAY_TO_TUPLE_17(arr), arr[17]
#define ARRAY_TO_TUPLE_19(arr) ARRAY_TO_TUPLE_18(arr), arr[18]
#define ARRAY_TO_TUPLE_20(arr) ARRAY_TO_TUPLE_19(arr), arr[19]
#define ARRAY_TO_TUPLE(n, arr) ARRAY_TO_TUPLE_##n(arr)

// macros to extract a tuple into an array
template <std::size_t I = 0, typename TupleT, typename ArrayT>
inline typename std::enable_if<I == std::tuple_size<TupleT>::value, void>::type extract_tuple_to_array (const TupleT&, ArrayT&) {}
template <std::size_t I = 0, typename TupleT, typename ArrayT>
        inline typename std::enable_if <
        I<std::tuple_size<TupleT>::value, void>::type extract_tuple_to_array (const TupleT& t, ArrayT& arr) {
    arr[I] = amrex::get<I>(t);
    extract_tuple_to_array<I + 1, TupleT, ArrayT>(t, arr);
}

using namespace amrex;
using namespace ExaEpi::Utils;

/*! Add runtime SoA attributes */
void AgentContainer::addAttributes () {
    const bool communicate_this_comp = true;
    for (int i = 0; i < m_num_diseases * RealIdxDisease::nattribs; i++) {
        AddRealComp(communicate_this_comp);
    }
    for (int i = 0; i < m_num_diseases * IntIdxDisease::nattribs; i++) {
        AddIntComp(communicate_this_comp);
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
                                const bool fast,                                 /*!< faster but non-deterministic computation*/
                                const short a_ic_type /*!< type of initialization */)
    : amrex::ParticleContainer<0, 0, RealIdx::nattribs, IntIdx::nattribs>(a_geom, a_dmap, a_ba),
      m_student_counts(a_ba, a_dmap, SchoolCensusIDType::total - 1, 0), comm_density_scale(a_ba, a_dmap, 1, 0),
      comm_density_scale_work(a_ba, a_dmap, 1, 0), m_mod_nborhood_day(true), m_mod_nborhood_night(false), m_mod_comm_day(true),
      m_mod_comm_night(false) {
    BL_PROFILE("AgentContainer::AgentContainer");

    ic_type = a_ic_type;

    m_num_diseases = a_num_diseases;
    AMREX_ASSERT(m_num_diseases < ExaEpi::max_num_diseases);

    m_student_counts.setVal(0);             // Initialize the MultiFab to zero
    comm_density_scale.setVal(1.0_rt);      // Default: no size scaling until main.cpp supplies real values
    comm_density_scale_work.setVal(1.0_rt); // Default: no work-size scaling until main.cpp supplies real values

    addAttributes();

    amrex::ParmParse pp("agent");

    pp.query("shelter_compliance", m_shelter_compliance);
    queryGpuArray<int, SchoolType::total>(pp, "student_teacher_ratio", m_student_teacher_ratio);

    queryGpuArray<Real, AgeGroups::total>(pp, "symptomatic_withdraw_compliance_day_0", m_symptomatic_withdraw_compliance_day_0);
    queryGpuArray<Real, AgeGroups::total>(pp, "symptomatic_withdraw_compliance_day_1", m_symptomatic_withdraw_compliance_day_1);
    queryGpuArray<Real, AgeGroups::total>(pp, "symptomatic_withdraw_compliance_day_2", m_symptomatic_withdraw_compliance_day_2);

    m_hospital = std::make_unique<HospitalModel<PCType, PTDType, PType>>();

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

    For each agent, set its position to the work community (IntIdx::work_i, IntIdx::work_j).

    If EXAEPI_DISABLE_SCHOOL is defined, agents enrolled in school (IntIdx::school_id > 0 --
    students and teachers alike, see InteractionModSchool.H) are left at their current (home)
    position instead, so they never commute to school for the day.
*/
void AgentContainer::moveAgentsToWork () {
    BL_PROFILE("AgentContainer::moveAgentsToWork");

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
            const auto& ptd = ptile.getParticleTileData();
            auto& aos = ptile.GetArrayOfStructs();
            ParticleType* pstruct = &(aos[0]);
            const size_t np = aos.numParticles();

            auto& soa = ptile.GetStructOfArrays();
            auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
            auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
#ifdef EXAEPI_DISABLE_SCHOOL
#warning School Interactions and student/teacher movement disabled
            auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
#endif

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
                if (!inHospital(ip, ptd) && !isOnTravel(ip, ptd)) {
#ifdef EXAEPI_DISABLE_SCHOOL
                    if (school_id_ptr[ip] > 0) { return; }
#endif
                    ParticleType& p = pstruct[ip];
                    p.pos(0) = static_cast<ParticleReal>((work_i_ptr[ip] + 0.5_rt) * dx[0]);
                    p.pos(1) = static_cast<ParticleReal>((work_j_ptr[ip] + 0.5_rt) * dx[1]);
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
                if (!inHospital(ip, ptd) && !isOnTravel(ip, ptd)) {
                    ParticleType& p = pstruct[ip];
                    p.pos(0) = static_cast<ParticleReal>((home_i_ptr[ip] + 0.5_rt) * dx[0]);
                    p.pos(1) = static_cast<ParticleReal>((home_j_ptr[ip] + 0.5_rt) * dx[1]);
                }
            });
        }
    }

    m_at_work = false;

    Redistribute();
    AMREX_ASSERT(OK());
}

/*! \brief Assign each school_id>0 agent to a class within its (community, school_id, grade) group --
    see IntIdx::school_class / IntIdx::school_class_group and TestParams::school_class_size. Splits
    each raw group into ceil(child_count / school_class_size) fixed-size classes. Educators teaching
    that grade are drawn into the same set of classes independently of the students, the same idiom
    already used to spread regular workers across a workplace's workgroup buckets (see
    agent.workgroup_size / UrbanPopData.cpp) -- not an attempt at exact balance.

    Called once, on a FRESH run only (never on restart): IntIdx::school_class/school_class_group are
    persistent per-agent attributes, checkpointed/restored like any other SoA attribute, so redrawing
    them on restart would silently reshuffle students into different classmates than the original run
    had, breaking continuity for restarted or branched runs. Tallies enrollment directly from the
    static work_i/work_j/school_id/school_grade attributes, not particle position. The resulting
    school_class_group numbering is computed identically on every rank (from a globally-reduced flat
    array), so it stays meaningful even if a later restart uses a different rank count. */
void AgentContainer::assignSchoolClasses (const ExaEpi::TestParams& params) {
    BL_PROFILE("AgentContainer::assignSchoolClasses");

    int max_school_id = getMaxGroup(IntIdx::school_id) + 1;
    int max_grade = getMaxGroup(IntIdx::school_grade) + 1;
    int ncomp = max_school_id * max_grade;

    const Box& domain = Geom(0).Domain();
    int domain_x = domain.length(0);
    int domain_y = domain.length(1);
    long n_flat = (long)domain_x * (long)domain_y * (long)ncomp;

    // Pass 1: tally student enrollment (naics == -1, i.e. not employed -- the same distinction
    // UrbanPopData.cpp already uses to tell students from educators, since school_grade alone doesn't
    // work here: college students are adults by age_group, but are not employees) and total
    // enrollment (students + educators) per (community, school_id, grade) raw group. Both are needed
    // in Pass 2: class *count* scales with student enrollment, but a group with educators and no
    // students still needs at least one class for them to land in, and a genuinely empty group needs
    // none (avoiding reserving a wasted slot for every unoccupied community/school_id/grade
    // combination in the domain, most of which no agent ever touches).
    Gpu::DeviceVector<Real> student_count_d(n_flat, 0.0_rt);
    Gpu::DeviceVector<Real> total_count_d(n_flat, 0.0_rt);
    auto* student_count_ptr = student_count_d.data();
    auto* total_count_ptr = total_count_d.data();

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            const size_t np = ptile.GetArrayOfStructs().numParticles();
            if (np == 0) { continue; }

            auto& soa = ptile.GetStructOfArrays();
            auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
            auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
            auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
            auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
            auto naics_ptr = soa.GetIntData(IntIdx::naics).data();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
                if (school_id_ptr[ip] > 0) {
                    long flat_comm = (long)work_j_ptr[ip] * domain_x + (long)work_i_ptr[ip];
                    long idx = flat_comm * ncomp + school_id_ptr[ip] * max_grade + school_grade_ptr[ip];
                    Gpu::Atomic::AddNoRet(&total_count_ptr[idx], 1.0_rt);
                    if (naics_ptr[ip] == -1) { Gpu::Atomic::AddNoRet(&student_count_ptr[idx], 1.0_rt); }
                }
            });
        }
    }
    Gpu::streamSynchronize();

    Vector<Real> student_count_h(n_flat);
    Vector<Real> total_count_h(n_flat);
    Gpu::copyAsync(Gpu::deviceToHost, student_count_d.begin(), student_count_d.end(), student_count_h.begin());
    Gpu::copyAsync(Gpu::deviceToHost, total_count_d.begin(), total_count_d.end(), total_count_h.begin());
    Gpu::streamSynchronize();
    ParallelDescriptor::ReduceRealSum(student_count_h.data(), (int)n_flat);
    ParallelDescriptor::ReduceRealSum(total_count_h.data(), (int)n_flat);

    // Pass 2: host-side compaction, identical on every rank -- decide how many classes each raw group
    // needs, and assign every raw group a base offset into a densely-packed, no-wasted-slots global
    // school_class_group space.
    Vector<int> n_classes_h(n_flat);
    Vector<long> class_group_base_h(n_flat);
    long max_class_group = 0;
    for (long idx = 0; idx < n_flat; ++idx) {
        int n_classes = 0;
        if (total_count_h[idx] > 0.0_rt) {
            n_classes = 1;
            if (student_count_h[idx] > 0.0_rt) {
                n_classes = std::max(1, (int)std::ceil(student_count_h[idx] / (Real)params.school_class_size));
            }
        }
        n_classes_h[idx] = n_classes;
        class_group_base_h[idx] = max_class_group;
        max_class_group += n_classes;
    }

    Gpu::DeviceVector<int> n_classes_d(n_flat);
    Gpu::DeviceVector<long> class_group_base_d(n_flat);
    Gpu::copyAsync(Gpu::hostToDevice, n_classes_h.begin(), n_classes_h.end(), n_classes_d.begin());
    Gpu::copyAsync(Gpu::hostToDevice, class_group_base_h.begin(), class_group_base_h.end(), class_group_base_d.begin());
    Gpu::streamSynchronize();
    auto* n_classes_ptr = n_classes_d.data();
    auto* class_group_base_ptr = class_group_base_d.data();

    // Pass 3: assign each school_id>0 agent (student or educator) a class within its raw group via an
    // independent per-agent random draw.
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = plev[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
            const size_t np = ptile.GetArrayOfStructs().numParticles();
            if (np == 0) { continue; }

            auto& soa = ptile.GetStructOfArrays();
            auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
            auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
            auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
            auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
            auto school_class_ptr = soa.GetIntData(IntIdx::school_class).data();
            auto school_class_group_ptr = soa.GetIntData(IntIdx::school_class_group).data();

            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int ip, RandomEngine const& engine) noexcept {
                if (school_id_ptr[ip] > 0) {
                    long flat_comm = (long)work_j_ptr[ip] * domain_x + (long)work_i_ptr[ip];
                    long raw_group = flat_comm * ncomp + school_id_ptr[ip] * max_grade + school_grade_ptr[ip];
                    int n_classes = n_classes_ptr[raw_group];
                    int school_class = Random_int(n_classes, engine);
                    school_class_ptr[ip] = school_class;
                    school_class_group_ptr[ip] = (int)(class_group_base_ptr[raw_group] + school_class);
                } else {
                    school_class_ptr[ip] = 0;
                    school_class_group_ptr[ip] = -1;
                }
            });
        }
    }
    Gpu::streamSynchronize();

    amrex::Print() << "SchoolClasses: " << max_class_group << " classes across " << max_school_id << " school_ids x " << max_grade
                   << " grades (school_class_size=" << params.school_class_size << ")\n";
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
            auto weatherIdxPtr = soa.GetIntData(IntIdx::weatherLookup).data();

            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
                if (!inHospital(i, ptd) && !withdrawn_ptr[i]) {
                    ParticleType& p = pstruct[i];
                    if (amrex::Random(engine) < random_travel_prob) {
                        random_travel_ptr[i] = i;
                        int i_random = int(amrex::Real(i_max) * amrex::Random(engine));
                        int j_random = int(amrex::Real(j_max) * amrex::Random(engine));
                        p.pos(0) = i_random;
                        p.pos(1) = j_random;
                        weatherIdxPtr[i] = -weatherIdxPtr[i] - 2;
                    }
                }
            });
        }
    }

    // Redistribute();
    // AMREX_ALWAYS_ASSERT(OK());
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
            auto weatherIdxPtr = soa.GetIntData(IntIdx::weatherLookup).data();

            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
                int unit = unit_arr(home_i_ptr[i], home_j_ptr[i], 0);
                if (!inHospital(i, ptd) && random_travel_ptr[i] < 0 && air_travel_ptr[i] < 0) {
                    if (withdrawn_ptr[i] == 1) { return; }
                    if (amrex::Random(engine) < air_travel_prob_ptr[unit]) {
                        ParticleType& p = pstruct[i];
                        p.pos(0) = trav_i_ptr[i];
                        p.pos(1) = trav_j_ptr[i];
                        air_travel_ptr[i] = i;
                        weatherIdxPtr[i] = -weatherIdxPtr[i] - 2;
                    }
                }
            });
        }
    }
    // Redistribute();
    // AMREX_ALWAYS_ASSERT(OK());
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
                            if (random1 < arrivalUnits_prob_ptr[low]) {
                                break; // low is the found airport index
                            }
                            // if random1 falls within (low, high), half the range
                            int mid = low + (high - low) / 2;
                            if (arrivalUnits_prob_ptr[mid] < random1) {
                                low = mid + 1;
                            } else {
                                high = mid;
                            }
                        }
                        destUnit = arrivalUnits_ptr[low];
                    }
                    if (destUnit >= 0 && (Start[destUnit + 1] > Start[destUnit])) { // skip size 0 comms
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
            auto weatherIdxPtr = soa.GetIntData(IntIdx::weatherLookup).data();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                if (random_travel_ptr[i] >= 0) {
                    ParticleType& p = pstruct[i];
                    random_travel_ptr[i] = -1;
                    p.pos(0) = static_cast<ParticleReal>((home_i_ptr[i] + 0.5_rt) * dx[0]);
                    p.pos(1) = static_cast<ParticleReal>((home_j_ptr[i] + 0.5_rt) * dx[1]);
                    weatherIdxPtr[i] = -weatherIdxPtr[i] - 2;
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
            auto weatherIdxPtr = soa.GetIntData(IntIdx::weatherLookup).data();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                if (air_travel_ptr[i] >= 0) {
                    ParticleType& p = pstruct[i];
                    air_travel_ptr[i] = -1;
                    p.pos(0) = static_cast<ParticleReal>((home_i_ptr[i] + 0.5_rt) * dx[0]);
                    p.pos(1) = static_cast<ParticleReal>((home_j_ptr[i] + 0.5_rt) * dx[1]);
                    weatherIdxPtr[i] = -weatherIdxPtr[i] - 2;
                }
            });
        }
    }
    Redistribute();
    AMREX_ALWAYS_ASSERT(OK());
}

void AgentContainer::initializeWeatherIndex (const iMultiFab& unit_mf, ActiveWeather* activeWeatherdata) {
    BL_PROFILE("AgentContainer::initializeWeatherIndex");
    awd = activeWeatherdata;

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = plev[{mfi.index(), mfi.LocalTileIndex()}];
            auto& aos = ptile.GetArrayOfStructs();
            const size_t np = aos.numParticles();
            auto& soa = ptile.GetStructOfArrays();
            const auto unit_arr = unit_mf[mfi].array();
            auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
            auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
            auto unitVec = awd->unitVec.data();
            int nUnits = awd->unitVec.size();
            auto weatherIdxPtr = soa.GetIntData(IntIdx::weatherLookup).data();

            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
                int unit = unit_arr(home_i_ptr[i], home_j_ptr[i], 0);
                int lunit = 0;
                bool found = false;
                for (; lunit < nUnits; lunit++) {
                    if (unitVec[lunit] == unit) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    weatherIdxPtr[i] = lunit;
                } else {
                    weatherIdxPtr[i] = -1;
                }
            });
        }
    }
}

void AgentContainer::initializeWeatherIndex_UrbanPop (const iMultiFab& geoid_mf, ActiveWeather* activeWeatherdata) {
    BL_PROFILE("AgentContainer::initializeWeatherIndex");
    awd = activeWeatherdata;

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);

        const auto& geom = Geom(0);
        const auto domain = geom.Domain();
        const auto plo = geom.ProbLoArray();
        const auto dxi = geom.InvCellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = plev[{mfi.index(), mfi.LocalTileIndex()}];
            auto& aos = ptile.GetArrayOfStructs();
            auto aos1 = &ptile.GetArrayOfStructs()[0];
            const size_t np = aos.numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto geoid_arr = geoid_mf[mfi].array();
            auto unitVec = awd->unitVec.data();
            int nUnits = awd->unitVec.size();
            auto weatherIdxPtr = soa.GetIntData(IntIdx::weatherLookup).data();

            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
                auto& p = aos1[i];
                CellAssignor assignor;
                IntVect iv2 = assignor(p, plo, dxi, domain);
                int fips = geoid_arr(iv2[0], iv2[1], 0);

                int lunit = 0;
                bool found = false;
                for (; lunit < nUnits; lunit++) {
                    if (unitVec[lunit] == fips) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    weatherIdxPtr[i] = lunit;
                } else {
                    weatherIdxPtr[i] = -1;
                }
            });
        }
    }
}

void AgentContainer::advanceWeatherIndex () {
    for (int lev = 0; lev <= finestLevel(); ++lev) {
        auto& plev = GetParticles(lev);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = plev[{mfi.index(), mfi.LocalTileIndex()}];
            auto& aos = ptile.GetArrayOfStructs();
            const size_t np = aos.numParticles();
            auto& soa = ptile.GetStructOfArrays();
            auto weatherIdxPtr = soa.GetIntData(IntIdx::weatherLookup).data();
            int nUnits = awd->unitVec.size();

#if 0
            if (soa.GetIntData(IntIdx::weatherLookup).size() != np) {
                amrex::Print() << "VEC SIZE " << soa.GetIntData(IntIdx::weatherLookup).size()<< " np " << np << "\n";
            }
#endif
            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine) noexcept {
                if (weatherIdxPtr[i] >= 0) {
                    weatherIdxPtr[i] += nUnits;
                } else if (weatherIdxPtr[i] <= -2) {
                    weatherIdxPtr[i] -= nUnits;
                }
            });
        }
    }
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
                    p.pos(0) = static_cast<ParticleReal>((hosp_i_ptr[ip] + 0.5_prt) * dx[0]);
                    p.pos(1) = static_cast<ParticleReal>((hosp_j_ptr[ip] + 0.5_prt) * dx[1]);
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

            auto shelter_compliance = m_shelter_compliance;
            amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine) noexcept {
                if (amrex::Random(engine) < shelter_compliance) { withdrawn_ptr[i] = 1; }
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

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                withdrawn_ptr[i] = 0;
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
                auto symptomatic_ptr = soa.GetIntData(i_RT + i0(d) + IntIdxDisease::symptomatic).data();

                auto prob_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::prob).data();
                auto counter_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::disease_counter).data();
                auto latent_period_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::latent_period).data();
                auto infectious_period_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::infectious_period).data();
                auto incubation_period_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::incubation_period).data();
                auto hospital_delay_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::hospital_delay).data();
                auto hospital_random_ptr = soa.GetRealData(r_RT + r0(d) + RealIdxDisease::hospital_random).data();
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
                            setInfected(&(status_ptr[i]), &(symptomatic_ptr[i]), &(counter_ptr[i]), &(latent_period_ptr[i]),
                                        &(infectious_period_ptr[i]), &(incubation_period_ptr[i]), &(hospital_delay_ptr[i]),
                                        &(hospital_random_ptr[i]), engine, lparm);
                            // ds_arr is this tile's local (no-ghost-cell) FAB, indexed by the agent's home
                            // cell; only safe when the agent is actually physically at home there (as
                            // moveAgentsToHome() guarantees for everyone except hospitalized/traveling
                            // agents -- for those, home_i/home_j can fall in a different, possibly
                            // remote-rank box, so skip the home-community attribution for them).
                            if (!inHospital(i, ptd) && !isOnTravel(i, ptd)) {
                                Gpu::Atomic::AddNoRet(&ds_arr(home_i_ptr[i], home_j_ptr[i], 0, DiseaseStats::new_cases), 1.0_rt);
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

/*! \brief Computes the total number of agents with each #OutputStatus

    Returns a vector with nattrib components corresponding to each value of #OutputStatus; each element is
    the total number of agents at a step with the corresponding #OutputStatus (in that order).
*/
std::array<Long, OutputStatus::nattribs> AgentContainer::getTotals (const int a_d /*!< disease index */) {
    BL_PROFILE("getTotals");
    static_assert(OutputStatus::nattribs == 20, "Expected nattribs == 20");

    const auto* disease_parm_d = getDiseaseParameters_d(a_d);
    amrex::ReduceOps<REPEAT(20, ReduceOpSum)> reduce_ops;
    auto r = amrex::ParticleReduce<ReduceData<REPEAT(20, int)>>(
            *this,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                                 const int i) noexcept -> amrex::GpuTuple<REPEAT(20, int)> {
                int s[OutputStatus::nattribs] = {};
                auto status = ptd.m_runtime_idata[i0(a_d) + IntIdxDisease::status][i];

                AMREX_ALWAYS_ASSERT(status >= 0);
                AMREX_ALWAYS_ASSERT(status <= 4);

                if (status == Status::never || status == Status::susceptible) { s[OutputStatus::Su] = 1; }
                if (status == Status::immune) { s[OutputStatus::R] = 1; }
                if (status == Status::dead) { s[OutputStatus::D] = 1; }

                if (isNewlyInfected(i, ptd, a_d)) { s[OutputStatus::NewI] = 1; }
                if (isNewlySymptomatic(i, ptd, a_d)) { s[OutputStatus::NewS] = 1; }
                if (isNewlyAsymptomatic(i, ptd, a_d)) { s[OutputStatus::NewA] = 1; }
                if (isNewlyPresymptomatic(i, ptd, a_d)) { s[OutputStatus::NewP] = 1; }
                if (isNewlyHospitalized(i, ptd, a_d)) { s[OutputStatus::NewH] = 1; }

                if (!inHospital(i, ptd)) {                           // do not include hospitalized agents in these counts
                    if (status == Status::infected) {                // exposed
                        if (notInfectiousButInfected(i, ptd, a_d)) { // exposed, but not infectious
                            if (isAsymptomatic(i, ptd, a_d)) {
                                s[OutputStatus::A_PI] = 1;
                            } else if (isPresymptomatic(i, ptd, a_d)) {
                                s[OutputStatus::PS_PI] = 1;
                            } else if (isSymptomatic(i, ptd, a_d)) {
                                if (willBeHospitalized(i, ptd, a_d, *disease_parm_d)) {
                                    s[OutputStatus::S_PI_H] = 1;
                                } else {
                                    s[OutputStatus::S_PI_NH] = 1;
                                }
                            } else {
                                amrex::Abort("how did I get here?");
                            }
                        } else { // currently infectious
                            if (isAsymptomatic(i, ptd, a_d)) {
                                s[OutputStatus::A_I] = 1;
                            } else if (isPresymptomatic(i, ptd, a_d)) {
                                s[OutputStatus::PS_I] = 1;
                            } else if (isSymptomatic(i, ptd, a_d)) {
                                if (willBeHospitalized(i, ptd, a_d, *disease_parm_d)) {
                                    s[OutputStatus::S_I_H] = 1;
                                } else {
                                    s[OutputStatus::S_I_NH] = 1;
                                }
                            } else {
                                amrex::Abort("how did I get here?");
                            }
                        }
                    }
                } else { // now the hospitalized categories
                    if (notInfectiousButInfected(i, ptd, a_d)) {
                        s[OutputStatus::H_NI] = 1;
                    } else {
                        s[OutputStatus::H_I] = 1;
                    }
                }
                return {ARRAY_TO_TUPLE(20, s)};
            },
            reduce_ops);
    std::array<Long, OutputStatus::nattribs> counts;
    extract_tuple_to_array(r, counts);
    ParallelDescriptor::ReduceLongSum(&counts[0], OutputStatus::nattribs);

    return counts;
}

/*! \brief Computes the total number of agents with a requested status, by age

    Returns a vector with counts by age.
*/
std::array<Long, AgeGroups::total> AgentContainer::getNewStatusByAge (const int a_d, const int output_status) {
    BL_PROFILE("getNewStatusByAge");
    static_assert(AgeGroups::total == 6, "Expected 6 total age groups");

    amrex::ReduceOps<REPEAT(6, ReduceOpSum)> reduce_ops;
    auto r = amrex::ParticleReduce<ReduceData<REPEAT(6, int)>>(
            *this,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                                 const int i) noexcept -> amrex::GpuTuple<REPEAT(6, int)> {
                int s[AgeGroups::total] = {};
                auto age = ptd.m_idata[IntIdx::age_group][i];

                if (output_status == OutputStatus::NewI && isNewlyInfected(i, ptd, a_d)) { s[age] = 1; }
                if (output_status == OutputStatus::NewS && isNewlySymptomatic(i, ptd, a_d)) { s[age] = 1; }
                if (output_status == OutputStatus::NewH && isNewlyHospitalized(i, ptd, a_d)) { s[age] = 1; }
                if (output_status == OutputStatus::NewA && isNewlyAsymptomatic(i, ptd, a_d)) { s[age] = 1; }
                if (output_status == OutputStatus::NewP && isNewlyPresymptomatic(i, ptd, a_d)) { s[age] = 1; }

                return {ARRAY_TO_TUPLE(6, s)};
            },
            reduce_ops);
    std::array<Long, AgeGroups::total> counts;
    extract_tuple_to_array(r, counts);
    ParallelDescriptor::ReduceLongSum(&counts[0], AgeGroups::total, ParallelDescriptor::IOProcessorNumber());

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
        int local_max = amrex::get<0>(r);
        // ParticleReduce only reduces over this rank's own particles -- AMReX performs no MPI
        // reduction internally (confirmed both by AMReX_ParticleReduce.H's own doc comments and by
        // direct observation: different ranks reported different local maxima for school_id on the
        // same run). Every caller relies on this being the true GLOBAL max (e.g. to size a MultiFab's
        // component count consistently across ranks), so reduce across ranks before caching.
        ParallelDescriptor::ReduceIntMax(local_max);
        max_attribute_values[group_idx] = local_max;
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
    moveAgentsToHome();
}

/*! \brief Interaction of agents during day time - work and school */
void AgentContainer::interactDay (MultiFab& a_mask_behavior /*!< Masking behavior */) {
    BL_PROFILE("AgentContainer::interactDay");
    interactWork(a_mask_behavior);
    interactHospital(a_mask_behavior);
    interactSchool(a_mask_behavior);
    interactNborhoodDay(a_mask_behavior);
    interactCommDay(a_mask_behavior);
}

void AgentContainer::interactWork (MultiFab& a_mask_behavior) {
    BL_PROFILE("AgentContainer::interactWork");
    m_mod_work.interactAgents(*this, a_mask_behavior);
}

void AgentContainer::interactHospital (MultiFab& a_mask_behavior) {
    BL_PROFILE("AgentContainer::interactHospital");
    m_hospital->interactAgents(*this, a_mask_behavior);
}

void AgentContainer::interactSchool (MultiFab& a_mask_behavior) {
    BL_PROFILE("AgentContainer::interactSchool");
#ifndef EXAEPI_DISABLE_SCHOOL
    m_mod_school.interactAgents(*this, a_mask_behavior);
#else
    amrex::ignore_unused(a_mask_behavior);
#endif
}

void AgentContainer::interactNborhoodDay (MultiFab& a_mask_behavior) {
    BL_PROFILE("AgentContainer::interactNborhoodDay");
    m_mod_nborhood_day.interactAgents(*this, a_mask_behavior);
}

void AgentContainer::interactCommDay (MultiFab& a_mask_behavior) {
    BL_PROFILE("AgentContainer::interactCommDay");
    m_mod_comm_day.interactAgents(*this, a_mask_behavior);
}

/*! \brief Interaction of agents during evening (after work) - social stuff */
void AgentContainer::interactEvening (MultiFab& /*a_mask_behavior*/ /*!< Masking behavior */) {
    BL_PROFILE("AgentContainer::interactEvening");
}

void AgentContainer::interactHH (MultiFab& a_mask_behavior) {
    BL_PROFILE("AgentContainer::interactHH");
    m_mod_hh.interactAgents(*this, a_mask_behavior);
}

void AgentContainer::interactNC (MultiFab& a_mask_behavior) {
    BL_PROFILE("AgentContainer::interactNC");
    m_mod_nc.interactAgents(*this, a_mask_behavior);
}

void AgentContainer::interactNborhoodNight (MultiFab& a_mask_behavior) {
    BL_PROFILE("AgentContainer::interactNborhoodNight");
    m_mod_nborhood_night.interactAgents(*this, a_mask_behavior);
}

void AgentContainer::interactCommNight (MultiFab& a_mask_behavior) {
    BL_PROFILE("AgentContainer::interactCommNight");
    m_mod_comm_night.interactAgents(*this, a_mask_behavior);
}

/*! \brief Interaction of agents during nighttime - at home */
void AgentContainer::interactNight (MultiFab& a_mask_behavior /*!< Masking behavior */) {
    BL_PROFILE("AgentContainer::interactNight");
    interactHH(a_mask_behavior);
    interactNC(a_mask_behavior);
    interactNborhoodNight(a_mask_behavior);
    interactCommNight(a_mask_behavior);
}

/*! \brief Sum of expected infections (1 - prob_ptr) over susceptible agents for disease d.
    Call after interactions but before infectAgents. The difference between successive calls
    gives the marginal expected infections from each interaction context. */
amrex::Real AgentContainer::sumExpectedInfections (int d) const {
    BL_PROFILE("AgentContainer::sumExpectedInfections");
    amrex::ReduceOps<amrex::ReduceOpSum> reduce_ops;
    auto r = amrex::ParticleReduce<amrex::ReduceData<amrex::Real>>(
            *this,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                                 const int i) noexcept -> amrex::GpuTuple<amrex::Real> {
                auto status = ptd.m_runtime_idata[i0(d) + IntIdxDisease::status][i];
                if (status == Status::never || status == Status::susceptible) {
                    amrex::Real prob = ptd.m_runtime_rdata[r0(d) + RealIdxDisease::prob][i];
                    return {amrex::max(0.0_rt, 1.0_rt - prob)};
                }
                return {0.0_rt};
            },
            reduce_ops);
    return amrex::get<0>(r);
}

/*! \brief Snapshot the current prob_ptr for all susceptible agents into m_prob_snapshot.
    Call immediately before an interaction; follow with sumContextInfections to get the
    order-independent contribution of that interaction. */
void AgentContainer::snapshotProbs (int d) {
    BL_PROFILE("AgentContainer::snapshotProbs");
    m_prob_snapshot.clear();
    for (int lev = 0; lev < numLevels(); ++lev) {
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = ParticlesAt(lev, mfi);
            auto& soa = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();
            auto& snap = m_prob_snapshot.emplace_back();
            snap.resize(np);
            if (np > 0) {
                auto& prob_vec = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob);
                amrex::Gpu::copy(amrex::Gpu::deviceToDevice, prob_vec.begin(), prob_vec.begin() + np, snap.begin());
            }
        }
    }
}

/*! \brief Order-independent contribution of the last interaction to expected new infections.
    Computes sum_i( max(0, 1 - prob_after[i] / prob_before[i]) ) over susceptible agents,
    where prob_before comes from the preceding snapshotProbs() call.
    The ratio recovers the standalone survival factor for each agent regardless of what
    prior interactions have already applied. */
amrex::Real AgentContainer::sumContextInfections (int d) {
    BL_PROFILE("AgentContainer::sumContextInfections");
    amrex::Real total = 0.0_rt;
    int snap_idx = 0;
    for (int lev = 0; lev < numLevels(); ++lev) {
        for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            auto& soa = ptile.GetStructOfArrays();
            const auto np = ptile.numParticles();

            if (np == 0) {
                ++snap_idx;
                continue;
            }

            AMREX_ASSERT(snap_idx < (int)m_prob_snapshot.size());
            AMREX_ASSERT((int)m_prob_snapshot[snap_idx].size() == np);

            auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
            auto before_ptr = m_prob_snapshot[snap_idx].dataPtr();

            amrex::Gpu::DeviceScalar<amrex::Real> tile_sum_d(0.0_rt);
            auto* sum_ptr = tile_sum_d.dataPtr();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                if (isSusceptible(i, ptd, d)) {
                    amrex::Real before = amrex::max(before_ptr[i], ParticleReal(1e-30));
                    amrex::Real contribution = amrex::max(0.0_rt, 1.0_rt - prob_ptr[i] / before);
                    amrex::Gpu::Atomic::AddNoRet(sum_ptr, contribution);
                }
            });
            amrex::Gpu::synchronize();

            total += tile_sum_d.dataValue();
            ++snap_idx;
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(&total, 1);
    return total;
}

void AgentContainer::printStudentTeacherCounts () const {
    ReduceOps<REPEAT(10, ReduceOpSum)> reduce_ops;
    auto r = ParticleReduce<ReduceData<REPEAT(10, int)>>(
            *this,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                                 const int i) noexcept -> GpuTuple<REPEAT(10, int)> {
                int counts[10] = {};
                if (ptd.m_idata[IntIdx::school_id][i] > 0) {
                    int pos = (ptd.m_idata[IntIdx::workgroup][i] > 0 ? 0 : 5);
                    int grade = ptd.m_idata[IntIdx::school_grade][i];
                    int school_type = getSchoolType(grade);
                    // always should have an allocated grade if we have a school
                    AMREX_ASSERT(school_type != SchoolType::none);
                    pos = pos + school_type - SchoolType::college;
                    AMREX_ASSERT(pos >= 0 && pos < 10);
                    counts[pos] = 1;
                }
                return {ARRAY_TO_TUPLE(10, counts)};
            },
            reduce_ops);
    std::array<Long, OutputStatus::nattribs> counts;
    extract_tuple_to_array(r, counts);
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
    ReduceOps<REPEAT(6, ReduceOpSum)> reduce_ops;
    auto r = ParticleReduce<ReduceData<REPEAT(6, int)>>(
            *this,
            [=] AMREX_GPU_DEVICE (const AgentContainer::ParticleTileType::ConstParticleTileDataType& ptd,
                                 const int i) noexcept -> GpuTuple<REPEAT(6, int)> {
                int counts[6] = {};
                int age_group = ptd.m_idata[IntIdx::age_group][i];
                counts[age_group] = 1;
                return {ARRAY_TO_TUPLE(6, counts)};
            },
            reduce_ops);

    std::array<Long, 6> counts;
    extract_tuple_to_array(r, counts);
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
