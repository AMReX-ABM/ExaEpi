/*! @file CensusData.cpp
*/

#include <AMReX_ParticleUtil.H>

#include "CensusData.H"


using namespace amrex;
using namespace ExaEpi;


/*! \brief Set computational domain, i.e., number of cells in each direction, from the
    demographic data (number of communities).
 *
 *  If the initialization type (ExaEpi::TestParams::ic_type) is ExaEpi::ICType::Census, then
 *  + The domain is a 2D square, where the total number of cells is the lowest square of an
 *    integer that is greater than #DemographicData::Ncommunity
 *  + The physical size is 1.0 in each dimension.
 *
 *  A periodic Cartesian grid is defined.
*/
Geometry get_geometry (const DemographicData& demo /*!< demographic data */) {
    int is_per[BL_SPACEDIM];
    for (int i = 0; i < BL_SPACEDIM; i++) {
        is_per[i] = true;
    }

    RealBox real_box;
    Box base_domain;
    Geometry geom;

    IntVect iv;
    iv[0] = iv[1] = (int) std::floor(std::sqrt((double) demo.Ncommunity));
    while (iv[0]*iv[1] <= demo.Ncommunity) {
        ++iv[0];
    }
    base_domain = Box(IntVect(AMREX_D_DECL(0, 0, 0)), iv-1);

    for (int n = 0; n < BL_SPACEDIM; n++)
    {
        real_box.setLo(n, 0.0);
        real_box.setHi(n, 1.0);
    }

    geom.define(base_domain, &real_box, CoordSys::cartesian, is_per);
    return geom;
}


void CensusData::init (ExaEpi::TestParams &params, Geometry &geom, BoxArray &ba, DistributionMapping &dm) {

    demo.InitFromFile(params.census_filename, params.workgroup_size);

    geom = get_geometry(demo);

    ba.define(geom.Domain());
    ba.maxSize(params.max_box_size);
    dm.define(ba);

    Print() << "Base domain is: " << geom.Domain() << "\n";
    Print() << "Max box size is: " << params.max_box_size << "\n";
    Print() << "Number of boxes is: " << ba.size() << " over " << ParallelDescriptor::NProcs() << " ranks. \n";

    num_residents_mf.define(ba, dm, 6, 0);
    unit_mf.define(ba, dm, 1, 0);
    FIPS_mf.define(ba, dm, 2, 0);
    comm_mf.define(ba, dm, 1, 0);

    num_residents_mf.setVal(0);
    unit_mf.setVal(-1);
    FIPS_mf.setVal(-1);
    comm_mf.setVal(-1);

}

/*! \brief Assigns school by taking a random number between 0 and 100, and using
 *  default distribution to choose elementary/middle/high school. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void assign_school (int* school_grade, int* school_id, const int age_group, const int nborhood, const RandomEngine& engine) {
    if (age_group == AgeGroups::u5) {
        // under 5
        // assume 50% in daycare
        if (Random_int(100, engine) < 50) {
            *school_grade = 0;
            *school_id = SchoolCensusIDType::daycare_5 + nborhood; // one daycare per nborhood
        } else {
            *school_grade = -1;
            *school_id = SchoolCensusIDType::none; // no school
        }
    } else if (age_group == AgeGroups::a5to17) {
        // 5 to 17
        int il4 = Random_int(100, engine);
        if (il4 < 36) {
            *school_id = SchoolCensusIDType::elem_3 + (nborhood / 2);  // elementary school, in neighborhood 1&2 or 3&4
            *school_grade = 5;
            AMREX_ALWAYS_ASSERT(*school_id < 5);
        } else if (il4 < 68) {
            *school_id = SchoolCensusIDType::middle_2;  // middle school, one for all neighborhoods
            *school_grade = 9;
        } else if (il4 < 93) {
            *school_id = SchoolCensusIDType::high_1;  // high school, one for all neighborhoods
            *school_grade = 12;
        } else {
            *school_id = SchoolCensusIDType::none;  // not in school, presumably 18-year-olds or some home-schooled, etc
            *school_grade = -1;
        }
    } else {
        *school_grade = -1;
        *school_id = SchoolCensusIDType::none; // no school
    }
}


/*! \brief Initialize agents for ExaEpi::ICType::Census

 *  + Define and allocate the following integer MultiFabs:
 *    + num_families: number of families; has 7 components, each component is the
 *      number of families of size (component+1)
 *    + fam_offsets: offset array for each family (i.e., each component of each grid cell), where the
 *      offset is the total number of people before this family while iterating over the grid.
 *    + fam_id: ID array for each family ()i.e., each component of each grid cell, where the ID is the
 *      total number of families before this family while iterating over the grid.
 *  + At each grid cell in each box/tile on each processor:
 *    + Set community number.
 *    + Find unit number for this community; specify that a part of this unit is on this processor;
 *      set unit number, FIPS code, and census tract number at this grid cell (community).
 *    + Set community size: 2000 people, unless this is the last community of a unit, in which case
 *      the remaining people if > 1000 (else 0).
 *    + Compute cumulative distribution (on a scale of 0-1000) of household size ranging from 1 to 7:
 *      initialize with default distributions, then compute from census data if available.
 *    + For each person in this community, generate a random integer between 0 and 1000; based on its
 *      value, assign this person to a household of a certain size (1-7) based on the cumulative
 *      distributions above.
 *  + Compute total number of agents (people), family offsets and IDs over the box/tile.
 *  + Allocate particle container AoS and SoA arrays for the computed number of agents.
 *  + At each grid cell in each box/tile on each processor, and for each component (where component
 *    corresponds to family size):
 *    + Compute percentage of school age kids (kids of age 5-17 as a fraction of total kids - under 5
 *      plus 5-17), if available in census data or set to default (76%).
 *    + For each agent at this grid cell and family size (component):
 *      + Find age group by generating a random integer (0-100) and using default age distributions.
 *        Look at code to see the algorithm for family size > 1.
 *      + Set agent position at the center of this grid cell.
 *      + Initialize status and day counters.
 *      + Set age group and family ID.
 *      + Set home location to current grid cell.
 *      + Initialize work location to current grid cell. Actual work location is set in
 *        ExaEpi::read_workerflow().
 *      + Set neighborhood and work neighborhood values. Actual work neighborhood is set
 *        in ExaEpi::read_workerflow().
 *      + Initialize workgroup to 0. It is set in ExaEpi::read_workerflow().
 *      + If age group is 5-17, assign a school based on neighborhood (#assign_school).
 *  + Copy everything to GPU device.
*/
void CensusData::initAgents (AgentContainer& pc,       /*!< Agents */
                             const int nborhood_size      /*!< Size of neighborhood */ )
{
    BL_PROFILE("CensusData::initAgents");

    const Box& domain = pc.Geom(0).Domain();

    auto &ba = num_residents_mf.boxArray();
    auto &dm = num_residents_mf.DistributionMap();

    iMultiFab num_families(ba, dm, 7, 0);
    iMultiFab fam_offsets(ba, dm, 7, 0);
    iMultiFab fam_id(ba, dm, 7, 0);
    num_families.setVal(0);

    auto Ncommunity = demo.Ncommunity;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf); mfi.isValid(); ++mfi) {
        auto unit_arr = unit_mf[mfi].array();
        auto FIPS_arr = FIPS_mf[mfi].array();
        auto comm_arr = comm_mf[mfi].array();
        auto nf_arr = num_families[mfi].array();
        auto nr_arr = num_residents_mf[mfi].array();

        auto unit_on_proc = demo.Unit_on_proc_d.data();
        auto Start = demo.Start_d.data();
        auto FIPS = demo.FIPS_d.data();
        auto Tract = demo.Tract_d.data();
        auto Population = demo.Population_d.data();

        auto H1 = demo.H1_d.data();
        auto H2 = demo.H2_d.data();
        auto H3 = demo.H3_d.data();
        auto H4 = demo.H4_d.data();
        auto H5 = demo.H5_d.data();
        auto H6 = demo.H6_d.data();
        auto H7 = demo.H7_d.data();

        auto N5  = demo.N5_d.data();
        auto N17 = demo.N17_d.data();

        auto bx = mfi.tilebox();
        ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& engine) noexcept
        {
            int community = (int) domain.index(IntVect(AMREX_D_DECL(i, j, k)));
            if (community >= Ncommunity) { return; }
            comm_arr(i, j, k) = community;

            int unit = 0;
            while (community >= Start[unit+1]) { unit++; }
            unit_on_proc[unit] = 1;
            unit_arr(i, j, k) = unit;
            FIPS_arr(i, j, k, 0) = FIPS[unit];
            FIPS_arr(i, j, k, 1) = Tract[unit];

            int community_size;
            if (Population[unit] < (1000 + DemographicData::COMMUNITY_SIZE * (community - Start[unit]))) {
                community_size = 0;  /* Don't set up any residents; workgroup-only */
            }
            else {
                community_size = DemographicData::COMMUNITY_SIZE;   /* Standard 2000-person community */
            }

            int p_hh[7] = {330, 670, 800, 900, 970, 990, 1000};
            int num_hh = H1[unit] + H2[unit] + H3[unit] +
                H4[unit] + H5[unit] + H6[unit] + H7[unit];
            if (num_hh) {
                p_hh[0] = 1000 * H1[unit] / num_hh;
                p_hh[1] = 1000* (H1[unit] + H2[unit]) / num_hh;
                p_hh[2] = 1000* (H1[unit] + H2[unit] + H3[unit]) / num_hh;
                p_hh[3] = 1000* (H1[unit] + H2[unit] + H3[unit] + H4[unit]) / num_hh;
                p_hh[4] = 1000* (H1[unit] + H2[unit] + H3[unit] +
                                 H4[unit] + H5[unit]) / num_hh;
                p_hh[5] = 1000* (H1[unit] + H2[unit] + H3[unit] +
                                 H4[unit] + H5[unit] + H6[unit]) / num_hh;
                p_hh[6] = 1000;
            }

            int npeople = 0;
            while (npeople < community_size + 1) {
                int il  = Random_int(1000, engine);

                int family_size = 1;
                while (il > p_hh[family_size]) { ++family_size; }
                AMREX_ASSERT(family_size > 0);
                AMREX_ASSERT(family_size <= 7);

                nf_arr(i, j, k, family_size-1) += 1;
                npeople += family_size;
            }

            AMREX_ASSERT(npeople == nf_arr(i, j, k, 0) +
                         2*nf_arr(i, j, k, 1) +
                         3*nf_arr(i, j, k, 2) +
                         4*nf_arr(i, j, k, 3) +
                         5*nf_arr(i, j, k, 4) +
                         6*nf_arr(i, j, k, 5) +
                         7*nf_arr(i, j, k, 6));

            nr_arr(i, j, k, 5) = npeople;
        });

        int nagents;
        int ncomp = num_families[mfi].nComp();
        int ncell = num_families[mfi].numPts();
        {
            BL_PROFILE("setPopulationCounts_prefixsum")
            const int* in = num_families[mfi].dataPtr();
            int* out = fam_offsets[mfi].dataPtr();
            nagents = Scan::PrefixSum<int>(ncomp*ncell,
                            [=] AMREX_GPU_DEVICE (int i) -> int {
                                int comp = i / ncell;
                                return (comp+1)*in[i];
                            },
                            [=] AMREX_GPU_DEVICE (int i, int const& x) { out[i] = x; },
                                               Scan::Type::exclusive, Scan::retSum);
        }
        {
            BL_PROFILE("setFamily_id_prefixsum")
            const int* in = num_families[mfi].dataPtr();
            int* out = fam_id[mfi].dataPtr();
            Scan::PrefixSum<int>(ncomp*ncell,
                                 [=] AMREX_GPU_DEVICE (int i) -> int {
                                     return in[i];
                                 },
                                 [=] AMREX_GPU_DEVICE (int i, int const& x) { out[i] = x; },
                                 Scan::Type::exclusive, Scan::retSum);
        }

        auto offset_arr = fam_offsets[mfi].array();
        auto fam_id_arr = fam_id[mfi].array();
        auto& agents_tile = pc.DefineAndReturnParticleTile(0, mfi);
        agents_tile.resize(nagents);
        auto aos = &agents_tile.GetArrayOfStructs()[0];
        auto& soa = agents_tile.GetStructOfArrays();

        auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
        auto family_ptr = soa.GetIntData(IntIdx::family).data();
        auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
        auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
        auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
        auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
        auto trav_i_ptr = soa.GetIntData(IntIdx::trav_i).data();
        auto trav_j_ptr = soa.GetIntData(IntIdx::trav_j).data();
        auto hosp_i_ptr = soa.GetIntData(IntIdx::hosp_i).data();
        auto hosp_j_ptr = soa.GetIntData(IntIdx::hosp_j).data();
        auto nborhood_ptr = soa.GetIntData(IntIdx::nborhood).data();
        auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
        auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
        auto school_closed_ptr = soa.GetIntData(IntIdx::school_closed).data();
        auto naics_ptr = soa.GetIntData(IntIdx::naics).data();
        auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
        auto work_nborhood_ptr = soa.GetIntData(IntIdx::work_nborhood).data();
        auto random_travel_ptr = soa.GetIntData(IntIdx::random_travel).data();
        auto air_travel_ptr = soa.GetIntData(IntIdx::air_travel).data();

        int i_RT = IntIdx::nattribs;
        int r_RT = RealIdx::nattribs;
        int n_disease = pc.m_num_diseases;

        GpuArray<int*,ExaEpi::max_num_diseases> status_ptrs;
        GpuArray<ParticleReal*,ExaEpi::max_num_diseases> counter_ptrs, timer_ptrs;
        for (int d = 0; d < n_disease; d++) {
            status_ptrs[d] = soa.GetIntData(i_RT+i0(d)+IntIdxDisease::status).data();
            counter_ptrs[d] = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::disease_counter).data();
            timer_ptrs[d] = soa.GetRealData(r_RT+r0(d)+RealIdxDisease::treatment_timer).data();
        }

        auto dx = pc.ParticleGeom(0).CellSizeArray();
        auto my_proc = ParallelDescriptor::MyProc();

        auto student_counts_arr = pc.m_student_counts[mfi].array();

        Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical (init_agents_nextid)
#endif
        {
            pid = AgentContainer::ParticleType::NextID();
            AgentContainer::ParticleType::NextID(pid+nagents);
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<Long>(pid + nagents) < LastParticleID,
            "Error: overflow on agent id numbers!");

        ParallelForRNG(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n, RandomEngine const& engine) noexcept
        {
            int nf = nf_arr(i, j, k, n);
            if (nf == 0) return;

            int unit = unit_arr(i, j, k);
            int community = comm_arr(i, j, k);
            int family_id_start = fam_id_arr(i, j, k, n);
            int family_size = n + 1;
            int num_to_add = family_size * nf;

            int community_size;
            if (Population[unit] < (1000 + DemographicData::COMMUNITY_SIZE * (community - Start[unit]))) {
                community_size = 0;  /* Don't set up any residents; workgroup-only */
            }
            else {
                community_size = DemographicData::COMMUNITY_SIZE;   /* Standard 2000-person community */
            }

            int p_schoolage = 0;
            if (community_size) {  // Only bother for residential communities
                if (N5[unit] + N17[unit]) {
                    p_schoolage = 100*N17[unit] / (N5[unit] + N17[unit]);
                }
                else {
                    p_schoolage = 76;
                }
            }

            int start = offset_arr(i, j, k, n);
            int nborhood = 0;
            for (int ii = 0; ii < num_to_add; ++ii) {
                int ip = start + ii;
                auto& agent = aos[ip];
                int il2 = Random_int(100, engine);
                if (ii % family_size == 0) {
                    nborhood = Random_int(DemographicData::COMMUNITY_SIZE / nborhood_size, engine);
                }
                int age_group = -1;

                if (family_size == 1) {
                    if (il2 < 28) age_group = AgeGroups::o65;      /* single adult age 65+   */
                    else if (il2 < 51) age_group = AgeGroups::a30to49; /* age 30-49 (ASSUME 40%) */
                    else if (il2 < 68) age_group = AgeGroups::a50to64;
                    else age_group = AgeGroups::a18to29;               /* single adult age 19-29 */
                    nr_arr(i, j, k, age_group) += 1;
                } else if (family_size == 2) {
                    if (il2 == 0) {
                        /* 1% probability of one parent + one child */
                        int il3 = Random_int(100, engine);
                        if (il3 < 2) age_group = AgeGroups::o65;        /* one parent, age 65+ */
                        else if (il3 < 36) age_group = AgeGroups::a30to49;  /* one parent 30-64 (ASSUME 60%) */
                        else if (il3 < 62) age_group = AgeGroups::a50to64;
                        else age_group = AgeGroups::a18to29;                /* one parent 19-29 */
                        nr_arr(i, j, k, age_group) += 1;
                        if (((int) Random_int(100, engine)) < p_schoolage) age_group = AgeGroups::a5to17;
                        else age_group = AgeGroups::u5;
                        nr_arr(i, j, k, age_group) += 1;
                    } else {
                        /* 2 adults, 28% over 65 (ASSUME both same age group) */
                        if (il2 < 28) age_group = AgeGroups::o65;      /* single adult age 65+ */
                        else if (il2 < 51) age_group = AgeGroups::a30to49; /* age 30-64 (ASSUME 40%) */
                        else if (il2 < 68) age_group = AgeGroups::a50to64;
                        else age_group = AgeGroups::a18to29;               /* single adult age 19-29 */
                        nr_arr(i, j, k, age_group) += 2;
                    }
                }

                if (family_size > 2) {
                    /* ASSUME 2 adults, of the same age group */
                    if (il2 < 2) age_group = AgeGroups::o65;  /* parents are age 65+ */
                    else if (il2 < 36) age_group = AgeGroups::a30to49;  /* parents 30-64 (ASSUME 60%) */
                    else if (il2 < 62) age_group = AgeGroups::a50to64;
                    else age_group = AgeGroups::a18to29;  /* parents 19-29 */
                    nr_arr(i, j, k, age_group) += 2;

                    /* Now pick the children's age groups */
                    for (int nc = 2; nc < family_size; ++nc) {
                        if (((int) Random_int(100, engine)) < p_schoolage) age_group = AgeGroups::a5to17;
                        else age_group = AgeGroups::u5;
                        nr_arr(i, j, k, age_group) += 1;
                    }
                }

                agent.pos(0) = static_cast<ParticleReal>((i + 0.5_rt) * dx[0]);
                agent.pos(1) = static_cast<ParticleReal>((j + 0.5_rt) * dx[1]);
                agent.id()  = pid+ip;
                agent.cpu() = my_proc;

                for (int d = 0; d < n_disease; d++) {
                    status_ptrs[d][ip] = 0;
                    counter_ptrs[d][ip] = 0.0_prt;
                    timer_ptrs[d][ip] = 0.0_prt;
                }
                age_group_ptr[ip] = age_group;
                family_ptr[ip] = family_id_start + (ii / family_size);
                home_i_ptr[ip] = i;
                home_j_ptr[ip] = j;
                work_i_ptr[ip] = i;
                work_j_ptr[ip] = j;
                trav_i_ptr[ip] = i;
                trav_j_ptr[ip] = j;
                hosp_i_ptr[ip] = -1;
                hosp_j_ptr[ip] = -1;
                nborhood_ptr[ip] = nborhood;
                work_nborhood_ptr[ip] = nborhood;
                workgroup_ptr[ip] = 0;
                naics_ptr[ip] = 0;
                random_travel_ptr[ip] = -1;
                air_travel_ptr[ip] = -1;

                assign_school(&school_grade_ptr[ip], &school_id_ptr[ip], age_group, nborhood, engine);

                school_closed_ptr[ip] = 0;

                // Increment the appropriate student counter based on the school assignment
                if (school_id_ptr[ip] >= SchoolCensusIDType::daycare_5) {
                    Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolCensusIDType::daycare_5 - 1), 1);
                } else if (school_id_ptr[ip] > SchoolCensusIDType::none) {
                    Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, school_id_ptr[ip] - 1), 1);
                }
            }
        });

    }

    demo.CopyToHostAsync(demo.Unit_on_proc_d, demo.Unit_on_proc);
    Gpu::streamSynchronize();

    pc.comm_mf.define(comm_mf.boxArray(), comm_mf.DistributionMap(), 1, 0);
    iMultiFab::Copy(pc.comm_mf, comm_mf, 0, 0, 1, 0);
}

/*! \brief Read worker flow data from file and set work location for agents

    *  Read in worker flow (home and work) data from a given binary file:
    *  + Initialize and allocate space for the worker-flow matrix with #DemographicData::Nunit rows
    *    and columns; note that only those rows are allocated where part of the unit resides on this
    *    processor.
    *  + Read worker flow data from #ExaEpi::TestParams::workerflow_filename: it is a binary file that
    *    contains 3 x (number of work patthers) unsigned integer data. The 3 integers are: from, to,
    *    and the number of workers with this from and to. The from and to are the IDs from the
    *    first column of the census data file (#DemographicData::myID).
    *  + For each work pattern: Read in the from, to, and number. If both the from and to ID values
    *    correspond to units that are on this processor, say, i and j, then set the worker-flow
    *    matrix element at [i][j] to the number. Note that DemographicData::myIDtoUnit() maps from
    *    ID value to unit number (from -> i, to -> j).
    *  + Comvert values in each row to row-wise cumulative values.
    *  + Scale these values to account for ~2% of people of vacation/sick leave.
    *  + For each agent (particle) in each box/tile on each processor:
    *    + Get the home (from) unit of the agent from its home cell index (i,j) and the input argument
    *      unit_mf.
    *    + Compute the number of workers in the "from" unit as 58.6% of the total population. If this
    *      number if greater than zero, continue with the following steps.
    *    + Find age group of this agent, and if it is either 18-29 or 30-64, continue with the
    *      following steps.
    *    + Assign a random work destination unit by picking a random number and placing it in the
    *      row-wise cumulative numbers in the "from" row of the worker flow matrix.
    *    + If the "to" unit is same as the "from" unit, then set the work community number same as
    *      the home community number witn 25% probability and some other random community number in
    *      the same unit with 75% probability.
    *    + Set the work location indices (i,j) for the agent to the values corresponding to this
    *      computed work community.
    *    + Find the number of workgroups in the work location unit, where one workgroup consists of
    *      20 workers; then assign a random workgroup to this agent.
*/
void CensusData::read_workerflow (AgentContainer& pc,           /*!< Agent container (particle container) */
                                  const std::string &workerflow_filename,
                                  const int workgroup_size)
{
    /* Allocate worker-flow matrix, only from units with nighttime
        communities on this processor (Unit_on_proc[] flag) */
    unsigned int** flow = (unsigned int **) The_Pinned_Arena()->alloc(demo.Nunit*sizeof(unsigned int *));
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            flow[i] = (unsigned int *) The_Pinned_Arena()->alloc(demo.Nunit*sizeof(unsigned int));
            for (int j = 0; j < demo.Nunit; j++) flow[i][j] = 0;
        }
    }

    VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

    std::ifstream ifs;
    ifs.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());

    ifs.open(workerflow_filename.c_str(), std::ios::in|std::ios::binary);
    if (!ifs.good()) {
        FileOpenFailed(workerflow_filename);
    }

    const std::streamoff CURPOS = ifs.tellg();
    ifs.seekg(0,std::ios::end);
    const std::streamoff ENDPOS = ifs.tellg();
    const long num_work = (ENDPOS - CURPOS) / (3*sizeof(unsigned int));

    ifs.seekg(CURPOS, std::ios::beg);

    for (int work = 0; work < num_work; ++work) {
        unsigned int from, to, number;
        ifs.read((char*)&from, sizeof(from));
        ifs.read((char*)&to, sizeof(to));
        ifs.read((char*)&number, sizeof(number));
        if (from > 65334) {continue;}
        int i = demo.myIDtoUnit[from];
        if (demo.Unit_on_proc[i]) {
            if (to > 65334) {continue;}
            int j = demo.myIDtoUnit[to];
            if (demo.Start[j+1] != demo.Start[j]) { // if there are communities in this unit
                flow[i][j] = number;
            }
        }
    }

    /* Convert to cumulative numbers to enable random selection */
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            for (int j = 1; j < demo.Nunit; j++) {
                flow[i][j] += flow[i][j-1];
            }
        }
    }

    /* These numbers were for the true population, and do not include
        the roughly 2% of people who were on vacation or sick during the
        Census 2000 reporting week.  We need to scale the worker flow to
        the model tract residential populations, and might as well add
        the 2% back in while we're at it.... */
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i] && demo.Population[i]) {
            unsigned int number = (unsigned int) rint(((double) demo.Population[i]) / 2000.0);
            double scale = 1.02 * (2000.0 * number) / ((double) demo.Population[i]);
            for (int j = 0; j < demo.Nunit; j++) {
                flow[i][j] = (unsigned int) rint((double) flow[i][j] * scale);
            }
        }
    }

    unsigned int** d_flow = (unsigned int **) The_Device_Arena()->alloc(demo.Nunit*sizeof(unsigned int *));
    Gpu::HostVector<unsigned int*> host_vector_flow(demo.Nunit, nullptr);
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            host_vector_flow[i] = (unsigned int *) The_Device_Arena()->alloc(demo.Nunit*sizeof(unsigned int));
        }
    }

    Gpu::copy(Gpu::hostToDevice, host_vector_flow.begin(), host_vector_flow.end(), d_flow);
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            Gpu::copy(Gpu::hostToDevice, flow[i], flow[i] + demo.Nunit, host_vector_flow[i]);
        }
    }

    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            The_Pinned_Arena()->free(flow[i]);
        }
    }
    The_Pinned_Arena()->free(flow);

    const Box& domain = pc.Geom(0).Domain();

    /* This is where workplaces should be assigned */
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf); mfi.isValid(); ++mfi) {
        auto& agents_tile = pc.GetParticles(0)[std::make_pair(mfi.index(),mfi.LocalTileIndex())];
        auto& soa = agents_tile.GetStructOfArrays();
        auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
        auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
        auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
        auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
        auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
        auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
        auto work_nborhood_ptr = soa.GetIntData(IntIdx::work_nborhood).data();
        auto np = soa.numParticles();

        auto unit_arr = unit_mf[mfi].array();
        auto comm_arr = comm_mf[mfi].array();

        auto Population = demo.Population_d.data();
        auto Start = demo.Start_d.data();
        auto Ndaywork = demo.Ndaywork_d.data();
        auto Ncommunity = demo.Ncommunity;
        auto Nunit = demo.Nunit;

        ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int ip, RandomEngine const& engine) noexcept
        {
            auto from = unit_arr(home_i_ptr[ip], home_j_ptr[ip], 0);

            /* Randomly assign the eligible working-age population */
            unsigned int number = (unsigned int) rint(((Real) Population[from]) / 2000.0);
            unsigned int nwork = (unsigned int) (2000.0 * number * .586); /* 58.6% of population is working-age */
            if (nwork == 0) { return; }

            int age_group = age_group_ptr[ip];
            /* Check working-age population */
            if ((age_group == AgeGroups::a18to29) || (age_group == AgeGroups::a30to49)) {
                unsigned int irnd = Random_int(nwork, engine);
                int to = 0;
                int comm_to = 0;
                if (irnd < d_flow[from][Nunit-1]) {
                    /* Choose a random destination unit */
                    to = 0;
                    while (irnd >= d_flow[from][to]) { to++; }
                }

                /*If from=to unit, 25% EXTRA chance of working in home community*/
                if ((from == to) && (Random(engine) < 0.25)) {
                    comm_to = comm_arr(home_i_ptr[ip], home_j_ptr[ip], 0);
                } else {
                    /* Choose a random community within that destination unit */
                    comm_to = Start[to] + Random_int(Start[to+1] - Start[to], engine);
                    AMREX_ALWAYS_ASSERT(comm_to < Ncommunity);
                }

                IntVect comm_to_iv = domain.atOffset(comm_to);
                work_i_ptr[ip] = comm_to_iv[0];
                work_j_ptr[ip] = comm_to_iv[1];

                number = (unsigned int) rint( ((Real) Ndaywork[to]) /
                            ((Real) workgroup_size * (Start[to+1] - Start[to])) );

                if (number) {
                    workgroup_ptr[ip] = 1 + Random_int(number, engine);
                    work_nborhood_ptr[ip] = workgroup_ptr[ip] % 4; // each workgroup is assigned to a neighborhood as well
                }
            }
        });
    }

    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            The_Device_Arena()->free(host_vector_flow[i]);
        }
    }
    The_Device_Arena()->free(d_flow);

    assignTeachersAndWorkgroup(pc, workgroup_size);
}

void CensusData::assignTeachersAndWorkgroup (AgentContainer& pc, const int workgroup_size) {
    const Box& domain = pc.Geom(0).Domain();

    auto Ncommunity = demo.Ncommunity;
    const int num_school_types = SchoolCensusIDType::total - 1;
    Gpu::DeviceVector<int> teacher_counts_array[num_school_types];
    int* teacher_counts_ptr[num_school_types];
    for (int i = 0; i < num_school_types; i++) {
        teacher_counts_array[i].resize(Ncommunity, 0);
        teacher_counts_ptr[i] = teacher_counts_array[i].data();
    }

    auto student_teacher_ratio = pc.m_student_teacher_ratio;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf); mfi.isValid(); ++mfi) {
        auto student_counts_arr = pc.m_student_counts[mfi].array();
        auto comm_arr = comm_mf[mfi].array();
        auto bx = mfi.tilebox();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            int comm = comm_arr(i, j, k);
            for (int s = 0; s < num_school_types; s++) {
                teacher_counts_ptr[s][comm] = student_counts_arr(i, j, k, s);
            }
            teacher_counts_ptr[SchoolCensusIDType::high_1 - 1][comm] /= student_teacher_ratio[SchoolType::high];
            teacher_counts_ptr[SchoolCensusIDType::middle_2 - 1][comm] /= student_teacher_ratio[SchoolType::middle];
            teacher_counts_ptr[SchoolCensusIDType::elem_3 - 1][comm] /= student_teacher_ratio[SchoolType::elem];
            teacher_counts_ptr[SchoolCensusIDType::elem_4 - 1][comm] /= student_teacher_ratio[SchoolType::elem];
            teacher_counts_ptr[SchoolCensusIDType::daycare_5 - 1][comm] /= student_teacher_ratio[SchoolType::daycare];
        });
    }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf); mfi.isValid(); ++mfi) {
        auto& agents_tile = pc.GetParticles(0)[std::make_pair(mfi.index(),mfi.LocalTileIndex())];
        auto& soa = agents_tile.GetStructOfArrays();

        auto np = soa.numParticles();

        auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
        auto workgroup_ptr = soa.GetIntData(IntIdx::workgroup).data();
        auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
        auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
        auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
        auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
        auto work_nborhood_ptr = soa.GetIntData(IntIdx::work_nborhood).data();

        ParallelForRNG (np, [=] AMREX_GPU_DEVICE (int ip, RandomEngine const& engine) noexcept {

            int comm_to = (int)domain.index(IntVect(AMREX_D_DECL(work_i_ptr[ip], work_j_ptr[ip], 0)));
            if (comm_to >= Ncommunity) return;
            // skip non-working age
            if (age_group_ptr[ip] < AgeGroups::a18to29 || age_group_ptr[ip] > AgeGroups::a50to64) return;
            // skip non-workers
            if (workgroup_ptr[ip] == 0) return;

            int choice = Random_int(num_school_types, engine);

            for (int k = 0; k < num_school_types; k++) {
                int pos = (k + choice) % num_school_types;
                AMREX_ALWAYS_ASSERT(pos < num_school_types);
                int count = Gpu::Atomic::Add(&(teacher_counts_ptr[pos][comm_to]), -1);
                if (count > 0) {
                    // school_id of 0 is reserved for no school
                    school_id_ptr[ip] = pos + 1;
                    // workgroup is the whole school, i.e. adults interact with all other adults in the school
                    workgroup_ptr[ip] = school_id_ptr[ip];
                    // teachers are assigned the grade they teach
                    switch (school_id_ptr[ip]) {
                        case SchoolCensusIDType::high_1:
                            school_grade_ptr[ip] = 12;  // 10th grade - generic for high school
                            work_nborhood_ptr[ip] = 3; // assuming the high school is located in Neighbordhood 4
                            break;
                        case SchoolCensusIDType::middle_2:
                            school_grade_ptr[ip] = 9;  // 7th grade - generic for middle
                            work_nborhood_ptr[ip] = 1; // assuming the middle school is located in Neighbordhood 2
                            break;
                        case SchoolCensusIDType::elem_3:
                            school_grade_ptr[ip] = 5;  // 3rd grade - generic for elementary
                            work_nborhood_ptr[ip] = 0; // assuming the first elementary school is located in Neighbordhood 1
                            break;
                        case SchoolCensusIDType::elem_4:
                            school_grade_ptr[ip] = 5;  // 3rd grade - generic for elementary
                            work_nborhood_ptr[ip] = 2; // assuming the first elementary school is located in Neighbordhood 3
                            break;
                        case SchoolCensusIDType::daycare_5:
                            school_grade_ptr[ip] = 0; // generic for daycare
                            work_nborhood_ptr[ip] = Random_int(4, engine); // randomly select nborhood
                            school_id_ptr[ip] += work_nborhood_ptr[ip];
                            break;
                    }
                    return;
                }
            }
        });
        Gpu::synchronize();
    }
}

