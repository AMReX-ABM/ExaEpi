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
Geometry get_geometry (const DemographicData&    demo   /*!< demographic data */) {
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
    ba.maxSize(params.max_grid_size);
    dm.define(ba);

    Print() << "Base domain is: " << geom.Domain() << "\n";
    Print() << "Max grid size is: " << params.max_grid_size << "\n";
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
int assign_school (const int nborhood, const amrex::RandomEngine& engine) {
    int il4 = amrex::Random_int(100, engine);
    int school = 0;

    if (il4 < 36) {
        school = 3 + (nborhood / 2);  // elementary school, in neighborhoods 1 and 2
        AMREX_ALWAYS_ASSERT(school < 5);
    }
    else if (il4 < 68) {
        school = 2;  // middle school, one for all neighborhoods
    }

    else if (il4 < 93) {
        school = 1;  // high school, one for all neighborhoods
    }
    else {
        school = 0;  // not in school, presumably 18-year-olds or some home-schooled, etc
    }
    return school;
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

    auto Nunit = demo.Nunit;
    auto Ncommunity = demo.Ncommunity;
    pc.m_unit_teacher_counts_d.resize(Nunit, 0);
    /* One can decide to define a iMultifab teachercounts -- but, data locality might be challenging*/
    pc.m_comm_teacher_counts_total_d.resize(Ncommunity, 0);
    pc.m_comm_teacher_counts_high_d.resize(Ncommunity, 0);
    pc.m_comm_teacher_counts_middle_d.resize(Ncommunity, 0);
    pc.m_comm_teacher_counts_elem3_d.resize(Ncommunity, 0);
    pc.m_comm_teacher_counts_elem4_d.resize(Ncommunity, 0);
    pc.m_comm_teacher_counts_daycr_d.resize(Ncommunity, 0);

    amrex::Gpu::DeviceVector<long> student_teacher_ratios_d(pc.m_student_teacher_ratios.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, pc.m_student_teacher_ratios.begin(), pc.m_student_teacher_ratios.end(),
                     student_teacher_ratios_d.begin());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
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
        //auto N29 = demo.N29_d.data();
        //auto N64 = demo.N64_d.data();
        //auto N65plus = demo.N65plus_d.data();

        auto ratios = student_teacher_ratios_d.dataPtr();
        auto unit_teacher_counts_d_ptr = pc.m_unit_teacher_counts_d.data();
        auto comm_teacher_counts_total_d_ptr = pc.m_comm_teacher_counts_total_d.data();
        auto comm_teacher_counts_high_d_ptr = pc.m_comm_teacher_counts_high_d.data();
        auto comm_teacher_counts_middle_d_ptr = pc.m_comm_teacher_counts_middle_d.data();
        auto comm_teacher_counts_elem3_d_ptr = pc.m_comm_teacher_counts_elem3_d.data();
        auto comm_teacher_counts_elem4_d_ptr = pc.m_comm_teacher_counts_elem4_d.data();
        auto comm_teacher_counts_daycr_d_ptr = pc.m_comm_teacher_counts_daycr_d.data();

        auto bx = mfi.tilebox();
        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
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
                int il  = amrex::Random_int(1000, engine);

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
        auto school_ptr = soa.GetIntData(IntIdx::school).data();
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

        amrex::ParallelForRNG(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n, amrex::RandomEngine const& engine) noexcept
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
                int il2 = amrex::Random_int(100, engine);
                if (ii % family_size == 0) {
                    nborhood = amrex::Random_int(DemographicData::COMMUNITY_SIZE / nborhood_size, engine);
                }
                int age_group = -1;

                if (family_size == 1) {
                    if (il2 < 28) { age_group = 4; }      /* single adult age 65+   */
                    else if (il2 < 68) { age_group = 3; } /* age 30-64 (ASSUME 40%) */
                    else { age_group = 2; }               /* single adult age 19-29 */
                    nr_arr(i, j, k, age_group) += 1;
                } else if (family_size == 2) {
                    if (il2 == 0) {
                        /* 1% probability of one parent + one child */
                        int il3 = amrex::Random_int(100, engine);
                        if (il3 < 2) { age_group = 4; }        /* one parent, age 65+ */
                        else if (il3 < 62) { age_group = 3; }  /* one parent 30-64 (ASSUME 60%) */
                        else { age_group = 2; }                /* one parent 19-29 */
                        nr_arr(i, j, k, age_group) += 1;
                        if (((int) amrex::Random_int(100, engine)) < p_schoolage) {
                            age_group = 1; /* 22.0% of total population ages 5-18 */
                        } else {
                            age_group = 0;   /* 6.8% of total population ages 0-4 */
                        }
                        nr_arr(i, j, k, age_group) += 1;
                    } else {
                        /* 2 adults, 28% over 65 (ASSUME both same age group) */
                        if (il2 < 28) { age_group = 4; }      /* single adult age 65+ */
                        else if (il2 < 68) { age_group = 3; } /* age 30-64 (ASSUME 40%) */
                        else { age_group = 2; }               /* single adult age 19-29 */
                        nr_arr(i, j, k, age_group) += 2;
                    }
                }

                if (family_size > 2) {
                    /* ASSUME 2 adults, of the same age group */
                    if (il2 < 2) { age_group = 4; }  /* parents are age 65+ */
                    else if (il2 < 62) { age_group = 3; }  /* parents 30-64 (ASSUME 60%) */
                    else { age_group = 2; }  /* parents 19-29 */
                    nr_arr(i, j, k, age_group) += 2;

                    /* Now pick the children's age groups */
                    for (int nc = 2; nc < family_size; ++nc) {
                        if (((int) amrex::Random_int(100, engine)) < p_schoolage) {
                            age_group = 1; /* 22.0% of total population ages 5-18 */
                        } else {
                            age_group = 0;   /* 6.8% of total population ages 0-4 */
                        }
                        nr_arr(i, j, k, age_group) += 1;
                    }
                }

                agent.pos(0) = (i + 0.5_rt)*dx[0];
                agent.pos(1) = (j + 0.5_rt)*dx[1];
                agent.id()  = pid+ip;
                agent.cpu() = my_proc;

                for (int d = 0; d < n_disease; d++) {
                    status_ptrs[d][ip] = 0;
                    counter_ptrs[d][ip] = 0.0_rt;
                    timer_ptrs[d][ip] = 0.0_rt;
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
                random_travel_ptr[ip] = -1;
                air_travel_ptr[ip] = -1;

                if (age_group == 0) {
                    school_ptr[ip] = 5; // note - need to handle playgroups
                } else if (age_group == 1) {
                    school_ptr[ip] = assign_school(nborhood, engine);
                } else {
                    school_ptr[ip] = 0; // only use negative values to indicate school closed
                }

                // Increment the appropriate student counter based on the school assignment
                if (school_ptr[ip] == SchoolType::elem_3) {
                    amrex::Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolType::elem_3), 1);
                } else if (school_ptr[ip] == SchoolType::elem_4) {
                    amrex::Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolType::elem_4), 1);
                } else if (school_ptr[ip] == SchoolType::middle) {
                    amrex::Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolType::middle), 1);
                } else if (school_ptr[ip] == SchoolType::high) {
                    amrex::Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolType::high), 1);
                } else if (school_ptr[ip] == SchoolType::day_care) {
                    amrex::Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolType::day_care), 1);
                }

                if (school_ptr[ip]>0) {amrex::Gpu::Atomic::AddNoRet(&student_counts_arr(i, j, k, SchoolType::total), 1); }

            }
        });

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int comm = comm_arr(i,j,k);

            comm_teacher_counts_high_d_ptr[comm]   = (int)((student_counts_arr(i, j, k, SchoolType::high))     / (ratios[SchoolType::high]));
            comm_teacher_counts_middle_d_ptr[comm] = (int)((student_counts_arr(i, j, k, SchoolType::middle))   / (ratios[SchoolType::middle]));
            comm_teacher_counts_elem3_d_ptr[comm]  = (int)((student_counts_arr(i, j, k, SchoolType::elem_3))   / (ratios[SchoolType::elem_3]));
            comm_teacher_counts_elem4_d_ptr[comm]  = (int)((student_counts_arr(i, j, k, SchoolType::elem_4))   / (ratios[SchoolType::elem_4]));
            comm_teacher_counts_daycr_d_ptr[comm]  = (int)((student_counts_arr(i, j, k, SchoolType::day_care)) / (ratios[SchoolType::day_care]));

            int total = comm_teacher_counts_high_d_ptr[comm]
                      + comm_teacher_counts_middle_d_ptr[comm]
                      + comm_teacher_counts_elem3_d_ptr[comm]
                      + comm_teacher_counts_elem4_d_ptr[comm]
                      + comm_teacher_counts_daycr_d_ptr[comm];
            comm_teacher_counts_total_d_ptr[comm] = total;
            amrex::Gpu::Atomic::AddNoRet(&unit_teacher_counts_d_ptr[unit_arr(i,j,k,0)], total);
        });
    }

    demo.CopyToHostAsync(demo.Unit_on_proc_d, demo.Unit_on_proc);
    amrex::Gpu::streamSynchronize();
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
    unsigned int** flow = (unsigned int **) amrex::The_Pinned_Arena()->alloc(demo.Nunit*sizeof(unsigned int *));
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            flow[i] = (unsigned int *) amrex::The_Pinned_Arena()->alloc(demo.Nunit*sizeof(unsigned int));
            for (int j = 0; j < demo.Nunit; j++) flow[i][j] = 0;
        }
    }

    VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

    std::ifstream ifs;
    ifs.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());

    ifs.open(workerflow_filename.c_str(), std::ios::in|std::ios::binary);
    if (!ifs.good()) {
        amrex::FileOpenFailed(workerflow_filename);
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

    unsigned int** d_flow = (unsigned int **) amrex::The_Device_Arena()->alloc(demo.Nunit*sizeof(unsigned int *));
    amrex::Gpu::HostVector<unsigned int*> host_vector_flow(demo.Nunit, nullptr);
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            host_vector_flow[i] = (unsigned int *) amrex::The_Device_Arena()->alloc(demo.Nunit*sizeof(unsigned int));
        }
    }

    amrex::Gpu::copy(Gpu::hostToDevice, host_vector_flow.begin(), host_vector_flow.end(), d_flow);
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            amrex::Gpu::copy(Gpu::hostToDevice, flow[i], flow[i] + demo.Nunit, host_vector_flow[i]);
        }
    }

    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            amrex::The_Pinned_Arena()->free(flow[i]);
        }
    }
    amrex::The_Pinned_Arena()->free(flow);

    const Box& domain = pc.Geom(0).Domain();

    /* This is where workplaces should be assigned */
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
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

        amrex::ParallelForRNG( np,
            [=] AMREX_GPU_DEVICE (int ip, RandomEngine const& engine) noexcept
        {
            auto from = unit_arr(home_i_ptr[ip], home_j_ptr[ip], 0);

            /* Randomly assign the eligible working-age population */
            unsigned int number = (unsigned int) rint(((Real) Population[from]) / 2000.0);
            unsigned int nwork = (unsigned int) (2000.0 * number * .586); /* 58.6% of population is working-age */
            if (nwork == 0) { return; }

            int age_group = age_group_ptr[ip];
            /* Check working-age population */
            if ((age_group == 2) || (age_group == 3)) {
                unsigned int irnd = amrex::Random_int(nwork, engine);
                int to = 0;
                int comm_to = 0;
                if (irnd < d_flow[from][Nunit-1]) {
                    /* Choose a random destination unit */
                    to = 0;
                    while (irnd >= d_flow[from][to]) { to++; }
                }

                /*If from=to unit, 25% EXTRA chance of working in home community*/
                if ((from == to) && (amrex::Random(engine) < 0.25)) {
                    comm_to = comm_arr(home_i_ptr[ip], home_j_ptr[ip], 0);
                } else {
                    /* Choose a random community within that destination unit */
                    comm_to = Start[to] + amrex::Random_int(Start[to+1] - Start[to], engine);
                    AMREX_ALWAYS_ASSERT(comm_to < Ncommunity);
                }

                IntVect comm_to_iv = domain.atOffset(comm_to);
                work_i_ptr[ip] = comm_to_iv[0];
                work_j_ptr[ip] = comm_to_iv[1];

                number = (unsigned int) rint( ((Real) Ndaywork[to]) /
                            ((Real) workgroup_size * (Start[to+1] - Start[to])) );

                if (number) {
                    workgroup_ptr[ip] = 1 + amrex::Random_int(number, engine);
                    work_nborhood_ptr[ip] = workgroup_ptr[ip] % 4; // each workgroup is assigned to a neighborhood as well
                }
            }
        });
    }

    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            amrex::The_Device_Arena()->free(host_vector_flow[i]);
        }
    }
    amrex::The_Device_Arena()->free(d_flow);

    assignTeachersAndWorkgroup(pc, workgroup_size);
}

void CensusData::assignTeachersAndWorkgroup (AgentContainer& pc       /*!< Agent container (particle container) */,
                                             const int workgroup_size)

{
    const Box& domain = pc.Geom(0).Domain();

    auto total_teacher_unit_d = pc.getUnitTeacherCounts();
    amrex::Gpu::HostVector<int> total_teacher_unit(total_teacher_unit_d.size());
    amrex::Gpu::copy(Gpu::deviceToHost, total_teacher_unit_d.begin(),
                     total_teacher_unit_d.end(), total_teacher_unit.begin());

    auto total_teacher_counts_d = pc.getCommTeacherCounts();
    amrex::Gpu::HostVector<int> total_teacher_counts(total_teacher_counts_d.size());
    amrex::Gpu::copy(Gpu::deviceToHost, total_teacher_counts_d.begin(),
                     total_teacher_counts_d.end(), total_teacher_counts.begin());
    amrex::Gpu::HostVector<int> total_teacher_counts_mod(total_teacher_counts.size(),0);
    auto total_teacher_counts_ptr = total_teacher_counts.data();

    auto daycr_teacher_counts_d = pc.getCommDayCrTeacherCounts();
    amrex::Gpu::HostVector<int> daycr_teacher_counts(daycr_teacher_counts_d.size());
    amrex::Gpu::copy(Gpu::deviceToHost, daycr_teacher_counts_d.begin(),
                     daycr_teacher_counts_d.end(), daycr_teacher_counts.begin());
    amrex::Gpu::HostVector<int> daycr_teacher_counts_mod(daycr_teacher_counts.size(),0);
    auto daycr_teacher_counts_ptr = daycr_teacher_counts_mod.data();

    auto high_teacher_counts_d = pc.getCommHighTeacherCounts();
    amrex::Gpu::HostVector<int> high_teacher_counts(high_teacher_counts_d.size());
    amrex::Gpu::copy(Gpu::deviceToHost, high_teacher_counts_d.begin(),
                     high_teacher_counts_d.end(), high_teacher_counts.begin());
    amrex::Gpu::HostVector<int> high_teacher_counts_mod(high_teacher_counts.size(),0);
    auto high_teacher_counts_ptr = high_teacher_counts_mod.data();

    auto middle_teacher_counts_d = pc.getCommMiddleTeacherCounts();
    amrex::Gpu::HostVector<int> middle_teacher_counts(middle_teacher_counts_d.size());
    amrex::Gpu::copy(Gpu::deviceToHost, middle_teacher_counts_d.begin(),
                     middle_teacher_counts_d.end(), middle_teacher_counts.begin());
    amrex::Gpu::HostVector<int> middle_teacher_counts_mod(middle_teacher_counts.size(),0);
    auto middle_teacher_counts_ptr = middle_teacher_counts_mod.data();

    auto elem3_teacher_counts_d = pc.getCommElem3TeacherCounts();
    amrex::Gpu::HostVector<int> elem3_teacher_counts(elem3_teacher_counts_d.size());
    amrex::Gpu::copy(Gpu::deviceToHost, elem3_teacher_counts_d.begin(),
                     elem3_teacher_counts_d.end(), elem3_teacher_counts.begin());
    amrex::Gpu::HostVector<int> elem3_teacher_counts_mod(elem3_teacher_counts.size(),0);
    auto elem3_teacher_counts_ptr = elem3_teacher_counts_mod.data();

    auto elem4_teacher_counts_d = pc.getCommElem4TeacherCounts();
    amrex::Gpu::HostVector<int> elem4_teacher_counts(elem4_teacher_counts_d.size());
    amrex::Gpu::copy(Gpu::deviceToHost, elem4_teacher_counts_d.begin(),
                     elem4_teacher_counts_d.end(), elem4_teacher_counts.begin());
    amrex::Gpu::HostVector<int> elem4_teacher_counts_mod(elem4_teacher_counts.size(),0);
    auto elem4_teacher_counts_ptr = elem4_teacher_counts_mod.data();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto& agents_tile = pc.GetParticles(0)[std::make_pair(mfi.index(),mfi.LocalTileIndex())];
        auto& soa = agents_tile.GetStructOfArrays();

        auto Ndaywork = demo.Ndaywork.data();
        auto Start = demo.Start.data();
        auto Ncommunity = demo.Ncommunity;

        auto np = soa.numParticles();

        amrex::Gpu::PinnedVector<int> age_group_h(np);
        amrex::Gpu::PinnedVector<int> workgroup_h(np);
        amrex::Gpu::PinnedVector<int> work_i_h(np);
        amrex::Gpu::PinnedVector<int> work_j_h(np);
        amrex::Gpu::PinnedVector<int> school_h(np);
        amrex::Gpu::PinnedVector<int> work_nborhood_h(np);

        amrex::Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::age_group).begin(),
                         soa.GetIntData(IntIdx::age_group).end(), age_group_h.begin());
        amrex::Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::workgroup).begin(),
                         soa.GetIntData(IntIdx::workgroup).end(), workgroup_h.begin());
        amrex::Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::work_i).begin(),
                         soa.GetIntData(IntIdx::work_i).end(), work_i_h.begin());
        amrex::Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::work_j).begin(),
                         soa.GetIntData(IntIdx::work_j).end(), work_j_h.begin());
        amrex::Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::school).begin(),
                         soa.GetIntData(IntIdx::school).end(), school_h.begin());
        amrex::Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::work_nborhood).begin(),
                         soa.GetIntData(IntIdx::work_nborhood).end(), work_nborhood_h.begin());

        auto age_group_ptr = age_group_h.data();
        auto workgroup_ptr = workgroup_h.data();
        auto work_i_ptr = work_i_h.data();
        auto work_j_ptr = work_j_h.data();
        auto school_ptr = school_h.data();
        auto work_nborhood_ptr = work_nborhood_h.data();

        for (int ip = 0; ip < np; ++ip) {

            int comm_to = (int) domain.index(IntVect(AMREX_D_DECL(work_i_ptr[ip],work_j_ptr[ip],0)));
            if (comm_to >= Ncommunity) {
                continue;
            }
            int to = 0;
            while (comm_to >= Start[to+1]) { to++; }

            if (total_teacher_unit.data()[to] && (age_group_ptr[ip] == 2 || age_group_ptr[ip] == 3) && workgroup_ptr[ip] > 0) {
                int elem3_teacher  = elem3_teacher_counts_ptr[comm_to];
                int elem4_teacher  = elem4_teacher_counts_ptr[comm_to];
                int middle_teacher = middle_teacher_counts_ptr[comm_to];
                int high_teacher   = high_teacher_counts_ptr[comm_to];
                int daycr_teacher  = daycr_teacher_counts_ptr[comm_to];
                int total          = total_teacher_counts_ptr[comm_to];

                // 50% chance of being a teacher if in working-age population (until max_teacher_numb is met)
                if ((elem3_teacher + elem4_teacher + middle_teacher + high_teacher + daycr_teacher) < total) {
                    int available_slots[5] = {
                        elem3_teacher  < elem3_teacher_counts.data()[comm_to],
                        elem4_teacher  < elem4_teacher_counts.data()[comm_to],
                        middle_teacher < middle_teacher_counts.data()[comm_to],
                        high_teacher   < high_teacher_counts.data()[comm_to],
                        daycr_teacher  < daycr_teacher_counts.data()[comm_to]
                    };

                    int total_available = available_slots[0] + available_slots[1] + available_slots[2] + available_slots[3] +
                                          available_slots[4];
                    if (total_available > 0) {
                        int choice = amrex::Random_int(total_available);
                        if (choice < available_slots[0]) {
                            school_ptr[ip] = 3;  // elementary school for kids in Neighbordhood 1 & 2
                            workgroup_ptr[ip] = 3 ;
                            work_nborhood_ptr[ip] = 1; // assuming the first elementary school is located in Neighbordhood 1
                            elem3_teacher_counts_ptr[comm_to]++;
                        } else if (choice < available_slots[0] + available_slots[1]) {
                            school_ptr[ip] = 4;  // elementary school for kids in Neighbordhood 3 & 4
                            workgroup_ptr[ip] = 4 ;
                            work_nborhood_ptr[ip] = 3; // assuming the first elementary school is located in Neighbordhood 3
                            elem4_teacher_counts_ptr[comm_to]++;
                        } else if (choice < available_slots[0] + available_slots[1] + available_slots[2]) {
                            school_ptr[ip] = 2;  // middle school for kids in all Neighbordhoods (1 through 4)
                            workgroup_ptr[ip] = 2 ;
                            work_nborhood_ptr[ip] = 3; // assuming the middle school is located in Neighbordhood 2
                            middle_teacher_counts_ptr[comm_to]++;
                        } else if (choice < available_slots[0] + available_slots[1] + available_slots[2] + available_slots[3]) {
                            school_ptr[ip] = 1;  // high school for kids in all Neighbordhoods (1 through 4)
                            workgroup_ptr[ip] = 1 ;
                            work_nborhood_ptr[ip] = 4; // assuming the high school is located in Neighbordhood 4
                            high_teacher_counts_ptr[comm_to]++;
                        } else if (choice < total_available) {
                            school_ptr[ip] = 5;  // day care
                            workgroup_ptr[ip] = 5 ;
                            work_nborhood_ptr[ip] = 1; // deal with daycare/playgroups later
                            daycr_teacher_counts_ptr[comm_to]++;
                        }
                    }
                } else {
                    AMREX_ALWAYS_ASSERT(Ndaywork[to] > total_teacher_unit.data()[to]);
                    unsigned int number = (unsigned int) rint( ((Real) Ndaywork[to] - total_teacher_unit.data()[to] ) /
                                ((Real) workgroup_size * (Start[to+1] - Start[to])) );

                    if (number) {
                        workgroup_ptr[ip] = 6 + amrex::Random_int(number);
                        work_nborhood_ptr[ip] = workgroup_ptr[ip] % 4; // each workgroup is assigned to a neighborhood as well
                    }
                }
            }
        }

        amrex::Gpu::copy(Gpu::hostToDevice, school_h.begin(), school_h.end(),
                         soa.GetIntData(IntIdx::school).begin());
        amrex::Gpu::copy(Gpu::hostToDevice, workgroup_h.begin(), workgroup_h.end(),
                         soa.GetIntData(IntIdx::workgroup).begin());
        amrex::Gpu::copy(Gpu::hostToDevice, work_nborhood_h.begin(), work_nborhood_h.end(),
                         soa.GetIntData(IntIdx::work_nborhood).begin());
    }
}

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
int infect_random_community (AgentContainer& pc, /*!< Agent container (particle container)*/
                             CensusData &censusData,
                             std::map<std::pair<int, int>,
                             DenseBins<AgentContainer::ParticleType> >& bin_map, /*!< Map of dense bins with agents */
                             int unit, /*!< Unit number to infect */
                             const int d_idx, /*!< Disease index */
                             int ninfect, /*!< Target number of agents to infect */
                             const bool fast_bin /*!< Use GPU binning - fast but non-deterministic */  ) {

    auto &demo = censusData.demo;
    // chose random community
    int ncomms = demo.Ncommunity;
    int comm_offset = 0;
    if (unit > 0) {
        ncomms = demo.Start[unit+1] - demo.Start[unit];
        comm_offset = demo.Start[unit];
    }

    int random_comm = -1;
    if (ParallelDescriptor::IOProcessor()) {
        random_comm = amrex::Random_int(ncomms) + comm_offset;
    }
    ParallelDescriptor::Bcast(&random_comm, 1);

    const Geometry& geom = pc.Geom(0);
    IntVect bin_size = {AMREX_D_DECL(1, 1, 1)};
    const auto dxi = geom.InvCellSizeArray();
    const auto plo = geom.ProbLoArray();
    const auto domain = geom.Domain();

    int num_infected = 0;
    for (MFIter mfi(censusData.unit_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        amrex::DenseBins<AgentContainer::ParticleType>& bins = bin_map[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto& agents_tile = pc.GetParticles(0)[std::make_pair(mfi.index(),mfi.LocalTileIndex())];
        auto& aos = agents_tile.GetArrayOfStructs();
        auto& soa = agents_tile.GetStructOfArrays();
        const size_t np = aos.numParticles();
        if (np == 0) { continue; }
        auto pstruct_ptr = aos().dataPtr();
        const Box& box = mfi.validbox();

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

        //auto unit_arr = pc.m_unit_mf[mfi].array();
        auto comm_arr = censusData.comm_mf[mfi].array();
        auto bx = mfi.tilebox();

        const auto* lparm = pc.getDiseaseParameters_d(d_idx);

        Gpu::DeviceScalar<int> num_infected_d(num_infected);
        int* num_infected_p = num_infected_d.dataPtr();
        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            int community = comm_arr(i, j, k);
            if (community != random_comm) { return; }

            Box tbx;
            int i_cell = getTileIndex({AMREX_D_DECL(i, j, k)}, box, true, bin_size, tbx);
            auto cell_start = offsets[i_cell];
            auto cell_stop  = offsets[i_cell+1];
            int num_this_community = cell_stop - cell_start;
            AMREX_ASSERT(num_this_community > 0);
            //AMREX_ASSERT(cell_stop < np);

            if (num_this_community == 0) { return;}

            int ntry = 0;
            int ni = 0;
            /*unsigned*/ int stop = std::min(cell_start + ninfect, cell_stop);
            for (/*unsigned*/ int ip = cell_start; ip < stop; ++ip) {
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
                    status_ptr[pindex] = Status::infected;
                    counter_ptr[pindex] = 0;
                    latent_period_ptr[pindex] = amrex::RandomNormal(lparm->latent_length_mean, lparm->latent_length_std, engine);
                    infectious_period_ptr[pindex] = amrex::RandomNormal(lparm->infectious_length_mean, lparm->infectious_length_std, engine);
                    incubation_period_ptr[pindex] = amrex::RandomNormal(lparm->incubation_length_mean, lparm->incubation_length_std, engine);
                    ++ni;
                }
            }
            *num_infected_p = ni;
        });

        Gpu::Device::streamSynchronize();
        num_infected += num_infected_d.dataValue();
        if (num_infected >= ninfect) {
            break;
        }
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
        FIPS code. See #ExaEpi::Initialization::infect_random_community().
*/
void CensusData::setInitialCasesFromFile (AgentContainer& pc, /*!< Agent container (particle container) */
                                          const std::vector<CaseData>& cases, /*!< Case data */
                                          const std::vector<std::string>& d_names, /*!< Disease names */
                                          const bool fast_bin)
{
    BL_PROFILE("setInitialCasesFromFile");

    std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType> > bin_map;

    for (size_t d = 0; d < cases.size(); d++) {
        amrex::Print() << "Initializing infections for " << d_names[d] << "\n";
        int ntry = 5;
        int ninf = 0;
        for (int ihub = 0; ihub < cases[d].N_hubs; ++ihub) {
            if (cases[d].Size_hubs[ihub] > 0) {
                int FIPS = cases[d].FIPS_hubs[ihub];
                std::vector<int> units;
                units.resize(0);
                for (int i = 0; i < demo.Nunit; ++i) if(demo.FIPS[i]==FIPS)units.push_back(i);
                //int unit = FIPS_code_to_i[FIPS];
                if (units.size() > 0) {
                    amrex::Print() << "    Attempting to infect: " << cases[d].Size_hubs[ihub] << " people in FIPS " << FIPS << "... ";
                    int u=0;
                    int i=0;
                    while (i < cases[d].Size_hubs[ihub]) {
                        int nSuccesses = infect_random_community(pc, *this, bin_map, units[u], d, ntry, fast_bin);
                        ninf += nSuccesses;
                        i+= nSuccesses;
                        u=(u+1)%units.size(); //sometimes we infect fewer than ntry, but switch to next unit anyway
                    }
                    amrex::Print() << "infected " << i<< " (total " << ninf << ") after processing. \n";
                }
            }
        }
        amrex::ignore_unused(ninf);
    }
}

void CensusData::setInitialCasesRandom (AgentContainer& pc, /*!< Agent container (particle container) */
                                        std::vector<int> num_cases, /*!< Number of initial cases */
                                        const std::vector<std::string>& d_names, /*!< Disease names */
                                        const bool fast_bin)
{
    BL_PROFILE("setInitialCasesRandom");

    std::map<std::pair<int, int>, amrex::DenseBins<AgentContainer::ParticleType> > bin_map;

    for (size_t d = 0; d < num_cases.size(); d++) {
        amrex::Print() << "Initializing infections for " << d_names[d] << "\n";

        int ninf = 0;
        for (int ihub = 0; ihub < num_cases[d]; ++ihub) {
            int i = 0;
            while (i < 1) {
                int nSuccesses = infect_random_community(pc, *this, bin_map, -1, d, 1, fast_bin);
                ninf += nSuccesses;
                i+= nSuccesses;
            }
        }
        amrex::ignore_unused(ninf);
    }
}




