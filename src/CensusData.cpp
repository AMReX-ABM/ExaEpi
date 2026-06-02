/*! @file CensusData.cpp
 */

#include <AMReX_ParticleUtil.H>

#include "CensusData.H"

using namespace amrex;
using namespace ExaEpi;

void CensusData::init (ExaEpi::TestParams& params, Geometry& geom, BoxArray& ba, DistributionMapping& dm) {

    demo.initFromFile(params.census_filename, params.workgroup_size);

    geom = getGeometry(demo.Ncommunity);

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
void assignSchool (int* school_grade, int* school_id, const int age_group, const int nborhood, const RandomEngine& engine) {
    if (age_group == AgeGroups::u5) {
        // under 5
        // assume 50% in daycare
        if (Random_int(100, engine) < 50) {
            *school_grade = schoolTypeToGrade(SchoolType::daycare);
            *school_id = SchoolCensusIDType::daycare_5 + nborhood; // one daycare per nborhood
        } else {
            *school_grade = -1;
            *school_id = SchoolCensusIDType::none;                 // no school
        }
    } else if (age_group == AgeGroups::a5to17) {
        // 5 to 17
        int il4 = Random_int(100, engine);
        if (il4 < 36) {
            *school_id = SchoolCensusIDType::elem_3 + (nborhood / 2); // elementary school, in neighborhood 1&2 or 3&4
            *school_grade = schoolTypeToGrade(SchoolType::elem);
            AMREX_ALWAYS_ASSERT(*school_id < 5);
        } else if (il4 < 68) {
            *school_id = SchoolCensusIDType::middle_2; // middle school, one for all neighborhoods
            *school_grade = schoolTypeToGrade(SchoolType::middle);
        } else if (il4 < 93) {
            *school_id = SchoolCensusIDType::high_1;   // high school, one for all neighborhoods
            *school_grade = schoolTypeToGrade(SchoolType::high);
        } else {
            *school_id = SchoolCensusIDType::none;     // not in school, presumably 18-year-olds or some home-schooled, etc
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
 *        ExaEpi::readWorkerflow().
 *      + Set neighborhood and work neighborhood values. Actual work neighborhood is set
 *        in ExaEpi::readWorkerflow().
 *      + Initialize workgroup to 0. It is set in ExaEpi::readWorkerflow().
 *      + If age group is 5-17, assign a school based on neighborhood (#assignSchool).
 *  + Copy everything to GPU device.
*/
void CensusData::initAgents (AgentContainer& pc, /*!< Agents */
                             const int nborhood_size /*!< Size of neighborhood */) {
    BL_PROFILE("CensusData::initAgents");

    const Box& domain = pc.Geom(0).Domain();

    auto& ba = num_residents_mf.boxArray();
    auto& dm = num_residents_mf.DistributionMap();

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

        auto N5 = demo.N5_d.data();
        auto N17 = demo.N17_d.data();

        auto bx = mfi.tilebox();
        ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& engine) noexcept {
            int community = (int)domain.index(IntVect(AMREX_D_DECL(i, j, k)));
            if (community >= Ncommunity) { return; }
            comm_arr(i, j, k) = community;

            int unit = 0;
            while (community >= Start[unit + 1]) {
                unit++;
            }
            unit_on_proc[unit] = 1;
            unit_arr(i, j, k) = unit;
            FIPS_arr(i, j, k, 0) = FIPS[unit];
            FIPS_arr(i, j, k, 1) = Tract[unit];

            int community_size;
            if (Population[unit] < (1000 + DemographicData::COMMUNITY_SIZE * (community - Start[unit]))) {
                community_size = 0;                               /* Don't set up any residents; workgroup-only */
            } else {
                community_size = DemographicData::COMMUNITY_SIZE; /* Standard 2000-person community */
            }

            int p_hh[7] = {330, 670, 800, 900, 970, 990, 1000};
            int num_hh = H1[unit] + H2[unit] + H3[unit] + H4[unit] + H5[unit] + H6[unit] + H7[unit];
            if (num_hh) {
                p_hh[0] = 1000 * H1[unit] / num_hh;
                p_hh[1] = 1000 * (H1[unit] + H2[unit]) / num_hh;
                p_hh[2] = 1000 * (H1[unit] + H2[unit] + H3[unit]) / num_hh;
                p_hh[3] = 1000 * (H1[unit] + H2[unit] + H3[unit] + H4[unit]) / num_hh;
                p_hh[4] = 1000 * (H1[unit] + H2[unit] + H3[unit] + H4[unit] + H5[unit]) / num_hh;
                p_hh[5] = 1000 * (H1[unit] + H2[unit] + H3[unit] + H4[unit] + H5[unit] + H6[unit]) / num_hh;
                p_hh[6] = 1000;
            }

            int npeople = 0;
            while (npeople < community_size + 1) {
                int il = Random_int(1000, engine);

                int family_size = 1;
                while (il > p_hh[family_size - 1]) {
                    ++family_size;
                }
                AMREX_ASSERT(family_size > 0);
                AMREX_ASSERT(family_size <= 7);

                nf_arr(i, j, k, family_size - 1) += 1;
                npeople += family_size;
            }

            AMREX_ASSERT(npeople == nf_arr(i, j, k, 0) + 2 * nf_arr(i, j, k, 1) + 3 * nf_arr(i, j, k, 2) +
                                            4 * nf_arr(i, j, k, 3) + 5 * nf_arr(i, j, k, 4) + 6 * nf_arr(i, j, k, 5) +
                                            7 * nf_arr(i, j, k, 6));

            nr_arr(i, j, k, 5) = npeople;
        });

        int nagents;
        int ncomp = num_families[mfi].nComp();
        int ncell = num_families[mfi].numPts();
        {
            BL_PROFILE("setPopulationCounts_prefixsum")
            const int* in = num_families[mfi].dataPtr();
            int* out = fam_offsets[mfi].dataPtr();
            nagents = Scan::PrefixSum<int>(
                    ncomp * ncell,
                    [=] AMREX_GPU_DEVICE (int i) -> int {
                        int comp = i / ncell;
                        return (comp + 1) * in[i];
                    },
                    [=] AMREX_GPU_DEVICE (int i, int const& x) {
                        out[i] = x;
                    },
                    Scan::Type::exclusive, Scan::retSum);
        }
        {
            BL_PROFILE("setFamily_id_prefixsum")
            const int* in = num_families[mfi].dataPtr();
            int* out = fam_id[mfi].dataPtr();
            Scan::PrefixSum<int>(
                    ncomp * ncell,
                    [=] AMREX_GPU_DEVICE (int i) -> int {
                        return in[i];
                    },
                    [=] AMREX_GPU_DEVICE (int i, int const& x) {
                        out[i] = x;
                    },
                    Scan::Type::exclusive, Scan::retSum);
        }

        auto offset_arr = fam_offsets[mfi].array();
        auto fam_id_arr = fam_id[mfi].array();
        auto student_counts_arr = pc.m_student_counts[mfi].array();
        auto& agents_tile = pc.DefineAndReturnParticleTile(0, mfi);
        agents_tile.resize(nagents);
        auto ptd = agents_tile.getParticleTileData();

        auto dx = pc.ParticleGeom(0).CellSizeArray();
        auto my_proc = ParallelDescriptor::MyProc();
        int n_disease = pc.m_num_diseases;

        Long pid;
#ifdef AMREX_USE_OMP
#pragma omp critical(init_agents_nextid)
#endif
        {
            pid = AgentContainer::ParticleType::NextID();
            AgentContainer::ParticleType::NextID(pid + nagents);
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<Long>(pid + nagents) < LastParticleID,
                                         "Error: overflow on agent id numbers!");

        ParallelForRNG(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n, RandomEngine const& engine) noexcept {
            int nf = nf_arr(i, j, k, n);
            if (nf == 0) { return; }

            int unit = unit_arr(i, j, k);
            int community = comm_arr(i, j, k);
            int family_id_start = fam_id_arr(i, j, k, n);
            int family_size = n + 1;

            int community_size;
            if (Population[unit] < (1000 + DemographicData::COMMUNITY_SIZE * (community - Start[unit]))) {
                community_size = 0;                               /* Don't set up any residents; workgroup-only */
            } else {
                community_size = DemographicData::COMMUNITY_SIZE; /* Standard 2000-person community */
            }

            int p_schoolage = 0;
            if (community_size) { // Only bother for residential communities
                if (N5[unit] + N17[unit]) {
                    p_schoolage = 100 * N17[unit] / (N5[unit] + N17[unit]);
                } else {
                    p_schoolage = 76;
                }
            }

            int start = offset_arr(i, j, k, n);
            int nborhood = 0;
            for (int ifam = 0; ifam < nf; ++ifam) {
                int family = family_id_start + ifam;
                int ii = family_size * ifam;
                int ip = start + ii;

                int il2 = Random_int(100, engine);
                nborhood = Random_int(DemographicData::COMMUNITY_SIZE / nborhood_size, engine);
                int age_group = -1;

                if (family_size == 1) {
                    if (il2 < 28) {
                        age_group = AgeGroups::o65;     /* single adult age 65+   */
                    } else if (il2 < 51) {
                        age_group = AgeGroups::a30to49; /* age 30-49 (ASSUME 40%) */
                    } else if (il2 < 68) {
                        age_group = AgeGroups::a50to64;
                    } else {
                        age_group = AgeGroups::a18to29; /* single adult age 19-29 */
                    }
                    setAgentDataAndAssignSchool(ptd, ip, i, j, k, dx, pid + ip, my_proc, age_group, family, nborhood, n_disease,
                                                nr_arr, student_counts_arr, engine);
                } else if (family_size == 2) {
                    if (il2 == 0) {
                        /* 1% probability of one parent + one child */
                        int il3 = Random_int(100, engine);
                        if (il3 < 2) {
                            age_group = AgeGroups::o65;     /* one parent, age 65+ */
                        } else if (il3 < 36) {
                            age_group = AgeGroups::a30to49; /* one parent 30-64 (ASSUME 60%) */
                        } else if (il3 < 62) {
                            age_group = AgeGroups::a50to64;
                        } else {
                            age_group = AgeGroups::a18to29; /* one parent 19-29 */
                        }
                        setAgentDataAndAssignSchool(ptd, ip, i, j, k, dx, pid + ip, my_proc, age_group, family, nborhood,
                                                    n_disease, nr_arr, student_counts_arr, engine);
                        if (((int)Random_int(100, engine)) < p_schoolage) {
                            age_group = AgeGroups::a5to17;
                        } else {
                            age_group = AgeGroups::u5;
                        }
                        setAgentDataAndAssignSchool(ptd, ip + 1, i, j, k, dx, pid + ip + 1, my_proc, age_group, family, nborhood,
                                                    n_disease, nr_arr, student_counts_arr, engine);
                    } else {
                        /* 2 adults, 28% over 65 (ASSUME both same age group) */
                        if (il2 < 28) {
                            age_group = AgeGroups::o65;     /* single adult age 65+ */
                        } else if (il2 < 51) {
                            age_group = AgeGroups::a30to49; /* age 30-64 (ASSUME 40%) */
                        } else if (il2 < 68) {
                            age_group = AgeGroups::a50to64;
                        } else {
                            age_group = AgeGroups::a18to29; /* single adult age 19-29 */
                        }
                        setAgentDataAndAssignSchool(ptd, ip, i, j, k, dx, pid + ip, my_proc, age_group, family, nborhood,
                                                    n_disease, nr_arr, student_counts_arr, engine);
                        setAgentDataAndAssignSchool(ptd, ip + 1, i, j, k, dx, pid + ip + 1, my_proc, age_group, family, nborhood,
                                                    n_disease, nr_arr, student_counts_arr, engine);
                    }
                }

                if (family_size > 2) {
                    /* ASSUME 2 adults, of the same age group */
                    if (il2 < 2) {
                        age_group = AgeGroups::o65;     /* parents are age 65+ */
                    } else if (il2 < 36) {
                        age_group = AgeGroups::a30to49; /* parents 30-64 (ASSUME 60%) */
                    } else if (il2 < 62) {
                        age_group = AgeGroups::a50to64;
                    } else {
                        age_group = AgeGroups::a18to29; /* parents 19-29 */
                    }
                    setAgentDataAndAssignSchool(ptd, ip, i, j, k, dx, pid + ip, my_proc, age_group, family, nborhood, n_disease,
                                                nr_arr, student_counts_arr, engine);
                    setAgentDataAndAssignSchool(ptd, ip + 1, i, j, k, dx, pid + ip + 1, my_proc, age_group, family, nborhood,
                                                n_disease, nr_arr, student_counts_arr, engine);

                    /* Now pick the children's age groups */
                    for (int nc = 2; nc < family_size; ++nc) {
                        if (((int)Random_int(100, engine)) < p_schoolage) {
                            age_group = AgeGroups::a5to17;
                        } else {
                            age_group = AgeGroups::u5;
                        }
                        setAgentDataAndAssignSchool(ptd, ip + nc, i, j, k, dx, pid + ip + nc, my_proc, age_group, family,
                                                    nborhood, n_disease, nr_arr, student_counts_arr, engine);
                    }
                }
            }
        });
    }

    demo.copyToHostAsync(demo.Unit_on_proc_d, demo.Unit_on_proc);
    demo.copyToHostAsync(demo.FIPS_d, demo.FIPS);
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
void CensusData::readWorkerflow (AgentContainer& pc, /*!< Agent container (particle container) */
                                 const std::string& workerflow_filename, const int workgroup_size) {
    /* Allocate worker-flow matrix, only from units with nighttime
        communities on this processor (Unit_on_proc[] flag) */
    unsigned int** flow = (unsigned int**)The_Pinned_Arena()->alloc(demo.Nunit * sizeof(unsigned int*));
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            flow[i] = (unsigned int*)The_Pinned_Arena()->alloc(demo.Nunit * sizeof(unsigned int));
            for (int j = 0; j < demo.Nunit; j++) {
                flow[i][j] = 0;
            }
        }
    }

    VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

    std::ifstream ifs;
    ifs.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());

    ifs.open(workerflow_filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.good()) { FileOpenFailed(workerflow_filename); }

    const std::streamoff CURPOS = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    const std::streamoff ENDPOS = ifs.tellg();
    const long num_work = (ENDPOS - CURPOS) / (3 * sizeof(unsigned int));

    ifs.seekg(CURPOS, std::ios::beg);

    for (int work = 0; work < num_work; ++work) {
        unsigned int from, to, number;
        ifs.read((char*)&from, sizeof(from));
        ifs.read((char*)&to, sizeof(to));
        ifs.read((char*)&number, sizeof(number));
        if (from > 65334) { continue; }
        int i = demo.myIDtoUnit[from];
        if (demo.Unit_on_proc[i]) {
            if (to > 65334) { continue; }
            int j = demo.myIDtoUnit[to];
            if (demo.Start[j + 1] != demo.Start[j]) { // if there are communities in this unit
                flow[i][j] = number;
            }
        }
    }

    /* Convert to cumulative numbers to enable random selection */
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            for (int j = 1; j < demo.Nunit; j++) {
                flow[i][j] += flow[i][j - 1];
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
            unsigned int number = (unsigned int)rint(((double)demo.Population[i]) / 2000.0);
            double scale = 1.02 * (2000.0 * number) / ((double)demo.Population[i]);
            for (int j = 0; j < demo.Nunit; j++) {
                flow[i][j] = (unsigned int)rint((double)flow[i][j] * scale);
            }
        }
    }

    unsigned int** d_flow = (unsigned int**)The_Device_Arena()->alloc(demo.Nunit * sizeof(unsigned int*));
    Gpu::HostVector<unsigned int*> host_vector_flow(demo.Nunit, nullptr);
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) {
            host_vector_flow[i] = (unsigned int*)The_Device_Arena()->alloc(demo.Nunit * sizeof(unsigned int));
        }
    }

    Gpu::copy(Gpu::hostToDevice, host_vector_flow.begin(), host_vector_flow.end(), d_flow);
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) { Gpu::copy(Gpu::hostToDevice, flow[i], flow[i] + demo.Nunit, host_vector_flow[i]); }
    }

    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) { The_Pinned_Arena()->free(flow[i]); }
    }
    The_Pinned_Arena()->free(flow);

    const Box& domain = pc.Geom(0).Domain();

    /* This is where workplaces should be assigned */
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf); mfi.isValid(); ++mfi) {
        auto& agents_tile = pc.GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto& soa = agents_tile.GetStructOfArrays();
        auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();
        auto home_i_ptr = soa.GetIntData(IntIdx::home_i).data();
        auto home_j_ptr = soa.GetIntData(IntIdx::home_j).data();
        auto work_i_ptr = soa.GetIntData(IntIdx::work_i).data();
        auto work_j_ptr = soa.GetIntData(IntIdx::work_j).data();
        auto naics_ptr = soa.GetIntData(IntIdx::naics).data();
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

        ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int ip, RandomEngine const& engine) noexcept {
            auto from = unit_arr(home_i_ptr[ip], home_j_ptr[ip], 0);

            /* Randomly assign the eligible working-age population */
            unsigned int number = (unsigned int)rint(((Real)Population[from]) / 2000.0);
            unsigned int nwork = (unsigned int)(2000.0 * number * .586); /* 58.6% of population is working-age */
            if (nwork == 0) { return; }

            int age_group = age_group_ptr[ip];
            /* Check working-age population */
            if (age_group >= AgeGroups::a18to29 && age_group <= AgeGroups::a50to64) {
                unsigned int irnd = Random_int(nwork, engine);
                int to = from; // Default: work in home unit
                int comm_to = 0;
                if (irnd < d_flow[from][Nunit - 1]) {
                    /* Choose a random destination unit */
                    to = 0;
                    while (irnd >= d_flow[from][to]) {
                        to++;
                    }
                }
                // else: irnd >= d_flow[from][Nunit - 1], stay in home unit (to = from)

                /*If from=to unit, 25% EXTRA chance of working in home community*/
                if ((from == to) && (Random(engine) < 0.25)) {
                    comm_to = comm_arr(home_i_ptr[ip], home_j_ptr[ip], 0);
                } else {
                    /* Choose a random community within that destination unit */
                    comm_to = Start[to] + Random_int(Start[to + 1] - Start[to], engine);
                    AMREX_ALWAYS_ASSERT(comm_to < Ncommunity);
                }

                IntVect comm_to_iv = domain.atOffset(comm_to);
                work_i_ptr[ip] = comm_to_iv[0];
                work_j_ptr[ip] = comm_to_iv[1];

                // Set default NAICS code as "uncategorized"
                naics_ptr[ip] = NAICSCodes::NAICS::uncat;

                number = (unsigned int)rint(((Real)Ndaywork[to]) / ((Real)workgroup_size * (Start[to + 1] - Start[to])));
                if (number) {
                    workgroup_ptr[ip] = 1 + Random_int(number, engine);
                    work_nborhood_ptr[ip] = workgroup_ptr[ip] % 4; // each workgroup is assigned to a neighborhood as well
                }
            }
        });
    }

    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) { The_Device_Arena()->free(host_vector_flow[i]); }
    }
    The_Device_Arena()->free(d_flow);

    assignTeachersAndWorkgroup(pc);
    assignMedicalWorkers(pc);
}

void CensusData::assignTeachersAndWorkgroup (AgentContainer& a_pc) {
    const Box& domain = a_pc.Geom(0).Domain();
    auto Ncommunity = demo.Ncommunity;
    Gpu::HostVector<int> high_teachers_array(Ncommunity, 0);
    Gpu::HostVector<int> middle_teachers_array(Ncommunity, 0);
    Gpu::HostVector<int> elem3_teachers_array(Ncommunity, 0);
    Gpu::HostVector<int> elem4_teachers_array(Ncommunity, 0);
    Gpu::HostVector<int> daycare_teachers_array(Ncommunity, 0);
    auto high_teachers_ptr = high_teachers_array.data();
    auto middle_teachers_ptr = middle_teachers_array.data();
    auto elem3_teachers_ptr = elem3_teachers_array.data();
    auto elem4_teachers_ptr = elem4_teachers_array.data();
    auto daycare_teachers_ptr = daycare_teachers_array.data();
    auto student_teacher_ratio = a_pc.m_student_teacher_ratio;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf); mfi.isValid(); ++mfi) {
        auto student_counts_arr = a_pc.m_student_counts[mfi].array();
        auto comm_arr = comm_mf[mfi].array();
        auto bx = mfi.tilebox();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            int comm = comm_arr(i, j, k);
            if (comm >= Ncommunity || comm < 0) { return; }
            high_teachers_ptr[comm] = int(std::round(double(student_counts_arr(i, j, k, SchoolCensusIDType::high_1 - 1)) /
                                                     student_teacher_ratio[SchoolType::high]));
            middle_teachers_ptr[comm] = int(std::round((double)student_counts_arr(i, j, k, SchoolCensusIDType::middle_2 - 1) /
                                                       student_teacher_ratio[SchoolType::middle]));
            elem3_teachers_ptr[comm] = int(std::round((double)student_counts_arr(i, j, k, SchoolCensusIDType::elem_3 - 1) /
                                                      student_teacher_ratio[SchoolType::elem]));
            elem4_teachers_ptr[comm] = int(std::round((double)student_counts_arr(i, j, k, SchoolCensusIDType::elem_4 - 1) /
                                                      student_teacher_ratio[SchoolType::elem]));
            daycare_teachers_ptr[comm] = int(std::round((double)student_counts_arr(i, j, k, SchoolCensusIDType::daycare_5 - 1) /
                                                        student_teacher_ratio[SchoolType::daycare]));
        });
        Gpu::synchronize();
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(unit_mf); mfi.isValid(); ++mfi) {
        auto& agents_tile = a_pc.GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto& soa = agents_tile.GetStructOfArrays();

        auto np = soa.numParticles();

        Gpu::HostVector<int> age_group_h(np);
        Gpu::HostVector<int> workgroup_h(np);
        Gpu::HostVector<int> work_i_h(np);
        Gpu::HostVector<int> work_j_h(np);
        Gpu::HostVector<int> school_grade_h(np);
        Gpu::HostVector<int> school_id_h(np);
        Gpu::HostVector<int> work_nborhood_h(np);
        Gpu::HostVector<int> naics_h(np);

        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::age_group).begin(), soa.GetIntData(IntIdx::age_group).end(),
                  age_group_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::workgroup).begin(), soa.GetIntData(IntIdx::workgroup).end(),
                  workgroup_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::work_i).begin(), soa.GetIntData(IntIdx::work_i).end(),
                  work_i_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::work_j).begin(), soa.GetIntData(IntIdx::work_j).end(),
                  work_j_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::school_grade).begin(), soa.GetIntData(IntIdx::school_grade).end(),
                  school_grade_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::school_id).begin(), soa.GetIntData(IntIdx::school_id).end(),
                  school_id_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::work_nborhood).begin(), soa.GetIntData(IntIdx::work_nborhood).end(),
                  work_nborhood_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::naics).begin(), soa.GetIntData(IntIdx::naics).end(), naics_h.begin());

        auto age_group_ptr = age_group_h.data();
        auto workgroup_ptr = workgroup_h.data();
        auto work_i_ptr = work_i_h.data();
        auto work_j_ptr = work_j_h.data();
        auto school_grade_ptr = school_grade_h.data();
        auto school_id_ptr = school_id_h.data();
        auto work_nborhood_ptr = work_nborhood_h.data();
        auto naics_ptr = naics_h.data();

        for (int ip = 0; ip < np; ++ip) {
            int comm = (int)domain.index(IntVect(AMREX_D_DECL(work_i_ptr[ip], work_j_ptr[ip], 0)));
            if (comm >= Ncommunity || comm < 0) { continue; }
            // skip non-working age
            if (age_group_ptr[ip] < AgeGroups::a18to29 || age_group_ptr[ip] > AgeGroups::a50to64) { continue; }
            // skip non-workers
            if (workgroup_ptr[ip] == 0) { continue; }
            // skip workers already assigned to other professions
            if (naics_ptr[ip] != NAICSCodes::NAICS::uncat) { continue; }

            int high_teachers = high_teachers_ptr[comm];
            int middle_teachers = middle_teachers_ptr[comm];
            int elem3_teachers = elem3_teachers_ptr[comm];
            int elem4_teachers = elem4_teachers_ptr[comm];
            int daycare_teachers = daycare_teachers_ptr[comm];
            int total_teachers = high_teachers + middle_teachers + elem3_teachers + elem4_teachers + daycare_teachers;
            if (total_teachers > 0) {
                int choice = Random_int(total_teachers);
                if (choice < high_teachers) {
                    school_grade_ptr[ip] = schoolTypeToGrade(SchoolType::high);
                    school_id_ptr[ip] = SchoolCensusIDType::high_1;
                    work_nborhood_ptr[ip] = 3; // assuming the high school is located in Neighbordhood 3
                    workgroup_ptr[ip] = 1;
                    high_teachers_ptr[comm]--;
                } else if (choice < high_teachers + middle_teachers) {
                    school_grade_ptr[ip] = schoolTypeToGrade(SchoolType::middle);
                    school_id_ptr[ip] = SchoolCensusIDType::middle_2;
                    work_nborhood_ptr[ip] = 1; // assuming the middle school is located in Neighbordhood 2
                    workgroup_ptr[ip] = 2;
                    middle_teachers_ptr[comm]--;
                } else if (choice < high_teachers + middle_teachers + elem3_teachers) {
                    school_grade_ptr[ip] = schoolTypeToGrade(SchoolType::elem);
                    school_id_ptr[ip] = SchoolCensusIDType::elem_3;
                    work_nborhood_ptr[ip] = 0; // assuming the first elementary school is located in Neighbordhood 1
                    workgroup_ptr[ip] = 3;
                    elem3_teachers_ptr[comm]--;
                } else if (choice < high_teachers + middle_teachers + elem3_teachers + elem4_teachers) {
                    school_grade_ptr[ip] = schoolTypeToGrade(SchoolType::elem);
                    school_id_ptr[ip] = SchoolCensusIDType::elem_4;
                    work_nborhood_ptr[ip] = 2; // assuming the first elementary school is located in Neighbordhood 3
                    workgroup_ptr[ip] = 4;
                    elem4_teachers_ptr[comm]--;
                } else {
                    school_grade_ptr[ip] = schoolTypeToGrade(SchoolType::daycare);
                    work_nborhood_ptr[ip] = Random_int(4); // randomly select nborhood
                    school_id_ptr[ip] = SchoolCensusIDType::daycare_5 + work_nborhood_ptr[ip];
                    workgroup_ptr[ip] = 5;
                    daycare_teachers_ptr[comm]--;
                }
            }
        }
        Gpu::copy(Gpu::hostToDevice, school_grade_h.begin(), school_grade_h.end(), soa.GetIntData(IntIdx::school_grade).begin());
        Gpu::copy(Gpu::hostToDevice, school_id_h.begin(), school_id_h.end(), soa.GetIntData(IntIdx::school_id).begin());
        Gpu::copy(Gpu::hostToDevice, workgroup_h.begin(), workgroup_h.end(), soa.GetIntData(IntIdx::workgroup).begin());
        Gpu::copy(Gpu::hostToDevice, work_nborhood_h.begin(), work_nborhood_h.end(),
                  soa.GetIntData(IntIdx::work_nborhood).begin());
    }
}

void CensusData::assignMedicalWorkers (AgentContainer& a_pc) {
    auto Ncommunity = demo.Ncommunity;

    Gpu::HostVector<int> totworkers_array(Ncommunity, 0);
    auto num_workers = countWorkersByComm(a_pc, totworkers_array);
    const auto totworkers_ptr = totworkers_array.data();

    Print() << "Assigning medical workers:\n";
    Print() << "  Number of workers = " << num_workers << "\n";

    Gpu::HostVector<int> medworkers_array(Ncommunity, 0);
    auto med_workers_prop = a_pc.medicalWorkersProportion();
    auto medworkers_ptr = medworkers_array.data();
    int tot_medworkers = 0;
    for (int comm = 0; comm < Ncommunity; comm++) {
        medworkers_ptr[comm] = int(std::round(med_workers_prop * double(totworkers_ptr[comm])));
        tot_medworkers += medworkers_ptr[comm];
    }
    Print() << "  Target number of medical workers: " << tot_medworkers << "\n";

    auto num_medworkers_local = setAsMedicalWorkers(a_pc, medworkers_array, true);
    ParallelDescriptor::ReduceIntSum(&num_medworkers_local, 1);
    ParallelDescriptor::ReduceIntMin(medworkers_array.begin(), medworkers_array.size());

    Vector<int> procs(ParallelDescriptor::NProcs());
    for (int i = 0; i < ParallelDescriptor::NProcs(); i++) {
        procs[i] = i;
    }
    {
        /* TODO: use AMReX's seed? */
        unsigned long int seed = 1024UL;
        std::mt19937 rndeng;
        rndeng.seed(seed);
        std::shuffle(procs.begin(), procs.end(), rndeng);
    }
    int num_medworkers_nonlocal = 0;
    for (int i = 0; i < ParallelDescriptor::NProcs(); i++) {
        if (ParallelDescriptor::MyProc() == procs[i]) {
            auto num_medworkers_nonlocal_proc = setAsMedicalWorkers(a_pc, medworkers_array, false);
            num_medworkers_nonlocal += num_medworkers_nonlocal_proc;
        }
        ParallelDescriptor::ReduceIntMin(medworkers_array.begin(), medworkers_array.size());
    }
    ParallelDescriptor::ReduceIntSum(&num_medworkers_nonlocal, 1);

    Print() << "  Total medical workers assigned: " << num_medworkers_local + num_medworkers_nonlocal << " (local - "
            << num_medworkers_local << ", nonlocal - " << num_medworkers_nonlocal << ")"
            << "\n";
}

int CensusData::countWorkersByComm (AgentContainer& a_pc, Gpu::HostVector<int>& a_workers_array, const int a_naics) {
    const Box& domain = a_pc.Geom(0).Domain();
    auto Ncommunity = demo.Ncommunity;
    AMREX_ASSERT(a_workers_array.size() == size_t(Ncommunity));
    auto workers_ptr = a_workers_array.data();
    for (MFIter mfi(unit_mf); mfi.isValid(); ++mfi) {
        auto& agents_tile = a_pc.GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto& soa = agents_tile.GetStructOfArrays();
        auto np = soa.numParticles();

        Gpu::HostVector<int> naics_h(np);
        Gpu::HostVector<int> work_i_h(np);
        Gpu::HostVector<int> work_j_h(np);

        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::naics).begin(), soa.GetIntData(IntIdx::naics).end(), naics_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::work_i).begin(), soa.GetIntData(IntIdx::work_i).end(),
                  work_i_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::work_j).begin(), soa.GetIntData(IntIdx::work_j).end(),
                  work_j_h.begin());

        const auto naics_ptr = naics_h.data();
        const auto work_i_ptr = work_i_h.data();
        const auto work_j_ptr = work_j_h.data();

        for (int ip = 0; ip < np; ++ip) {
            if (naics_ptr[ip] < 0) { continue; }
            if ((a_naics >= 0) && (naics_ptr[ip] != a_naics)) { continue; }
            auto iv_work = IntVect(AMREX_D_DECL(work_i_ptr[ip], work_j_ptr[ip], 0));
            int comm = (int)domain.index(iv_work);
            if (comm >= Ncommunity || comm < 0) { continue; }
            workers_ptr[comm]++;
        }
    }
    ParallelDescriptor::ReduceIntSum(a_workers_array.begin(), a_workers_array.size());

    int total = 0;
    for (int i = 0; i < Ncommunity; i++) {
        total += workers_ptr[i];
    }
    return total;
}

AMREX_GPU_HOST AMREX_FORCE_INLINE static bool boxesContainIV (const IntVect& a_iv, const Gpu::HostVector<Box>& a_boxes) {
    bool retval = false;
    for (size_t i = 0; i < a_boxes.size(); i++) {
        if (a_boxes[i].contains(a_iv)) { retval = true; }
    }
    return retval;
}

int CensusData::setAsMedicalWorkers (AgentContainer& a_pc, Gpu::HostVector<int>& a_medworkers_array, bool a_local) {
    Gpu::HostVector<Box> local_boxes(0);
    for (MFIter mfi(unit_mf); mfi.isValid(); ++mfi) {
        local_boxes.push_back(mfi.tilebox());
    }

    const Box& domain = a_pc.Geom(0).Domain();
    auto Ncommunity = demo.Ncommunity;
    auto medworkers_ptr = a_medworkers_array.data();
    int total = 0;
    for (MFIter mfi(unit_mf); mfi.isValid(); ++mfi) {
        auto& agents_tile = a_pc.GetParticles(0)[std::make_pair(mfi.index(), mfi.LocalTileIndex())];
        auto& soa = agents_tile.GetStructOfArrays();

        auto np = soa.numParticles();

        Gpu::HostVector<int> age_group_h(np);
        Gpu::HostVector<int> workgroup_h(np);
        Gpu::HostVector<int> work_nborhood_h(np);
        Gpu::HostVector<int> work_i_h(np);
        Gpu::HostVector<int> work_j_h(np);
        Gpu::HostVector<int> naics_h(np);

        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::age_group).begin(), soa.GetIntData(IntIdx::age_group).end(),
                  age_group_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::workgroup).begin(), soa.GetIntData(IntIdx::workgroup).end(),
                  workgroup_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::work_nborhood).begin(), soa.GetIntData(IntIdx::work_nborhood).end(),
                  work_nborhood_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::work_i).begin(), soa.GetIntData(IntIdx::work_i).end(),
                  work_i_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::work_j).begin(), soa.GetIntData(IntIdx::work_j).end(),
                  work_j_h.begin());
        Gpu::copy(Gpu::deviceToHost, soa.GetIntData(IntIdx::naics).begin(), soa.GetIntData(IntIdx::naics).end(), naics_h.begin());

        const auto age_group_ptr = age_group_h.data();
        const auto work_i_ptr = work_i_h.data();
        const auto work_j_ptr = work_j_h.data();
        auto workgroup_ptr = workgroup_h.data();
        auto work_nborhood_ptr = work_nborhood_h.data();
        auto naics_ptr = naics_h.data();

        Vector<int> idx(np);
        for (int ip = 0; ip < np; ++ip) {
            idx[ip] = ip;
        }
        {
            /* TODO: use AMReX's seed? */
            unsigned long int seed = 1024UL;
            std::mt19937 rndeng;
            rndeng.seed(seed);
            std::shuffle(idx.begin(), idx.end(), rndeng);
        }

        for (int i = 0; i < np; ++i) {
            int ip = idx[i];
            // skip non-working age
            if (age_group_ptr[ip] < AgeGroups::a18to29 || age_group_ptr[ip] > AgeGroups::a50to64) { continue; }
            // skip non-workers
            if (workgroup_ptr[ip] == 0) { continue; }
            // skip workers already assigned to other professions
            if (naics_ptr[ip] != NAICSCodes::NAICS::uncat) { continue; }

            auto iv_work = IntVect(AMREX_D_DECL(work_i_ptr[ip], work_j_ptr[ip], 0));
            if (((a_local) && (!boxesContainIV(iv_work, local_boxes))) ||
                ((!a_local) && (boxesContainIV(iv_work, local_boxes)))) {
                continue;
            }

            int comm = (int)domain.index(iv_work);
            if (comm >= Ncommunity || comm < 0) { continue; }

            int n_medworkers = medworkers_ptr[comm];
            if (n_medworkers > 0) {
                naics_h[ip] = NAICSCodes::NAICS::med_sca;

                // Right now, each community has a medical facility;
                // let it be in neighborhood 0
                work_nborhood_ptr[ip] = 0;

                // Assuming one medical facility per community, let all
                // the medical workers be in one workgroup
                workgroup_ptr[ip] = 1;

                medworkers_ptr[comm]--;
                total++;
            }
        }
        Gpu::copy(Gpu::hostToDevice, workgroup_h.begin(), workgroup_h.end(), soa.GetIntData(IntIdx::workgroup).begin());
        Gpu::copy(Gpu::hostToDevice, work_nborhood_h.begin(), work_nborhood_h.end(),
                  soa.GetIntData(IntIdx::work_nborhood).begin());
        Gpu::copy(Gpu::hostToDevice, naics_h.begin(), naics_h.end(), soa.GetIntData(IntIdx::naics).begin());
    }
    return total;
}
