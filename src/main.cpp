/*! @file main.cpp
    \brief **Main**: Contains main() and runAgent()
*/

#include <chrono>
#include <filesystem>

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_iMultiFab.H>

#include "AgentContainer.H"
#include "AirTravelFlow.H"
#include "CaseData.H"
#include "DemographicData.H"
#include "IO.H"
#include "InitializeInfections.H"
#include "UrbanPopData.H"
#include "Utils.H"

#include "version.h"

using namespace amrex;
using namespace ExaEpi;

void runAgent();

/*! \brief Set ExaEpi-specific defaults for memory-management and tiling */
void overrideAmrexDefaults () {
    amrex::ParmParse pp("amrex");
    // ExaEpi should never require mananaged memory in the Arena
    bool the_arena_is_managed = true;
    pp.queryAdd("the_arena_is_managed", the_arena_is_managed);

    bool use_comms_arena = true;
    pp.queryAdd("use_comms_arena", use_comms_arena);

    amrex::ParmParse pp2("particles");
    // enable for CPUs, disable for GPUs
    bool do_tiling = TilingIfNotGPU();
    pp2.queryAdd("do_tiling", do_tiling);
}

/*! \brief Main function: initializes AMReX, calls runAgent(), finalizes AMReX */
int main (int argc, /*!< Number of command line arguments */
          char* argv[] /*!< Command line arguments */) {
    amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, overrideAmrexDefaults);

    Print() << "ExaEpi version " << EXAEPI_VERSION << " (built on " << __DATE__ << ")\n";

    runAgent();

    amrex::Finalize();
}

/*! \brief Run agent-based simulation:

    \b Initialization
    + Read test parameters (#ExaEpi::TestParams) from command line input file
    + If initialization type (#ExaEpi::TestParams::ic_type) is ExaEpi::ICType::Census,
      + Read #DemographicData from #ExaEpi::TestParams::census_filename
        (see DemographicData::initFromFile)
      + Read #CaseData from #ExaEpi::TestParams::case_filename
        (see CaseData::initFromFile)
    + Get computational domain from ExaEpi::Utils::getGeometry. Each grid cell corresponds to
      a community.
    + Create box arrays and distribution mapping based on #ExaEpi::TestParams::max_box_size.
    + Initialize the following MultiFabs:
      + Number of residents: 6 components - number of residents in age groups under-5, 5-17,
        18-29, 30-64, 65+, total.
      + Unit number of the community at each grid cell (1 component).
      + FIPS code of the community at each grid cell (2 components - FIPS code, census tract ID).
      + Community number of the community at each grid cell.
      + Disease statistics with 4 components (hospitalization, ICU, ventilator, deaths)
      + Masking behavior
    + Initialize agents (AgentContainer::initAgentsCensus).
      If ExaEpi::TestParams::ic_type is ExaEpi::ICType::Census, then
      + Read worker flow (ExaEpi::Initialization::readWorkerflow)
      + Initialize cases (ExaEpi::Initialization::setInitialCases)


    \b Evolution
    At each step from 0 to #ExaEpi::TestParams::nsteps-1:
    + IO:
      + if the current step number is a multiple of #ExaEpi::TestParams::plot_int, then write
        out plot file - see ExaEpi::IO::writePlotFile()
      + if current step number is a multiple of #ExaEpi::TestParams::aggregated_diag_int, then write
        out aggregated diagnostic data - see ExaEpi::IO::writeFIPSData().
    + Agents behavior:
      + Update agent #Status based on their age, number of days since infection, hospitalization,
        etc. - see AgentContainer::updateStatus().
      + Move agents to work - see AgentContainer::moveAgentsToWork().
      + Let agents interact at work - see AgentContainer::interactAgentsHomeWork().
      + Move agents to home - see AgentContainer::moveAgentsToHome().
      + Let agents interact at home - see AgentContainer::interactAgentsHomeWork().
      + Infect agents based on their movements during the day - see AgentContainer::infectAgents().
    + Get disease statistics counts - see AgentContainer::printTotals() - and update the
      peak number of infections and cumulative deaths.

    \b Finalize
    + Report peak infections, day of peak infections, and cumulative deaths.
    + Write out final plot file - see ExaEpi::IO::writePlotFile()
    + Write out final aggregated diagnostic data - see ExaEpi::IO::writeFIPSData().
*/
void runAgent () {
    BL_PROFILE("runAgent");
    TestParams params;
    ExaEpi::Utils::getTestParams(params, "agent");

    amrex::Print() << "Tracking " << params.num_diseases << " diseases:\n";
    for (int d = 0; d < params.num_diseases; d++) {
        amrex::Print() << "    " << params.disease_names[d] << "\n";
    }

    Geometry geom;
    BoxArray ba;
    DistributionMapping dm;
    CensusData censusData;
    UrbanPopData urbanPopData;

    if (params.ic_type == ICType::Census) {
        censusData.init(params, geom, ba, dm);
    } else if (params.ic_type == ICType::UrbanPop) {
        urbanPopData.init(params, geom, ba, dm);
    }

    AirTravelFlow air;
    if (params.air_travel_int > 0) {
        air.readAirports(params.airports_filename, censusData.demo);
        air.readAirTravelFlow(params.air_traffic_filename);
        air.computeTravelProbs(censusData.demo);
    }

    // The default output filename is:
    // output.dat for a single disease
    // output_<disease_name>.dat for multiple diseases
    std::vector<std::string> output_filename;
    output_filename.resize(params.num_diseases);
    if (params.num_diseases == 1) {
        output_filename[0] = "output.dat";
    } else {
        for (int d = 0; d < params.num_diseases; d++) {
            output_filename[d] = "output_" + params.disease_names[d] + ".dat";
        }
    }
    ParmParse pp("diag");
    pp.queryarr("output_filename", output_filename, 0, params.num_diseases);

    if (params.restart_chkfile == "") {
        for (int d = 0; d < params.num_diseases; d++) {
            if (ParallelDescriptor::IOProcessor()) {
                std::ofstream File;
                File.open(output_filename[d].c_str(), std::ios::out | std::ios::trunc);

                if (!File.good()) { amrex::FileOpenFailed(output_filename[d]); }

                File << std::setw(5) << "Day";
                File << std::setw(12) << "Susceptible";
                File << std::setw(12) << "Infected";
                File << std::setw(12) << "Recovered";
                File << std::setw(12) << "Deaths";
                File << std::setw(15) << "Hospitalized";
                File << std::setw(15) << "ICU";
                File << std::setw(12) << "Ventilated";
                File << std::setw(12) << "Exposed";
                File << std::setw(15) << "Asymptomatic";
                File << std::setw(15) << "Presymptomatic";
                File << std::setw(15) << "Symptomatic\n";

                File.flush();

                File.close();

                if (!File.good()) { amrex::Abort("problem writing output file"); }
            }
        }
    }

    amrex::Vector<std::unique_ptr<MultiFab>> disease_stats;
    disease_stats.resize(params.num_diseases);
    for (int d = 0; d < params.num_diseases; d++) {
        disease_stats[d] = std::make_unique<MultiFab>(ba, dm, 5, 0);
        disease_stats[d]->setVal(0);
    }

    MultiFab mask_behavior(ba, dm, 1, 0);
    mask_behavior.setVal(1);

    AgentContainer pc(geom, dm, ba, params.num_diseases, params.disease_names, params.fast, params.ic_type);
    bool stable_redistribute = !params.fast;
    pc.setStableRedistribute(stable_redistribute);
    pc.setTileSize(censusData.unit_mf.mfiter_tile_size);

    amrex::Real cur_time = 0;
    int start_day = 0;
    {
        BL_PROFILE_REGION("Initialization");
        if (params.restart_chkfile.empty()) {
            if (params.ic_type == ICType::Census) {
                censusData.initAgents(pc, params.nborhood_size);
                censusData.readWorkerflow(pc, params.workerflow_filename, params.workgroup_size);
            } else if (params.ic_type == ICType::UrbanPop) {
                urbanPopData.initAgents(pc, params);
            } else {
                Abort("Unimplemented ic_type");
            }

            for (int d = 0; d < params.num_diseases; d++) {
                auto disease_params = pc.getDiseaseParameters_h(d);
                if (disease_params->initial_case_type == CaseTypes::file) {
                    CaseData cases;
                    cases.initFromFile(disease_params->disease_name, std::string(disease_params->case_filename));
                    setInitialCasesFromFile(pc, cases, disease_params->disease_name, d,
                                            (params.ic_type == ICType::Census ? censusData.demo.FIPS : urbanPopData.FIPS_codes),
                                            (params.ic_type == ICType::Census ? censusData.demo.Start
                                                                              : urbanPopData.fips_community_start),
                                            (params.ic_type == ICType::Census ? censusData.comm_mf : urbanPopData.community_mf),
                                            params.fast);
                } else {
                    setInitialCasesRandom(pc, disease_params->num_initial_cases, disease_params->disease_name, d,
                                          (params.ic_type == ICType::Census ? censusData.demo.Start
                                                                            : urbanPopData.fips_community_start),
                                          (params.ic_type == ICType::Census ? censusData.comm_mf : urbanPopData.community_mf),
                                          params.fast);
                }
            }

            pc.printStudentTeacherCounts();
            pc.printAgeGroupCounts();

            if (params.ic_type == ICType::Census && params.air_travel_int > 0) {
                pc.setAirTravel(censusData.unit_mf, air, censusData.demo);
            }
        } else {
            IO::readCheckpointFile(params.restart_chkfile, pc, disease_stats, &(censusData.unit_mf), &(censusData.FIPS_mf),
                                   &(censusData.comm_mf), cur_time, start_day);
        }
    }

    // if we are doing a restart, we need to fix up the output_file
    if (params.restart_chkfile != "") {
        for (int d = 0; d < params.num_diseases; d++) {
            if (ParallelDescriptor::IOProcessor()) {

                if (amrex::FileExists(output_filename[d])) {
                    std::string newoldname(output_filename[d] + ".old." + amrex::UniqueString());
                    amrex::Print() << output_filename[d] << " exists.  Renaming to:  " << newoldname << '\n';
                    std::filesystem::copy(output_filename[d], newoldname);
                }

                std::ifstream inFile;
                inFile.open(output_filename[d].c_str(), std::ios::in);

                if (!inFile.good()) { amrex::FileOpenFailed(output_filename[d]); }

                std::vector<std::string> lines;
                std::string line;
                while (std::getline(inFile, line)) {
                    lines.push_back(line);
                }
                inFile.close();

                AMREX_ALWAYS_ASSERT(std::size_t(start_day + 1) <= lines.size());
                lines.erase(lines.begin() + start_day + 1, lines.end());

                std::ofstream outFile;
                outFile.open(output_filename[d].c_str(), std::ios::out | std::ios::trunc);

                if (!outFile.good()) { amrex::FileOpenFailed(output_filename[d]); }
                for (auto li : lines) {
                    outFile << li << "\n";
                }

                outFile.flush();

                outFile.close();

                if (!outFile.good()) { amrex::Abort("problem writing output file"); }
            }
        }
    }

// #define DUMP_INITIAL_AGENTS_ASCII
#ifdef DUMP_INITIAL_AGENTS_ASCII
    string agents_fname = std::string("agents.") + (params.ic_type == ICType::UrbanPop ? "urbanpop" : "census") + ".csv";
    pc.WriteAsciiFile(agents_fname);
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream agents_f(agents_fname, std::ios_base::app);
        agents_f << "#posx posy id cpu " << "treatment_timer " << "disease_counter " << "prob " << "incubation_period "
                 << "infectious_period " << "symptomdev_period " << "age_group " << "family " << "home_i " << "home_j "
                 << "work_i " << "work_j " << "hosp_i " << "hosp_j " << "nborhood " << "school " << "naics " << "workgroup "
                 << "work_nborhood " << "withdrawn " << "random_travel " << "status " << "symptomatic\n";
        agents_f.close();
    }
#endif

    std::vector<int> step_of_peak(params.num_diseases, 0);
    std::vector<Long> num_infected_peak(params.num_diseases, 0);
    std::vector<Long> cumulative_deaths(params.num_diseases, 0);
    for (int d = 0; d < params.num_diseases; d++) {
        auto counts = pc.getTotals(d);
        if (counts[1] > num_infected_peak[d]) {
            num_infected_peak[d] = counts[1];
            step_of_peak[d] = 0;
        }
        cumulative_deaths[d] = counts[4];
    }

    Vector<Long> num_infected(params.num_diseases, 0);

    amrex::ParmParse::QueryUnusedInputs();

    {
        BL_PROFILE_REGION("Evolution");
        for (int i = start_day; i < params.nsteps; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            if ((params.plot_int > 0) && (i % params.plot_int == 0)) {
                if (params.ic_type == ICType::Census) {
                    ExaEpi::IO::writePlotFile(pc, disease_stats, &censusData.unit_mf, &censusData.FIPS_mf, &censusData.comm_mf,
                                              params.num_diseases, params.disease_names, cur_time, i);
                } else {
                    ExaEpi::IO::writePlotFile(pc, disease_stats, nullptr, &urbanPopData.geoid_mf, &urbanPopData.community_mf,
                                              params.num_diseases, params.disease_names, cur_time, i);
                }
            }

            if ((params.check_int > 0) && (i % params.check_int == 0) && ((params.restart_chkfile == "") || (i != start_day))) {
                if (params.ic_type == ICType::Census) {
                    ExaEpi::IO::writeCheckpointFile(pc, disease_stats, &censusData.unit_mf, &censusData.FIPS_mf,
                                                    &censusData.comm_mf, params.num_diseases, params.disease_names, cur_time, i);
                } else {
                    ExaEpi::IO::writeCheckpointFile(pc, disease_stats, nullptr, &urbanPopData.geoid_mf,
                                                    &urbanPopData.community_mf, params.num_diseases, params.disease_names,
                                                    cur_time, i);
                }
            }

            if ((params.aggregated_diag_int > 0) && (i % params.aggregated_diag_int == 0)) {
                if (params.ic_type == ICType::Census) {
                    ExaEpi::IO::writeFIPSData(pc, censusData, params.aggregated_diag_prefix, params.num_diseases,
                                              params.disease_names, i);
                } else {
                    ExaEpi::IO::writeAggregatedData(pc, urbanPopData, params.aggregated_diag_prefix, params.num_diseases,
                                                    params.disease_names, i);
                }
            }

            // Update agents' disease status
            pc.updateStatus(disease_stats);

            for (int d = 0; d < params.num_diseases; d++) {
                auto counts = pc.getTotals(d);
                if (counts[1] > num_infected_peak[d]) {
                    num_infected_peak[d] = counts[1];
                    step_of_peak[d] = i;
                }
                cumulative_deaths[d] = counts[4];
                num_infected[d] = counts[1];

                Real mmc[4] = {0, 0, 0, 0};
#ifdef AMREX_USE_GPU
                if (Gpu::inLaunchRegion()) {
                    auto const& ma = disease_stats[d]->const_arrays();
                    GpuTuple<Real, Real, Real, Real> mm =
                            ParReduce(TypeList<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum>{},
                                      TypeList<Real, Real, Real, Real>{}, *(disease_stats[d]), IntVect(0, 0),
                                      [=] AMREX_GPU_DEVICE (int box_no, int ii, int jj,
                                                           int kk) noexcept -> GpuTuple<Real, Real, Real, Real> {
                                          return {ma[box_no](ii, jj, kk, 0), ma[box_no](ii, jj, kk, 1), ma[box_no](ii, jj, kk, 2),
                                                  ma[box_no](ii, jj, kk, 3)};
                                      });
                    mmc[0] = amrex::get<0>(mm);
                    mmc[1] = amrex::get<1>(mm);
                    mmc[2] = amrex::get<2>(mm);
                    mmc[3] = amrex::get<3>(mm);
                } else
#endif
                {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (!system::regtest_reduction) reduction(+ : mmc[ : 4])
#endif
                    for (MFIter mfi(*(disease_stats[d])); mfi.isValid(); ++mfi) {
                        Box const& bx = mfi.tilebox();
                        auto const& dfab = disease_stats[d]->const_array(mfi);
                        AMREX_LOOP_3D(bx, ii, jj, kk, {
                            mmc[0] += dfab(ii, jj, kk, 0);
                            mmc[1] += dfab(ii, jj, kk, 1);
                            mmc[2] += dfab(ii, jj, kk, 2);
                            mmc[3] += dfab(ii, jj, kk, 3);
                        });
                    }
                }

                ParallelDescriptor::ReduceRealSum(&mmc[0], 4, ParallelDescriptor::IOProcessorNumber());

                if (ParallelDescriptor::IOProcessor()) {
                    // total number of deaths computed on agents and on mesh should be the same...
                    if (mmc[3] != counts[4]) { amrex::Print() << mmc[3] << " " << counts[4] << "\n"; }
                    AMREX_ALWAYS_ASSERT(mmc[3] == counts[4]);

                    // the total number of infected should equal the sum of
                    //     exposed but not infectious
                    //     infectious and asymptomatic
                    //     infectious and pre-symptomatic
                    //     infectious and symptomatic
                    AMREX_ALWAYS_ASSERT(counts[1] == counts[5] + counts[6] + counts[7] + counts[8]);

                    std::ofstream File;
                    File.open(output_filename[d].c_str(), std::ios::out | std::ios::app);

                    if (!File.good()) { amrex::FileOpenFailed(output_filename[d]); }

                    File << std::setw(5) << i;
                    File << std::setw(12) << counts[0];
                    File << std::setw(12) << counts[1];
                    File << std::setw(12) << counts[2];
                    File << std::setw(12) << counts[4];
                    File << std::setw(15) << mmc[0];
                    File << std::setw(15) << mmc[1];
                    File << std::setw(12) << mmc[2];
                    File << std::setw(12) << counts[5];
                    File << std::setw(15) << counts[6];
                    File << std::setw(15) << counts[7];
                    File << std::setw(15) << counts[8] << "\n";

                    File.flush();

                    File.close();

                    if (!File.good()) { amrex::Abort("problem writing output file"); }
                }
            }

            if (params.shelter_start > 0 && params.shelter_start == i) { pc.shelterStart(); }

            if (params.shelter_start > 0 && params.shelter_start + params.shelter_length == i) { pc.shelterStop(); }

            if ((params.random_travel_int > 0) && (i % params.random_travel_int == 0)) {
                pc.moveRandomTravel(params.random_travel_prob);
            }

            if ((params.air_travel_int > 0) && (i % params.air_travel_int == 0)) {
                pc.moveAirTravel(censusData.unit_mf, air, censusData.demo);
            }

            // Typical day
            pc.morningCommute(mask_behavior);
            pc.interactDay(mask_behavior);
            pc.eveningCommute(mask_behavior);
            pc.interactEvening(mask_behavior);
            pc.interactNight(mask_behavior);

            if ((params.random_travel_int > 0) && (i % params.random_travel_int == 0)) { pc.returnRandomTravel(); }

            if ((params.air_travel_int > 0) && (i % params.air_travel_int == 0)) { pc.returnAirTravel(); }

            // Infect agents based on their interactions
            pc.infectAgents(disease_stats);

            std::chrono::duration<double> elapsed_time = std::chrono::high_resolution_clock::now() - start_time;

            Print() << "[Day " << cur_time << " " << std::fixed << std::setprecision(1) << elapsed_time.count()
                    << "s] infected: ";
            for (int d = 0; d < params.num_diseases; d++) {
                if (d > 0) { Print() << ", "; }
                Print() << params.disease_names[d] << " " << num_infected[d];
            }
            // the cumulative deaths are not tracked separately for each disease
            Print() << "; deaths: " << cumulative_deaths[0] << "\n";

            cur_time += 1.0_rt; // time step is one day
        }
    }

    if (params.num_diseases == 1) {
        amrex::Print() << "\n \n";
        amrex::Print() << "Peak number of infected: " << num_infected_peak[0] << "\n";
        amrex::Print() << "Day of peak: " << step_of_peak[0] << "\n";
        amrex::Print() << "Cumulative deaths: " << cumulative_deaths[0] << "\n";
        amrex::Print() << "\n \n";
    } else {
        amrex::Print() << "\n \n";
        for (int d = 0; d < params.num_diseases; d++) {
            amrex::Print() << "Disease " << params.disease_names[d] << ":\n";
            amrex::Print() << "    Peak number of infected: " << num_infected_peak[d] << "\n";
            amrex::Print() << "    Day of peak: " << step_of_peak[d] << "\n";
            amrex::Print() << "    Cumulative deaths: " << cumulative_deaths[d] << "\n";
        }
        amrex::Print() << "\n \n";
    }

    if (params.plot_int > 0) {
        if (params.ic_type == ICType::Census) {
            ExaEpi::IO::writePlotFile(pc, disease_stats, &censusData.unit_mf, &censusData.FIPS_mf, &censusData.comm_mf,
                                      params.num_diseases, params.disease_names, cur_time, params.nsteps);
        } else {
            ExaEpi::IO::writePlotFile(pc, disease_stats, nullptr, &urbanPopData.geoid_mf, &urbanPopData.community_mf,
                                      params.num_diseases, params.disease_names, cur_time, params.nsteps);
        }
    }

    if (params.check_int > 0) {
        if (params.ic_type == ICType::Census) {
            ExaEpi::IO::writeCheckpointFile(pc, disease_stats, &censusData.unit_mf, &censusData.FIPS_mf, &censusData.comm_mf,
                                            params.num_diseases, params.disease_names, cur_time, params.nsteps);
        } else {
            ExaEpi::IO::writeCheckpointFile(pc, disease_stats, nullptr, &urbanPopData.geoid_mf, &urbanPopData.community_mf,
                                            params.num_diseases, params.disease_names, cur_time, params.nsteps);
        }
    }

    if ((params.aggregated_diag_int > 0) && (params.nsteps % params.aggregated_diag_int == 0)) {
        if (params.ic_type == ICType::Census) {
            ExaEpi::IO::writeFIPSData(pc, censusData, params.aggregated_diag_prefix, params.num_diseases, params.disease_names,
                                      params.nsteps);
        } else {
            ExaEpi::IO::writeAggregatedData(pc, urbanPopData, params.aggregated_diag_prefix, params.num_diseases,
                                            params.disease_names, params.nsteps);
        }
    }
}
