/*! @file main.cpp
    \brief **Main**: Contains main() and runAgent()
*/

#include <chrono>

#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>

#include "AgentContainer.H"
#include "CaseData.H"
#include "AirTravelFlow.H"
#include "DemographicData.H"
#include "IO.H"
#include "Utils.H"


using namespace amrex;
using namespace ExaEpi;

void runAgent();

/*! \brief Set ExaEpi-specific defaults for memory-management */
void override_amrex_defaults ()
{
    amrex::ParmParse pp("amrex");

    // ExaEpi currently assumes we have mananaged memory in the Arena
    bool the_arena_is_managed = true;
    pp.queryAdd("the_arena_is_managed", the_arena_is_managed);
}

/*! \brief Main function: initializes AMReX, calls runAgent(), finalizes AMReX */
int main (int argc, /*!< Number of command line arguments */
          char* argv[] /*!< Command line arguments */)
{
    amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, override_amrex_defaults);

    runAgent();

    amrex::Finalize();
}

/*! \brief Run agent-based simulation:

    \b Initialization
    + Read test parameters (#ExaEpi::TestParams) from command line input file
    + If initialization type (#ExaEpi::TestParams::ic_type) is ExaEpi::ICType::Census,
      + Read #DemographicData from #ExaEpi::TestParams::census_filename
        (see DemographicData::InitFromFile)
      + Read #CaseData from #ExaEpi::TestParams::case_filename
        (see CaseData::InitFromFile)
    + Get computational domain from ExaEpi::Utils::get_geometry. Each grid cell corresponds to
      a community.
    + Create box arrays and distribution mapping based on #ExaEpi::TestParams::max_grid_size.
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
      + Read worker flow (ExaEpi::Initialization::read_workerflow)
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
void runAgent ()
{
    BL_PROFILE("runAgent");
    TestParams params;
    ExaEpi::Utils::get_test_params(params, "agent");

    amrex::Print() << "Tracking " << params.num_diseases << " diseases:\n";
    for (int d = 0; d < params.num_diseases; d++) {
        amrex::Print() << "    " << params.disease_names[d] << "\n";
    }

    std::vector<CaseData> cases;
    cases.resize(params.num_diseases);
    for (int d = 0; d < params.num_diseases; d++) {
        if (params.initial_case_type[d] == "file") {
            cases[d].InitFromFile(params.disease_names[d],params.case_filename[d]);
        }
    }

    Geometry geom;
    BoxArray ba;
    DistributionMapping dm;
    CensusData censusData;

    if (params.ic_type == ICType::Census) {
        censusData.init(params, geom, ba, dm);
    } else if (params.ic_type == ICType::UrbanPop) {
        Abort("UrbanPop not yet implemented");
        return;
    }

    AirTravelFlow air;
    if (params.air_travel_int > 0){
        air.ReadAirports(params.airports_filename, censusData.demo);
        air.ReadAirTravelFlow(params.air_traffic_filename);
        air.ComputeTravelProbs(censusData.demo);
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
    pp.queryarr("output_filename",output_filename,0,params.num_diseases);

    for (int d = 0; d < params.num_diseases; d++) {
        if (ParallelDescriptor::IOProcessor())
        {
            std::ofstream File;
            File.open(output_filename[d].c_str(), std::ios::out|std::ios::trunc);

            if (!File.good()) {
                amrex::FileOpenFailed(output_filename[d]);
            }

            File << std::setw(5) << "Day"
                 << std::setw(12) << "Susceptible"
                 << std::setw(12) << "Infected"
                 << std::setw(12) << "Recovered"
                 << std::setw(12) << "Deaths"
                 << std::setw(15) << "Hospitalized"
                 << std::setw(15) << "Ventilated"
                 << std::setw(12) << "ICU"
                 << std::setw(12) << "Exposed"
                 << std::setw(15) << "Asymptomatic"
                 << std::setw(15) << "Presymptomatic"
                 << std::setw(15) << "Symptomatic\n";

            File.flush();

            File.close();

            if (!File.good()) {
                amrex::Abort("problem writing output file");
            }
        }
    }

    amrex::Vector< std::unique_ptr<MultiFab> > disease_stats;
    disease_stats.resize(params.num_diseases);
    for (int d = 0; d < params.num_diseases; d++) {
        disease_stats[d] = std::make_unique<MultiFab>(ba, dm, 4, 0);
        disease_stats[d]->setVal(0);
    }

    MultiFab mask_behavior(ba, dm, 1, 0);
    mask_behavior.setVal(1);

    AgentContainer pc(geom, dm, ba, params.num_diseases, params.disease_names, params.fast);
    if (params.air_travel_int > 0) pc.setAirTravel(censusData.unit_mf, air, censusData.demo);

    {
        BL_PROFILE_REGION("Initialization");
        if (params.ic_type == ICType::Census) {
            censusData.initAgents(pc, params.nborhood_size);
            censusData.read_workerflow(pc, params.workerflow_filename, params.workgroup_size);
            if (params.initial_case_type[0] == "file") {
                censusData.setInitialCasesFromFile(pc, cases, params.disease_names, params.fast);
            } else {
                censusData.setInitialCasesRandom(pc, params.num_initial_cases, params.disease_names, params.fast);
            }
        } else if (params.ic_type == ICType::UrbanPop) {
            Abort("UrbanPop not yet implemented");
        } else {
            Abort("Unimplemented ic_type");
        }
    }

    std::vector<int>  step_of_peak(params.num_diseases, 0);
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

    amrex::Real cur_time = 0;

    Vector<Long> num_infected(params.num_diseases, 0);

    {
        BL_PROFILE_REGION("Evolution");
        for (int i = 0; i < params.nsteps; ++i)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            if ((params.plot_int > 0) && (i % params.plot_int == 0)) {
                ExaEpi::IO::writePlotFile(pc, censusData, params.num_diseases, params.disease_names, cur_time, i);
            }

            if ((params.aggregated_diag_int > 0) && (i % params.aggregated_diag_int == 0)) {
                ExaEpi::IO::writeFIPSData(pc, censusData, params.aggregated_diag_prefix, params.num_diseases, params.disease_names, i);
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
                    GpuTuple<Real,Real,Real,Real> mm = ParReduce(
                             TypeList<ReduceOpSum,ReduceOpSum,ReduceOpSum,ReduceOpSum>{},
                             TypeList<Real,Real,Real,Real>{},
                             *(disease_stats[d]), IntVect(0, 0),
                             [=] AMREX_GPU_DEVICE (int box_no, int ii, int jj, int kk) noexcept
                             -> GpuTuple<Real,Real,Real,Real>
                             {
                                 return { ma[box_no](ii,jj,kk,0),
                                          ma[box_no](ii,jj,kk,1),
                                          ma[box_no](ii,jj,kk,2),
                                          ma[box_no](ii,jj,kk,3) };
                             });
                    mmc[0] = amrex::get<0>(mm);
                    mmc[1] = amrex::get<1>(mm);
                    mmc[2] = amrex::get<2>(mm);
                    mmc[3] = amrex::get<3>(mm);
                } else
#endif
                    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (!system::regtest_reduction) reduction(+:mmc[:4])
#endif
                        for (MFIter mfi(*(disease_stats[d]),true); mfi.isValid(); ++mfi)
                        {
                            Box const& bx = mfi.tilebox();
                            auto const& dfab = disease_stats[d]->const_array(mfi);
                            AMREX_LOOP_3D(bx, ii, jj, kk,
                            {
                                mmc[0] += dfab(ii,jj,kk,0);
                                mmc[1] += dfab(ii,jj,kk,1);
                                mmc[2] += dfab(ii,jj,kk,2);
                                mmc[3] += dfab(ii,jj,kk,3);
                            });
                        }
                    }

                ParallelDescriptor::ReduceRealSum(&mmc[0], 4,
                                                  ParallelDescriptor::IOProcessorNumber());

                if (ParallelDescriptor::IOProcessor())
                {
                    // total number of deaths computed on agents and on mesh should be the same...
                    if (mmc[3] != counts[4]) {
                        amrex::Print() << mmc[3] << " " << counts[4] << "\n";
                    }
                    AMREX_ALWAYS_ASSERT(mmc[3] == counts[4]);

                    // the total number of infected should equal the sum of
                    //     exposed but not infectious
                    //     infectious and asymptomatic
                    //     infectious and pre-symptomatic
                    //     infectious and symptomatic
                    AMREX_ALWAYS_ASSERT(counts[1] == counts[5] + counts[6] + counts[7] + counts[8]);

                    std::ofstream File;
                    File.open(output_filename[d].c_str(), std::ios::out|std::ios::app);

                    if (!File.good()) {
                        amrex::FileOpenFailed(output_filename[d]);
                    }

                    File << std::setw(5) << i
                         << std::setw(12) << counts[0]
                         << std::setw(12) << counts[1]
                         << std::setw(12) << counts[2]
                         << std::setw(12) << counts[4]
                         << std::setw(15) << mmc[0]
                         << std::setw(15) << mmc[1]
                         << std::setw(12) << mmc[2]
                         << std::setw(12) << counts[5]
                         << std::setw(15) << counts[6]
                         << std::setw(15) << counts[7]
                         << std::setw(15) << counts[8] << "\n";

                    File.flush();

                    File.close();

                    if (!File.good()) {
                        amrex::Abort("problem writing output file");
                    }
                }
            }

            if (params.shelter_start > 0 && params.shelter_start == i) {
                pc.shelterStart();
            }

            if (params.shelter_start > 0 && params.shelter_start + params.shelter_length == i) {
                pc.shelterStop();
            }

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

            if ((params.random_travel_int > 0) && (i % params.random_travel_int == 0)) {
                pc.returnRandomTravel();
            }

            if ((params.air_travel_int > 0) && (i % params.air_travel_int == 0)){
                pc.returnAirTravel();
            }

            // Infect agents based on their interactions
            pc.infectAgents();

            std::chrono::duration<double> elapsed_time = std::chrono::high_resolution_clock::now() - start_time;

            Print() << "[Day " << cur_time <<  " " << std::fixed << std::setprecision(1) << elapsed_time.count() << "s] ";
            for (int d = 0; d < params.num_diseases; d++) {
                if (d > 0) Print() << "; ";
                Print() << params.disease_names[d] << ": " << num_infected[d] << " infected, " << cumulative_deaths[d] << " deaths";
            }
            Print() << "\n";

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
        ExaEpi::IO::writePlotFile(pc, censusData, params.num_diseases, params.disease_names, cur_time, params.nsteps);
    }

    if ((params.aggregated_diag_int > 0) && (params.nsteps % params.aggregated_diag_int == 0)) {
        ExaEpi::IO::writeFIPSData(pc, censusData, params.aggregated_diag_prefix, params.num_diseases,
                                  params.disease_names, params.nsteps);
    }
}
