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
#include "WeatherData.H"

#include "version.h"

using namespace amrex;
using namespace ExaEpi;

void runAgent();

/*! \brief Print usage and the list of recognized "agent.*" input-file parameters, with defaults */
void printHelp (const char* prog) {
    std::cout
            << "Usage: \n"
               "  "
            << prog
            << " --version               Print the ExaEpi and AMReX versions\n"
               "  "
            << prog
            << " -h | --help              Print this help message\n"
               "  "
            << prog
            << " <inputs_file> [overrides...]\n"
               "                            Run the model, reading parameters from <inputs_file>.\n"
               "                            Any parameter may be overridden on the command line as\n"
               "                            key=value pairs, e.g. agent.nsteps=10\n"
               "\n"
               "Recognized \"agent.*\" parameters (name, default, description):\n"
               "  nsteps                          (1)          number of simulation steps\n"
               "  plot_int                        (-1)         plot file interval, in steps; <=0 disables\n"
               "  check_int                       (-1)         checkpoint file interval, in steps; <=0 disables\n"
               "  random_travel_int               (-1)         interval between random travel events, in steps; <=0 disables\n"
               "  random_travel_prob              (0.0001)     probability of an agent going on random travel\n"
               "  air_travel_int                  (-1)         interval between air travel events, in steps; <=0 disables\n"
               "  number_of_diseases              (1)          number of diseases to track\n"
               "  disease_names                   (default00, default01, ...)\n"
               "                                               names of the diseases (array of number_of_diseases strings)\n"
               "  weather_int                     (-1)         interval for weather effects, in steps; <=0 disables\n"
               "  weather_filename                (required if weather_int > 0)\n"
               "                                               weather data file\n"
               "  startdate                       (\"\")         simulation start date (YYYY-MM-DD)\n"
               "  ic_type                         (urbanpop)   initial condition type: \"census\" or \"urbanpop\"\n"
               "  census_filename                 (required if ic_type=census)\n"
               "                                               census data file\n"
               "  workerflow_filename             (required if ic_type=census)\n"
               "                                               worker flow binary file\n"
               "  air_traffic_filename            (required if ic_type=census and air_travel_int > 0)\n"
               "                                               air traffic flow file\n"
               "  airports_filename               (required if ic_type=census and air_travel_int > 0)\n"
               "                                               airports file\n"
               "  urbanpop_filename               (required if ic_type=urbanpop)\n"
               "                                               UrbanPop data file\n"
               "  workgroup_size_filename         (\"\")         optional per-(state, NAICS-code) work-group target size "
               "table (urbanpop only); falls back to workgroup_size for any (state, NAICS) pair not in the file\n"
               "  size_scale_enabled              (true)       enable population-size-based transmission scaling (urbanpop "
               "only)\n"
               "  school_class_size               (15)         fallback target students-per-class for raw groups with "
               "no identified teachers\n"
               "  school_class_size_min           (5)          floor on average class size (bounds class count from "
               "above)\n"
               "  school_class_size_max           (50)         cap on average class size (bounds class count from "
               "below)\n"
               "  college_instructional_fraction  (0.1)        fraction of a college-level raw group's reported "
               "teacher/staff headcount assumed to actually be instructors, before class-size clamping\n"
               "  max_box_size                    (16)         box size for domain decomposition\n"
               "  aggregated_diag_int             (-1)         interval for aggregated diagnostic output, in steps; <=0 "
               "disables\n"
               "  aggregated_diag_prefix          (cases)      filename prefix for aggregated diagnostic output\n"
               "  restart                         (\"\")         checkpoint file to restart from\n"
               "  shelter_start                   (-1)         step at which to start sheltering; <=0 disables\n"
               "  shelter_length                  (0)          number of steps to shelter for\n"
               "  nborhood_size                   (500)        target neighborhood size\n"
               "  workgroup_size                  (20)         target workgroup size\n"
               "  seed                            (unset)      RNG seed\n"
               "  fast                            (false)      use fast, non-bitwise-reproducible implementations\n"
               "  context_diag                    (false)      attribute infections to interaction contexts in the output file\n"
               "  shelter_compliance              (0.95)       shelter-in-place compliance rate\n"
               "  student_teacher_ratio           (0, 15, 15, 15, 15, 15)\n"
               "                                               students-per-teacher, census only, by school type\n"
               "                                               (none, college, high, middle, elem, daycare)\n"
               "  symptomatic_withdraw_compliance_day_0  (0.3, 0.3, 0.3, 0.3, 0.3, 0.3)\n"
               "  symptomatic_withdraw_compliance_day_1  (0.8, 0.6, 0.5, 0.5, 0.5, 0.5)\n"
               "  symptomatic_withdraw_compliance_day_2  (0.9, 0.8, 0.7, 0.7, 0.7, 0.7)\n"
               "                                               symptomatic-withdrawal compliance rate on the 1st/2nd/3rd+ day\n"
               "                                               of symptoms, by age group (u5, 5-17, 18-29, 30-49, 50-64, 65+)\n"
               "\n"
               "Other \"agent.*\" parameters:\n"
               "  diag.output_filename            (output.dat, or output_<disease>.dat for multiple diseases)\n"
               "                                               output filename(s)\n"
               "\n"
               "Recognized \"disease.*\" parameters, common to all diseases (name, default, description).\n"
               "Every parameter below may also be overridden per-disease as \"disease_<disease_name>.*\"\n"
               "(disease_name is one of agent.disease_names), which takes precedence over \"disease.*\":\n"
               "  initial_case_type               (random)     how initial cases are seeded: \"random\" or \"file\"\n"
               "  case_filename                   (required if initial_case_type=file)\n"
               "                                               initial cases file: FIPS code, current cases, cumulative cases\n"
               "  num_initial_cases               (0)          number of initial cases (if initial_case_type=random)\n"
               "  xmit_comm                       (0.000021, 0.000062, 0.000165, 0.000165, 0.000165, 0.000247)\n"
               "                                               community transmission prob, by age group of receiver\n"
               "  xmit_comm_scale                 (1.0)        overall magnitude of community transmission (population-size-\n"
               "                                               scaled, urbanpop only; does not affect xmit_hood)\n"
               "  xmit_hood                       (0.000082, 0.000247, 0.00066, 0.00066, 0.00066, 0.001)\n"
               "                                               neighborhood transmission prob, by age group of receiver\n"
               "  xmit_hh_adult                   (0.3, 0.3, 0.4, 0.4, 0.4, 0.4)\n"
               "                                               within-household transmission prob, transmitter is an adult\n"
               "  xmit_hh_child                   (0.6, 0.6, 0.3, 0.3, 0.3, 0.3)\n"
               "                                               within-household transmission prob, transmitter is a child\n"
               "  xmit_nc_adult                   (0.132, 0.132, 0.165, 0.165, 0.165, 0.165)\n"
               "                                               neighborhood-cluster transmission prob, transmitter is an adult\n"
               "  xmit_nc_child                   (0.25, 0.25, 0.132, 0.132, 0.132, 0.132)\n"
               "                                               neighborhood-cluster transmission prob, transmitter is a child\n"
               "  xmit_school                     (0, 0.0002, 0.0024, 0.0027, 0.0088, 0.0125)\n"
               "                                               child-to-child school transmission prob, by school type\n"
               "                                               (none, college, high, middle, elem, daycare)\n"
               "  xmit_school_a2c                 (0, 0.001, 0.01, 0.013, 0.043, 0.025)\n"
               "                                               adult-to-child school transmission prob, by school type\n"
               "  xmit_school_c2a                 (0, 0.001, 0.01, 0.013, 0.043, 0.025)\n"
               "                                               child-to-adult school transmission prob, by school type\n"
               "  xmit_school_scale               (1.0)        overall scale factor applied to all xmit_school* values\n"
               "  xmit_work                       (0.0575)     workgroup transmission prob\n"
               "  p_trans                         (0.20)       probability of transmission given contact\n"
               "  p_asymp                         (0.30)       fraction of cases that are asymptomatic\n"
               "  asymp_relative_inf              (0.7)        relative infectiousness of asymptomatic individuals\n"
               "  vac_eff                         (0.0)        vaccine efficacy (unsupported; must be 0)\n"
               "  compare_to_epicast              (false)      sample latent/incubation/infectious periods from Epicast's\n"
               "                                               fixed CDFs instead of the Gamma distributions below\n"
               "  latent_length_alpha             (6.0)        Gamma distribution alpha for latent period length\n"
               "  latent_length_beta              (0.73)       Gamma distribution beta for latent period length\n"
               "  infectious_length_alpha         (3.54)       Gamma distribution alpha for infectious period length\n"
               "  infectious_length_beta          (1.22)       Gamma distribution beta for infectious period length\n"
               "  infectious_length_loc           (2.5)        location (shift) for infectious period length\n"
               "  incubation_length_alpha         (6.0)        Gamma distribution alpha for incubation period length\n"
               "  incubation_length_beta          (0.73)       Gamma distribution beta for incubation period length\n"
               "  incubation_length_loc           (1.0)        location (shift) for incubation period length\n"
               "  hospital_delay_length_alpha     (0.1)        Gamma distribution alpha for hospital admission delay\n"
               "  hospital_delay_length_beta      (0.1)        Gamma distribution beta for hospital admission delay\n"
               "  hospital_delay_length_loc       (1.0)        location (shift) for hospital admission delay\n"
               "  immune_length_alpha             (1.0)        Gamma distribution alpha for immunity period length\n"
               "  immune_length_beta              (1.0)        Gamma distribution beta for immunity period length\n"
               "  immune_length_loc               (10000.0)    location (shift) for immunity period length\n"
               "  hospital_stay_type              (constant)   hospital stay length model: \"constant\" or \"random\"\n"
               "  hospitalization_days            (3, 3, 3, 3, 8, 7)\n"
               "                                               hospital stay length in days, by age group\n"
               "                                               (used if hospital_stay_type=constant)\n"
               "  hospitalization_days_alpha      (1.33, 1.33, 1.33, 1.33, 0.5, 0.57)\n"
               "  hospitalization_days_beta       (1.5, 1.5, 1.5, 1.5, 4.0, 3.5)\n"
               "                                               Gamma distribution alpha/beta for hospital stay length,\n"
               "                                               by age group (used if hospital_stay_type=random)\n"
               "  CHR                             (.0104, .0104, .070, .28, .28, 1.0)\n"
               "                                               symptomatic -> hospitalized probability, by age group\n"
               "  CIC                             (.24, .24, .24, .36, .36, .35)\n"
               "                                               hospitalized -> ICU probability, by age group\n"
               "  CVE                             (.12, .12, .12, .22, .22, .22)\n"
               "                                               ICU -> ventilator probability, by age group\n"
               "  hospCVF                         (0, 0, 0, 0, 0, 0)\n"
               "                                               probability of dying in hospital (non-ICU), by age group\n"
               "  icuCVF                          (0, 0, 0, 0, 0, 0.26)\n"
               "                                               probability of dying in ICU (non-ventilated), by age group\n"
               "  ventCVF                         (0.20, 0.20, 0.20, 0.45, 0.45, 1.0)\n"
               "                                               probability of dying while ventilated, by age group\n"
               "\n"
               "Recognized \"disease_coupling.*\" parameters (only relevant if number_of_diseases > 1):\n"
               "  coimmunity_matrix               (identity: 1.0 on diagonal, 0.0 off-diagonal)\n"
               "                                               number_of_diseases x number_of_diseases row-major matrix;\n"
               "                                               immunity to disease j conferred by recovery from disease i\n"
               "  cosusceptibility_matrix         (all 1.0)    number_of_diseases x number_of_diseases row-major matrix;\n"
               "                                               susceptibility to disease j while currently infected with i\n";
}

/*! \brief Set ExaEpi-specific defaults for memory-management and tiling */
void overrideAmrexDefaults () {
    amrex::ParmParse pp("amrex");
    // ExaEpi should not require managed memory in the Arena.
    bool the_arena_is_managed = false;
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

    int my_rank;
#ifdef AMREX_USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#else
    my_rank = 0;
#endif

    if (argc < 2) {
        if (my_rank == 0) { printHelp(argv[0]); }
#ifdef AMREX_USE_MPI
        MPI_Finalize();
#endif
        return 0;
    }

    if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        if (my_rank == 0) { printHelp(argv[0]); }
#ifdef AMREX_USE_MPI
        MPI_Finalize();
#endif
        return 0;
    }

    if (std::string(argv[1]) == "--version") {
        if (my_rank == 0) {
            std::cout << "AMReX version " << amrex::Version() << "\n";
            std::cout << "ExaEpi version " << EXAEPI_VERSION << " (built on " << __DATE__ << ")\n";
        }
#ifdef AMREX_USE_MPI
        MPI_Finalize();
#endif
        return 0;
    }

    amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, overrideAmrexDefaults);

    Print() << "ExaEpi version " << EXAEPI_VERSION << " (built on " << __DATE__ << ")\n";

    runAgent();

    amrex::Finalize();

#ifdef AMREX_USE_MPI
    MPI_Finalize();
#endif
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

    WeatherData wd;
    if (params.weather_int > 0) { wd.readDataFromFile(params.weather_filename); }

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
                Vector<string> headers = {"Day",   "Su",   "PS/PI", "S/PI/NH", "S/PI/H", "PS/I", "S/I/NH",
                                          "S/I/H", "A/PI", "A/I",   "H/NI",    "H/I",    "ICU",  "V",
                                          "R",     "D",    "NewI",  "NewS",    "NewH",   "NewA", "NewP"};
                if (params.context_diag) {
                    headers.insert(headers.end(),
                                   {"EWork", "EHosp", "ESchool", "ENbhD", "ECommD", "EHH", "ENC", "ENbhN", "ECommN"});
                }
                for (const auto& header : headers) {
                    File << std::setw(header == "Day" ? 5 : 12) << header;
                }
                Vector<string> age_headers = {"U5", "5to17", "18to29", "30to49", "50to64", "O64"};
                for (const auto& header : age_headers) {
                    File << std::setw(12) << "Symp" + header;
                }
                for (const auto& header : age_headers) {
                    File << std::setw(12) << "Hosp" + header;
                }
                File << "\n";

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

#ifdef AMREX_DEBUG
            //  dump a text file of the initial agent fields for debugging purposes
            string agents_fname =
                    std::string("initial_agents.") + (params.ic_type == ICType::UrbanPop ? "urbanpop" : "census") + ".csv";
            pc.WriteAsciiFile(agents_fname);
            if (ParallelDescriptor::IOProcessor()) {
                std::ofstream agents_f(agents_fname, std::ios_base::app);
                agents_f << "#posx posy id cpu " << "treatment_timer " << "disease_counter " << "prob " << "latent_period "
                         << "infectious_period " << "incubation_period " << "hospital_delay " << "age_group " << "family "
                         << "home_i " << "home_j " << "work_i " << "work_j " << "hosp_i " << "hosp_j " << "trav_i " << "trav_j "
                         << "nborhood " << "hh_cluster " << "school_grade " << "school_id " << "school_closed " << "naics "
                         << "workgroup " << "work_nborhood " << "withdrawn " << "random_travel " << "air_travel " << "status "
                         << "symptomatic\n";
                agents_f.close();
            }
#endif

            // Build a per-unit (county) population-weighted cumulative distribution over
            // communities, so that index cases are drawn independently and proportional to each
            // community's population -- matching Epicast's scheme of giving every susceptible
            // agent in a county equal probability of being seeded, rather than concentrating
            // batches of cases in a single, uniformly-chosen community. For the deprecated Census
            // path, a uniform placeholder reproduces the previous (uniform) seeding behavior.
            const Vector<int>& unit_community_start_seed =
                    (params.ic_type == ICType::Census ? censusData.demo.Start : urbanPopData.fips_community_start);
            Vector<int> community_population;
            if (params.ic_type == ICType::UrbanPop) {
                community_population.resize(urbanPopData.block_groups.size(), 0);
                for (int c = 0; c < (int)urbanPopData.block_groups.size(); ++c) {
                    community_population[c] = urbanPopData.block_groups[c].home_population;
                }
            } else {
                community_population.assign(censusData.demo.Ncommunity, 1);
            }
            Vector<float> community_cum_prob =
                    ExaEpi::Initialization::buildCommunityCumProb(unit_community_start_seed, community_population);

            for (int d = 0; d < params.num_diseases; d++) {
                auto disease_params = pc.getDiseaseParameters_h(d);
                if (disease_params->initial_case_type == CaseTypes::file) {
                    CaseData cases;
                    cases.initFromFile(disease_params->disease_name, std::string(disease_params->case_filename));
                    setInitialCasesFromFile(pc, cases, disease_params->disease_name, d,
                                            (params.ic_type == ICType::Census ? censusData.demo.FIPS : urbanPopData.FIPS_codes),
                                            unit_community_start_seed, community_cum_prob,
                                            (params.ic_type == ICType::Census ? censusData.comm_mf : urbanPopData.community_mf),
                                            params.fast);
                } else {
                    setInitialCasesRandom(pc, disease_params->num_initial_cases, disease_params->disease_name, d,
                                          (params.ic_type == ICType::Census ? censusData.demo.FIPS : urbanPopData.FIPS_codes),
                                          unit_community_start_seed, community_cum_prob,
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
            if (params.ic_type == ICType::Census) {
                IO::readCheckpointFile(params.restart_chkfile, pc, disease_stats, &(censusData.unit_mf), &(censusData.FIPS_mf),
                                       &(censusData.comm_mf), cur_time, start_day);
            } else {
                IO::readCheckpointFile(params.restart_chkfile, pc, disease_stats, nullptr, &(urbanPopData.geoid_mf),
                                       &(urbanPopData.community_mf), cur_time, start_day);
            }
        }

        // Populate pc.comm_density_scale from the population-size scale. Done after both the
        // fresh-init and restart branches above (urbanPopData.community_mf is valid either way), so
        // restarted runs get the same scaling as a fresh run rather than silently reverting to flat
        // 1.0.
        if (params.ic_type == ICType::UrbanPop && params.size_scale_enabled) {
            Vector<Real> comm_scale = computeCommunitySizeScale(urbanPopData.block_groups, params);
            Gpu::DeviceVector<Real> comm_scale_d(comm_scale.size());
            Gpu::copyAsync(Gpu::hostToDevice, comm_scale.begin(), comm_scale.end(), comm_scale_d.begin());
            Gpu::streamSynchronize();
            auto* comm_scale_ptr = comm_scale_d.data();
            for (MFIter mfi(urbanPopData.community_mf); mfi.isValid(); ++mfi) {
                auto comm_arr = urbanPopData.community_mf.const_array(mfi);
                auto scale_arr = pc.comm_density_scale.array(mfi);
                ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    int c = comm_arr(i, j, k);
                    scale_arr(i, j, k, 0) = (c >= 0) ? comm_scale_ptr[c] : 1.0_rt;
                });
            }
            Gpu::streamSynchronize();
        }

        // Populate pc.comm_density_scale_work from the work-population size scale -- switches on
        // together with the (home-population-based) block above under the same size_scale_enabled
        // flag, since it's one mechanism applied with the covariate appropriate to each time of day.
        // Used only for daytime interactions (see InteractionModComm.H / InteractionModNborhood.H),
        // since home_population isn't a meaningful covariate for the population physically present
        // in a community during the day.
        if (params.ic_type == ICType::UrbanPop && params.size_scale_enabled) {
            Vector<Real> work_scale = computeCommunityWorkSizeScale(urbanPopData.day_population, params);
            Gpu::DeviceVector<Real> work_scale_d(work_scale.size());
            Gpu::copyAsync(Gpu::hostToDevice, work_scale.begin(), work_scale.end(), work_scale_d.begin());
            Gpu::streamSynchronize();
            auto* work_scale_ptr = work_scale_d.data();
            for (MFIter mfi(urbanPopData.community_mf); mfi.isValid(); ++mfi) {
                auto comm_arr = urbanPopData.community_mf.const_array(mfi);
                auto scale_arr = pc.comm_density_scale_work.array(mfi);
                ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    int c = comm_arr(i, j, k);
                    scale_arr(i, j, k, 0) = (c >= 0) ? work_scale_ptr[c] : 1.0_rt;
                });
            }
            Gpu::streamSynchronize();
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

    std::vector<int> step_of_peak(params.num_diseases, 0);
    std::vector<Long> num_infected_peak(params.num_diseases, 0);
    std::vector<Long> cumulative_deaths(params.num_diseases, 0);
    std::vector<Long> cumulative_infected(params.num_diseases, 0);
    for (int d = 0; d < params.num_diseases; d++) {
        auto counts = pc.getTotals(d);
        auto total_infected = totalInfected(counts);
        if (total_infected > num_infected_peak[d]) {
            num_infected_peak[d] = total_infected;
            step_of_peak[d] = 0;
        }
        cumulative_deaths[d] = counts[OutputStatus::D];
        // initial infections seeded before the time loop starts are not caught by NewI below
        cumulative_infected[d] = total_infected;
    }

    const Long total_population = pc.TotalNumberOfParticles();

    Vector<Long> num_infected(params.num_diseases, 0);

    amrex::ParmParse::QueryUnusedInputs();
    date startdate(params.startdate);
    if (params.startdate.size()) {
        if (ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "SIMULATION START DATE ";
            startdate.print();
        }
    }
    int weatherWeekIndex = -1;
    int firstWeatherWeekIndex = -1;
    int daysToWeatherWeekend = -1;
    if (params.weather_int > 0) {
        wd.computeIndex(startdate, weatherWeekIndex, daysToWeatherWeekend);
        if (weatherWeekIndex >= 0) {
            firstWeatherWeekIndex = weatherWeekIndex;
            if (ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "Extracting " << params.nsteps / 7 + 1 << " Weeks of Weather Data \n";
            }
            // extract weather data for the simulation timeframe
            if (params.ic_type == ICType::Census) {
                wd.extractActiveData(censusData.demo, weatherWeekIndex, params.nsteps / 7 + 1);
                pc.initializeWeatherIndex(censusData.unit_mf, &wd.activeWeather);
            } else {
                wd.extractActiveData(urbanPopData, weatherWeekIndex, params.nsteps / 7 + 1);
                pc.initializeWeatherIndex_UrbanPop(urbanPopData.geoid_mf, &wd.activeWeather);
            }
        }
    }

    {
        BL_PROFILE_REGION("Evolution");
        // Per-context expected-infection diagnostics (1-step lag: written on day i+1).
        amrex::Real diag_exp_work = 0.0;
        amrex::Real diag_exp_hosp = 0.0;
        amrex::Real diag_exp_school = 0.0;
        amrex::Real diag_exp_nbhd = 0.0;
        amrex::Real diag_exp_commd = 0.0;
        amrex::Real diag_exp_hh = 0.0;
        amrex::Real diag_exp_nc = 0.0;
        amrex::Real diag_exp_nbhn = 0.0;
        amrex::Real diag_exp_commn = 0.0;
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
            if (weatherWeekIndex >= 0) {
                if ((weatherWeekIndex + 1) < wd.numWeeks) {
                    if ((i - start_day) % 7 == daysToWeatherWeekend) {
                        weatherWeekIndex++;
                        pc.advanceWeatherIndex();
                    }
                }
                pc.setActiveWeatherWeek(weatherWeekIndex - firstWeatherWeekIndex);
            }

            // Update agents' disease status
            pc.updateStatus(disease_stats);

            for (int d = 0; d < params.num_diseases; d++) {
                auto counts = pc.getTotals(d);
                if (totalInfected(counts) > num_infected_peak[d]) {
                    num_infected_peak[d] = totalInfected(counts);
                    step_of_peak[d] = i;
                }
                cumulative_deaths[d] = counts[OutputStatus::D];
                cumulative_infected[d] += counts[OutputStatus::NewI];
                num_infected[d] = totalInfected(counts);

                Real mmc[5] = {0, 0, 0, 0, 0};
#ifdef AMREX_USE_GPU
                if (Gpu::inLaunchRegion()) {
                    auto const& ma = disease_stats[d]->const_arrays();
                    GpuTuple<Real, Real, Real, Real, Real> mm =
                            ParReduce(TypeList<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum>{},
                                      TypeList<Real, Real, Real, Real, Real>{}, *(disease_stats[d]), IntVect(0, 0),
                                      [=] AMREX_GPU_DEVICE (int box_no, int ii, int jj,
                                                           int kk) noexcept -> GpuTuple<Real, Real, Real, Real, Real> {
                                          return {ma[box_no](ii, jj, kk, 0), ma[box_no](ii, jj, kk, 1), ma[box_no](ii, jj, kk, 2),
                                                  ma[box_no](ii, jj, kk, 3), ma[box_no](ii, jj, kk, 4)};
                                      });
                    mmc[0] = amrex::get<0>(mm);
                    mmc[1] = amrex::get<1>(mm);
                    mmc[2] = amrex::get<2>(mm);
                    mmc[3] = amrex::get<3>(mm);
                    mmc[4] = amrex::get<4>(mm);
                } else
#endif
                {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (!system::regtest_reduction) reduction(+ : mmc[ : 5])
#endif
                    for (MFIter mfi(*(disease_stats[d])); mfi.isValid(); ++mfi) {
                        Box const& bx = mfi.tilebox();
                        auto const& dfab = disease_stats[d]->const_array(mfi);
                        AMREX_LOOP_3D(bx, ii, jj, kk, {
                            mmc[0] += dfab(ii, jj, kk, 0);
                            mmc[1] += dfab(ii, jj, kk, 1);
                            mmc[2] += dfab(ii, jj, kk, 2);
                            mmc[3] += dfab(ii, jj, kk, 3);
                            mmc[4] += dfab(ii, jj, kk, 4);
                        });
                    }
                }

                ParallelDescriptor::ReduceRealSum(&mmc[0], 5, ParallelDescriptor::IOProcessorNumber());

                auto symp_age_counts = pc.getNewStatusByAge(d, OutputStatus::NewS);
                auto hosp_age_counts = pc.getNewStatusByAge(d, OutputStatus::NewH);

                if (ParallelDescriptor::IOProcessor()) {
                    // total number of deaths computed on agents and on mesh should be the same...
                    if (mmc[3] != counts[OutputStatus::D]) {
                        amrex::Print() << "ERROR in death counts: " << mmc[3] << " != " << counts[OutputStatus::D] << "\n";
                    }
                    AMREX_ALWAYS_ASSERT(mmc[3] == counts[OutputStatus::D]);

                    // the total number of infected should equal the sum of
                    //     those that are hospitalized
                    //     exposed but not infectious
                    //     infectious and asymptomatic
                    //     infectious and pre-symptomatic
                    //     infectious and symptomatic
                    // AMREX_ALWAYS_ASSERT(counts[1] == mmc[0] + counts[5] + counts[6] + counts[7] + counts[8]);

                    std::ofstream File;
                    File.open(output_filename[d].c_str(), std::ios::out | std::ios::app);

                    if (!File.good()) { amrex::FileOpenFailed(output_filename[d]); }

                    File << std::setw(5) << i;
                    for (int j = 0; j < OutputStatus::ICU; ++j) {
                        AMREX_ALWAYS_ASSERT(counts[j] >= 0);
                        File << std::setw(11) << counts[j];
                    }

                    AMREX_ALWAYS_ASSERT(mmc[1] >= 0);
                    AMREX_ALWAYS_ASSERT(mmc[2] >= 0);
                    File << std::setw(11) << mmc[1];
                    File << std::setw(11) << mmc[2];
                    for (int j = OutputStatus::R; j < OutputStatus::nattribs; ++j) {
                        AMREX_ALWAYS_ASSERT(counts[j] >= 0);
                        File << std::setw(11) << counts[j];
                    }
                    if (params.context_diag) {
                        File << std::setw(12) << std::fixed << std::setprecision(1) << diag_exp_work;
                        File << std::setw(12) << std::fixed << std::setprecision(1) << diag_exp_hosp;
                        File << std::setw(12) << std::fixed << std::setprecision(1) << diag_exp_school;
                        File << std::setw(12) << std::fixed << std::setprecision(1) << diag_exp_nbhd;
                        File << std::setw(12) << std::fixed << std::setprecision(1) << diag_exp_commd;
                        File << std::setw(12) << std::fixed << std::setprecision(1) << diag_exp_hh;
                        File << std::setw(12) << std::fixed << std::setprecision(1) << diag_exp_nc;
                        File << std::setw(12) << std::fixed << std::setprecision(1) << diag_exp_nbhn;
                        File << std::setw(12) << std::fixed << std::setprecision(1) << diag_exp_commn;
                    }
                    for (int j = 0; j < AgeGroups::total; j++) {
                        File << std::setw(12) << symp_age_counts[j];
                    }
                    for (int j = 0; j < AgeGroups::total; j++) {
                        File << std::setw(12) << hosp_age_counts[j];
                    }
                    // File << std::setw(12) << mmc[4];

                    File << "\n";

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

            using InteractFn = void (AgentContainer::*)(amrex::MultiFab&);
            auto interact = [&] (InteractFn fn, amrex::Real& diag) {
                if (params.context_diag) { pc.snapshotProbs(0); }
                (pc.*fn)(mask_behavior);
                if (params.context_diag) { diag = pc.sumContextInfections(0); }
            };

            pc.morningCommute(mask_behavior);

            // Split each (community, school_id, grade) group into fixed-size classes (see
            // AgentContainer::assignSchoolClasses). Only needs to happen once, on a fresh start:
            // IntIdx::school_class/school_class_group are persistent, checkpointed attributes, so a
            // restart must keep whatever classes the original run assigned rather than redrawing them.
            if (i == start_day && params.restart_chkfile.empty()) { pc.assignSchoolClasses(params); }

            interact(&AgentContainer::interactWork, diag_exp_work);
            interact(&AgentContainer::interactHospital, diag_exp_hosp);
            interact(&AgentContainer::interactSchool, diag_exp_school);
            interact(&AgentContainer::interactNborhoodDay, diag_exp_nbhd);
            interact(&AgentContainer::interactCommDay, diag_exp_commd);
            pc.eveningCommute(mask_behavior);
            pc.interactEvening(mask_behavior);
            interact(&AgentContainer::interactHH, diag_exp_hh);
            interact(&AgentContainer::interactNC, diag_exp_nc);
            interact(&AgentContainer::interactNborhoodNight, diag_exp_nbhn);
            interact(&AgentContainer::interactCommNight, diag_exp_commn);

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

            // early exit if no more spreading or deaths can occur
            if (num_infected[0] == 0) { break; }
        }
    }

    if (params.num_diseases == 1) {
        amrex::Print() << "\n \n";
        amrex::Print() << "Peak number of infected: " << num_infected_peak[0] << "\n";
        amrex::Print() << "Day of peak: " << step_of_peak[0] << "\n";
        amrex::Print() << "Cumulative deaths: " << cumulative_deaths[0] << "\n";
        amrex::Print() << "Cumulative infected: " << cumulative_infected[0] << " (attack rate "
                        << std::fixed << std::setprecision(2)
                        << (total_population > 0 ? 100.0 * (double)cumulative_infected[0] / (double)total_population : 0.0)
                        << "%)\n";
        amrex::Print() << "\n \n";
    } else {
        amrex::Print() << "\n \n";
        for (int d = 0; d < params.num_diseases; d++) {
            amrex::Print() << "Disease " << params.disease_names[d] << ":\n";
            amrex::Print() << "    Peak number of infected: " << num_infected_peak[d] << "\n";
            amrex::Print() << "    Day of peak: " << step_of_peak[d] << "\n";
            amrex::Print() << "    Cumulative deaths: " << cumulative_deaths[d] << "\n";
            amrex::Print() << "    Cumulative infected: " << cumulative_infected[d] << " (attack rate "
                            << std::fixed << std::setprecision(2)
                            << (total_population > 0 ? 100.0 * (double)cumulative_infected[d] / (double)total_population : 0.0)
                            << "%)\n";
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
