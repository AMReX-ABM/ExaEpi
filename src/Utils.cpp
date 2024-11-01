/*! @file Utils.cpp
    \brief Contains function implementations for the #ExaEpi::Utils namespace
*/

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_CoordSys.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParmParse.H>
#include <AMReX_RealBox.H>


#include "DemographicData.H"
#include "Utils.H"

#include <cmath>
#include <string>

using namespace amrex;
using namespace ExaEpi;

/*! \brief Read in test parameters in #ExaEpi::TestParams from input file */
void ExaEpi::Utils::get_test_params (   TestParams& params,         /*!< Test parameters */
                                        const std::string& prefix   /*!< ParmParse prefix */ )
{
    ParmParse pp(prefix);

    pp.query("nsteps", params.nsteps);
    pp.query("plot_int", params.plot_int);
    pp.query("random_travel_int", params.random_travel_int);
    pp.query("random_travel_prob", params.random_travel_prob);
    pp.query("air_travel_int", params.air_travel_int);
    pp.query("number_of_diseases", params.num_diseases);

    params.disease_names.resize(params.num_diseases);
    for (int d = 0; d < params.num_diseases; d++) {
        params.disease_names[d] = amrex::Concatenate("default", d, 2);
    }
    pp.queryarr("disease_names", params.disease_names,0,params.num_diseases);

    params.initial_case_type.resize(params.num_diseases);
    params.num_initial_cases.resize(params.num_diseases);
    params.case_filename.resize(params.num_diseases);

    std::string ic_type = "census";
    pp.query( "ic_type", ic_type );
    if (ic_type == "census") {
        params.ic_type = ICType::Census;
        pp.get("census_filename", params.census_filename);
        pp.get("workerflow_filename", params.workerflow_filename);
        params.max_grid_size = 16;
    } else if (ic_type == "urbanpop") {
        params.ic_type = ICType::UrbanPop;
        pp.get("urbanpop_filename", params.urbanpop_filename);
#ifdef AMREX_USE_CUDA
        params.max_grid_size = 500;
#else
        params.max_grid_size = 100;
#endif
    } else {
        amrex::Abort("ic_type not recognized (currently supported 'census')");
    }

    pp.query("max_grid_size", params.max_grid_size);

    for (int i = 0; i < params.num_diseases; i++) {
        // defaults
        params.initial_case_type[i] = "random";
        params.num_initial_cases[i] = 0;
    }
    pp.queryarr("initial_case_type", params.initial_case_type, 0, params.num_diseases);
    if (params.num_diseases == 1) {
        if (params.initial_case_type[0] == "file") {
            if (pp.contains("case_filename")) {
                pp.get("case_filename", params.case_filename[0]);
            } else {
                std::string key = "case_filename_" + params.disease_names[0];
                if (pp.contains(key.c_str())) pp.get(key.c_str(), params.case_filename[0]);
            }
        } else if (params.initial_case_type[0] == "random") {
            if (pp.contains("num_initial_cases")) {
                pp.get("num_initial_cases", params.num_initial_cases[0]);
            } else {
                std::string key = "num_initial_cases_" + params.disease_names[0];
                if (pp.contains(key.c_str())) pp.get(key.c_str(), params.num_initial_cases[0]);
            }
        } else {
            amrex::Abort("initial case type not recognized");
        }
    } else {
        for (int d = 0; d < params.num_diseases; d++) {
            if (params.initial_case_type[d] == "file") {
                std::string key = "case_filename_" + params.disease_names[d];
                if (pp.contains(key.c_str())) pp.get(key.c_str(), params.case_filename[d]);
            } else if (params.initial_case_type[d] == "random") {
                std::string key = "num_initial_cases_" + params.disease_names[d];
                if (pp.contains(key.c_str())) pp.get(key.c_str(), params.num_initial_cases[d]);
            } else {
                amrex::Abort("initial case type not recognized");
            }
        }
    }

    pp.query("aggregated_diag_int", params.aggregated_diag_int);
    if (params.aggregated_diag_int >= 0) {
        params.aggregated_diag_prefix = "cases";
        pp.get("aggregated_diag_prefix", params.aggregated_diag_prefix);
    }

    pp.query("shelter_start",  params.shelter_start);
    pp.query("shelter_length", params.shelter_length);

    pp.query("nborhood_size", params.nborhood_size);
    pp.query("workgroup_size", params.workgroup_size);

    Long seed = 0;
    bool reset_seed = pp.query("seed", seed);
    if (reset_seed) {
        ULong gpu_seed = (ULong) seed;
        ULong cpu_seed = (ULong) seed;
        amrex::ResetRandomSeed(cpu_seed, gpu_seed);
    }

    pp.query("fast", params.fast);
}

