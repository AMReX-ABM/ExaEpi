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

    std::string ic_type = "census";
    pp.query( "ic_type", ic_type );
    if (ic_type == "census") {
        params.ic_type = ICType::Census;
        pp.get("census_filename", params.census_filename);
        pp.get("workerflow_filename", params.workerflow_filename);
        if (params.air_travel_int > 0) {
            pp.get("air_traffic_filename", params.air_traffic_filename);
            pp.get("airports_filename", params.airports_filename);
        }
        params.max_box_size = 16;
    } else if (ic_type == "urbanpop") {
        params.ic_type = ICType::UrbanPop;
        pp.get("urbanpop_filename", params.urbanpop_filename);
#ifdef AMREX_USE_CUDA
        params.max_box_size = 500;
#else
        params.max_box_size = 100;
#endif
    } else {
        amrex::Abort("ic_type not recognized (currently supported 'census')");
    }

    pp.query("max_box_size", params.max_box_size);

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

