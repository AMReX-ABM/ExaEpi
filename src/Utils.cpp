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
#include <cstdint>

using namespace amrex;
using namespace ExaEpi;

/*! \brief Parses school closure policies from the input file. */
void ExaEpi::Utils::ParseSchoolPolicies(amrex::ParmParse& pp, std::vector<ExaEpi::SchoolPolicy>& policies, int nsteps)
{
    int num_policies = 0;
    pp.query("num_school_policies", num_policies);

    if (num_policies == 0) {
        policies.clear();
        return;
    }

    std::vector<ExaEpi::SchoolPolicy> valid_policies;
    valid_policies.reserve(num_policies);

    for (int i = 0; i < num_policies; ++i) {
        ExaEpi::SchoolPolicy current_policy;
        std::string policy_prefix = "school_policy_" + std::to_string(i);

        if (!pp.query((policy_prefix + ".schedule_pattern").c_str(), current_policy.schedule_pattern) || current_policy.schedule_pattern.empty()) {
            amrex::Print() << "WARNING: Policy " << i << " is missing a 'schedule_pattern'. Skipping this policy.\n";
            continue;
        }

        for (char const &c : current_policy.schedule_pattern) {
            if (c != 'A' && c != 'B' && c != 'C' && c != 'D' && c != '_') {
                std::string err_msg = "Policy " + std::to_string(i) +
                                      " contains an invalid character '" + c +
                                      "' in its schedule_pattern. Only 'A', 'B', 'C', 'D', and '_' are allowed " + 
                                      "(No need to put pattern in quotes).";
                amrex::Abort(err_msg);
            }
        }

        pp.query((policy_prefix + ".school_type").c_str(), current_policy.school_type);
        const bool is_valid_type = (current_policy.school_type > SchoolType::college &&
                                    current_policy.school_type < SchoolType::total);

        if (current_policy.school_type != -1 && !is_valid_type) {
            amrex::Print() << "WARNING: Policy " << i
                            << " has an invalid 'school_type' (" << current_policy.school_type
                            << "). Skipping this policy (ps: College not supported yet).\n";
            continue;
        }

        if (!pp.query((policy_prefix + ".start_day").c_str(), current_policy.start_day))
            current_policy.start_day = 0;
        if (!pp.query((policy_prefix + ".end_day").c_str(), current_policy.end_day))
            current_policy.end_day = nsteps;
        current_policy.start_day = std::max(0, current_policy.start_day);
        current_policy.end_day = std::min(nsteps, current_policy.end_day);

        if (current_policy.start_day >= current_policy.end_day) {
            amrex::Abort("Policy " + std::to_string(i) + " has start_day >= end_day.");
        }

        pp.query((policy_prefix + ".fips").c_str(), current_policy.fips);

        valid_policies.push_back(current_policy);
    }
    policies = std::move(valid_policies);
    amrex::Print() << "Parsed " << policies.size() << " school policies.\n";
}

/*! \brief Read in test parameters in #ExaEpi::TestParams from input file */
void ExaEpi::Utils::getTestParams (TestParams& params, /*!< Test parameters */
                                   const std::string& prefix /*!< ParmParse prefix */) {
    ParmParse pp(prefix);

    pp.query("nsteps", params.nsteps);
    pp.query("plot_int", params.plot_int);
    pp.query("check_int", params.check_int);
    pp.query("random_travel_int", params.random_travel_int);
    pp.query("random_travel_prob", params.random_travel_prob);
    pp.query("air_travel_int", params.air_travel_int);
    pp.query("number_of_diseases", params.num_diseases);

    params.disease_names.resize(params.num_diseases);
    for (int d = 0; d < params.num_diseases; d++) {
        params.disease_names[d] = amrex::Concatenate("default", d, 2);
    }
    pp.queryarr("disease_names", params.disease_names, 0, params.num_diseases);

    std::string ic_type = "census";
    pp.query("ic_type", ic_type);
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

    pp.query("restart", params.restart_chkfile);

    pp.query("shelter_start", params.shelter_start);
    pp.query("shelter_length", params.shelter_length);

    pp.query("nborhood_size", params.nborhood_size);
    pp.query("workgroup_size", params.workgroup_size);

    Long seed = 0;
    bool reset_seed = pp.query("seed", seed);
    if (reset_seed) {
        ULong gpu_seed = (ULong)seed;
        ULong cpu_seed = (ULong)seed;
        amrex::ResetRandomSeed(cpu_seed, gpu_seed);
    }

    pp.query("fast", params.fast);

    pp.query("set_school_closure", params.set_school_closure);

    pp.query("num_school_policies", params.num_school_policies);


    if (params.num_school_policies > 0) {
        ExaEpi::Utils::ParseSchoolPolicies(pp, params.school_policies, params.nsteps);
        params.num_school_policies = params.school_policies.size(); 
    }
}
