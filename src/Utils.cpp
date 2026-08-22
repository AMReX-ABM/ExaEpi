/*! @file Utils.cpp
    \brief Contains function implementations for the #ExaEpi::Utils namespace
*/

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_CoordSys.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>

#include "DemographicData.H"
#include "Utils.H"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <string>

using namespace amrex;
using namespace ExaEpi;

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
    pp.query("weather_int", params.weather_int);
    pp.query("startdate", params.startdate);

    params.disease_names.resize(params.num_diseases);
    for (int d = 0; d < params.num_diseases; d++) {
        params.disease_names[d] = amrex::Concatenate("default", d, 2);
    }
    pp.queryarr("disease_names", params.disease_names, 0, params.num_diseases);

    if (params.weather_int > 0) { pp.get("weather_filename", params.weather_filename); }

    std::string ic_type = "urbanpop";
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
        params.max_box_size = 16;

        pp.query("workgroup_size_filename", params.workgroup_size_filename);

        pp.query("size_scale_enabled", params.size_scale_enabled);
        ParmParse pp_disease("disease");
        pp_disease.query("xmit_comm_scale", params.xmit_comm_scale);
    } else {
        amrex::Abort("ic_type not recognized (currently supported 'census')");
    }

    pp.query("school_class_size", params.school_class_size);
    pp.query("school_class_size_min", params.school_class_size_min);
    pp.query("school_class_size_max", params.school_class_size_max);

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
    pp.query("context_diag", params.context_diag);
}

void ExaEpi::Utils::printHistogram (const std::string& label, const std::map<Long, Long>& value_counts,
                                    int max_distinct_buckets, int max_bar_width, Long bin_width) {
    if (!ParallelDescriptor::IOProcessor()) { return; }

    if (value_counts.empty()) {
        Print() << label << " histogram: no data\n";
        return;
    }

    Long total = 0;
    Long sum_val = 0;
    for (auto& kv : value_counts) {
        total += kv.second;
        sum_val += kv.first * kv.second;
    }
    Real mean = (Real)sum_val / (Real)total;

    // weighted median: average of the values at the two middle observation positions (1-indexed);
    // these coincide for an odd total, giving the usual single middle value
    Long pos_lo = (total + 1) / 2;
    Long pos_hi = (total % 2 == 0) ? (total / 2 + 1) : pos_lo;
    Long median_lo = value_counts.begin()->first, median_hi = median_lo;
    Long cum = 0;
    bool found_lo = false, found_hi = false;
    for (auto& kv : value_counts) {
        cum += kv.second;
        if (!found_lo && cum >= pos_lo) { median_lo = kv.first; found_lo = true; }
        if (!found_hi && cum >= pos_hi) { median_hi = kv.first; found_hi = true; }
    }
    Real median = (Real)(median_lo + median_hi) / 2.0_rt;

    Long min_val = value_counts.begin()->first;
    Long max_val = value_counts.rbegin()->first;
    Long range = max_val - min_val + 1;

    int nbuckets;
    Real auto_width = 1.0_rt;
    Long lo_aligned = min_val;
    if (bin_width > 0) {
        // fixed-width bins requested by the caller, aligned to a multiple of bin_width so
        // buckets look natural (e.g. 0-4, 5-9, ...) rather than starting at an arbitrary min_val
        lo_aligned = (min_val / bin_width) * bin_width;
        nbuckets = (int)((max_val - lo_aligned) / bin_width) + 1;
    } else {
        // one bucket per distinct integer value in [min_val, max_val], or max_distinct_buckets
        // equal-width range bins spanning the same interval, whichever is narrower
        nbuckets = (range <= (Long)max_distinct_buckets) ? (int)range : max_distinct_buckets;
        auto_width = (Real)range / (Real)nbuckets;
    }

    Vector<Long> bucket_lo(nbuckets), bucket_hi(nbuckets), bucket_counts(nbuckets, 0);
    for (int b = 0; b < nbuckets; ++b) {
        if (bin_width > 0) {
            bucket_lo[b] = lo_aligned + (Long)b * bin_width;
            bucket_hi[b] = bucket_lo[b] + bin_width - 1;
        } else {
            bucket_lo[b] = min_val + (Long)std::floor(b * auto_width);
            bucket_hi[b] = (b == nbuckets - 1) ? max_val : (min_val + (Long)std::floor((b + 1) * auto_width) - 1);
        }
    }
    for (auto& kv : value_counts) {
        int b = (bin_width > 0) ? (int)((kv.first - lo_aligned) / bin_width)
                                 : std::min(nbuckets - 1, (int)std::floor((Real)(kv.first - min_val) / auto_width));
        bucket_counts[b] += kv.second;
    }

    Long max_count = *std::max_element(bucket_counts.begin(), bucket_counts.end());
    Long scale = std::max((Long)1, (Long)std::ceil((Real)max_count / (Real)max_bar_width));

    Print() << label << " histogram (" << total << " observations, mean=" << std::fixed << std::setprecision(2) << mean
            << ", median=" << median;
    if (scale > 1) { Print() << ", each * = " << scale; }
    Print() << "):\n";
    // collapse runs of 2+ consecutive empty buckets into a single "..." gap line noting the
    // skipped value range, rather than printing every empty bucket -- a lone empty bucket is
    // still printed normally (not worth collapsing, and keeps the shape continuous)
    for (int b = 0; b < nbuckets; ++b) {
        if (bucket_counts[b] == 0) {
            int run_end = b;
            while (run_end < nbuckets && bucket_counts[run_end] == 0) { run_end++; }
            if (run_end - b >= 2) {
                Print() << "  " << std::setw(12) << "..." << " | " << std::setw(9) << 0 << " (" << (run_end - b)
                        << " empty buckets, " << bucket_lo[b] << "-" << bucket_hi[run_end - 1] << ")\n";
                b = run_end - 1;
                continue;
            }
        }
        std::string value_label = (bucket_lo[b] == bucket_hi[b])
            ? std::to_string(bucket_lo[b])
            : (std::to_string(bucket_lo[b]) + "-" + std::to_string(bucket_hi[b]));
        std::string bar(static_cast<size_t>(bucket_counts[b] / scale), '*');
        Print() << "  " << std::setw(12) << value_label << " | " << std::setw(9) << bucket_counts[b] << " " << bar << "\n";
    }
}

std::map<amrex::Long, amrex::Long> ExaEpi::Utils::gatherHistogramCounts (const std::map<Long, Long>& local_counts) {
    int root = ParallelDescriptor::IOProcessorNumber();
    int nprocs = ParallelDescriptor::NProcs();
    int myproc = ParallelDescriptor::MyProc();

    Vector<Long> local_flat;
    local_flat.reserve(local_counts.size() * 2);
    for (auto& kv : local_counts) {
        local_flat.push_back(kv.first);
        local_flat.push_back(kv.second);
    }
    int local_n = (int)local_flat.size();

    Vector<int> all_n(nprocs, 0);
    ParallelDescriptor::Gather(&local_n, 1, all_n.data(), 1, root);

    Vector<int> disp(nprocs, 0);
    int total_n = 0;
    if (myproc == root) {
        for (int p = 0; p < nprocs; ++p) {
            disp[p] = total_n;
            total_n += all_n[p];
        }
    }
    Vector<Long> all_flat(myproc == root ? total_n : 1);
    ParallelDescriptor::Gatherv(local_flat.data(), local_n, all_flat.data(), all_n, disp, root);

    std::map<Long, Long> merged;
    if (myproc == root) {
        for (int i = 0; i < total_n; i += 2) { merged[all_flat[i]] += all_flat[i + 1]; }
    }
    return merged;
}
