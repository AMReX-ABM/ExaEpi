/*! @file IO.cpp
    \brief Contains IO functions in #ExaEpi::IO namespace
*/

#include <AMReX_GpuContainers.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include "IO.H"

#include <array>
#include <vector>

using namespace amrex;

namespace ExaEpi {
namespace IO {

/*! \brief Write plotfile of computational domain with disease spread and geographic data at a
    given step.

    Writes the current disease spread information and geographic data (FIPS code, census tract
    ID, and community number) to a plotfile:
    + Create an output MultiFab (with the same domain and distribution map as the particle container)
      with 5*(number of diseases)+4 components:

      For each disease (0 <= d < n, d being the disease index, n being the number of diseases):
      + component 5*d+0: total
      + component 5*d+1: never infected (#Status::never)
      + component 5*d+2: infected (#Status::infected)
      + component 5*d+3: immune (#Status::immune)
      + component 5*d+4: susceptible (#Status::susceptible)

      Then, for each disease, we write the number of new cases each day at
      + component 5*n+d (d being the disease index and n the number of diseases)

      Then (n being the number of diseases):
      + component 6*n+0: FIPS ID
      + component 6*n+1: census tract number
      + component 6*n+2: community number
    + Get disease spread data (first 7*n components) from AgentContainer::generateCellData() and
    + also the disease_stats multifab, which tracts the number of new cases each day.
    + Copy FIPS code, census tract ID, and community number from the input MultiFabs to
      the remaining components.
    + Write the output MultiFab to file.
    + Write agents to file - see AgentContainer::WritePlotFile().

    \p unit_mf_ptr is always nullptr today (UrbanPop has no separate "unit" concept distinct from
    FIPS); the parameter is kept as a nullable-pointer interface in case a future IC source needs
    it again.
*/
void writePlotFile (const AgentContainer& pc,                      /*!< Agent (particle) container */
                    const MFPtrVec& a_disease_stats,               /*!< Disease stats tracker */
                    const iMultiFab* unit_mf_ptr,                  /*!< MultiFabs to write out */
                    const iMultiFab* FIPS_mf_ptr,                  /*!< MultiFabs to write out */
                    const iMultiFab* comm_mf_ptr,                  /*!< MultiFabs to write out */
                    const int num_diseases,                        /*!< Number of diseases */
                    const std::vector<std::string>& disease_names, /*!< Names of diseases */
                    const Real cur_time,                           /*!< current time */
                    const int step,                                /*!< Current step */
                    const int verbose /*!< at Verbosity::detail, print a message when writing the plotfile */) {
    if (verbose >= ExaEpi::Verbosity::detail) { amrex::Print() << "Writing plotfile \n"; }

    // make sure status_names are in the same order as the struct Status in AgentDefinitions.H
    // these are the names per disease, which do not include "dead", which will be added once at the end of all the diseases
    static const Vector<std::string> status_names = {"total", "never_infected", "infected", "immune", "susceptible"};

    static const int ncomp_d = status_names.size();
    // unit_mf_ptr is always nullptr today (see doc comment above)
    // the +4 (+3) is for new_cases, FIPS, Tract, Unit (new_cases, FIPS, Tract)
    static const int ncomp = ncomp_d * num_diseases + num_diseases + (unit_mf_ptr != nullptr ? 4 : 3);

    MultiFab output_mf(pc.ParticleBoxArray(0), pc.ParticleDistributionMap(0), ncomp, 0);
    output_mf.setVal(0.0);
    pc.generateCellData(output_mf, ncomp_d);

    for (int d = 0; d < num_diseases; d++) {
        amrex::Copy(output_mf, *a_disease_stats[d], DiseaseStats::new_cases, ncomp_d * num_diseases + d, 1, 0);
    }

    amrex::Copy(output_mf, *FIPS_mf_ptr, 0, ncomp_d * num_diseases + num_diseases, 2, 0);
    amrex::Copy(output_mf, *comm_mf_ptr, 0, ncomp_d * num_diseases + num_diseases + 2, 1, 0);
    if (unit_mf_ptr != nullptr) { amrex::Copy(output_mf, *unit_mf_ptr, 0, ncomp_d * num_diseases + num_diseases + 3, 1, 0); }

    {
        Vector<std::string> plt_varnames = {};
        if (num_diseases == 1) {
            for (auto status_name : status_names) {
                plt_varnames.push_back(status_name);
            }
            plt_varnames.push_back("new_cases");
        } else {
            for (int d = 0; d < num_diseases; d++) {
                for (auto status_name : status_names) {
                    plt_varnames.push_back(disease_names[d] + "_" + status_name);
                }
            }
            for (int d = 0; d < num_diseases; d++) {
                plt_varnames.push_back(disease_names[d] + "_new_cases");
            }
        }
        plt_varnames.push_back("FIPS");
        plt_varnames.push_back("Tract");
        plt_varnames.push_back("comm");
        if (unit_mf_ptr != nullptr) { plt_varnames.push_back("unit"); }

        AMREX_ASSERT(plt_varnames.size() == output_mf.nComp());

#ifdef AMREX_USE_HDF5
        WriteSingleLevelPlotfileHDF5MultiDset(amrex::Concatenate("plt", step, 5), output_mf, plt_varnames, pc.ParticleGeom(0),
                                              cur_time, step, "ZLIB@3");
#else
        WriteSingleLevelPlotfile(amrex::Concatenate("plt", step, 5), output_mf, plt_varnames, pc.ParticleGeom(0), cur_time, step);
#endif
    }

    {
        Vector<int> write_real_comp = {}, write_int_comp = {};
        Vector<std::string> real_varnames = {}, int_varnames = {};
        // non-disease-specific attributes
        int_varnames.push_back("age_group");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("family");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("home_i");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("home_j");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("work_i");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("work_j");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("hosp_i");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("hosp_j");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("trav_i");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("trav_j");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("nborhood");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("hh_cluster");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("school_grade");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("school_id");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("school_closed");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("school_class");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("school_class_group");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("naics");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("workgroup");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("work_nborhood");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("withdrawn");
        write_int_comp.push_back(1);
        int_varnames.push_back("random_travel");
        write_int_comp.push_back(1);
        int_varnames.push_back("air_travel");
        write_int_comp.push_back(1);
        int_varnames.push_back("weatherLookup");
        write_int_comp.push_back(1);
        // disease-specific (runtime-added) attributes
        if (num_diseases == 1) {
            real_varnames.push_back("treatment_timer");
            write_real_comp.push_back(1);
            real_varnames.push_back("disease_counter");
            write_real_comp.push_back(1);
            real_varnames.push_back("infection_prob");
            write_real_comp.push_back(1);
            real_varnames.push_back("latent_period");
            write_real_comp.push_back(static_cast<int>(step == 0));
            real_varnames.push_back("infectious_period");
            write_real_comp.push_back(static_cast<int>(step == 0));
            real_varnames.push_back("incubation_period");
            write_real_comp.push_back(static_cast<int>(step == 0));
            real_varnames.push_back("hospital_delay");
            write_real_comp.push_back(static_cast<int>(step == 0));
            real_varnames.push_back("hospital_random");
            write_real_comp.push_back(0);
            int_varnames.push_back("status");
            write_int_comp.push_back(1);
            int_varnames.push_back("symptomatic");
            write_int_comp.push_back(1);
        } else {
            for (int d = 0; d < num_diseases; d++) {
                real_varnames.push_back(disease_names[d] + "treatment_timer");
                write_real_comp.push_back(1);
                real_varnames.push_back(disease_names[d] + "_disease_counter");
                write_real_comp.push_back(1);
                real_varnames.push_back(disease_names[d] + "_infection_prob");
                write_real_comp.push_back(1);
                real_varnames.push_back(disease_names[d] + "_latent_period");
                write_real_comp.push_back(static_cast<int>(step == 0));
                real_varnames.push_back(disease_names[d] + "_infectious_period");
                write_real_comp.push_back(static_cast<int>(step == 0));
                real_varnames.push_back(disease_names[d] + "_incubation_period");
                write_real_comp.push_back(static_cast<int>(step == 0));
                real_varnames.push_back(disease_names[d] + "_hospital_delay");
                write_real_comp.push_back(static_cast<int>(step == 0));
                real_varnames.push_back(disease_names[d] + "_hospital_random");
                write_real_comp.push_back(0);
                int_varnames.push_back(disease_names[d] + "_status");
                write_int_comp.push_back(1);
                int_varnames.push_back(disease_names[d] + "_symptomatic");
                write_int_comp.push_back(1);
            }
        }

#ifdef AMREX_USE_HDF5
        pc.WritePlotFileHDF5(amrex::Concatenate("plt", step, 5), "agents", write_real_comp, write_int_comp, real_varnames,
                             int_varnames, "ZLIB@3");
#else
        pc.WritePlotFile(amrex::Concatenate("plt", step, 5), "agents", write_real_comp, write_int_comp, real_varnames,
                         int_varnames);
#endif
    }
}

void readCheckpointFile (const std::string restart_chkfile, /*!< checkpoint filename */
                         AgentContainer& pc,                /*!< Agent (particle) container */
                         MFPtrVec& a_disease_stats,         /*!< Disease stats tracker */
                         iMultiFab* unit_mf_ptr,            /*!< MultiFabs to write out */
                         iMultiFab* FIPS_mf_ptr,            /*!< MultiFabs to write out */
                         iMultiFab* comm_mf_ptr,            /*!< MultiFabs to write out */
                         Real& cur_time,                    /*!< current time */
                         int& step /*!< Current step */) {
    amrex::Print() << "Restarting from " << restart_chkfile << "\n";
    const std::string level_prefix{"Level_"};
    const int lev = 0;

    // Header
    {
        const std::string File(restart_chkfile + "/ExaEpiHeader");

        const VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

        Vector<char> fileCharPtr;
        ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
        const std::string fileCharPtrString(fileCharPtr.dataPtr());
        std::istringstream is(fileCharPtrString, std::istringstream::in);
        is.exceptions(std::ios_base::failbit | std::ios_base::badbit);

        std::string line, word;

        std::getline(is, line);

        is >> cur_time;
        is >> step;
    }

    if (unit_mf_ptr != nullptr) {
        auto unit = amrex::cast<MultiFab>(*unit_mf_ptr);
        VisMF::Read(unit, amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "unit"));
        *unit_mf_ptr = amrex::cast<iMultiFab>(unit);
    }

    auto fips = amrex::cast<MultiFab>(*FIPS_mf_ptr);
    VisMF::Read(fips, amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "FIPS"));
    *FIPS_mf_ptr = amrex::cast<iMultiFab>(fips);

    auto comm = amrex::cast<MultiFab>(*comm_mf_ptr);
    VisMF::Read(comm, amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "comm"));
    *comm_mf_ptr = amrex::cast<iMultiFab>(comm);

    for (std::size_t i = 0; i < a_disease_stats.size(); ++i) {
        VisMF::Read(*a_disease_stats[i],
                    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "disease_stats_" + std::to_string(i)));
    }

    pc.Restart(restart_chkfile, "agents");

    pc.comm_mf.define(comm_mf_ptr->boxArray(), comm_mf_ptr->DistributionMap(), 1, 0);
    iMultiFab::Copy(pc.comm_mf, *comm_mf_ptr, 0, 0, 1, 0);
}

void writeCheckpointFile (const AgentContainer& pc,                      /*!< Agent (particle) container */
                          const MFPtrVec& a_disease_stats,               /*!< Disease stats tracker */
                          const iMultiFab* unit_mf_ptr,                  /*!< MultiFabs to write out */
                          const iMultiFab* FIPS_mf_ptr,                  /*!< MultiFabs to write out */
                          const iMultiFab* comm_mf_ptr,                  /*!< MultiFabs to write out */
                          const int num_diseases,                        /*!< Number of diseases */
                          const std::vector<std::string>& disease_names, /*!< Names of diseases */
                          const Real cur_time,                           /*!< current time */
                          const int step /*!< Current step */) {

    amrex::Print() << "Writing checkfile \n";

    const int nlev = 1;
    const int lev = 0;
    const std::string& checkpointname = amrex::Concatenate("chk", step, 5);
    const std::string default_level_prefix{"Level_"};

    amrex::PreBuildDirectorHierarchy(checkpointname, default_level_prefix, nlev, true);

    if (ParallelDescriptor::IOProcessor()) {
        VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
        std::ofstream HeaderFile;
        HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
        const std::string HeaderFileName(checkpointname + "/ExaEpiHeader");
        HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
        if (!HeaderFile.good()) { amrex::FileOpenFailed(HeaderFileName); }

        HeaderFile.precision(17);

        HeaderFile << "Checkpoint version: 1\n";

        HeaderFile << cur_time << "\n";

        HeaderFile << step << "\n";
    }

    // write the mesh data
    {
        auto fips = amrex::cast<MultiFab>(*FIPS_mf_ptr);
        VisMF::Write(fips, amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "FIPS"));
        auto comm = amrex::cast<MultiFab>(*comm_mf_ptr);
        VisMF::Write(comm, amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "comm"));
        if (unit_mf_ptr != nullptr) {
            auto unit = amrex::cast<MultiFab>(*unit_mf_ptr);
            VisMF::Write(unit, amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "unit"));
        }
        for (std::size_t i = 0; i < a_disease_stats.size(); ++i) {
            VisMF::Write(*a_disease_stats[i], amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix,
                                                                            "disease_stats_" + std::to_string(i)));
        }
    }

    pc.Checkpoint(checkpointname, "agents");
}

/*! \brief Writes diagnostic data aggregated by block group

    Writes a CSV file per disease (suffixed by disease name if there is more than one disease),
    named <prefix><step, 5 digits>[_<disease_name>], with a header row (GEOID,total,
    never_infected,infected,immune) followed by one row per census block group community giving
    that community's GEOID and its total/never_infected/infected/immune agent counts -- everything
    needed to reconstruct the plotfile's own per-community grid fields, without writing a full
    AMReX plotfile.
    + Creates an output MultiFab of size #UrbanPopData::num_communities x 4 stats
    + Gets the disease status in agents from AgentContainer::generateCellData().
    + On each processor, sets the block-group-th element of each stat's output vector to that
      stat's count in the block group on this processor.
    + Sum across all processors and write GEOID + all 4 stats to file, one community per row.
*/
void writeAggregatedData (const AgentContainer& agents,                  /*!< Agents (particle) container */
                          const UrbanPopData& urbanpopData,              /*!< UrbanPop data */
                          const std::string& prefix,                     /*!< Filename prefix */
                          const int num_diseases,                        /*!< Number of diseases */
                          const std::vector<std::string>& disease_names, /*!< Names of diseases */
                          const int step /*!< Current step */) {
    static const int ncomp_d = 5;
    static const int ncomp = ncomp_d * num_diseases + 4;

    static const int nlevs = std::max(0, agents.finestLevel() + 1);
    std::vector<std::unique_ptr<MultiFab>> mf_vec;
    mf_vec.resize(nlevs);
    for (int lev = 0; lev < nlevs; ++lev) {
        mf_vec[lev] = std::make_unique<MultiFab>(agents.ParticleBoxArray(lev), agents.ParticleDistributionMap(lev), ncomp, 0);
        mf_vec[lev]->setVal(0.0);
        agents.generateCellData(*mf_vec[lev], ncomp_d);
    }

    // Component offsets (within each disease's ncomp_d-sized block) of the 4 stats written out --
    // matches writePlotFile's status_names order {"total","never_infected","infected","immune",
    // "susceptible"}; susceptible is skipped since no diagnostic reads it today.
    static const int n_stats = 4;
    static const std::array<const char*, n_stats> stat_names = {"total", "never_infected", "infected", "immune"};

    const long n_comm = urbanpopData.block_groups.size();

    for (int d = 0; d < num_diseases; d++) {
        amrex::Print() << "Generating diagnostic data by census block group " << "for " << disease_names[d] << "\n";
        std::array<amrex::Gpu::DeviceVector<amrex::Real>, n_stats> d_data;
        amrex::GpuArray<amrex::Real*, n_stats> data_ptr_arr;
        for (int c = 0; c < n_stats; ++c) {
            d_data[c].resize(n_comm, 0.0);
            data_ptr_arr[c] = d_data[c].dataPtr();
        }

        for (int lev = 0; lev < nlevs; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            {
                for (MFIter mfi(*mf_vec[lev]); mfi.isValid(); ++mfi) {
                    auto block_group_indices_arr = urbanpopData.community_mf[mfi].array();
                    auto cell_data_arr = (*mf_vec[lev])[mfi].array();

                    auto bx = mfi.tilebox();
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        int block_group_i = block_group_indices_arr(i, j, k);
                        if (block_group_i == -1) { return; }
                        // This should not require an atomic operation because each block group is at a separate i,j location
                        for (int c = 0; c < n_stats; ++c) {
                            data_ptr_arr[c][block_group_i] = cell_data_arr(i, j, k, ncomp_d * d + c);
                        }
                    });
                }
            }
        }

        std::array<std::vector<amrex::Real>, n_stats> data;
        for (int c = 0; c < n_stats; ++c) {
            // blocking copy from device to host
            data[c].resize(n_comm);
            amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_data[c].begin(), d_data[c].end(), data[c].begin());

            // reduced sum over mpi ranks
            ParallelDescriptor::ReduceRealSum(data[c].data(), data[c].size(), ParallelDescriptor::IOProcessorNumber());
        }

        if (ParallelDescriptor::IOProcessor()) {
            std::string fn = amrex::Concatenate(prefix, step, 5);
            if (num_diseases > 1) { fn += ("_" + disease_names[d]); }
            std::ofstream ofs{fn, std::ofstream::out};

            ofs << "GEOID";
            for (const auto* stat_name : stat_names) {
                ofs << "," << stat_name;
            }
            ofs << "\n";

            ofs << std::fixed << std::setprecision(0);
            for (long ci = 0; ci < n_comm; ++ci) {
                ofs << urbanpopData.block_groups[ci].geoid;
                for (int c = 0; c < n_stats; ++c) {
                    ofs << "," << data[c][ci];
                }
                ofs << "\n";
            }
            ofs.close();
        }
    }
}

/*! \brief Compute PopulationBreakdown from pc's agents' CURRENT positions -- see
    AgentContainer::generatePopulationBreakdown()'s doc comment for the home/work timing
    requirement this relies on. Collapses the 3-component mesh result down to one row per
    community via urbanpopData.community_mf, exactly like writeAggregatedData's own collapse
    above, just with a fixed 3-stat (total/workers/students) layout instead of a per-disease one.
    Result is valid (non-empty) on the IOProcessor only.
*/
PopulationBreakdown computePopulationBreakdownCsvData (const AgentContainer& pc, const UrbanPopData& urbanpopData) {
    static const int n_stats = 3;

    MultiFab mf(pc.ParticleBoxArray(0), pc.ParticleDistributionMap(0), n_stats, 0);
    mf.setVal(0.0);
    pc.generatePopulationBreakdown(mf);

    const long n_comm = urbanpopData.block_groups.size();
    std::array<Gpu::DeviceVector<Real>, n_stats> d_data;
    GpuArray<Real*, n_stats> data_ptr_arr;
    for (int c = 0; c < n_stats; ++c) {
        d_data[c].resize(n_comm, 0.0);
        data_ptr_arr[c] = d_data[c].dataPtr();
    }

    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto block_group_indices_arr = urbanpopData.community_mf[mfi].array();
        auto cell_data_arr = mf[mfi].array();
        auto bx = mfi.tilebox();
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            int bi = block_group_indices_arr(i, j, k);
            if (bi == -1) { return; }
            // Each block group is at a separate i,j location, so no atomic needed.
            for (int c = 0; c < n_stats; ++c) {
                data_ptr_arr[c][bi] = cell_data_arr(i, j, k, c);
            }
        });
    }

    std::array<std::vector<Real>, n_stats> data;
    for (int c = 0; c < n_stats; ++c) {
        data[c].resize(n_comm);
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_data[c].begin(), d_data[c].end(), data[c].begin());
        ParallelDescriptor::ReduceRealSum(data[c].data(), data[c].size(), ParallelDescriptor::IOProcessorNumber());
    }

    PopulationBreakdown result;
    if (ParallelDescriptor::IOProcessor()) {
        result.total.resize(n_comm);
        result.workers.resize(n_comm);
        result.students.resize(n_comm);
        for (long ci = 0; ci < n_comm; ++ci) {
            result.total[ci] = (Long)data[0][ci];
            result.workers[ci] = (Long)data[1][ci];
            result.students[ci] = (Long)data[2][ci];
        }
    }
    return result;
}

/*! \brief Write the static (run-long-constant), once-per-run aggregated diagnostics -- see IO.H
    for the exact file names/formats. IOProcessor only (day/night/groups are only populated there
    to begin with -- see computePopulationBreakdownCsvData / AgentContainer::computeGroupSizeDistributions).
*/
void writeStaticAggregatedData (const PopulationBreakdown& day, const PopulationBreakdown& night,
                                const GroupSizeAggregates& groups, const NborhoodSizeAggregates& nborhoods,
                                const UrbanPopData& urbanpopData, const std::string& prefix) {
    if (!ParallelDescriptor::IOProcessor()) { return; }

    const long n_comm = urbanpopData.block_groups.size();
    {
        std::ofstream ofs{prefix + "_day_night_population.csv", std::ofstream::out};
        ofs << "GEOID,night_total,night_workers,night_students,day_total,day_workers,day_students\n";
        for (long ci = 0; ci < n_comm; ++ci) {
            ofs << urbanpopData.block_groups[ci].geoid << "," << night.total[ci] << "," << night.workers[ci] << ","
                << night.students[ci] << "," << day.total[ci] << "," << day.workers[ci] << "," << day.students[ci] << "\n";
        }
    }

    auto write_counts = [&] (const std::string& suffix, const std::vector<amrex::Long>& counts) {
        std::ofstream ofs{prefix + "_" + suffix + ".txt", std::ofstream::out};
        for (auto c : counts) {
            ofs << c << "\n";
        }
    };
    write_counts("workgroup_sizes", groups.workgroup_sizes);
    write_counts("class_sizes", groups.school_class_sizes);
    write_counts("school_sizes", groups.school_sizes);
    write_counts("nborhood_sizes", nborhoods.nborhood_sizes);
    write_counts("nborhoods_per_community", nborhoods.nborhoods_per_community);
}

} // namespace IO
} // namespace ExaEpi
