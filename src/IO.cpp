/*! @file IO.cpp
    \brief Contains IO functions in #ExaEpi::IO namespace
*/

#include <AMReX_GpuContainers.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>

#include "IO.H"

#include <vector>

using namespace amrex;

namespace ExaEpi {
namespace IO {

/*! \brief Write plotfile of computational domain with disease spread and census data at a given step.

    Writes the current disease spread information and census data (unit, FIPS code, census tract ID,
    and community number) to a plotfile:
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
      + component 6*n+0: unit number
      + component 6*n+1: FIPS ID
      + component 6*n+2: census tract number
      + component 6*n+3: community number
    + Get disease spread data (first 7*n components) from AgentContainer::generateCellData() and
    + also the disease_stats multifab, which tracts the number of new cases each day.
    + Copy unit number, FIPS code, census tract ID, and community number from the input MultiFabs to
      the remaining components.
    + Write the output MultiFab to file.
    + Write agents to file - see AgentContainer::WritePlotFile().
*/
void writePlotFile (const AgentContainer& pc,                      /*!< Agent (particle) container */
                    const MFPtrVec& a_disease_stats,               /*!< Disease stats tracker */
                    const iMultiFab* unit_mf_ptr,                  /*!< MultiFabs to write out */
                    const iMultiFab* FIPS_mf_ptr,                  /*!< MultiFabs to write out */
                    const iMultiFab* comm_mf_ptr,                  /*!< MultiFabs to write out */
                    const int num_diseases,                        /*!< Number of diseases */
                    const std::vector<std::string>& disease_names, /*!< Names of diseases */
                    const Real cur_time,                           /*!< current time */
                    const int step /*!< Current step */) {
    amrex::Print() << "Writing plotfile \n";

    // make sure status_names are in the same order as the struct Status in AgentDefinitions.H (do not include "dead")
    static const Vector<std::string> status_names = {"total", "never_infected", "infected", "immune", "susceptible"};

    static const int ncomp_d = status_names.size();
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
        int_varnames.push_back("school_grade");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("school_id");
        write_int_comp.push_back(static_cast<int>(step == 0));
        int_varnames.push_back("school_closed");
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

    auto unit = amrex::cast<MultiFab>(*unit_mf_ptr);
    VisMF::Read(unit, amrex::MultiFabFileFullPrefix(lev, restart_chkfile, level_prefix, "unit"));
    *unit_mf_ptr = amrex::cast<iMultiFab>(unit);

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
        auto unit = amrex::cast<MultiFab>(*unit_mf_ptr);
        VisMF::Write(unit, amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix, "unit"));
        for (std::size_t i = 0; i < a_disease_stats.size(); ++i) {
            VisMF::Write(*a_disease_stats[i], amrex::MultiFabFileFullPrefix(lev, checkpointname, default_level_prefix,
                                                                            "disease_stats_" + std::to_string(i)));
        }
    }

    pc.Checkpoint(checkpointname, "agents");
}

/*! \brief Writes diagnostic data by FIPS code

    Writes a file with the total number of infected agents for each unit;
    it writes out the number of infected agents in the same order as the units in the
    census data file.
    + Creates a output vector of size #DemographicData::Nunit (total number of units).
    + Gets the disease status in agents from AgentContainer::generateCellData().
    + On each processor, sets the unit-th element of the output vector to the number of
      infected agents in the communities on this processor belonging to that unit.
    + Sum across all processors and write to file.
*/
void writeFIPSData (const AgentContainer& agents,                  /*!< Agents (particle) container */
                    const CensusData& censusData,                  /*!< Census data */
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

    for (int d = 0; d < num_diseases; d++) {

        amrex::Print() << "Generating diagnostic data by FIPS code " << "for " << disease_names[d] << "\n";

        std::vector<amrex::Real> data(censusData.demo.Nunit, 0.0);
        amrex::Gpu::DeviceVector<amrex::Real> d_data(data.size(), 0.0);
        amrex::Real* const AMREX_RESTRICT data_ptr = d_data.dataPtr();

        for (int lev = 0; lev < nlevs; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            {
                for (MFIter mfi(*mf_vec[lev]); mfi.isValid(); ++mfi) {
                    auto unit_arr = censusData.unit_mf[mfi].array();
                    auto cell_data_arr = (*mf_vec[lev])[mfi].array();

                    auto bx = mfi.tilebox();
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        int unit = unit_arr(i, j, k); // which FIPS
                        int num_infected = int(cell_data_arr(i, j, k, 2));
                        amrex::Gpu::Atomic::AddNoRet(&data_ptr[unit], (amrex::Real)num_infected);
                    });
                }
            }
        }

        // blocking copy from device to host
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_data.begin(), d_data.end(), data.begin());

        // reduced sum over mpi ranks
        ParallelDescriptor::ReduceRealSum(data.data(), data.size(), ParallelDescriptor::IOProcessorNumber());

        if (ParallelDescriptor::IOProcessor()) {
            std::string fn = amrex::Concatenate(prefix, step, 5);
            if (num_diseases > 1) { fn += ("_" + disease_names[d]); }
            std::ofstream ofs{fn, std::ofstream::out | std::ofstream::app};

            // set precision
            ofs << std::fixed << std::setprecision(14) << std::scientific;

            // loop over data size and write
            for (const auto& item : data) {
                ofs << " " << item;
            }

            ofs << std::endl;
            ofs.close();
        }
    }
}

/*! \brief Writes diagnostic data aggregated by block group

    Writes a file with the total number of infected agents for each census block group;
    it writes out the number of infected agents in the same order as the block groups in the UrbanPop .idx input file.
    + Creates a output vector of size #UrbanPopData::num_communities
    + Gets the disease status in agents from AgentContainer::generateCellData().
    + On each processor, sets the block-group-th element of the output vector to the number of
      infected agents in the block group on this processor belonging to that unit.
    + Sum across all processors and write to file.
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

    for (int d = 0; d < num_diseases; d++) {
        amrex::Print() << "Generating diagnostic data by census block group " << "for " << disease_names[d] << "\n";
        std::vector<amrex::Real> data(urbanpopData.num_communities, 0.0);
        amrex::Gpu::DeviceVector<amrex::Real> d_data(data.size(), 0.0);
        amrex::Real* const AMREX_RESTRICT data_ptr = d_data.dataPtr();

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
                        int num_infected = int(cell_data_arr(i, j, k, 2));
                        // This should not require an atomic operation because each block group is at a separate i,j location
                        // amrex::Gpu::Atomic::AddNoRet(&data_ptr[block_group_i], (amrex::Real)num_infected);
                        data_ptr[block_group_i] = (amrex::Real)num_infected;
                    });
                }
            }
        }

        // blocking copy from device to host
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, d_data.begin(), d_data.end(), data.begin());

        // reduced sum over mpi ranks
        ParallelDescriptor::ReduceRealSum(data.data(), data.size(), ParallelDescriptor::IOProcessorNumber());

        if (ParallelDescriptor::IOProcessor()) {
            std::string fn = amrex::Concatenate(prefix, step, 5);
            if (num_diseases > 1) { fn += ("_" + disease_names[d]); }
            std::ofstream ofs{fn, std::ofstream::out | std::ofstream::app};

            // set precision
            ofs << std::fixed << std::setprecision(14) << std::scientific;

            // loop over data size and write
            for (const auto& item : data) {
                ofs << " " << item;
            }

            ofs << std::endl;
            ofs.close();
        }
    }
}

} // namespace IO
} // namespace ExaEpi
