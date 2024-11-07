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

namespace ExaEpi
{
namespace IO
{

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

      Then (n being the number of diseases):
      + component 5*n+0: unit number
      + component 5*n+1: FIPS ID
      + component 5*n+2: census tract number
      + component 5*n+3: community number
    + Get disease spread data (first 5*n components) from AgentContainer::generateCellData().
    + Copy unit number, FIPS code, census tract ID, and community number from the input MultiFabs to
      the remaining components.
    + Write the output MultiFab to file.
    + Write agents to file - see AgentContainer::WritePlotFile().
*/
void writePlotFile (const AgentContainer& pc, /*!< Agent (particle) container */
                    const CensusData& censusData,  /*!< Contains census data */
                    const int num_diseases, /*!< Number of diseases */
                    const std::vector<std::string>& disease_names, /*!< Names of diseases */
                    const Real cur_time, /*!< current time */
                    const int step /*!< Current step */) {
    amrex::Print() << "Writing plotfile \n";

    static const int ncomp_d = 5;
    static const int ncomp = ncomp_d*num_diseases + 4;

    MultiFab output_mf(pc.ParticleBoxArray(0),
                       pc.ParticleDistributionMap(0), ncomp, 0);
    output_mf.setVal(0.0);
    pc.generateCellData(output_mf);

    amrex::Copy(output_mf, censusData.unit_mf, 0, ncomp_d*num_diseases  , 1, 0);
    amrex::Copy(output_mf, censusData.FIPS_mf, 0, ncomp_d*num_diseases+1, 2, 0);
    amrex::Copy(output_mf, censusData.comm_mf, 0, ncomp_d*num_diseases+3, 1, 0);

    {
        Vector<std::string> plt_varnames = {};
        if (num_diseases == 1) {
            plt_varnames.push_back("total");
            plt_varnames.push_back("never_infected");
            plt_varnames.push_back("infected");
            plt_varnames.push_back("immune");
            plt_varnames.push_back("susceptible");
        } else {
            for (int d = 0; d < num_diseases; d++) {
                plt_varnames.push_back(disease_names[d]+"_total");
                plt_varnames.push_back(disease_names[d]+"_never_infected");
                plt_varnames.push_back(disease_names[d]+"_infected");
                plt_varnames.push_back(disease_names[d]+"_immune");
                plt_varnames.push_back(disease_names[d]+"_susceptible");
            }
        }
        plt_varnames.push_back("unit");
        plt_varnames.push_back("FIPS");
        plt_varnames.push_back("Tract");
        plt_varnames.push_back("comm");

#ifdef AMREX_USE_HDF5
        WriteSingleLevelPlotfileHDF5MultiDset(  amrex::Concatenate("plt", step, 5),
                                                output_mf,
                                                plt_varnames,
                                                pc.ParticleGeom(0),
                                                cur_time,
                                                step,
                                                "ZLIB@3" );
#else
        WriteSingleLevelPlotfile(   amrex::Concatenate("plt", step, 5),
                                    output_mf,
                                    plt_varnames,
                                    pc.ParticleGeom(0),
                                    cur_time,
                                    step );
#endif
    }

    {
        Vector<int> write_real_comp = {}, write_int_comp = {};
        Vector<std::string> real_varnames = {}, int_varnames = {};
        // non-disease-specific attributes
        int_varnames.push_back ("age_group"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("family"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("home_i"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("home_j"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("work_i"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("work_j"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("hosp_i"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("hosp_j"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("trav_i"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("trav_j"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("nborhood"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("school"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("workgroup"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("work_nborhood"); write_int_comp.push_back(static_cast<int>(step==0));
        int_varnames.push_back ("withdrawn"); write_int_comp.push_back(1);
        int_varnames.push_back ("random_travel"); write_int_comp.push_back(1);
        int_varnames.push_back ("air_travel"); write_int_comp.push_back(1);
        // disease-specific (runtime-added) attributes
        if (num_diseases == 1) {
            real_varnames.push_back("treatment_timer"); write_real_comp.push_back(1);
            real_varnames.push_back("disease_counter"); write_real_comp.push_back(1);
            real_varnames.push_back("infection_prob"); write_real_comp.push_back(1);
            real_varnames.push_back("latent_period"); write_real_comp.push_back(static_cast<int>(step==0));
            real_varnames.push_back("infectious_period"); write_real_comp.push_back(static_cast<int>(step==0));
            real_varnames.push_back("incubation_period"); write_real_comp.push_back(static_cast<int>(step==0));
            int_varnames.push_back ("status"); write_int_comp.push_back(1);
            int_varnames.push_back ("strain"); write_int_comp.push_back(static_cast<int>(step==0));
            int_varnames.push_back ("symptomatic"); write_int_comp.push_back(1);
        } else {
            for (int d = 0; d < num_diseases; d++) {
                real_varnames.push_back(disease_names[d]+"treatment_timer"); write_real_comp.push_back(1);
                real_varnames.push_back(disease_names[d]+"_disease_counter"); write_real_comp.push_back(1);
                real_varnames.push_back(disease_names[d]+"_infection_prob"); write_real_comp.push_back(1);
                real_varnames.push_back(disease_names[d]+"_latent_period"); write_real_comp.push_back(static_cast<int>(step==0));
                real_varnames.push_back(disease_names[d]+"_infectious_period"); write_real_comp.push_back(static_cast<int>(step==0));
                real_varnames.push_back(disease_names[d]+"_incubation_period"); write_real_comp.push_back(static_cast<int>(step==0));
                int_varnames.push_back (disease_names[d]+"_status"); write_int_comp.push_back(1);
                int_varnames.push_back (disease_names[d]+"_strain"); write_int_comp.push_back(static_cast<int>(step==0));
                int_varnames.push_back (disease_names[d]+"_symptomatic"); write_int_comp.push_back(1);
            }
        }

#ifdef AMREX_USE_HDF5
        pc.WritePlotFileHDF5(   amrex::Concatenate("plt", step, 5),
                                "agents",
                                write_real_comp,
                                write_int_comp,
                                real_varnames,
                                int_varnames,
                                "ZLIB@3" );
#else
        pc.WritePlotFile(   amrex::Concatenate("plt", step, 5),
                            "agents",
                            write_real_comp,
                            write_int_comp,
                            real_varnames,
                            int_varnames );
#endif
    }
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
void writeFIPSData (const AgentContainer& agents, /*!< Agents (particle) container */
                    const CensusData& censusData, /*!< Census data */
                    const std::string& prefix, /*!< Filename prefix */
                    const int num_diseases, /*!< Number of diseases */
                    const std::vector<std::string>& disease_names, /*!< Names of diseases */
                    const int step /*!< Current step */)
{
    static const int ncomp_d = 5;
    static const int ncomp = ncomp_d*num_diseases + 4;

    static const int nlevs = std::max(0, agents.finestLevel()+1);
    std::vector<std::unique_ptr<MultiFab>> mf_vec;
    mf_vec.resize(nlevs);
    for (int lev = 0; lev < nlevs; ++lev) {
        mf_vec[lev] = std::make_unique<MultiFab>(   agents.ParticleBoxArray(lev),
                                                    agents.ParticleDistributionMap(lev),
                                                    ncomp,
                                                    0 );
        mf_vec[lev]->setVal(0.0);
        agents.generateCellData(*mf_vec[lev]);
    }

    for (int d = 0; d < num_diseases; d++) {

        amrex::Print() << "Generating diagnostic data by FIPS code "
                       << "for " << disease_names[d] << "\n";

        std::vector<amrex::Real> data(censusData.demo.Nunit, 0.0);
        amrex::Gpu::DeviceVector<amrex::Real> d_data(data.size(), 0.0);
        amrex::Real* const AMREX_RESTRICT data_ptr = d_data.dataPtr();

        for (int lev = 0; lev < nlevs; ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            {
                for (MFIter mfi(*mf_vec[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    auto unit_arr = censusData.unit_mf[mfi].array();
                    auto cell_data_arr = (*mf_vec[lev])[mfi].array();

                    auto bx = mfi.tilebox();
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            int unit = unit_arr(i, j, k);  // which FIPS
                            int num_infected = int(cell_data_arr(i, j, k, 2));
                            amrex::Gpu::Atomic::AddNoRet(&data_ptr[unit], (amrex::Real) num_infected);
                        });
                }
            }
        }

        // blocking copy from device to host
        amrex::Gpu::copy(amrex::Gpu::deviceToHost,
                         d_data.begin(), d_data.end(), data.begin());

        // reduced sum over mpi ranks
        ParallelDescriptor::ReduceRealSum
            (data.data(), data.size(), ParallelDescriptor::IOProcessorNumber());

        if (ParallelDescriptor::IOProcessor())
        {
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

}
}
