/*! @file DemographicData.cpp
    \brief Function implementations of #DemographicData class
*/

#include "DemographicData.H"

#include <AMReX_BLassert.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <string>
#include <sstream>

using namespace amrex;

/*! Initializes by reading in demographic data from a given
    filename. Calls DemographicData::InitFromFile(). */

DemographicData::DemographicData (const::std::string& fname, /*!< Name of file containing demographic data */
                                  const int workgroup_size)
{
    InitFromFile(fname, workgroup_size);
}

/*! \brief Read in demographic data from given file.
 *
 *  + The first line of the file contains the number of units.
 *  + The following lines have the following data:
 *    + ID (US-wide census tract ID)
 *    + Population
 *    + Number of day workers
 *    + FIPS code
 *    + Census tract number
 *    + Numbers of people in age groups: under 5, 5-17, 18-29, 30-64, and 65+
 *    + Number of households with: 1, 2, 3, 4, 5, 6, and 7 member(s)
 *
 *  This function reads the number of units and allocates the data arrays. Then, for each unit:
 *  + Read in the above data.
 *  + Compute the number of communities, where a community comprises 2000 people.
 *    + If there are no residential communities but a significant daytime worker population (> 20),
 *      a community is defined for these workers.
 *    + If the number of daytime workers exceed 1000, then compute the number of 1000-worker communities.
 *  + Save the starting community number of each unit
 *  + Set up the mapping: given by ID, what is the unit number?
 *  + Compute total population and number of daytime workers.
 *  + Copy data to GPU device memory.
 */

void DemographicData::InitFromFile (const std::string& fname, /*!< Name of file containing demographic data */
                                    const int workgroup_size  /*!< The size of workgroups */)
{
    BL_PROFILE("DemographicData::InitFromFile");

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line, word;

    std::getline(is, line);
    Nunit = std::stoi(line);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Nunit >= 0, "Number of units can't be negative");

    myID.resize(Nunit);
    FIPS.resize(Nunit);
    Tract.resize(Nunit);
    Start.resize(Nunit+1);
    Population.resize(Nunit);
    N5.resize(Nunit);
    N17.resize(Nunit);
    N29.resize(Nunit);
    N64.resize(Nunit);
    N65plus.resize(Nunit);

    H1.resize(Nunit);
    H2.resize(Nunit);
    H3.resize(Nunit);
    H4.resize(Nunit);
    H5.resize(Nunit);
    H6.resize(Nunit);
    H7.resize(Nunit);

    Ndaywork.resize(Nunit);
    myIDtoUnit.resize(65334);
    Unit_on_proc.resize(Nunit);

    Ncommunity = 0;
    for (int i = 0; i < Nunit; ++i) {
        Start[i] = Ncommunity;
        Unit_on_proc[i] = 0;
        AMREX_ALWAYS_ASSERT(is.good());
        std::getline(is, line);
        std::istringstream lis(line);
        lis >> myID[i] >> Population[i] >> Ndaywork[i] >> FIPS[i] >> Tract[i];
        lis >> N5[i] >> N17[i] >> N29[i] >> N64[i] >> N65plus[i];
        lis >> H1[i] >> H2[i] >> H3[i] >> H4[i] >> H5[i] >> H6[i] >> H7[i];
        myIDtoUnit[myID[i]] = i;
        if(CountyPop.find(FIPS[i]) ==CountyPop.end()) CountyPop[FIPS[i]]= Population[i];
        else CountyPop[FIPS[i]] += Population[i];

        /*   How many 2000-person communities does this require?   */
        int ncomm = (int) std::rint(((double) Population[i]) / DemographicData::COMMUNITY_SIZE);
        //amrex::Print() << Population[i] << " " << ncomm << " " << ncomm * DemographicData::COMMUNITY_SIZE << "\n";

        /*
          Note that some census tracts have little or no residential
          population, but a large daytime worker population, in which
          case we need to set up a dummy (unpopulated) community with
          some number of workgroups during the daytime.  When setting
          up 2000-person communities, check Population[i] to see if
          this is the case, and set Nworkgroup accordingly.

          It is often possible that multiple workplace-only
          communities are needed, if there are more than 255*WG_size
          workers (ca. 5000).  E.g. in Cook County, FIPS 17031-760900
          has only 22 residents but 33,541 daytime workers.
          Or FIPS 6059-62610 has 1,509 residents (1 community), but
          76,820 daytime workers (3841 workgroups x 20).

          Need to check for cases like this -- it would be better to
          limit the daytime population than have one very unbalanced
          community (e.g. FIPS 48201-100000 has 151,161 daytime workers)!
          Let's limit each community to 1000 workers, which would be
          about 50 workgroups:
        */
        if ( (Ndaywork[i] >= workgroup_size) && !ncomm ) {
            ncomm = 1;
            //amrex::Print() << "myID " << myID[i] << ": " << ncomm << " community to accommodate " << Ndaywork[i] << " daytime workers\n";
        }
        if ( Ndaywork[i] > (ncomm * 1000) ) {
            ncomm = Ndaywork[i] / 1000;
            //amrex::Print() << "myID " << myID[i] << ": " << ncomm << " communities to accommodate " << Ndaywork[i] << " daytime workers\n";
        }
        //amrex::Print() << ncomm << " " << Start[i] << "\n";
        Ncommunity += ncomm;
    }
    Start[Nunit] = Ncommunity;

    long total_pop = 0;
    long total_workers = 0;
    for (int i = 0; i < Nunit; ++i) {
        total_pop += Population[i];
        total_workers += Ndaywork[i];
    }
    amrex::Print() << "Total pop " << total_pop << "\n";
    amrex::Print() << "Total workers " << total_workers << "\n";
    amrex::Print() << "Number of communities: " << Ncommunity << "\n";

    CopyDataToDevice();
    amrex::Gpu::streamSynchronize();
}

/*! \brief Prints demographic data to screen:

 *  For each unit, print
 *  + ID (US-wide census tract ID)
 *  + Population
 *  + Number of day workers
 *  + FIPS code
 *  + Census tract number
 *  + Numbers of people in age groups: under 5, 5-17, 18-29, 30-64, and 65+
 *  + Number of households with: 1, 2, 3, 4, 5, 6, and 7 member(s)
*/
void DemographicData::Print () const {
    amrex::Print() << Nunit << "\n";
    for (int i = 0; i < Nunit; ++i) {
        amrex::Print() << myID[i] << " " << Population[i] << " " << Ndaywork[i] << " " << FIPS[i] << " " << Tract[i] << " ";
        amrex::Print() << N5[i] << " " << N17[i] << " " << N29[i] << " " << N64[i] << " " << N65plus[i] << " ";
        amrex::Print() << H1[i] << " " << H2[i] << " " << H3[i] << " " << H4[i] << " " << H5[i] << " " << H6[i] << " " << H7[i] << "\n";
    }
}

/*! \brief Copy array from host to device */
void DemographicData::CopyToDeviceAsync (const amrex::Vector<int>& h_vec, /*!< host vector */
                                         amrex::Gpu::DeviceVector<int>& d_vec /*!< device vector */) {
    d_vec.resize(0);
    d_vec.resize(h_vec.size());
    Gpu::copyAsync(Gpu::hostToDevice, h_vec.begin(), h_vec.end(), d_vec.begin());
}

/*! \brief Copy array from device to host */
void DemographicData::CopyToHostAsync (const amrex::Gpu::DeviceVector<int>& d_vec, /*!< device vector */
                                       amrex::Vector<int>& h_vec /*!< host vector */) {
    h_vec.resize(0);
    h_vec.resize(d_vec.size());
    Gpu::copyAsync(Gpu::deviceToHost, d_vec.begin(), d_vec.end(), h_vec.begin());
}

/*! \brief Copies member arrays of #DemographicData from host to device */
void DemographicData::CopyDataToDevice () {
    CopyToDeviceAsync(myID, myID_d);
    CopyToDeviceAsync(FIPS, FIPS_d);
    CopyToDeviceAsync(Tract, Tract_d);
    CopyToDeviceAsync(Start, Start_d);
    CopyToDeviceAsync(Population, Population_d);

    CopyToDeviceAsync(N5,  N5_d);
    CopyToDeviceAsync(N17, N17_d);
    CopyToDeviceAsync(N29, N29_d);
    CopyToDeviceAsync(N64, N64_d);
    CopyToDeviceAsync(N65plus, N65plus_d);

    CopyToDeviceAsync(H1, H1_d);
    CopyToDeviceAsync(H2, H2_d);
    CopyToDeviceAsync(H3, H3_d);
    CopyToDeviceAsync(H4, H4_d);
    CopyToDeviceAsync(H5, H5_d);
    CopyToDeviceAsync(H6, H6_d);
    CopyToDeviceAsync(H7, H7_d);

    CopyToDeviceAsync(Ndaywork, Ndaywork_d);
    CopyToDeviceAsync(myIDtoUnit, myIDtoUnit_d);
    CopyToDeviceAsync(Unit_on_proc, Unit_on_proc_d);
}
