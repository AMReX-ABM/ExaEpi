/*! @file CaseData.cpp
    \brief Function implementations for #CaseData class
*/

#include "CaseData.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_BLassert.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <sstream>
#include <string>

using namespace amrex;

/*! \brief Constructor that initializes by reading in data from given file

    See CaseData::initFromFile()
*/
CaseData::CaseData (const std::string& dname, /*!< Disease name */
                    const ::std::string& fname /*!< Filename to read case data from */) {
    initFromFile(dname, fname);
}

/*! \brief Read case data from a given file

    The case data file is an ASCII text file with three columns of numbers:
    FIPS code, current number of cases, and cumulative number of cases till date.

    + Initialize the member data (and their GPU versions) to empty vectors:
      + #CaseData::FIPS_hubs
      + #CaseData::Size_hubs
      + #CaseData::num_cases
      + #CaseData::num_cases2date
    + Initialize #CaseData::num_cases and #CaseData::num_cases2date as arrays of size
      57,000 with 0 values (56999 is the largest FIPS code)
    + Initialize #CaseData::N_hubs to 0.
    + Read the file: till reaching end-of-file, read each line that contains the FIPS code,
      current number of cases, and cumulative number of cases till date.
      + Set the #CaseData::num_cases and #CaseData::num_cases2date values for this FIPS code
      + Increment #CaseData::N_hubs
    + Resize #CaseData::FIPS_hubs and #CaseData::Size_hubs to #CaseData::N_hubs.
    + For each FIPS code value (0 to 56999), if the number of cases for that FIPS code is
      greater than zero,
      + Add the FIPS code to the #CaseData::FIPS_hubs array.
      + Add the number of cases to the #CaseData::Size_hubs array.
    + Copy the arrays to device

    \b Note: The code runs even if the case data file lacks the 3rd column. In this case, the
    #CaseData::num_cases2date will contain junk values (or maybe zero).
*/
void CaseData::initFromFile (const std::string& a_disease_name, /*!< Name of disease */
                             const std::string& fname /*!< Filename to read case data from */) {
    BL_PROFILE("CaseData::initFromFile");
    m_disease_name = a_disease_name;

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    FIPS_hubs.resize(0);
    Size_hubs.resize(0);
    num_cases.resize(0);
    num_cases2date.resize(0);

    FIPS_hubs_d.resize(0);
    Size_hubs_d.resize(0);
    num_cases_d.resize(0);
    num_cases2date_d.resize(0);

    num_cases.resize(57000, 0);
    num_cases2date.resize(57000, 0);

    int fips = 1;
    int last_fips = -1;
    int i, j;
    N_hubs = 0;
    int ntot = 0;
    std::string line;
    while ((is.good()) && (fips > 0)) {
        std::getline(is, line);
        std::istringstream lis(line);
        lis >> fips >> i >> j;
        if (fips != last_fips) {
            if (fips >= 57000) { amrex::Abort("FIPS too large when reading case data."); }
            num_cases[fips] = i;
            ntot += i;
            num_cases2date[fips] = j;
            N_hubs++;
            last_fips = fips;
        } else {
            fips = -1; // don't read another line
        }
    }

    amrex::Print() << "Setting initial case counts for " << m_disease_name << " in " << N_hubs << " disease hubs. \n";

    FIPS_hubs.resize(N_hubs, 0);
    Size_hubs.resize(N_hubs, 0);
    j = 0;
    for (i = 0; i < 57000; i++) {
        if (num_cases[i]) {
            FIPS_hubs[j] = i;
            Size_hubs[j++] = num_cases[i];
        }
    }

    amrex::ignore_unused(ntot);
    copyDataToDevice();
    amrex::Gpu::streamSynchronize();
}

/*! \brief Prints case data to screen

    For each disease hub (#CaseData::N_hubs), print the FIPS code, current number of cases,
    and cumulative number of cases till date.
*/
void CaseData::Print () const {
    amrex::Print() << "Disease name: " << m_disease_name << "\n";
    for (amrex::Long i = 0; i < FIPS_hubs.size(); ++i) {
        amrex::Print() << FIPS_hubs[i] << " " << num_cases[i] << " " << num_cases2date[i] << "\n";
    }
}

/*! \brief Copy a vector from host to device */
void CaseData::copyToDeviceAsync (const amrex::Vector<int>& h_vec, /*!< Host vector */
                                  amrex::Gpu::DeviceVector<int>& d_vec /*!< Device vector */) {
    d_vec.resize(0);
    d_vec.resize(h_vec.size());
    Gpu::copyAsync(Gpu::hostToDevice, h_vec.begin(), h_vec.end(), d_vec.begin());
}

/*! \brief Copy a vector from device to host */
void CaseData::copyToHostAsync (const amrex::Gpu::DeviceVector<int>& d_vec, /*!< Device vector */
                                amrex::Vector<int>& h_vec /*!< Host vector */) {
    h_vec.resize(0);
    h_vec.resize(d_vec.size());
    Gpu::copyAsync(Gpu::deviceToHost, d_vec.begin(), d_vec.end(), h_vec.begin());
}

/*! \brief Copy all member data from host to device */
void CaseData::copyDataToDevice () {
    copyToDeviceAsync(FIPS_hubs, FIPS_hubs_d);
    copyToDeviceAsync(Size_hubs, Size_hubs_d);
    copyToDeviceAsync(num_cases, num_cases_d);
    copyToDeviceAsync(num_cases2date, num_cases2date_d);
}
