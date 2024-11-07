/*! @file DemographicData.H
    \brief #DemographicData class definition
*/

#ifndef DEMOGRAPHICDATA_H_
#define DEMOGRAPHICDATA_H_

#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include <string>

/*! \brief Variables and functions for reading and storing demographic data */
struct DemographicData
{
    /*! \brief Null constructor */
    DemographicData () {}

    /*! \brief Constructor with initializes from a given census file */
    DemographicData (const::std::string& fname, const int workgroup_size);

    void InitFromFile (const std::string& fname, const int workgroup_size);

    void Print () const;

    void CopyToDeviceAsync (const amrex::Vector<int>& h_vec, amrex::Gpu::DeviceVector<int>& d_vec);

    void CopyToHostAsync (const amrex::Gpu::DeviceVector<int>& d_vec, amrex::Vector<int>& h_vec);

    void CopyDataToDevice ();

    int Ncommunity;     /*!< number of communities required */
    int Nunit;          /*!< number of county/state units */
    amrex::Vector<int>  myID,   /*!< ID array */
                        FIPS,   /*!< FIPS code array */
                        Tract;  /*!< Census tract array */
    amrex::Vector<int> Start;         /*!< Starting community number for each unit */
    amrex::Vector<int> Population;    /*!< Population of each unit */
    std::map<int, int> CountyPop; /*!< Population of each county */
    amrex::Vector<int> N5,      /*!< number of people in age-group: under-5 */
                       N17,     /*!< number of people in age-group: 5-17 */
                       N29,     /*!< number of people in age-group: 18-29 */
                       N64,     /*!< number of people in age-group: 30-64 */
                       N65plus; /*!< number of people in age-group: 65+ */
    amrex::Vector<int> H1, /*!< Number of households with 1 member  */
                       H2, /*!< Number of households with 2 members */
                       H3, /*!< Number of households with 3 members */
                       H4, /*!< Number of households with 4 members */
                       H5, /*!< Number of households with 5 members */
                       H6, /*!< Number of households with 6 members */
                       H7; /*!< Number of households with 7 members */
    amrex::Vector<int> Ndaywork; /*!< Number of daytime workers */
    amrex::Vector<int> myIDtoUnit; /*!< Given myID #, what Unit # is it? */
    amrex::Vector<int> Unit_on_proc; /*!< Is any part of this unit on this processor? */

    /* The following are device copies of the above arrays */
    amrex::Gpu::DeviceVector<int> myID_d,   /*!< ID array (GPU device) */
                                  FIPS_d,   /*!< FIPS code array (GPU device) */
                                  Tract_d;  /*!< Census tract array (GPU device) */
    amrex::Gpu::DeviceVector<int> Start_d;  /*!< Starting community number for each unit (GPU device)*/
    amrex::Gpu::DeviceVector<int> Population_d; /*!< Population of each unit (GPU device) */
    amrex::Gpu::DeviceVector<int> N5_d,         /*!< number of people in age-group: under-5 (GPU device) */
                                  N17_d,        /*!< number of people in age-group: 5-17 (GPU device) */
                                  N29_d,        /*!< number of people in age-group: 18-29 (GPU device) */
                                  N64_d,        /*!< number of people in age-group: 30-64 (GPU device) */
                                  N65plus_d;    /*!< number of people in age-group: 65+ (GPU device) */
    amrex::Gpu::DeviceVector<int> H1_d, /*!< Number of households with 1 member  (GPU device) */
                                  H2_d, /*!< Number of households with 2 members (GPU device) */
                                  H3_d, /*!< Number of households with 3 members (GPU device) */
                                  H4_d, /*!< Number of households with 4 members (GPU device) */
                                  H5_d, /*!< Number of households with 5 members (GPU device) */
                                  H6_d, /*!< Number of households with 6 members (GPU device) */
                                  H7_d; /*!< Number of households with 7 members (GPU device) */
    amrex::Gpu::DeviceVector<int> Ndaywork_d; /*!< Number of daytime workers (GPU device) */
    amrex::Gpu::DeviceVector<int> myIDtoUnit_d; /*!< Given myID #, what Unit # is it? (GPU device) */
    amrex::Gpu::DeviceVector<int> Unit_on_proc_d; /*!< Is any part of this unit on this processor? (GPU device) */

    static const int COMMUNITY_SIZE = 2000;
};

#endif
