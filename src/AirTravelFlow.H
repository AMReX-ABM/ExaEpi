/*! @file AirTravelFlow.H
    \brief #AirTravelFlow class definition
*/

#ifndef AIR_TRAFFIC_H_
#define AIR_TRAFFIC_H_

#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>
#include "DemographicData.H"

#include <string>
typedef std::map<std::string, std::map<std::string, float> > airTravelType;
using namespace amrex;

/*! \brief Structure containing information of traffic among major airport hubs*/
struct AirTravelFlow
{
    /*! \brief Null constructor */
    AirTravelFlow (){}

    AirTravelFlow (const::std::string fname);

    void ReadAirTravelFlow (const std::string fname);
    void ReadAirports (const std::string fname, DemographicData& demo);
    void ComputeTravelProbs(DemographicData& demo);
    void Print () const;

    void CopyToDeviceAsync (const amrex::Vector<int>& h_vec, /*!< host vector */
                                         amrex::Gpu::DeviceVector<int>& d_vec /*!< device vector */) {
        d_vec.resize(0);
        d_vec.resize(h_vec.size());
        Gpu::copyAsync(Gpu::hostToDevice, h_vec.begin(), h_vec.end(), d_vec.begin());
    }

    void CopyToDeviceAsync (const amrex::Vector<float>& h_vec, /*!< host vector */
                                         amrex::Gpu::DeviceVector<float>& d_vec /*!< device vector */) {
        d_vec.resize(0);
        d_vec.resize(h_vec.size());
        Gpu::copyAsync(Gpu::hostToDevice, h_vec.begin(), h_vec.end(), d_vec.begin());
    }

    void CopyDataToDevice () {
        CopyToDeviceAsync(assigned_airport, assigned_airport_d);
        CopyToDeviceAsync(air_travel_prob, air_travel_prob_d);
        CopyToDeviceAsync(dest_airports, dest_airports_d);
        CopyToDeviceAsync(dest_airports_offset, dest_airports_offset_d);
        CopyToDeviceAsync(dest_airports_prob, dest_airports_prob_d);
        CopyToDeviceAsync(arrivalUnits,arrivalUnits_d);
        CopyToDeviceAsync(arrivalUnits_offset, arrivalUnits_offset_d);
        CopyToDeviceAsync(arrivalUnits_prob, arrivalUnits_prob_d);
        amrex::Gpu::streamSynchronize();
    }

//data that processed by only host
    int nairports; /*! Total number of airports */
    int n_counties_with_airports;
    std::map<std::string, int> airport_id;                /*! integer id for airport code*/
    std::map<int, std::string> id_to_airport;                /*! integer id to airport code*/
    std::map<int, std::string> FIPS_to_airport;   /*!< map from FIPS to airport code */
    std::map<int, int> FIPS_to_county;  /*!< map from FIPS to county id */
    int nconnections; /*! The total number of airport connections */
    std::map<std::string, std::vector<std::string> > destAirportMap;
    std::map<std::string, std::vector<int> > inAirportRangeUnitMap; /*!< All units served by an airport */
    std::map<std::string, int> inAirportRangePop; /*!< population of all counties served by an airport */
    std::map<std::string, int> originPax; /*!<number of origin passengers*/
    std::map<std::string, int> destPax;/*!<number of destination passengers*/
    std::map<std::string/*org*/, std::map<std::string/*dest*/, float/*distribution*/> > travel_path_prob; /*!< probability of traveling between two airports */

//data that can be used by either host or GPU
    amrex::Vector<int> assigned_airport;//county id to airport
    amrex::Vector<float> air_travel_prob;
    amrex::Vector<int> dest_airports;
    amrex::Vector<int> dest_airports_offset;
    amrex::Vector<float> dest_airports_prob;
    amrex::Vector<int> arrivalUnits;
    amrex::Vector<int> arrivalUnits_offset;
    amrex::Vector<float> arrivalUnits_prob;

    amrex::Gpu::DeviceVector<int> assigned_airport_d;
    amrex::Gpu::DeviceVector<float> air_travel_prob_d;
    amrex::Gpu::DeviceVector<int> dest_airports_d;
    amrex::Gpu::DeviceVector<int> dest_airports_offset_d;
    amrex::Gpu::DeviceVector<float> dest_airports_prob_d;
    amrex::Gpu::DeviceVector<int> arrivalUnits_d;
    amrex::Gpu::DeviceVector<int> arrivalUnits_offset_d;
    amrex::Gpu::DeviceVector<float> arrivalUnits_prob_d;
};
#endif
