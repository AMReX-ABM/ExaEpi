/*! @file AirTravelFlow.cpp
    \brief Function implementations for AirTravelFlow class
*/

#include "AirTravelFlow.H"
#include <AMReX_BLassert.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>
#include <cmath>
#include <string>
#include <sstream>

using namespace amrex;

/*! \brief Constructor that initializes by reading in data from given file

    See AirTravelFlow::InitFromFile()
*/
AirTravelFlow::AirTravelFlow (const::std::string fname /*!< Filename to read case data from */)
{
}

void AirTravelFlow::ReadAirports(const std::string fname, DemographicData& demo){
    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line;
    n_counties_with_airports=0;
    if(!is.eof()) {
        getline(is, line);
        std::istringstream lis(line);
        lis >> n_counties_with_airports;
    }

    int FIPS = 0;
    std::string airportCode =  "";
    int cnt=0;
    int air_i=0;
    while (n_counties_with_airports>cnt) {
        std::getline(is, line);
        std::istringstream lis(line);
        lis >> FIPS >> airportCode;
        FIPS_to_airport[FIPS]= airportCode;
        if(inAirportRangePop.find(airportCode) == inAirportRangePop.end()){//the first time we see this airport
            airport_id[airportCode]=air_i;
            id_to_airport[air_i++]= airportCode;
            inAirportRangePop[airportCode]= demo.CountyPop[FIPS];
        }else {
            inAirportRangePop[airportCode]+= demo.CountyPop[FIPS];
        }
        cnt++;
    }
    //now convert the FIPS_to_airport map to vectors to offload to the GPU
    for(int i=0; i<n_counties_with_airports; i++){
        int fips= demo.FIPS[i];
        FIPS_to_county[fips]= i;
    }
    for(int i=0; i<demo.Nunit; i++){
        int fips= demo.FIPS[i];
        std::string airport= FIPS_to_airport[fips];
        inAirportRangeUnitMap[airport].push_back(i);
    }
}

/*! \brief Read case data from a given file
    Description TBW
*/
void AirTravelFlow::ReadAirTravelFlow (const std::string fname /*!< Filename to read case data from */)
{
    BL_PROFILE("AirTravelFlow::InitFromFile");

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line;
    nairports=0;
    nconnections=0;
    if(!is.eof()) {
        getline(is, line);
        std::istringstream lis(line);
        lis >> nairports >> nconnections;
        //amrex::Print() << "AIRPORTS " << nairports<< "  CONNECTION "<< nconnections <<"\n";
    }
    if(!nairports) return;
    if(!nconnections) return;

    std::string dest = "";
    std::string org =  "";
    int pax = 0;
    int c=0;
    while (nconnections>c++) {
        std::getline(is, line);
        std::istringstream lis(line);
        lis >> dest >> org >> pax;
        //amrex::Print() << "ORIGIN AIRPORT " << org<< "  DEST AIRPORT "<< dest <<" PASSENGERS "<< pax<<"\n";
        destAirportMap[org].push_back(dest);
        int pax_per_day= (int)(pax/365);
        travel_path_prob[org][dest]= (float)pax_per_day;//just initialize, will be finalized later in ComputeTravelProbs(DemographicData& demo)

        if(originPax.find(org)== originPax.end()) originPax[org]= pax_per_day;
        else originPax[org]+= pax_per_day;

        if(destPax.find(dest)== destPax.end()) destPax[dest]= pax_per_day;
        else destPax[dest]+= pax_per_day;
    }
}

void AirTravelFlow::ComputeTravelProbs(DemographicData& demo){
    air_travel_prob.resize(demo.Nunit);
    assigned_airport.resize(demo.Nunit);
    for(int i=0; i<demo.Nunit; i++){
        int fips=demo.FIPS[i];
        if(originPax.find(FIPS_to_airport[fips]) == originPax.end()) air_travel_prob[i]=0;
        else air_travel_prob[i]= (float)originPax[FIPS_to_airport[fips]]/inAirportRangePop[FIPS_to_airport[fips]];
        //amrex::Print() <<" Unit "<<i<<" FIPS "<<fips<< " travel prob "<<air_travel_prob[i]<<"\n";
        if(FIPS_to_airport.find(fips) != FIPS_to_airport.end()){
            std::string airportCode= FIPS_to_airport[fips];
            assigned_airport[i]= airport_id[airportCode];
        }else assigned_airport[i]= -1;//a unit that does not have any airport assigned to it
    }

    {
        //now we calculate the chance of traveling to a destination airport from an original one
        for(airTravelType::iterator it= travel_path_prob.begin(); it!= travel_path_prob.end(); it++)
        {
            std::string org= it->first;
            //find all paths from the origin airport
            float sum=0.;
            for(std::map<std::string, float>::iterator it1= it->second.begin(); it1!= it->second.end(); it1++){
                std::string dest= it1->first;
                sum+=  travel_path_prob[org][dest];
            }
            //compute chance for each path (we will sort these paths and compute prefix sums later)
            for(std::map<std::string, float>::iterator it1= it->second.begin(); it1!= it->second.end(); it1++){
                std::string dest= it1->first;
                travel_path_prob[org][dest]= travel_path_prob[org][dest]/sum;
                //amrex::Print() <<" Original Airport "<< org<< " Dest Airport "<< dest<< " travel path prob threshold "<< travel_path_prob[org][dest]<<"\n";
            }
        }
        //pack the travel_path_prob map to 3 vectors that can be offloaded to the GPU
        dest_airports_offset.resize(nairports+1);
        dest_airports.resize(nconnections);
        dest_airports_prob.resize(nconnections);
        int curOffset=0;
        for(int i=0; i<nairports; i++){
           dest_airports_offset[i]= curOffset;
           std::string airport= id_to_airport[i];
           int numDest= destAirportMap[airport].size();
           for(int j=0; j<numDest; j++){
               std::string destAirport= destAirportMap[airport][j];
               dest_airports[curOffset]= airport_id[destAirport];
               if(j==0) dest_airports_prob[curOffset]= travel_path_prob[airport][destAirport];
               else dest_airports_prob[curOffset]= dest_airports_prob[curOffset-1]+ travel_path_prob[airport][destAirport];
               //amrex::Print() <<" Original Airport "<< airport<<" Dest Airport "<< destAirport<< " travel path prob threshold "<< dest_airports_prob[curOffset]<<"\n";
               curOffset++;
           }
           dest_airports_prob[numDest-1]=1.0;//to avoid rounding error, we set the last option prob to 1.0
       }
       dest_airports_offset[nairports]= curOffset;
   }

   {
       //create 3 vectors of the arrivalUnits map that can be offloaded to the GPU
       arrivalUnits_offset.resize(nairports+1);
       arrivalUnits.resize(demo.Nunit);
       arrivalUnits_prob.resize(demo.Nunit);
       int curOffset=0;
       for(int i=0; i<nairports; i++){
           arrivalUnits_offset[i]= curOffset;
           std::string airport= id_to_airport[i];
           int nUnits= inAirportRangeUnitMap[airport].size();
           //all units served by this airport
           for(int j=0; j<nUnits; j++){
               int destUnit= inAirportRangeUnitMap[airport][j];
               arrivalUnits[curOffset]= destUnit;
               if(j==0)arrivalUnits_prob[curOffset]= (float)demo.Population[destUnit]/inAirportRangePop[airport];
               else arrivalUnits_prob[curOffset]=  arrivalUnits_prob[curOffset-1] + (float)demo.Population[destUnit]/inAirportRangePop[airport];
               //amrex::Print() <<" At Airport "<< airport <<" prob threshold to visit unit "<< destUnit<<" is "<< arrivalUnits_prob[curOffset] <<"\n";
               curOffset++;
           }
           arrivalUnits_prob[nUnits-1]= 1.0; //to avoid rounding error, we set the last option prob to 1.0
       }
       arrivalUnits_offset[nairports]= curOffset;
    }
    CopyDataToDevice();
}

/*! \brief Prints case data to screen

    For each disease hub (#CaseData::N_hubs), print the FIPS code, current number of cases,
    and cumulative number of cases till date.
*/
void AirTravelFlow::Print () const {
}
