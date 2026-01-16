#pragma once

#include <AMReX_GpuContainers.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>
#include <string>
#include "Utils.H"
#include "CensusData.H"

struct weatherVars{
    double hurs;
    double huss;
    double pr;
    double rlds;
    double rsds;
    double sfcWind;
    double tas;
    double tasmax;
    double tasmin;
    void print(){
        std::cout<<hurs <<" " <<huss <<" " <<pr <<" " <<rlds <<" " <<rsds <<" " <<sfcWind <<" " <<tas <<" " <<tasmax <<" " <<tasmin <<" ";
    }
};

struct date{
    int month;
    int day;
    int year;
    void print(){
        std::cout<<year <<"-" <<month <<"-" <<day <<"\n";
    }
    date():month(-1), day(-1), year(-1){}
    date(std::string s){
        std::istringstream iss(s);
	char hyphen;
        iss >> year; iss >> hyphen; iss >> month; iss >> hyphen; iss >> day;
    }
};

struct week{
    date begin;
    date end;
    bool straddleYears(){
        return end.year - begin.year == 1;
    }
    bool straddleMonths(){
        if (straddleYears()) return true;
        return end.month - begin.month ==1;
    }
    void print(){
        std::cout<<begin.year <<"-" <<begin.month <<"-" <<begin.day <<"/";
        std::cout<<end.year <<"-" <<end.month <<"-" <<end.day <<"\n";
    }
};

struct WeatherData {
    //keep a map on the host for quick data lookup
    std::map<int, std::vector<weatherVars> > varMap;
    std::vector<week> weekVec; //observation weeks
    int numWeeks;
    week firstWeek; //first observation weeks
    week lastWeek; //last observation weeks
    //WeatherData () {}
    amrex::Gpu::DeviceVector<weatherVars> activeWeather;//part of the weather data affecting the simulation
    void readDataFromFile(const std::string& weatherFilename);
    bool lookupIndex(date d, int& weekIndex, int& daysToWeatherWeekend);
    bool lookupWeatherVars(int stateFP, int countyFP, date d, int& weekIndex, weatherVars& vars);
    void extractData(CensusData& census, int startWeek, int numWeeks);
    void print(){
        std::cout<<"STATEFP " <<"COUNTYFP " <<"hurs " <<"huss " <<"pr " <<"rlds " <<"rsds " <<"sfcWind "<< "tas "<< "tasmax "<< "tasmin "<<"Week\n";   
        std::map<int, std::vector<weatherVars> >::iterator it;
        for(it= varMap.begin(); it!= varMap.end(); it++){
            for(int i=0; i<it->second.size(); i++){
	        std::cout<<it->first/1000 <<" "<<it->first%1000<<" ";
	        weatherVars vars= it->second[i];
	        std::cout<<vars.hurs <<" " <<vars.huss <<" " <<vars.pr <<" " <<vars.rlds <<" " <<vars.rsds <<" " <<vars.sfcWind <<" " <<vars.tas <<" " <<vars.tasmax <<" " <<vars.tasmin <<" ";
                weekVec[i].print();
	    }
        }
    }
};
