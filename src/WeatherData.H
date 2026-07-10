#ifndef WEATHER_H_
#define WEATHER_H_

#include "DemographicData.H"
#include "UrbanPopData.H"
#include "Utils.H"
#include <AMReX_GpuContainers.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>
#include <string>

struct weatherVars {
    double hurs;
    double huss;
    double pr;
    double rlds;
    double rsds;
    double sfcWind;
    double tas;
    double tasmax;
    double tasmin;
    void print () {
        amrex::Print() << hurs << " " << huss << " " << pr << " " << rlds << " " << rsds << " " << sfcWind << " " << tas << " "
                       << tasmax << " " << tasmin << " ";
    }
};

struct date {
    int month;
    int day;
    int year;
    void print () { amrex::Print() << year << "-" << month << "-" << day << "\n"; }
    date () : month(-1), day(-1), year(-1) {}
    // date format: year-month-day
    date (std::string s) {
        std::istringstream iss(s);
        char hyphen;
        iss >> year;
        iss >> hyphen;
        iss >> month;
        iss >> hyphen;
        iss >> day;
    }
};

struct week {
    date begin;
    date end;
    bool straddle2Years () { return end.year - begin.year == 1; }
    bool straddle2Months () {
        if (straddle2Years()) { return true; }
        return end.month - begin.month == 1;
    }
    void print () {
        amrex::Print() << begin.year << "-" << begin.month << "-" << begin.day << "/";
        amrex::Print() << end.year << "-" << end.month << "-" << end.day << "\n";
    }
};

struct ActiveWeather {
    amrex::Gpu::DeviceVector<weatherVars> varVec; // keep weather variables during the simulation time
    amrex::Vector<weatherVars> h_varVec;
    amrex::Gpu::DeviceVector<int> unitVec;
    amrex::Vector<int> h_unitVec;
    int numUnitsWithDataOnThisProc;
    weatherVars& operator[](int index) { return varVec[index]; }
    weatherVars& nextWeekWeather (int curIdx) { return varVec[curIdx + numUnitsWithDataOnThisProc]; }
    inline int computeIndex (int unit, int simWeek) {
        bool found = false;
        size_t lunit = 0;
        for (; lunit < h_unitVec.size(); lunit++) {
            if (h_unitVec[lunit] == unit) {
                found = true;
                break;
            }
        }
        if (!found) { return -1; }
        return simWeek * numUnitsWithDataOnThisProc + lunit;
    }
    weatherVars* lookupWeatherVars (int unit, int simWeek) {
        if (computeIndex(unit, simWeek) == -1) { return NULL; }
        return &varVec[computeIndex(unit, simWeek)];
    }
    weatherVars* h_lookupWeatherVars (int unit, int simWeek) {
        if (computeIndex(unit, simWeek) == -1) { return NULL; }
        return &h_varVec[computeIndex(unit, simWeek)];
    }
    void print (int unit, int simWeek) { // gpu
        h_lookupWeatherVars(unit, simWeek)->print();
    }
};

struct WeatherData {
    // keep a map on the host for quick data lookup
    std::map<int, std::vector<weatherVars>> varMap;
    std::vector<week> weekVec;   // observation weeks
    int numWeeks;
    week firstWeek;              // first observation weeks
    week lastWeek;               // last observation weeks
    week simStartWeek;           // first simulation week
    ActiveWeather activeWeather; // part of the weather data affecting the simulation
    void readDataFromFile(const std::string& weatherFilename);
    bool computeIndex(date d, int& weekIndex, int& daysToWeatherWeekend);
    bool lookupWeatherVars(int stateFP, int countyFP, date d, int& weatherWeekIndex, weatherVars& vars);
    void extractActiveData(DemographicData& demo, int startWeek, int numSimWeeks);
    void extractActiveData(UrbanPopData& upop, int startWeek, int numSimWeeks);
    void print () {
        amrex::Print() << "STATEFP " << "COUNTYFP " << "hurs " << "huss " << "pr " << "rlds " << "rsds " << "sfcWind " << "tas "
                       << "tasmax " << "tasmin " << "Week\n";
        std::map<int, std::vector<weatherVars>>::iterator it;
        for (it = varMap.begin(); it != varMap.end(); it++) {
            for (size_t i = 0; i < it->second.size(); i++) {
                amrex::Print() << it->first / 1000 << " " << it->first % 1000 << " ";
                weatherVars vars = it->second[i];
                amrex::Print() << vars.hurs << " " << vars.huss << " " << vars.pr << " " << vars.rlds << " " << vars.rsds << " "
                               << vars.sfcWind << " " << vars.tas << " " << vars.tasmax << " " << vars.tasmin << " ";
                weekVec[i].print();
            }
        }
    }
};
#endif
