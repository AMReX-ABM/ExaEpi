#include "WeatherData.H"
#include <AMReX_ParticleUtil.H>

using namespace amrex;
using namespace ExaEpi;

void WeatherData::readDataFromFile (const std::string& fname) {
    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);
    std::string line;
    if (!is.eof()) {
        getline(is, line); // read header line
        std::istringstream lis(line);
    }
    while (!is.eof()) {
        std::getline(is, line);
        if (line.size() == 0) { break; }
        std::istringstream lis(line);
        std::string temp[17];
        int i = 0;
        while (i < 17 && std::getline(lis, temp[i], ',')) {
            i++;
        }
        weatherVars vars;
        int STATEFP = std::stoi(temp[1]);
        int COUNTYFP = std::stoi(temp[2]);
        vars.hurs = temp[3].empty() ? 0. : std::stod(temp[3]);
        vars.huss = temp[4].empty() ? 0. : std::stod(temp[4]);
        vars.pr = temp[7].empty() ? 0. : std::stod(temp[7]);
        vars.rlds = temp[8].empty() ? 0. : std::stod(temp[8]);
        vars.rsds = temp[9].empty() ? 0. : std::stod(temp[9]);
        vars.sfcWind = temp[11].empty() ? 0. : std::stod(temp[11]);
        vars.tas = temp[12].empty() ? 0. : std::stod(temp[12]);
        vars.tasmax = temp[13].empty() ? 0. : std::stod(temp[13]);
        vars.tasmin = temp[14].empty() ? 0. : std::stod(temp[14]);
        varMap[STATEFP * 1000 + COUNTYFP].push_back(vars);
        if (weekVec.size() < varMap[STATEFP * 1000 + COUNTYFP].size()) {
            date d0, d1;
            char ch;
            std::string week = temp[15];
            std::istringstream iss(week);
            iss >> ch;
            iss >> ch;
            iss >> d0.year;
            iss >> ch;
            iss >> d0.month;
            iss >> ch;
            iss >> d0.day;
            iss >> ch;
            iss >> d1.year;
            iss >> ch;
            iss >> d1.month;
            iss >> ch;
            iss >> d1.day;
            weekVec.push_back({d0, d1});
        }
    }
    if (varMap.size() > 0) {
        numWeeks = varMap.begin()->second.size();
        firstWeek = weekVec[0];
        lastWeek = weekVec.back();
    } else {
        numWeeks = 0;
    }
}

bool WeatherData::computeIndex (date d, int& weekIndex, int& daysToWeatherWeekend) {
    if ((d.year < firstWeek.begin.year) || (d.year > lastWeek.end.year)) {
        return false;
    } else {
        if (((d.year == firstWeek.begin.year) && (d.month < firstWeek.begin.month)) ||
            ((d.year == lastWeek.end.year) && (d.month > lastWeek.end.month))) {
            return false;
        } else {
            if (((d.year == firstWeek.begin.year) && (d.month == firstWeek.begin.month) && (d.day < firstWeek.begin.day)) ||
                ((d.year == lastWeek.end.year) && (d.month == lastWeek.end.month) && (d.day > lastWeek.end.day))) {
                return false;
            }
        }
    }
    int approxStartWeeks = std::max(d.year - 1 - firstWeek.begin.year, 0) * 52;
    int i = approxStartWeeks;
    for (; i < numWeeks; i++) {
        if (weekVec[i].begin.year == d.year) {
            if (weekVec[i].begin.month == d.month) {
                if (weekVec[i].straddle2Months()) {
                    if (d.day >= weekVec[i].begin.day) {
                        daysToWeatherWeekend = 7 - (d.day - weekVec[i].begin.day);
                        weekIndex = i;
                        return true;
                    }
                } else {
                    if (d.day >= weekVec[i].begin.day && d.day <= weekVec[i].end.day) {
                        daysToWeatherWeekend = weekVec[i].end.day - d.day + 1;
                        weekIndex = i;
                        return true;
                    }
                }
            } else {
                if (weekVec[i].straddle2Months()) {
                    if (weekVec[i].end.month == d.month) {
                        if (d.day <= weekVec[i].end.day) {
                            daysToWeatherWeekend = weekVec[i].end.day - d.day + 1;
                            weekIndex = i;
                            return true;
                        }
                    }
                }
            }
        } else {
            if (weekVec[i].straddle2Years()) {
                if (d.month == 1 && d.day <= weekVec[i].end.day) {
                    daysToWeatherWeekend = weekVec[i].end.day - d.day + 1;
                    weekIndex = i;
                    return true;
                }
            }
        }
    }
    return false;
}

bool WeatherData::lookupWeatherVars (int stateFP, int countyFP, date d, int& weekIndex, weatherVars& vars) {
    int daysToWeatherWeekend;
    bool found = computeIndex(d, weekIndex, daysToWeatherWeekend);
    if (found) {
        vars = varMap[stateFP * 1000 + countyFP][weekIndex];
#if 0
        amrex::Print() << "STATEFP " << "COUNTYFP " << "hurs " << "huss " << "pr " << "rlds " << "rsds " << "sfcWind " << "tas " << "tasmax " << "tasmin " << "WeekBeginDate\n";
        amrex::Print() << stateFP << " " << countyFP << " ";
        amrex::Print() << vars.hurs << " " << vars.huss << " " << vars.pr << " " <<vars.rlds << " " << vars.rsds << " " << vars.sfcWind << " " << vars.tas << " " << vars.tasmax << " " << vars.tasmin << " ";
#endif
        return true;
    }
    return false;
}

void WeatherData::extractActiveData (DemographicData& demo, int startWeek, int numSimWeeks) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(startWeek + numSimWeeks <= numWeeks,
                                     "Weather data does not cover the full simulation period: "
                                     "reduce nsteps or provide a weather file with more data.");
    int numUnitsOnThisProc = 0;
    for (int i = 0; i < demo.Nunit; i++) {
        if (demo.Unit_on_proc[i]) { numUnitsOnThisProc++; }
    }
    activeWeather.h_unitVec.resize(numUnitsOnThisProc);
    // set up the map from local unit index to global unit index
    int idx = 0;
    activeWeather.numUnitsWithDataOnThisProc = 0; // we are going to ignore units that don't have weather data
    for (int unit = 0; unit < demo.Nunit; unit++) {
        if (demo.Unit_on_proc[unit]) {
            int FIPS = demo.FIPS[unit];
            if (varMap.find(FIPS) != varMap.end()) {
                activeWeather.h_unitVec[idx] = unit;
                activeWeather.numUnitsWithDataOnThisProc++;
                idx++;
            }
        }
    }
    activeWeather.h_unitVec.resize(activeWeather.numUnitsWithDataOnThisProc);
    activeWeather.h_varVec.resize(activeWeather.numUnitsWithDataOnThisProc * numSimWeeks);
    for (int week = startWeek; week < startWeek + numSimWeeks; week++) {
        int offset = (week - startWeek) * activeWeather.numUnitsWithDataOnThisProc;
        int idx1 = 0;
        for (int unit = 0; unit < demo.Nunit; unit++) {
            if (demo.Unit_on_proc[unit]) {
                int FIPS = demo.FIPS[unit];
                if (varMap.find(FIPS) != varMap.end()) {
                    activeWeather.h_varVec[offset + idx1] = varMap[FIPS][week];
                    idx1++;
                } else {
                    // amrex::Print() << "Weather data NOT available in county with FIPS code " << FIPS << "\n";
                }
            }
        }
    }
    activeWeather.copyToDevice();
}

void WeatherData::extractActiveData (UrbanPopData& upop, int startWeek, int numSimWeeks) {
    int numWeatherUnits = upop.FIPS_codes.size();
    activeWeather.h_unitVec.resize(numWeatherUnits);
    // set up the map from local unit index to global unit index
    int idx = 0;
    activeWeather.numUnitsWithDataOnThisProc = 0; // we are going to ignore units that don't have weather data
    for (int unit = 0; unit < numWeatherUnits; unit++) {
        if (upop.County_on_proc[unit]) {
            int FIPS = upop.FIPS_codes[unit];
            if (varMap.find(FIPS) != varMap.end()) {
                activeWeather.h_unitVec[idx] = FIPS;
                activeWeather.numUnitsWithDataOnThisProc++;
                idx++;
            }
        }
    }

    activeWeather.h_unitVec.resize(activeWeather.numUnitsWithDataOnThisProc);
    activeWeather.h_varVec.resize(activeWeather.numUnitsWithDataOnThisProc * numSimWeeks);

    for (int week = startWeek; week < startWeek + numSimWeeks; week++) {
        int offset = (week - startWeek) * activeWeather.numUnitsWithDataOnThisProc;
        int idx1 = 0;
        for (int unit = 0; unit < numWeatherUnits; unit++) {
            if (upop.County_on_proc[unit]) {
                int FIPS = upop.FIPS_codes[unit];
                if (varMap.find(FIPS) != varMap.end()) {
                    activeWeather.h_varVec[offset + idx1] = varMap[FIPS][week];
                    idx1++;
                } else {
                    // amrex::Print() << "Weather data NOT available in county with FIPS code " << FIPS << "\n";
                }
            }
        }
    }
    activeWeather.copyToDevice();
}
