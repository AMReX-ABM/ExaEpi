/*! @file WeatherTransmissionModel.cpp
    \brief Function implementations for #WeatherTransmissionModel
*/

#include "WeatherTransmissionModel.H"

#include <AMReX_ParmParse.H>

/*! \brief Read weather-transmission model parameters from the inputs file. */
void WeatherTransmissionModel::readInputs (const std::string& pp_str)
{
    amrex::ParmParse pp(pp_str);
    pp.query("p_max",   p_max);
    pp.query("beta_AH", beta_AH);
    pp.query("T0",      T0);
    pp.query("alpha_T", alpha_T);
}
