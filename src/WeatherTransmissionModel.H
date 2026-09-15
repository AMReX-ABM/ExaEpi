/*! @file WeatherTransmissionModel.H
    \brief #WeatherTransmissionModel struct definition
*/

#ifndef WEATHER_TRANSMISSION_MODEL_H_
#define WEATHER_TRANSMISSION_MODEL_H_

#include <AMReX_Algorithm.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include "WeatherData.H"

#include <cmath>
#include <string>

/*! \brief Weather-dependent transmission probability model.
 *
 *  Computes a weather-based scale factor for the transmission probability as a
 *  function of near-surface air temperature and absolute humidity:
 *
 *  \f[
 *    p(T,AH) = \min\left(1,\; p_max \exp(-beta_AH (AH - AH_{ref}) - alpha_T \max(0,T - T_0))\right)
 *  \f]
 *
 *  where \f$T\f$ is temperature in degrees Celsius and \f$AH\f$ is absolute
 *  humidity in g/m^3.  The exponential is referenced at \f$AH_{ref}\f$ (rather
 *  than at \f$AH=0\f$, which is never reached in practice): with \c p_max = 1
 *  the scale factor equals 1 at/below \f$AH_{ref}\f$ and \f$T \le T_0\f$, so the
 *  calibrated baseline transmission probability is recovered during flu-season
 *  conditions and weather only attenuates it toward summer.  The result is
 *  clamped to 1 so that \f$1 - infect \cdot xmit \cdot p \ge 0\f$ always holds
 *  (the value is used as a multiplier on the baseline \c disease.p_trans /
 *  \c xmit_* probabilities).  The compute function (#operator()) is decorated
 *  with AMREX_GPU_HOST_DEVICE so it may be called from within amrex::ParallelFor
 *  kernels on both CPU and GPU backends.
 *
 *  Parameters are parsed from the inputs file under the prefix supplied to
 *  #readInputs (typically \c "weather_transmission"):
 *  | Key                             | Default | Units       |
 *  |---------------------------------|---------|-------------|
 *  | \c weather_transmission.p_max   | 1.00    | —           |
 *  | \c weather_transmission.beta_AH | 0.18    | m^3/g       |
 *  | \c weather_transmission.T0      | 5.0     | °C          |
 *  | \c weather_transmission.alpha_T | 0.015   | °C^{-1}     |
 *  | \c weather_transmission.AH_ref  | 6.0     | g/m^3       |
 *
 *  Set \c AH_ref = 0 (with \c p_max = 1) to recover the un-normalised behaviour.
 */
struct WeatherTransmissionModel {

    /*! Maximum transmission probability (dimensionless). */
    amrex::Real p_max = amrex::Real(1.0);

    /*! Absolute-humidity sensitivity coefficient (m^3/g).
     *  Higher values make transmission drop faster with increasing humidity. */
    amrex::Real beta_AH = amrex::Real(0.18);

    /*! Temperature threshold above which the temperature penalty applies (°C). */
    amrex::Real T0 = amrex::Real(5.0);

    /*! Temperature sensitivity coefficient (°C^{-1}).
     *  Higher values make transmission drop faster above T0. */
    amrex::Real alpha_T = amrex::Real(0.015);

    /*! Reference absolute humidity (g/m^3) at which the weather scale factor
     *  equals p_max (i.e. the clamped scale factor is 1 for p_max = 1 when
     *  T <= T0).  Set this near the flu-season AH so the calibrated baseline
     *  transmission probability is recovered during the epidemic season.
     *  Set to 0 to recover the un-normalised AH=0 reference. */
    amrex::Real AH_ref = amrex::Real(6.0);

    /*! \brief Parse model parameters from the AMReX inputs file.
     *
     *  Call this on the host during initialisation, passing the ParmParse
     *  prefix (e.g. \c "weather_transmission").  All four parameters are
     *  optional; unspecified keys retain their default values.
     *
     *  \param pp_str  ParmParse prefix string.
     */
    void readInputs(const std::string& pp_str);

    /*! \brief Evaluate the transmission probability.
     *
     *  This function is callable from device code and may therefore be used
     *  inside \c amrex::ParallelFor lambdas.
     *
     *  \param T_celsius    Near-surface air temperature (°C).
     *  \param AH_g_per_m3  Absolute humidity (g/m^3).
     *  \return             Weather transmission scale factor \f$p(T,AH) \in (0,1]\f$.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator()(amrex::Real T_celsius, amrex::Real AH_g_per_m3) const noexcept {
        amrex::Real T_excess = amrex::max(amrex::Real(0), T_celsius - T0);
        amrex::Real rho = p_max * std::exp(-beta_AH * (AH_g_per_m3 - AH_ref) - alpha_T * T_excess);
        return amrex::min(amrex::Real(1), rho); // rho <= 1 keeps 1 - infect*xmit*rho >= 0
    }

    /*! \brief Evaluate the transmission probability directly from a #weatherVars record.
     *
     *  Converts \c wvar.tas (surface temperature, K) to Celsius and
     *  \c wvar.huss (specific humidity, kg/kg) to absolute humidity (g/m^3)
     *  using a standard sea-level air density of 1.225 kg/m^3, then
     *  delegates to the two-argument overload.
     *
     *  \param wvar  Weather variables for the agent's location.
     *  \return      Transmission probability \f$p(T,AH)\f$.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real operator()(const weatherVars& wvar) const noexcept {
        // tas is in Kelvin; convert to Celsius
        amrex::Real T_celsius = static_cast<amrex::Real>(wvar.tas) - amrex::Real(273.15);
        // huss is specific humidity [kg vapor / kg moist air]; multiply by standard
        // air density (1.225 kg/m^3) and convert to g/m^3 to get absolute humidity
        amrex::Real AH_g_per_m3 = static_cast<amrex::Real>(wvar.huss) * amrex::Real(1225.0);
        return (*this)(T_celsius, AH_g_per_m3);
    }
};

#endif
