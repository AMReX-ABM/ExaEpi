/*! @file RandomTravel.H
    \brief Irregular (long-distance) travel, after Epicast 2.0's irregular travel model

    Epicast 2.0 (Alexander et al. 2025, sec. 2.5.2) models infrequent long trips -- vacations,
    business travel, family visits -- with three parts: when a trip starts, where it goes, and how
    long it lasts. This reproduces it as closely as the paper and Epicast's event logs allow:

    - Start: an agent who is not travelling, hospitalized or withdrawn starts a trip with
      probability p_travel * P_travel(age), P_travel from Epicast's Table 3, once a day by default.
      The paper gives these per time step (Epicast's are 12 hours); steps_per_day = 2 applies them
      twice a day instead, as 1 - (1 - p)^2.
    - Destination: Epicast draws a destination state from its home state's row of a 2019 flight
      count matrix, then a uniformly random tract in that state and a uniformly random community in
      that tract. Here a trip goes to a uniformly random tract of the simulated region, then a
      uniformly random block group (ExaEpi's community) in it. in_state_frac < 1 sends the rest of
      the trips out of the region (in Epicast's flights.csv the home state's own share is NM 0.116,
      CA 0.244): out_of_state = remove takes such a traveller out of the simulation for the trip --
      it neither infects nor can be infected -- and skip doesn't take the trip.
    - Duration: whole days, from the eCDF of Epicast's Figure 4B (digitized; 2 to 22 days, mean
      ~6.2).
    - At the destination, the traveller joins a uniformly random household cluster, nighttime
      neighborhood and daytime neighborhood of the destination block group, and its community, day
      and night -- Epicast logs travellers' infections only in the household-cluster and
      neighborhood/community contexts. It leaves its household, workplace and school behind.

    The defaults (once a day, every trip in the region) are what reproduces Epicast's own NM
    single-state run with p_travel = 1 (seed 0, against its 30 runs without travel; ExaEpi with the
    NM transmission tune, seed 10): attack rate -1.3 points in Epicast, -1.4 here; statewide peak
    +7.8% and +9.8%; infections on trips >= 0.68% of all in Epicast (it logs only the night ones)
    and 1.2% here, day and night. Epicast logged no traveller outside NM. Twice a day gave -2.9
    points and 2.5% of infections on trips; removing the out-of-state trips at NM's 0.116 gave -6.0
    (twice a day) or -3.2 (once) and a lower peak, the opposite of Epicast.

    The trip state lives in the agent's IntIdx::random_travel (days left, -1 when home),
    IntIdx::trav_i/trav_j (destination cell, -1 when out of state) and IntIdx::trav_nborhood /
    trav_work_nborhood / trav_hh_cluster (destination groups). See AgentContainer::initRandomTravel
    and AgentContainer::updateRandomTravel.
*/

#ifndef RANDOM_TRAVEL_H_
#define RANDOM_TRAVEL_H_

#include <vector>

#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include "AgentDefinitions.H"

struct RandomTravel {
    /*! Epicast Table 3 (ages 0-5, 6-17, 18-29, 30-64, 65+) on ExaEpi's age groups; 30-49 and
        50-64 both take 30-64's value. Per Epicast time step. */
    static constexpr amrex::GpuArray<amrex::Real, AgeGroups::total> default_prob_per_step = {amrex::Real(0.0023),
                                                                                             amrex::Real(0.0023),
                                                                                             amrex::Real(0.0050),
                                                                                             amrex::Real(0.0053),
                                                                                             amrex::Real(0.0053),
                                                                                             amrex::Real(0.0028)};
    /*! Epicast Figure 4B, digitized: probability that a trip lasts at most d days, d = 1, 2, ... */
    static inline const std::vector<amrex::Real> default_duration_cdf = {0.000, 0.324, 0.407, 0.493, 0.575, 0.658, 0.743, 0.767,
                                                                         0.791, 0.814, 0.838, 0.861, 0.885, 0.909, 0.932, 0.944,
                                                                         0.956, 0.968, 0.979, 0.985, 0.994, 1.000};

    /*! Per-step start probability by age group (agent.random_travel_prob) */
    amrex::GpuArray<amrex::Real, AgeGroups::total> prob_per_step = default_prob_per_step;
    amrex::Real scale = 1.0;         /*!< Epicast's p_travel (agent.random_travel_scale) */
    int steps_per_day = 1;           /*!< start draws per ExaEpi day (agent.random_travel_steps_per_day) */
    amrex::Real in_state_frac = 1.0; /*!< share of trips to the simulated state (agent.random_travel_in_state_frac) */
    bool remove_out_of_state = true; /*!< agent.random_travel_out_of_state: "remove" (true) or "skip" */
    std::vector<amrex::Real> duration_cdf = default_duration_cdf; /*!< agent.random_travel_duration_cdf */

    /*! Derived: daily start probability by age group */
    amrex::GpuArray<amrex::Real, AgeGroups::total> prob_per_day;

    /*! Destination tables (device), built by AgentContainer::initRandomTravel: block groups with
        residents grouped by tract (bg_by_tract, offsets tract_offset), and per block group the
        number of nighttime neighborhoods, daytime neighborhoods and household clusters */
    amrex::Gpu::DeviceVector<amrex::Real> duration_cdf_d;
    amrex::Gpu::DeviceVector<int> tract_offset_d;
    amrex::Gpu::DeviceVector<int> bg_by_tract_d;
    amrex::Gpu::DeviceVector<int> n_nborhood_d;
    amrex::Gpu::DeviceVector<int> n_work_nborhood_d;
    amrex::Gpu::DeviceVector<int> n_hh_cluster_d;
    int num_tracts = 0;
    int nx = 0; /*!< grid width: block group b is at cell (b % nx, b / nx) */

    /*! Run totals for the end-of-run summary (agent-days and expected infections, disease 0) */
    double agent_days_in_state = 0.0;
    double agent_days_away = 0.0;
    double infections_on_trip = 0.0;
    double infections_all = 0.0;
    int days = 0;

    /*! Read the agent.random_travel_* inputs and derive prob_per_day */
    void readInputs();
};

#endif
