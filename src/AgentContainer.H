/*! @file AgentContainer.H
    \brief Contains #AgentContainer class and related structs
*/
#ifndef AGENT_CONTAINER_H_
#define AGENT_CONTAINER_H_

#include <array>
#include <string>
#include <vector>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include "AgentDefinitions.H"
#include "AirTravelFlow.H"
#include "DiseaseCoupling.H"
#include "DiseaseParm.H"
#include "DiseaseStatus.H"
#include "HospitalModel.H"
#include "InteractionModelLibrary.H"
#include "Utils.H"
#include "WeatherData.H"

struct UrbanPopData; // fwd decl only -- avoids a circular include with UrbanPopData.H, which
                     // itself forward-declares AgentContainer for the same reason

/*! \brief Realized group-size distributions for workgroups, schools, and school classes -- one
    entry per nonzero-count group (the group's size), not a per-community scalar. See
    AgentContainer::computeGroupSizeDistributions(). Populated (non-empty) on the IOProcessor
    only. */
struct GroupSizeAggregates {
    std::vector<amrex::Long> workgroup_sizes, school_sizes, school_class_sizes;
};

/*! \brief Realized neighborhood distributions -- one entry per nonempty neighborhood in
    nborhood_sizes (that neighborhood's agent count) and one per populated community in
    nborhoods_per_community (how many nonempty neighborhoods it has). Describes either the night
    (home) or the day (wherever the day is spent) neighborhoods, depending on which attribute it
    was computed from. Separate from GroupSizeAggregates because each has to be counted with
    agents in the matching position -- see AgentContainer::computeNborhoodSizeDistributions().
    Populated (non-empty) on the IOProcessor only. */
struct NborhoodSizeAggregates {
    std::vector<amrex::Long> nborhood_sizes, nborhoods_per_community;
};

/*! \brief Derived class from ParticleContainer that defines agents and their functions */
class AgentContainer : public amrex::ParticleContainer<0, 0, RealIdx::nattribs, IntIdx::nattribs> {

    using PCType = AgentContainer;
    using PType = ParticleType;
    using PTileType = ParticleTileType;
    using PTDType = PTileType::ParticleTileDataType;

    using MFPtr = std::unique_ptr<MultiFab>;
    using MFPtrVec = std::vector<MFPtr>;

  public:
    AgentContainer(const amrex::Geometry& a_geom, const amrex::DistributionMapping& a_dmap, const amrex::BoxArray& a_ba,
                   const int& a_num_diseases, const std::vector<std::string>& a_disease_names, const bool fast,
                   const int verbose = ExaEpi::Verbosity::quiet);

    void morningCommute(amrex::MultiFab&);

    void eveningCommute(amrex::MultiFab&);

    void interactDay(amrex::MultiFab&);

    void interactWork(amrex::MultiFab&);
    void interactHospital(amrex::MultiFab&);
    void interactSchool(amrex::MultiFab&);
    void interactNborhoodDay(amrex::MultiFab&);
    void interactCommDay(amrex::MultiFab&);

    void interactEvening(amrex::MultiFab&);

    void interactNight(amrex::MultiFab&);

    void interactHH(amrex::MultiFab&);
    void interactNC(amrex::MultiFab&);
    void interactNborhoodNight(amrex::MultiFab&);
    void interactCommNight(amrex::MultiFab&);

    void moveAirTravel(const amrex::iMultiFab& unit_mf, AirTravelFlow& air);

    void setAirTravel(const iMultiFab& unit_mf, AirTravelFlow& air, const AirTravelDemoData& demo);

    void moveAgentsRandomWalk();

    void moveRandomTravel(const amrex::Real random_travel_prob);

    void returnRandomTravel();

    void returnAirTravel();

    void updateStatus(MFPtrVec&);

    void infectAgents(MFPtrVec&);

    amrex::Real sumExpectedInfections(int d) const;

    void snapshotProbs(int d);
    amrex::Real sumContextInfections(int d);

    void shelterStart();

    void shelterStop();

    void generateCellData(amrex::MultiFab& mf, const int ncomp) const;

    /*! \brief Fill a 3-component MultiFab (same BoxArray/DistributionMapping as this container)
        with, per grid cell, the count of agents CURRENTLY located there, broken out as:
        component 0 = everyone, 1 = workers (naics != -1), 2 = students (naics == -1 &&
        school_id != 0). Uses each agent's own current physical position (via ParticleToMesh), so
        the result reflects home population if called while agents are at home, or work/school
        population if called right after moveAgentsToWork() -- see main.cpp's static-aggregated-data
        block, which relies on the same guarantee: moveAgentsToWork()'s
        Redistribute() ensures every agent's tile matches its current (work) position, so a
        per-tile community lookup (e.g. UrbanPopData::community_mf) is valid there exactly as it
        is for the resting (home) state generateCellData() already assumes. */
    void generatePopulationBreakdown(amrex::MultiFab& mf) const;

    std::array<amrex::Long, OutputStatus::nattribs> getTotals(const int);

    std::array<amrex::Long, AgeGroups::total> getNewStatusByAge(const int a_d, const int output_status);

    std::array<amrex::Long, AgeGroups::total> getPopulationByAge();

    std::array<amrex::Long, AgeGroups::total> getInfectedByAge(const int a_d);

    int getMaxGroup(const int group_idx);

    /*! \brief Compute the realized size distributions of workgroups, schools, and school classes
        (see GroupSizeAggregates) from agents' static work_i/work_j/naics/workgroup/school_id/
        school_class_group attributes. NOT const: calls getMaxGroup(), which mutates a cached
        MPI-reduced value the first time each attribute is queried. Must be called once, while
        agents are at their work location (see main.cpp's call site, immediately after
        morningCommute()) -- workgroup/school_id are only unique within a work community, so
        this relies on the same current-position-is-work-cell guarantee generatePopulationBreakdown()
        documents above. Reduces to the IOProcessor only (not all ranks), since only the
        IOProcessor writes the resulting output files. */
    GroupSizeAggregates computeGroupSizeDistributions(const UrbanPopData& urbanpopData);

    /*! \brief Compute the realized neighborhood distributions (see NborhoodSizeAggregates) from
        `nborhood_attr`, which selects the night or the day neighborhood exactly as
        InteractionModNborhood.H's two instances do: IntIdx::nborhood for the home neighborhood,
        IntIdx::work_nborhood for the one the agent spends the day in.

        Agents must be in the position matching the attribute -- at HOME for nborhood (between
        eveningCommute() and morningCommute()), at WORK for work_nborhood -- because a neighborhood
        ID is only unique within the community it is read in, so counting it from the wrong
        position silently merges unrelated neighborhoods. See main.cpp's two call sites, one on
        each side of the commute.

        NOT const, for the same getMaxGroup() reason as computeGroupSizeDistributions() above.
        Reduces to the IOProcessor only. */
    NborhoodSizeAggregates computeNborhoodSizeDistributions(const UrbanPopData& urbanpopData,
                                                            int nborhood_attr = IntIdx::nborhood);

    void moveAgentsToWork();

    void moveAgentsToHome();

    void setTileSize (IntVect newTileSize) { tile_size = newTileSize; }

    void setActiveWeatherWeek (int week) { activeWeatherWeek = week; }
    void initializeWeatherIndex_UrbanPop(const iMultiFab&, ActiveWeather*);
    void advanceWeatherIndex();

    /*! \brief Return flag indicating if agents are at work */
    inline bool isAtWork () const { return m_at_work; }

    /*! \brief Return disease parameters object pointer (host) */
    inline const DiseaseParm* getDiseaseParameters_h (int d /*!< disease index */) const { return m_h_parm[d]; }

    /*! \brief Return disease parameters object pointer (device) */
    inline const DiseaseParm* getDiseaseParameters_d (int d /*!< disease index */) const { return m_d_parm[d]; }

    /*! \brief Return the number of diseases */
    inline int numDiseases () const { return m_num_diseases; }

    void printStudentTeacherCounts() const;

    void printAgeGroupCounts() const;

    /*! Default values for the members below, exposed as named constants (rather than bare
     *  initializers) so that main.cpp's printHelp() can print the actual defaults instead of
     *  a hardcoded copy that can drift out of sync. */
    static constexpr amrex::GpuArray<amrex::Real, AgeGroups::total> default_symptomatic_withdraw_compliance_day_0 =
            {0.3_rt, 0.3_rt, 0.3_rt, 0.3_rt, 0.3_rt, 0.3_rt};
    static constexpr amrex::GpuArray<amrex::Real, AgeGroups::total> default_symptomatic_withdraw_compliance_day_1 =
            {0.8_rt, 0.7_rt, 0.5_rt, 0.5_rt, 0.5_rt, 0.5_rt};
    static constexpr amrex::GpuArray<amrex::Real, AgeGroups::total> default_symptomatic_withdraw_compliance_day_2 =
            {0.9_rt, 0.8_rt, 0.7_rt, 0.7_rt, 0.7_rt, 0.7_rt};
    static constexpr amrex::Real default_shelter_compliance = 0.95_rt; /*!< Shelter-in-place compliance rate */

    /*!< Symptomatic withdrawal compliance rate, day 0, AgeGroups u5, 5-17, 18-29, 30-49, 50-64, 65+*/
    amrex::GpuArray<amrex::Real, AgeGroups::total> m_symptomatic_withdraw_compliance_day_0 =
            default_symptomatic_withdraw_compliance_day_0;
    /*!< Symptomatic withdrawal compliance rate, day 1 */
    amrex::GpuArray<amrex::Real, AgeGroups::total> m_symptomatic_withdraw_compliance_day_1 =
            default_symptomatic_withdraw_compliance_day_1;
    /*!< Symptomatic withdrawal compliance rate, day 2 */
    amrex::GpuArray<amrex::Real, AgeGroups::total> m_symptomatic_withdraw_compliance_day_2 =
            default_symptomatic_withdraw_compliance_day_2;

    int m_num_diseases; /*!< Number of diseases */

    amrex::iMultiFab comm_mf;

    /*! Per-community size scale factor multiplying xmit_comm/xmit_hood for NIGHTTIME interactions
        (UrbanPop only), keyed on home_population -- see TestParams::size_scale_enabled /
        computeCommunitySizeScale. Always valid (flat 1.0 unless size-scale supplies real values in
        main.cpp) -- see InteractionModComm.H / InteractionModNborhood.H. */
    amrex::MultiFab comm_density_scale;

    /*! Per-community scale factor multiplying xmit_comm/xmit_hood for DAYTIME interactions
        (UrbanPop only), keyed on work_populations[0] instead of home_population -- see
        TestParams::size_scale_enabled / computeCommunityWorkSizeScale. Always valid (flat 1.0
        unless work-size-scale supplies real values in main.cpp). */
    amrex::MultiFab comm_density_scale_work;

    int activeWeatherWeek;
    ActiveWeather* awd;

  protected:
    amrex::Real m_shelter_compliance = default_shelter_compliance;

    std::vector<DiseaseParm*> m_h_parm; /*!< Disease parameters */
    std::vector<DiseaseParm*> m_d_parm; /*!< Disease parameters (GPU device) */

    std::unique_ptr<DiseaseCouplingParm<PTDType>> m_disease_coupling;

    InteractionModWork<PCType, PTDType, PType> m_mod_work;
    InteractionModSchool<PCType, PTDType, PType> m_mod_school;
    InteractionModHH<PCType, PTDType, PType> m_mod_hh;
    InteractionModNC<PCType, PTDType, PType> m_mod_nc;
    InteractionModNborhood<PCType, PTDType, PType> m_mod_nborhood_day;
    InteractionModNborhood<PCType, PTDType, PType> m_mod_nborhood_night;
    InteractionModComm<PCType, PTDType, PType> m_mod_comm_day;
    InteractionModComm<PCType, PTDType, PType> m_mod_comm_night;
    InteractionModAirTravel<PCType, PTDType, PType> m_mod_air_travel;
    std::unique_ptr<HospitalModel<PCType, PTDType, PType>> m_hospital; /*!< hospital model */

    /*! Flag to indicate if agents are at work */
    bool m_at_work;

    /*! Disease status update model */
    DiseaseStatus<PCType, PTileType, PTDType, PType> m_disease_status;

    Array<int, IntIdx::nattribs> max_attribute_values;

    std::vector<amrex::Gpu::DeviceVector<amrex::ParticleReal>>
            m_prob_snapshot; /*!< per-tile prob snapshot for context diagnostics */

    /*! \brief Add runtime SoA attributes */
    void addAttributes();
};

using AgentIterator = typename AgentContainer::ParIterType;

#endif
