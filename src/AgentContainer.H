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
                   const short a_ic_type);

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

    void moveAirTravel(const amrex::iMultiFab& unit_mf, AirTravelFlow& air, DemographicData& demo);

    void setAirTravel(const iMultiFab& unit_mf, AirTravelFlow& air, DemographicData& demo);

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

    std::array<amrex::Long, OutputStatus::nattribs> getTotals(const int);

    std::array<amrex::Long, AgeGroups::total> getNewStatusByAge(const int a_d, const int output_status);

    int getMaxGroup(const int group_idx);

    void moveAgentsToWork();

    void moveAgentsToHome();

    void assignSchoolClasses(const ExaEpi::TestParams& params);

    void setTileSize (IntVect newTileSize) { tile_size = newTileSize; }

    void setActiveWeatherWeek (int week) { activeWeatherWeek = week; }
    void initializeWeatherIndex(const iMultiFab&, ActiveWeather*);
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

    amrex::iMultiFab m_student_counts;
    /// Used only for Census data. A ratio for each school type: none, college, high, middle, elem, daycare
    amrex::GpuArray<int, SchoolType::total> m_student_teacher_ratio = {0, 15, 15, 15, 15, 15};

    /*!< Symptomatic withdrawal compliance rate, day 0, AgeGroups u5, 5-17, 18-29, 30-49, 50-64, 65+*/
    amrex::GpuArray<amrex::Real, AgeGroups::total> m_symptomatic_withdraw_compliance_day_0 = {0.3_rt, 0.3_rt, 0.3_rt,
                                                                                              0.3_rt, 0.3_rt, 0.3_rt};
    /*!< Symptomatic withdrawal compliance rate, day 1 */
    amrex::GpuArray<amrex::Real, AgeGroups::total> m_symptomatic_withdraw_compliance_day_1 = {0.8_rt, 0.6_rt, 0.5_rt,
                                                                                              0.5_rt, 0.5_rt, 0.5_rt};
    /*!< Symptomatic withdrawal compliance rate, day 2 */
    amrex::GpuArray<amrex::Real, AgeGroups::total> m_symptomatic_withdraw_compliance_day_2 = {0.9_rt, 0.8_rt, 0.7_rt,
                                                                                              0.7_rt, 0.7_rt, 0.7_rt};

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

    short ic_type;

    int activeWeatherWeek;
    ActiveWeather* awd;

  protected:
    amrex::Real m_shelter_compliance = 0.95_rt; /*!< Shelter-in-place compliance rate */

    std::vector<DiseaseParm*> m_h_parm;         /*!< Disease parameters */
    std::vector<DiseaseParm*> m_d_parm;         /*!< Disease parameters (GPU device) */

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

    std::vector<amrex::Gpu::DeviceVector<amrex::Real>> m_prob_snapshot; /*!< per-tile prob snapshot for context diagnostics */

    /*! \brief Add runtime SoA attributes */
    void addAttributes();
};

using AgentIterator = typename AgentContainer::ParIterType;

#endif
