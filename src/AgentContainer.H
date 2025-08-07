/*! @file AgentContainer.H
    \brief Contains #AgentContainer class and related structs
*/
#ifndef AGENT_CONTAINER_H_
#define AGENT_CONTAINER_H_

#include <array>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_IntVect.H>
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

struct LngLatToGrid {
    Real min_lng, min_lat;
    Real gspacing_x, gspacing_y;

    LngLatToGrid (Real _min_lng, Real _min_lat, Real _gspacing_x, Real _gspacing_y) {
        init(_min_lng, _min_lat, _gspacing_x, _gspacing_y);
    }

    LngLatToGrid () {}

    void init (Real _min_lng, Real _min_lat, Real _gspacing_x, Real _gspacing_y) {
        min_lng = _min_lng;
        min_lat = _min_lat;
        gspacing_x = _gspacing_x;
        gspacing_y = _gspacing_y;
    }

    AMREX_GPU_HOST_DEVICE
    void operator()(Real lng, Real lat, int& x, int& y) const {
        Real rx = (lng - min_lng) / gspacing_x;
        Real ry = (lat - min_lat) / gspacing_y;
        x = static_cast<int>(rx + 0.001_rt);
        y = static_cast<int>(ry + 0.001_rt);
    }
};

/*! Convert grid coordinates to latitude and longitude for UrbanPop data */
struct GridToLngLat {
    Real min_lng, min_lat;
    Real gspacing_x, gspacing_y;

    GridToLngLat (Real _min_lng, Real _min_lat, Real _gspacing_x, Real _gspacing_y) {
        init(_min_lng, _min_lat, _gspacing_x, _gspacing_y);
    }

    GridToLngLat () {}

    void init (Real _min_lng, Real _min_lat, Real _gspacing_x, Real _gspacing_y) {
        min_lng = _min_lng;
        min_lat = _min_lat;
        gspacing_x = _gspacing_x;
        gspacing_y = _gspacing_y;
    }

    AMREX_GPU_HOST_DEVICE
    void operator()(int x, int y, Real& lng, Real& lat) const {
        lng = (Real)x * gspacing_x + min_lng + 0.001_rt;
        lat = (Real)y * gspacing_y + min_lat + 0.001_rt;
    }
};

/*! \brief Derived class from ParticleContainer that defines agents and their functions */
class AgentContainer : public amrex::ParticleContainer<0, 0, RealIdx::nattribs, IntIdx::nattribs> {

    using PCType = AgentContainer;
    using PType = ParticleType;
    using PTileType = ParticleTileType;
    using PTDType = PTileType::ParticleTileDataType;
    using IntModel = InteractionModel<PCType, PTDType, PType>;

    using MFPtr = std::unique_ptr<MultiFab>;
    using MFPtrVec = std::vector<MFPtr>;

  public:
    AgentContainer(const amrex::Geometry& a_geom, const amrex::DistributionMapping& a_dmap, const amrex::BoxArray& a_ba,
                   const int& a_num_diseases, const std::vector<std::string>& a_disease_names, const std::vector<int>& a_fips_codes, const bool fast,
                   const short a_ic_type);

    void morningCommute(amrex::MultiFab&);

    void eveningCommute(amrex::MultiFab&);

    void interactDay(amrex::MultiFab&);

    void interactEvening(amrex::MultiFab&);

    void interactNight(amrex::MultiFab&);

    void moveAirTravel(const amrex::iMultiFab& unit_mf, AirTravelFlow& air, DemographicData& demo);

    void setAirTravel(const iMultiFab& unit_mf, AirTravelFlow& air, DemographicData& demo);

    void moveAgentsRandomWalk();

    void moveRandomTravel(const amrex::Real random_travel_prob);

    void returnRandomTravel();

    void returnAirTravel();

    void updateStatus(MFPtrVec&);

    void updateSchoolInfection(iMultiFab& a_school_stats);

    void infectAgents(MFPtrVec&);

    void shelterStart ();

    void shelterStop();

    void generateCellData(amrex::MultiFab& mf, const int ncomp) const;

    std::array<amrex::Long, 9> getTotals(const int);

    int getMaxGroup(const int group_idx);

    void moveAgentsToWork();

    void moveAgentsToHome();

    void setTileSize (IntVect newTileSize) { tile_size = newTileSize; }

    /*! \brief Return flag indicating if agents are at work */
    inline bool isAtWork () const { return m_at_work; }

    /*! \brief Return disease parameters object pointer (host) */
    inline const DiseaseParm* getDiseaseParameters_h (int d /*!< disease index */) const { return m_h_parm[d]; }

    /*! \brief Return disease parameters object pointer (device) */
    inline const DiseaseParm* getDiseaseParameters_d (int d /*!< disease index */) const { return m_d_parm[d]; }

    /*! \brief Return the number of diseases */
    inline int numDiseases () const { return m_num_diseases; }

    /*! \brief return the symptomatic withdrawal compliance */
    inline amrex::Real symptomaticWithdrawCompliance () const { return m_symptomatic_withdraw_compliance; }

    void printStudentTeacherCounts() const;

    void printAgeGroupCounts() const;

    void printStudentTeacherCountsPerCommunity() const;

    void printStudentTeacherCountsPerUnit(const amrex::iMultiFab& FIPS_mf, const amrex::iMultiFab& geoid_mf, bool use_census, int num_units, const std::vector<int>& FIPS_codes) const;

    void setCurrentDay(int day);

    inline int getCurrentDay() const { return m_current_day; }

    void updateAttendanceSchedules(int current_day, const std::vector<ExaEpi::SchoolPolicy>& policies);

    inline int getLastMaskUpdateDay() const { return m_last_mask_update_day; }

    void assignAttendanceFromPolicies(const std::vector<ExaEpi::SchoolPolicy>& policies, int time_offset);

    void initializeSchoolAttendanceForAll(int window_size);

    void countAttendingStudents(int current_day) const;



    amrex::iMultiFab m_student_counts;
    std::vector<int> m_fips_codes;
    amrex::Gpu::DeviceVector<int> m_fips_student_stats; // FIPS-level student counts
    amrex::iMultiFab m_school_stats; // Community and school-level school status and infection counts

    /// Used only for Census data. A ratio for each school type: none, college, high, middle, elem, daycare
    amrex::GpuArray<int, SchoolType::total> m_student_teacher_ratio = {0, 15, 15, 15, 15, 15};

    int m_num_diseases; /*!< Number of diseases */

    amrex::iMultiFab comm_mf;

    LngLatToGrid lnglat_to_grid;
    GridToLngLat grid_to_lnglat;

    short ic_type;

  protected:
    amrex::Real m_shelter_compliance = 0.95_rt;              /*!< Shelter-in-place compliance rate */
    amrex::Real m_symptomatic_withdraw_compliance = 0.95_rt; /*!< Symptomatic withdrawal compliance rate */

    std::vector<DiseaseParm*> m_h_parm;                      /*!< Disease parameters */
    std::vector<DiseaseParm*> m_d_parm;                      /*!< Disease parameters (GPU device) */

    std::unique_ptr<DiseaseCouplingParm<PTDType>> m_disease_coupling;

    std::map<ExaEpi::InteractionNames, IntModel*> m_interactions;      /*!< Map of interaction models */
    std::unique_ptr<HospitalModel<PCType, PTDType, PType>> m_hospital; /*!< hospital model */

    /*! Flag to indicate if agents are at work */
    bool m_at_work;

    /*! Disease status update model */
    DiseaseStatus<PCType, PTileType, PTDType, PType> m_disease_status;

    Array<int, IntIdx::nattribs> max_attribute_values;

    /*! \brief queries if a given interaction type (model) is available */
    inline bool haveInteractionModel (ExaEpi::InteractionNames a_mod_name) const {
        return (m_interactions.find(a_mod_name) != m_interactions.end());
    }

    /*! \brief Add runtime SoA attributes */
    void addAttributes();

    int m_current_day = 0;
    int m_last_mask_update_day = -1;
};

namespace SchoolPolicy {
    // Holds pre-calculated policy data for the GPU kernel.
    struct PolicyGPUData {
        int start_day;
        int end_day;
        int school_type;
        int num_groups;
        int fips;
        // Pre-calculated 32-bit mask for each possible group (e.g., 'A', 'B', 'C', 'D')
        uint32_t group_masks[4];
    };
    struct SchoolDismissalType
    {
        enum {
            ByCommunity = 0,   /*!< dismiss school with respect to total infection in a community */
            BySchool,   /*!< dismiss school with respect to total infection in specific school in a community */
            ByFIPS  /*!< dismiss school with respect to total infection in a FIPS */
        };
    };
    struct SchoolStats
    {
        enum {
            SchoolDismissal = 0,   /*!< whether school is open or closed */
            SchoolInfectionCount,  /*!< total infected students in community if school open */
            SchoolStatusDayCount,  /*!< days since the last status change */
            SchoolReopenDay,       /*!< day the school was last reopened */
            nattribs
        };
    };
}
using AgentIterator = typename AgentContainer::ParIterType;

#endif
