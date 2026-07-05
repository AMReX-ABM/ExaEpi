/*! @file InteractionModSchool.H
 * \brief Contains the class describing agent interactions at school
 */

#ifndef _INTERACTION_MOD_SCHOOL_H_
#define _INTERACTION_MOD_SCHOOL_H_

#include "AgentDefinitions.H"
#include "InteractionModel.H"

using namespace amrex;

template <typename PTDType>
struct SchoolCandidate {
    AMREX_GPU_HOST_DEVICE
    bool operator()(const int idx, const PTDType& ptd) const noexcept {
        return !inHospital(idx, ptd) && ptd.m_idata[IntIdx::school_id][idx] > 0 && ptd.m_idata[IntIdx::school_closed][idx] == 0 &&
               !ptd.m_idata[IntIdx::withdrawn][idx] && ptd.m_idata[IntIdx::air_travel][idx] < 0 &&
               ptd.m_idata[IntIdx::random_travel][idx] < 0;
    }
};

/*! \brief Class describing agent interactions at school */
template <typename PCType, typename PTDType, typename PType>
class InteractionModSchool {
  public:
    /*! \brief Simulate agent interaction at school */
    void interactAgents (PCType& agents, MultiFab&) { fastInteractSchool(agents); }

    void fastInteractSchool(PCType& agents);
};

template <typename PCType, typename PTDType, typename PType>
void InteractionModSchool<PCType, PTDType, PType>::fastInteractSchool (PCType& agents) {
    BL_PROFILE(__func__);
    int n_disease = agents.numDiseases();

    SchoolCandidate<PTDType> isCandidate;

    Vector<Gpu::DeviceVector<int>> infected_school_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_daycare_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_school_asymp_d(OMP_MAX_THREADS);
    Vector<Gpu::DeviceVector<int>> infected_daycare_asymp_d(OMP_MAX_THREADS);

    for (int lev = 0; lev < agents.numLevels(); ++lev) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi = agents.MakeMFIter(lev); mfi.isValid(); ++mfi) {
            auto& ptile = agents.ParticlesAt(lev, mfi);
            const auto& ptd = ptile.getParticleTileData();
            const auto np = ptile.GetArrayOfStructs().numParticles();
            if (np == 0) { continue; }

            auto& soa = ptile.GetStructOfArrays();
            auto school_grade_ptr = soa.GetIntData(IntIdx::school_grade).data();
            auto school_id_ptr = soa.GetIntData(IntIdx::school_id).data();
            auto age_group_ptr = soa.GetIntData(IntIdx::age_group).data();

            GetCommunityIndex<PTDType> getCommunityIndex;
            getCommunityIndex.init(agents.Geom(lev), mfi.tilebox(), agents.comm_mf[mfi].array());
            auto plo = getCommunityIndex.plo;
            auto dxi = getCommunityIndex.dxi;
            auto domain = getCommunityIndex.domain;
            auto valid_box = getCommunityIndex.valid_box;
            auto bin_size = getCommunityIndex.bin_size;
            auto comm_to_local_index_d = getCommunityIndex.comm_to_local_index_d;

            int max_communities = getCommunityIndex.max();
            int max_school_grade = agents.getMaxGroup(IntIdx::school_grade) + 1;
            int max_school_id = agents.getMaxGroup(IntIdx::school_id) + 1;

            infected_school_d[OMP_THREAD_NUM].resize(max_communities * max_school_id * max_school_grade);
            infected_daycare_d[OMP_THREAD_NUM].resize(max_communities * max_school_id * max_school_grade);
            infected_school_asymp_d[OMP_THREAD_NUM].resize(max_communities * max_school_id * max_school_grade);
            infected_daycare_asymp_d[OMP_THREAD_NUM].resize(max_communities * max_school_id * max_school_grade);
            auto infected_school_d_ptr = infected_school_d[OMP_THREAD_NUM].data();
            auto infected_daycare_d_ptr = infected_daycare_d[OMP_THREAD_NUM].data();
            auto infected_school_asymp_d_ptr = infected_school_asymp_d[OMP_THREAD_NUM].data();
            auto infected_daycare_asymp_d_ptr = infected_daycare_asymp_d[OMP_THREAD_NUM].data();

            /*
            Print() << mfi.tilebox() << " np " << np << " max communities " << max_communities
                    << " max school id " << max_school_id << " max school grade " << max_school_grade
                    << " school vector size " << max_communities * max_school_id * max_school_grade
                    << "\n";
            */
            for (int d = 0; d < n_disease; d++) {
                for (auto adults : {true, false}) {
                    devMemset(infected_school_d_ptr, 0, infected_school_d[OMP_THREAD_NUM].size() * sizeof(int));
                    devMemset(infected_daycare_d_ptr, 0, infected_daycare_d[OMP_THREAD_NUM].size() * sizeof(int));

                    devMemset(infected_school_asymp_d_ptr, 0, infected_school_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));
                    devMemset(infected_daycare_asymp_d_ptr, 0, infected_daycare_asymp_d[OMP_THREAD_NUM].size() * sizeof(int));

                    auto prob_ptr = soa.GetRealData(RealIdx::nattribs + r0(d) + RealIdxDisease::prob).data();
                    auto lparm = agents.getDiseaseParameters_d(d);
                    auto lparm_h = agents.getDiseaseParameters_h(d);
                    Real scale = 1.0_rt; // TODO this should vary based on cell
                    Real infect = (1.0_rt - lparm_h->vac_eff);
                    auto d_ptr = getCommunityIndex.comm_to_local_index_d.data();

                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isInfectious(i, ptd, d) && isCandidate(i, ptd) && isAnAdult(i, ptd) == adults) {
                            Box tbx;
                            auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                            auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                            auto community = d_ptr[tidx];
                            int pos = (community * max_school_id + school_id_ptr[i]) * max_school_grade + school_grade_ptr[i];
                            if (getSchoolType(school_grade_ptr[i]) == SchoolType::daycare) {
                                if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                                    Gpu::Atomic::AddNoRet(&infected_daycare_asymp_d_ptr[pos], 1);
                                } else {
                                    Gpu::Atomic::AddNoRet(&infected_daycare_d_ptr[pos], 1);
                                }
                            } else {
                                if (isAsymptomatic(i, ptd, d) || isPresymptomatic(i, ptd, d)) {
                                    Gpu::Atomic::AddNoRet(&infected_school_asymp_d_ptr[pos], 1);
                                } else {
                                    Gpu::Atomic::AddNoRet(&infected_school_d_ptr[pos], 1);
                                }
                            }
                        }
                    });
                    Gpu::synchronize();

                    ParallelFor(np, [=] AMREX_GPU_DEVICE (int i) noexcept {
                        if (isSusceptible(i, ptd, d) && isCandidate(i, ptd)) {
                            Box tbx;
                            auto iv = getParticleCell(ptd, i, plo, dxi, domain);
                            auto tidx = getTileIndex(iv, valid_box, true, bin_size, tbx);
                            auto community = d_ptr[tidx];
                            int pos = (community * max_school_id + school_id_ptr[i]) * max_school_grade + school_grade_ptr[i];
                            if (getSchoolType(school_grade_ptr[i]) == SchoolType::daycare) {
                                if (adults && age_group_ptr[i] > AgeGroups::a5to17) { // adults-adults interaction accounted for
                                                                                      // in InteractionModWork.H
                                    // do nothing!
                                } else {
                                    int num_infected_daycare = infected_daycare_d_ptr[pos];
                                    Real daycare_prob = 1.0_rt - infect * lparm->xmit_school[SchoolType::daycare] * scale;
                                    prob_ptr[i] *= static_cast<ParticleReal>(std::pow(daycare_prob, num_infected_daycare));
                                    int num_infected_daycare_asymp = infected_daycare_asymp_d_ptr[pos];
                                    Real daycare_prob_asymp = 1.0_rt - lparm->asymp_relative_inf * infect *
                                                                               lparm->xmit_school[SchoolType::daycare] * scale;
                                    prob_ptr[i] *=
                                            static_cast<ParticleReal>(std::pow(daycare_prob_asymp, num_infected_daycare_asymp));
                                }
                            } else {
                                int num_infected_school = infected_school_d_ptr[pos];
                                int num_infected_school_asymp = infected_school_asymp_d_ptr[pos];
                                Real xmit = 0.0_rt;
                                if (adults) {                                    // transmitters are adults
                                    if (age_group_ptr[i] <= AgeGroups::a5to17) { // Adult teacher/staff -> child student
                                        xmit = lparm->xmit_school_a2c[getSchoolType(school_grade_ptr[i])];
                                    } else { //  adults-adults interaction accounted for in InteractionModWork.H // adult to adult
                                             //  - teachers also have grades (the grade they teach)
                                        // do nothing!
                                        // xmit = lparm->xmit_school[getSchoolType(school_grade_ptr[i])];
                                    }
                                } else {                                         // transmitters are children
                                    if (age_group_ptr[i] <= AgeGroups::a5to17) { // Receiver j is a child
                                        xmit = lparm->xmit_school[getSchoolType(school_grade_ptr[i])];
                                    } else { // Child student -> adult teacher/staff transmission
                                        xmit = lparm->xmit_school_c2a[getSchoolType(school_grade_ptr[i])];
                                    }
                                }
                                Real school_prob = 1.0_rt - infect * xmit * scale;
                                prob_ptr[i] *= static_cast<ParticleReal>(std::pow(school_prob, num_infected_school));
                                Real school_prob_asymp = 1.0_rt - lparm->asymp_relative_inf * infect * xmit * scale;
                                prob_ptr[i] *= static_cast<ParticleReal>(std::pow(school_prob_asymp, num_infected_school_asymp));
                            }
                        }
                    });
                    Gpu::synchronize();
                }
            }
        }
    }
}

#endif
