agent.ic_type = "census"
agent.census_filename = "../../data/CensusData/MA.dat"
agent.workerflow_filename = "../../data/CensusData/MA-wf.dat"


agent.nsteps = 120
agent.plot_int = 10
agent.random_travel_int = 24

agent.aggregated_diag_int = -1

#agent.shelter_start = 7
#agent.shelter_length = 30
#agent.shelter_compliance = 0.85

#disease.initial_case_type = "random"
#disease.num_initial_cases = 5
disease.initial_case_type = "file"
disease.case_filename = "../../data/CaseData/July4.cases"
disease.p_trans = 0.20
disease.p_asymp = 0.40

disease.incubation_length_alpha = 9.0
disease.incubation_length_beta = 0.33