# List of source files
set(_sources
         main.cpp
         AgentDefinitions.H
         AgentContainer.H
         AgentContainer.cpp
         CaseData.H
         CaseData.cpp
         AirTravelFlow.H
         AirTravelFlow.cpp
         CensusData.H
         CensusData.cpp
         DiseaseParm.H
         DiseaseParm.cpp
         DemographicData.H
         DemographicData.cpp
         IO.H
         IO.cpp
         InteractionModel.H
         InteractionModHome.H
         InteractionModHomeNborhood.H
         InteractionModWorkNborhood.H
         InteractionModSchool.H
         InteractionModWork.H
         InteractionModelLibrary.H
         InitializeInfections.H
         InitializeInfections.cpp
         UrbanPopAgentStruct.H
         UrbanPopData.H
         UrbanPopData.cpp
         Utils.H
         Utils.cpp)

# List of input files
set(_input_files )

setup_agent(_sources _input_files)

unset( _sources )
unset( _input_files )
