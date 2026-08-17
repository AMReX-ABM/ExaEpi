# List of source files
set(_sources
    main.cpp
    AgentDefinitions.H
    AgentContainer.H
    AgentContainer.cpp
    AirTravelFlow.H
    AirTravelFlow.cpp
    CaseData.H
    CaseData.cpp
    WeatherData.H
    WeatherData.cpp
    DiseaseCoupling.H
    CensusData.H
    CensusData.cpp
    DiseaseParm.H
    DiseaseParm.cpp
    DemographicData.H
    DemographicData.cpp
    IO.H
    IO.cpp
    InteractionModel.H
    InteractionModHH.H
    InteractionModNC.H
    InteractionModNborhood.H
    InteractionModComm.H
    InteractionModSchool.H
    InteractionModWork.H
    InteractionModAirTravel.H
    InteractionModelLibrary.H
    InitializeInfections.H
    InitializeInfections.cpp
    UrbanPopAgentStruct.H
    UrbanPopData.H
    UrbanPopData.cpp
    Utils.H
    Utils.cpp)

# List of input files
set(_input_files)

set(_extra_definitions)
if(EXAEPI_DISABLE_SCHOOL)
  list(APPEND _extra_definitions EXAEPI_DISABLE_SCHOOL)
endif()

setup_agent(_sources _input_files EXTRA_DEFINITIONS "${_extra_definitions}")

unset(_sources)
unset(_input_files)
