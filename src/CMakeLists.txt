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
    DiseaseParm.H
    DiseaseParm.cpp
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

# The UrbanPop .bin stores each block group's agents as an independently-deflated frame, so the
# reader inflates them (see readFrame in UrbanPopData.cpp). zlib is ubiquitous -- including on
# Perlmutter -- so this needs no module to be loaded. Linked with the plain signature, since
# that is what setup_agent already used for amrex and the two forms cannot be mixed.
find_package(ZLIB REQUIRED)
target_link_libraries(agent ZLIB::ZLIB)

unset(_sources)
unset(_input_files)
