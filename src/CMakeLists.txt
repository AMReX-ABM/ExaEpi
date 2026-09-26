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
    DenseLinAlg.H
    DenseLinAlg.cpp
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
    KeyedRNG.H
    PopGenStages.H
    PopulationBundle.H
    PopulationBundle.cpp
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

# Population generation's solver runs its matrix products on the vendor BLAS (DenseLinAlg.cpp):
# cuBLAS and rocBLAS ship with their toolkits; CPU builds use DenseLinAlg's own kernel.
function(exaepi_link_gpu_blas _target)
  if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${_target} CUDA::cublas)
  elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    find_package(rocblas REQUIRED)
    target_link_libraries(${_target} roc::rocblas)
  endif()
endfunction()
exaepi_link_gpu_blas(agent)

# Unit checks for the population-generation building blocks that need AMReX (the standalone ones
# -- kr64_kat, bundle_read -- build with a plain compiler; see their headers).
option(ExaEpi_POPGEN_TESTS "Build population-generation unit checks" OFF)
if(ExaEpi_POPGEN_TESTS)
  add_executable(popgen_gemm_check ${PROJECT_SOURCE_DIR}/utilities/tests/gemm_check.cpp DenseLinAlg.cpp)
  set_target_properties(popgen_gemm_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  target_include_directories(popgen_gemm_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(popgen_gemm_check amrex)
  if(AMReX_CUDA)
    setup_target_for_cuda_compilation(popgen_gemm_check)
  endif()
  exaepi_link_gpu_blas(popgen_gemm_check)
endif()

unset(_sources)
unset(_input_files)
