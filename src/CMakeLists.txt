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
    PmedmProblem.H
    PmedmProblem.cpp
    PmedmSolver.H
    PmedmSolver.cpp
    PopGenStages.H
    PopulationBundle.H
    PopulationBundle.cpp
    PopulationGenerator.H
    PopulationGenerator.cpp
    Sha256.H
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

# The population stages must round exactly as the Python reference generator does (bit-for-bit
# digests), so no fused multiply-adds: on the host compiler in CUDA builds, directly otherwise.
set_source_files_properties(PopulationGenerator.cpp PROPERTIES COMPILE_OPTIONS
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>;$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>")

# The UrbanPop .bin stores each block group's agents as an independently-deflated frame, so the
# reader inflates them (see readFrame in UrbanPopData.cpp). zlib is ubiquitous -- including on
# Perlmutter -- so this needs no module to be loaded. Linked with the plain signature, since
# that is what setup_agent already used for amrex and the two forms cannot be mixed.
find_package(ZLIB REQUIRED)
target_link_libraries(agent ZLIB::ZLIB)

# Population generation's solver runs its matrix products on the vendor BLAS (DenseLinAlg.cpp):
# cuBLAS and rocBLAS ship with their toolkits. CPU builds use OpenBLAS when it is found (held to
# one thread per call, for results independent of the thread count) and DenseLinAlg's own, much
# slower kernel otherwise. Only OpenBLAS is looked for, because only its thread count can be set
# portably from the code.
option(ExaEpi_POPGEN_OPENBLAS "Use OpenBLAS for population generation in CPU builds, if found" ON)
if(ExaEpi_POPGEN_OPENBLAS AND (NOT AMReX_GPU_BACKEND OR AMReX_GPU_BACKEND STREQUAL "NONE"))
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS QUIET)
  unset(BLA_VENDOR)
  find_path(EXAEPI_CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas openblas-pthread openblas-openmp)
  if(BLAS_FOUND AND EXAEPI_CBLAS_INCLUDE_DIR)
    message(STATUS "Population generation: OpenBLAS ${BLAS_LIBRARIES}, cblas.h in ${EXAEPI_CBLAS_INCLUDE_DIR}")
    set(EXAEPI_POPGEN_USE_OPENBLAS ON)
  else()
    message(STATUS "Population generation: OpenBLAS not found; CPU builds use the built-in matrix kernel")
  endif()
endif()

function(exaepi_link_blas _target)
  if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${_target} CUDA::cublas)
  elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    find_package(rocblas REQUIRED)
    target_link_libraries(${_target} roc::rocblas)
  elseif(EXAEPI_POPGEN_USE_OPENBLAS)
    target_link_libraries(${_target} ${BLAS_LIBRARIES})
    target_include_directories(${_target} PRIVATE ${EXAEPI_CBLAS_INCLUDE_DIR})
    target_compile_definitions(${_target} PRIVATE EXAEPI_POPGEN_OPENBLAS)
  endif()
endfunction()
exaepi_link_blas(agent)

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
  exaepi_link_blas(popgen_gemm_check)

  add_executable(popgen_pmedm_check ${PROJECT_SOURCE_DIR}/utilities/tests/pmedm_check.cpp DenseLinAlg.cpp
                                    PmedmProblem.cpp PmedmSolver.cpp PopulationBundle.cpp)
  set_target_properties(popgen_pmedm_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  target_include_directories(popgen_pmedm_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(popgen_pmedm_check amrex ZLIB::ZLIB)
  if(AMReX_CUDA)
    setup_target_for_cuda_compilation(popgen_pmedm_check)
  endif()
  exaepi_link_blas(popgen_pmedm_check)

  # Built like the agent (through nvcc in CUDA builds), so it checks the stages as ExaEpi compiles them.
  add_executable(popgen_stages_check ${PROJECT_SOURCE_DIR}/utilities/tests/popgen_stages_check.cpp
                                     PopulationGenerator.cpp PmedmProblem.cpp PopulationBundle.cpp)
  set_target_properties(popgen_stages_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  target_include_directories(popgen_stages_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(popgen_stages_check amrex ZLIB::ZLIB)
  if(AMReX_CUDA)
    setup_target_for_cuda_compilation(popgen_stages_check)
  endif()
endif()

unset(_sources)
unset(_input_files)
