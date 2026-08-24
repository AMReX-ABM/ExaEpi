cmake_minimum_required(VERSION 3.14)

project(ExaEpi
        DESCRIPTION "An agent-based simulation code for epidemiology modeling using the AMReX adaptive mesh refinement framework"
        # VERSION  ${AMREX_PKG_VERSION}
)

# Defines a DEBUG symbol for use in the code
add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

# Experimental: keep students/teachers home during the day (skip their daytime commute to school)
# and skip school transmission entirely -- see InteractionModSchool.H / AgentContainer::interactSchool
# and AgentContainer::moveAgentsToWork
option(EXAEPI_DISABLE_SCHOOL "Disable daytime school commute and school transmission (experimental)" OFF)

# Prevent MPI headers from declaring the deprecated MPI C++ bindings (MPI::Comm etc.) --
# unused by AMReX/ExaEpi, but Open MPI's and MPICH's mpi.h declare them by default unless
# told not to, which then requires linking libmpi_cxx even though nothing calls any MPI::
# API. Open MPI 5.x+ removed that library entirely, breaking portability without this.
# Defining both macros is harmless on an implementation that doesn't recognize the other's.
add_compile_definitions(OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

set(AMReX_PARTICLES ON)
set(AMReX_FORTRAN OFF)
set(AMReX_LINEAR_SOLVERS OFF)
set(AMReX_AMRLEVEL OFF)
set(AMReX_TINY_PROFILE ON CACHE BOOL "Enable AMReX tiny profiler")
set(AMReX_SPACEDIM
    2
    CACHE INTERNAL "")
set(AMReX_PRECISION
    SINGLE
    CACHE INTERNAL "")
set(AMReX_PARTICLES_PRECISION
    SINGLE
    CACHE INTERNAL "")

#
# Fetch amrex repo
#
set(AMReX_GIT_BRANCH
    "development"
    CACHE STRING "The AMReX branch to checkout")
set(AMReX_INSTALL
    "NO"
    CACHE INTERNAL "Disable install target for amrex")

include(FetchContent)
set(FETCHCONTENT_QUIET OFF) # Verbose ON

FetchContent_Declare(
  amrex
  GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git/
  GIT_TAG ${AMReX_GIT_BRANCH}
  GIT_SUBMODULES "")

if(NOT ${amrex}_POPULATED)
  FetchContent_Populate(amrex)

  # Workaround for an AMReX CMake bug: it unconditionally requests the deprecated MPI C++
  # bindings component (Tools/CMake/AMReXParallelBackends.cmake), which nothing in
  # AMReX/ExaEpi actually uses, but which Open MPI 5.x+ no longer ships at all -- breaking
  # the build on any host with a newer Open MPI. Patch it out of the freshly-fetched
  # source here so this survives every fresh AMReX fetch (only C is needed; see also the
  # OMPI_SKIP_MPICXX/MPICH_SKIP_MPICXX compile definitions above, which are still required
  # even with this fix, since MPI headers otherwise declare those C++ classes regardless of
  # whether the library ends up linked).
  set(_amrex_mpi_cmake ${amrex_SOURCE_DIR}/Tools/CMake/AMReXParallelBackends.cmake)
  if(EXISTS ${_amrex_mpi_cmake})
    file(READ ${_amrex_mpi_cmake} _amrex_mpi_cmake_contents)
    string(REPLACE "set(_mpi_comps C CXX)" "set(_mpi_comps C)" _amrex_mpi_cmake_contents
                   "${_amrex_mpi_cmake_contents}")
    file(WRITE ${_amrex_mpi_cmake} "${_amrex_mpi_cmake_contents}")
  endif()
  unset(_amrex_mpi_cmake)
  unset(_amrex_mpi_cmake_contents)

  list(APPEND CMAKE_MODULE_PATH ${amrex_SOURCE_DIR}/Tools/CMake)

  # Load amrex options here so that they are available to the entire project
  include(AMReXOptions)

  if(AMReX_FORTRAN)
    enable_language(Fortran)
  endif()

  if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    # include(AMReX_SetupCUDA)
  endif()

  # Bring the populated content into the build
  add_subdirectory(${amrex_SOURCE_DIR} ${amrex_BINARY_DIR})
endif()

#
# List of subdirectories to search for CMakeLists.
#
set(AMREX_AGENT_SUBDIRS src)

list(TRANSFORM AMREX_AGENT_SUBDIRS PREPEND "${CMAKE_CURRENT_LIST_DIR}/")

#
# Search for CMakelists.txt in the subdirectories defined in the list above and include those that contain one into the build
#
include(SetupAgent)
foreach(_subdir IN LISTS AMREX_AGENT_SUBDIRS)
  file(GLOB_RECURSE _tests "${_subdir}/*CMakeLists.txt")
  foreach(_item IN LISTS _tests)
    get_filename_component(_dir ${_item} DIRECTORY)
    add_subdirectory(${_dir})
  endforeach()
endforeach()
