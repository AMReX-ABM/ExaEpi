cmake_minimum_required(VERSION 3.14)

project( ExaEpi
   DESCRIPTION "An agent-based simulation code for epidemiology modeling using the AMReX adaptive mesh refinement framework"
   #VERSION  ${AMREX_PKG_VERSION}
   )

# Defines a DEBUG symbol for use in the code
add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake )

set(AMReX_PARTICLES       ON)
set(AMReX_FORTRAN        OFF)
set(AMReX_LINEAR_SOLVERS OFF)
set(AMReX_AMRLEVEL       OFF)
set(AMReX_TINY_PROFILE    ON)
set(AMReX_SPACEDIM 2 CACHE INTERNAL "")
set(AMReX_PRECISION SINGLE CACHE INTERNAL "")
set(AMReX_PARTICLES_PRECISION SINGLE CACHE INTERNAL "")

#
# Fetch amrex repo
#
set(AMReX_GIT_BRANCH "development" CACHE STRING "The AMReX branch to checkout")
set(AMReX_INSTALL  "NO" CACHE INTERNAL "Disable install target for amrex")

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)  # Verbose ON

FetchContent_Declare(
  amrex
  GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git/
  GIT_TAG        ${AMReX_GIT_BRANCH}
  )

if(NOT ${amrex}_POPULATED)
  FetchContent_Populate(amrex)

  list(APPEND CMAKE_MODULE_PATH ${amrex_SOURCE_DIR}/Tools/CMake)

  # Load amrex options here so that they are
  # available to the entire project
  include(AMReXOptions)

  if (AMReX_FORTRAN)
     enable_language(Fortran)
  endif ()

  if (AMReX_GPU_BACKEND STREQUAL "CUDA")
     enable_language(CUDA)
#     include(AMReX_SetupCUDA)
  endif ()

  # Bring the populated content into the build
  add_subdirectory(${amrex_SOURCE_DIR} ${amrex_BINARY_DIR})
endif()

#
# List of subdirectories to search for CMakeLists.
#
set( AMREX_AGENT_SUBDIRS src)

list(TRANSFORM AMREX_AGENT_SUBDIRS PREPEND "${CMAKE_CURRENT_LIST_DIR}/")

#
# Search for CMakelists.txt in the subdirectories defined in the list above
# and include those that contain one into the build
#
include(SetupAgent)
foreach (_subdir IN LISTS AMREX_AGENT_SUBDIRS)
   file( GLOB_RECURSE _tests "${_subdir}/*CMakeLists.txt" )
   foreach ( _item  IN LISTS _tests)
      get_filename_component(_dir ${_item} DIRECTORY )
      add_subdirectory(${_dir})
   endforeach ()
endforeach ()

