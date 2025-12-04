# Automatically detect source files in subdirectories
# Find all subdirectories in src/
file(GLOB _subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
list(FILTER _subdirs INCLUDE REGEX "^[^.]+$")  # Exclude hidden files
set(_subdir_sources "")

# For each subdirectory, collect all .cpp and .H files
foreach(_dir IN LISTS _subdirs)
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${_dir})
        file(GLOB_RECURSE _dir_sources
            "${_dir}/*.cpp"
            "${_dir}/*.H"
        )
        list(APPEND _subdir_sources ${_dir_sources})
    endif()
endforeach()

# Main source file list includes main.cpp and all subdirectory sources
set(_sources
    main.cpp
    ${_subdir_sources}
)

# List of input files
set(_input_files)

setup_agent(_sources _input_files)

unset(_sources)
unset(_input_files)
