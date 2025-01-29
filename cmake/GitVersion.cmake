function(get_version_from_git)
    find_package(Git QUIET)
    if(NOT Git_FOUND)
        message(WARNING "Git not found")
        return()
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE GIT_RESULT
    )

    if(NOT GIT_RESULT EQUAL 0)
        message(STATUS "No git tags found, using v0.0.1 instead")
        set(GIT_TAG "v0.0.1")
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --always --dirty
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_SHORT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    execute_process(
        COMMAND ${GIT_EXECUTABLE} branch --show-current
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set(EXAEPI_VERSION "${GIT_TAG}-g${GIT_COMMIT_SHORT_HASH} (${GIT_BRANCH})")
    set(EXAEPI_VERSION "${GIT_TAG}-g${GIT_COMMIT_SHORT_HASH} (${GIT_BRANCH})" PARENT_SCOPE)

    message(STATUS "Building version ${EXAEPI_VERSION}")
    configure_file(${CMAKE_SOURCE_DIR}/src/version.h.in ${CMAKE_SOURCE_DIR}/src/version.h)

endfunction()
