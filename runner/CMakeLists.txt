# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Add tinyxml2
add_library(tinyxml2 STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tinyxml2/tinyxml2.cpp
)
target_include_directories(tinyxml2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tinyxml2
)

# Add miniz using its CMake build (from submodule)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_HEADER_ONLY OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(INSTALL_PROJECT OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/miniz 
                 ${CMAKE_CURRENT_BINARY_DIR}/miniz)

# Add FTXUI
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(FTXUI_QUIET ON CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/FTXUI 
                 ${CMAKE_CURRENT_BINARY_DIR}/ftxui)

# Shared runner library for both executables
add_library(hip_cts_runner_lib STATIC
    test_runner.cpp
    test_runner.hpp
    xml_aggregator.cpp
    xml_aggregator.hpp
    text_reporter.cpp
    text_reporter.hpp
    html_reporter.cpp
    html_reporter.hpp
)

target_link_libraries(hip_cts_runner_lib PUBLIC
    tinyxml2
    miniz
)

target_include_directories(hip_cts_runner_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# HIP CTS Runner executable (CLI version)
add_executable(hip_cts_runner
    main.cpp
)

target_link_libraries(hip_cts_runner PRIVATE
    hip_cts_runner_lib
)

# HIP CTS Runner executable (FTXUI interactive version)
add_executable(hip_cts_runner_ftx
    main_ftx.cpp
)

target_link_libraries(hip_cts_runner_ftx PRIVATE
    hip_cts_runner_lib
    ftxui::screen
    ftxui::dom
    ftxui::component
)

# Install the runners alongside tests
install(TARGETS hip_cts_runner hip_cts_runner_ftx DESTINATION bin)
