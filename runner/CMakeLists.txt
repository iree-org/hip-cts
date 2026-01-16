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

# HIP CTS Runner executable
add_executable(hip_cts_runner
    main.cpp
    test_runner.cpp
    test_runner.hpp
    xml_aggregator.cpp
    xml_aggregator.hpp
    text_reporter.cpp
    text_reporter.hpp
    html_reporter.cpp
    html_reporter.hpp
)

target_link_libraries(hip_cts_runner PRIVATE
    tinyxml2
    miniz
)

target_include_directories(hip_cts_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install the runner alongside tests
install(TARGETS hip_cts_runner DESTINATION bin)
