# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.20.0)
project(hip_cts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python for the embed_binary script
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Add Catch2
add_subdirectory(third_party/Catch2)

# Include backend configuration and kernel compilation helper functions
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/HipKernels.cmake)

# Core libraries
add_subdirectory(core)

# Tests
include(CTest)
add_subdirectory(tests)

# Test runner
add_subdirectory(runner)
