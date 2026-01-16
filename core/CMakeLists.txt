# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# HIP CTS core library (loader and types)
add_library(hip_cts_core STATIC
    hip_loader.cpp
)
target_include_directories(hip_cts_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${hip_cts_BINARY_DIR}/generated  # Generated backend config header
)
target_link_libraries(hip_cts_core PUBLIC ${CMAKE_DL_LIBS})

# Custom main for Catch2 with HIP library argument
add_library(hip_cts_main STATIC
    main.cpp
)
target_link_libraries(hip_cts_main PUBLIC 
    Catch2::Catch2
    hip_cts_core
)

