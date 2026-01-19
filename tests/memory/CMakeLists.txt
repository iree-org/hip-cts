# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_hip_cts_test(memory_malloc_test
    SOURCES malloc_test.cpp
)

add_hip_cts_test(memory_memcpy_test
    SOURCES memcpy_test.cpp
)

add_hip_cts_test(memory_memset_test
    SOURCES memset_test.cpp
)

add_hip_cts_test(memory_host_memory_test
    SOURCES host_memory_test.cpp
)

add_hip_cts_test(memory_managed_memory_test
    SOURCES managed_memory_test.cpp
)

add_hip_cts_test(memory_kernel_test
    SOURCES kernel_memory_test.cpp
    KERNELS
        kernels/vector_set.hip
        kernels/increment.hip
        kernels/vector_add.hip
)

add_hip_cts_test(memory_symbol_test
    SOURCES symbol_test.cpp
    KERNELS
        kernels/symbol_test.hip
)
