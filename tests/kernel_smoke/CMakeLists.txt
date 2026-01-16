# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_hip_cts_test(kernel_smoke_test
    SOURCES kernel_smoke_test.cpp
    KERNELS 
        kernels/vector_add.hip
        kernels/vector_scale.hip
        kernels/memset_kernel.hip
)
