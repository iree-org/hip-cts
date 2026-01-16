// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Kernel to increment all elements of a vector by 1.
// Used to verify memory was initialized correctly and kernel can modify it.

#include <hip/hip_runtime.h>

extern "C" __global__ void increment_int(int* data, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] += 1;
    }
}

extern "C" __global__ void increment_char(char* data, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] += 1;
    }
}
