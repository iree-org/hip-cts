// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Kernel to set all elements of a vector to a specific value.

#include <hip/hip_runtime.h>

extern "C" __global__ void vector_set_int(int* data, int value, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = value;
    }
}

extern "C" __global__ void vector_set_char(char* data, char value, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = value;
    }
}
