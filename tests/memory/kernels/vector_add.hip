// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Kernel to add two vectors element-wise.
// C[i] = A[i] + B[i]

#include <hip/hip_runtime.h>

extern "C" __global__ void vector_add_int(const int* A, const int* B, int* C, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" __global__ void vector_add_float(const float* A, const float* B, float* C, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = A[idx] + B[idx];
    }
}
