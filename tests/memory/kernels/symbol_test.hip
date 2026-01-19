// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Kernel with device symbols for testing hipModuleGetGlobal and symbol access.

#include <hip/hip_runtime.h>

// Device global variable - can be accessed via hipModuleGetGlobal
__device__ int d_globalValue = 0;

// Device constant variable - read-only on device, can be set from host
__constant__ int d_constantValue = 0;

// Device array for more complex tests
__device__ int d_globalArray[256];

// Kernel that reads from the global variable and writes to output
extern "C" __global__ void read_global_value(int* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = d_globalValue;
    }
}

// Kernel that writes to the global variable
extern "C" __global__ void write_global_value(int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_globalValue = value;
    }
}

// Kernel that reads from constant memory
extern "C" __global__ void read_constant_value(int* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = d_constantValue;
    }
}

// Kernel that adds the global value to each element of an array
extern "C" __global__ void add_global_to_array(int* data, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] += d_globalValue;
    }
}

// Kernel that copies from global array to output
extern "C" __global__ void read_global_array(int* output, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && idx < 256) {
        output[idx] = d_globalArray[idx];
    }
}
