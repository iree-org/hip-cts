// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>

// Simple kernel for occupancy testing
extern "C" __global__ void simple_kernel() {
    // Empty kernel - just for occupancy calculation
}

// Kernel that sets output values
extern "C" __global__ void set_value(int* output, int value, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = value;
    }
}

// Kernel with shared memory usage for occupancy testing
extern "C" __global__ void shared_mem_kernel(int* output, int n) {
    __shared__ int sdata[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < 256) {
        sdata[tid] = tid;
    }
    __syncthreads();
    
    if (gid < n) {
        output[gid] = sdata[tid % 256];
    }
}
