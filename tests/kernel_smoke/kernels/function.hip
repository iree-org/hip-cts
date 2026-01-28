// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>

// Simple kernel for function attribute testing
extern "C" __global__ void simple_kernel() {
    // Empty kernel
}

// Kernel with static shared memory
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

// Kernel with local memory usage
extern "C" __global__ void local_mem_kernel(int* output, int n) {
    int local_array[16];  // Local memory
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use local array
    for (int i = 0; i < 16; ++i) {
        local_array[i] = tid + i;
    }
    
    if (gid < n) {
        int sum = 0;
        for (int i = 0; i < 16; ++i) {
            sum += local_array[i];
        }
        output[gid] = sum;
    }
}

// Kernel with many registers
extern "C" __global__ void many_regs_kernel(float* output, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use many registers
    float r0 = 0.0f, r1 = 1.0f, r2 = 2.0f, r3 = 3.0f;
    float r4 = 4.0f, r5 = 5.0f, r6 = 6.0f, r7 = 7.0f;
    float r8 = 8.0f, r9 = 9.0f, r10 = 10.0f, r11 = 11.0f;
    float r12 = 12.0f, r13 = 13.0f, r14 = 14.0f, r15 = 15.0f;
    
    // Compute something
    float result = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 +
                   r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
    
    if (gid < n) {
        output[gid] = result + (float)gid;
    }
}
