// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernels for testing shared memory functionality

#include <hip/hip_runtime.h>

// Static shared memory: simple copy through shared memory
extern "C" __global__ void shared_copy(int* input, int* output, int n) {
    __shared__ int shared_data[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load from global to shared
    if (idx < n) {
        shared_data[tid] = input[idx];
    }
    
    __syncthreads();
    
    // Store from shared to global
    if (idx < n) {
        output[idx] = shared_data[tid];
    }
}

// Shared memory with neighbor access (tests __syncthreads correctness)
extern "C" __global__ void shared_neighbor_sum(int* input, int* output, int n) {
    __shared__ int shared_data[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Load from global to shared
    if (idx < n) {
        shared_data[tid] = input[idx];
    } else {
        shared_data[tid] = 0;
    }
    
    __syncthreads();
    
    // Sum with left and right neighbors (within block)
    if (idx < n) {
        int left = (tid > 0) ? shared_data[tid - 1] : 0;
        int right = (tid < block_size - 1) ? shared_data[tid + 1] : 0;
        output[idx] = left + shared_data[tid] + right;
    }
}

// Block-level reduction using shared memory
extern "C" __global__ void shared_reduce_sum(int* input, int* output, int n) {
    extern __shared__ int sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load element or 0 if out of bounds
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Dynamic shared memory test: fill with thread-specific values
extern "C" __global__ void shared_dynamic_fill(int* output, int n) {
    extern __shared__ int dyn_shared[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread writes its thread ID to shared memory
    dyn_shared[tid] = tid * 10 + blockIdx.x;
    
    __syncthreads();
    
    // Read back from shared to global (verifies shared memory works)
    if (idx < n) {
        output[idx] = dyn_shared[tid];
    }
}

// Shared memory reverse within block
extern "C" __global__ void shared_reverse(int* data, int n) {
    __shared__ int shared_data[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Load to shared
    if (idx < n) {
        shared_data[tid] = data[idx];
    }
    
    __syncthreads();
    
    // Write back in reverse order within block
    if (idx < n) {
        data[idx] = shared_data[block_size - 1 - tid];
    }
}

// Multiple syncthreads test
extern "C" __global__ void shared_multi_sync(int* data, int n) {
    __shared__ int shared_data[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        shared_data[tid] = data[idx];
    }
    __syncthreads();
    
    // Add 1
    if (idx < n) {
        shared_data[tid] += 1;
    }
    __syncthreads();
    
    // Add 10
    if (idx < n) {
        shared_data[tid] += 10;
    }
    __syncthreads();
    
    // Add 100
    if (idx < n) {
        shared_data[tid] += 100;
    }
    __syncthreads();
    
    // Write back (original + 111)
    if (idx < n) {
        data[idx] = shared_data[tid];
    }
}
