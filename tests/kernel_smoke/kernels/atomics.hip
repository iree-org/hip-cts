// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernels for testing atomic operations

#include <hip/hip_runtime.h>

// Each thread atomically adds 1 to a counter
extern "C" __global__ void atomic_add_int(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(counter, 1);
    }
}

// Each thread atomically adds its index to result
extern "C" __global__ void atomic_add_indices(int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, idx);
    }
}

// Atomic add with unsigned int
extern "C" __global__ void atomic_add_uint(unsigned int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(counter, 1u);
    }
}

// Atomic add with float
extern "C" __global__ void atomic_add_float(float* result, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, value);
    }
}

// Atomic subtraction
extern "C" __global__ void atomic_sub_int(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicSub(counter, 1);
    }
}

// Atomic exchange - each thread exchanges with counter, returns old value
extern "C" __global__ void atomic_exch_int(int* counter, int* old_values, int new_value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        old_values[idx] = atomicExch(counter, new_value);
    }
}

// Atomic min - find minimum value among all threads
extern "C" __global__ void atomic_min_int(int* result, int* values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMin(result, values[idx]);
    }
}

// Atomic max - find maximum value among all threads
extern "C" __global__ void atomic_max_int(int* result, int* values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMax(result, values[idx]);
    }
}

// Atomic min with unsigned
extern "C" __global__ void atomic_min_uint(unsigned int* result, unsigned int* values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMin(result, values[idx]);
    }
}

// Atomic max with unsigned  
extern "C" __global__ void atomic_max_uint(unsigned int* result, unsigned int* values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMax(result, values[idx]);
    }
}

// Atomic AND
extern "C" __global__ void atomic_and_int(int* result, int mask, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAnd(result, mask);
    }
}

// Atomic OR
extern "C" __global__ void atomic_or_int(int* result, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicOr(result, value);
    }
}

// Atomic XOR
extern "C" __global__ void atomic_xor_int(int* result, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicXor(result, value);
    }
}

// Atomic Compare-And-Swap: increment only if current value matches expected
extern "C" __global__ void atomic_cas_increment(int* counter, int expected, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int old = *counter;
        // Keep trying to increment
        while (old == expected) {
            int prev = atomicCAS(counter, expected, expected + 1);
            if (prev == expected) break;
            old = prev;
        }
    }
}

// Atomic increment (returns old value)
extern "C" __global__ void atomic_inc_test(unsigned int* counter, unsigned int* old_values, unsigned int max_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        old_values[idx] = atomicInc(counter, max_val);
    }
}

// Atomic decrement (returns old value)
extern "C" __global__ void atomic_dec_test(unsigned int* counter, unsigned int* old_values, unsigned int max_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        old_values[idx] = atomicDec(counter, max_val);
    }
}

// Histogram using atomics - count occurrences of values 0-255
extern "C" __global__ void atomic_histogram(unsigned char* data, int* histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&histogram[data[idx]], 1);
    }
}

// Multi-location atomic add - each thread adds to a different slot
extern "C" __global__ void atomic_add_slots(int* slots, int num_slots, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int slot = idx % num_slots;
        atomicAdd(&slots[slot], 1);
    }
}
