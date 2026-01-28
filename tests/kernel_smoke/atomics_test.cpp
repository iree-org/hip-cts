// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for kernel atomic operations

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

#include "kernel_atomics_test_atomics.hpp"

#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <set>

//=============================================================================
// Test Constants
//=============================================================================

static constexpr int kBlockSize = 256;

static unsigned int calculateGridSize(int n, int blockSize) {
    return static_cast<unsigned int>((n + blockSize - 1) / blockSize);
}

//=============================================================================
// Atomic Add Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "atomicAdd int - count threads", "[kernel][atomic][add]") {
    int n = GENERATE(256, 1024, 4096, 16384);
    
    DYNAMIC_SECTION("N = " << n) {
        int* d_counter = nullptr;
        REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_counter), sizeof(int)) == hipSuccess);
        REQUIRE(hip().hipMemset(d_counter, 0, sizeof(int)) == hipSuccess);
        
        // Load module
        auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
        hipModule_t module = nullptr;
        REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
        
        hipFunction_t kernel = nullptr;
        REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_add_int") == hipSuccess);
        
        void* args[] = { &d_counter, &n };
        unsigned int gridSize = calculateGridSize(n, kBlockSize);
        
        REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                            0, nullptr, args, nullptr) == hipSuccess);
        REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
        
        int result = 0;
        REQUIRE(hip().hipMemcpy(&result, d_counter, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
        
        // Each of n threads added 1
        REQUIRE(result == n);
        
        REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
        REQUIRE(hip().hipFree(d_counter) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "atomicAdd int - sum of indices", "[kernel][atomic][add]") {
    int n = 1000;
    
    int* d_result = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_result), sizeof(int)) == hipSuccess);
    REQUIRE(hip().hipMemset(d_result, 0, sizeof(int)) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_add_indices") == hipSuccess);
    
    void* args[] = { &d_result, &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    int result = 0;
    REQUIRE(hip().hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
    
    // Sum of 0 + 1 + 2 + ... + (n-1) = n*(n-1)/2
    int expected = n * (n - 1) / 2;
    REQUIRE(result == expected);
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_result) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "atomicAdd unsigned int", "[kernel][atomic][add]") {
    int n = 4096;
    
    unsigned int* d_counter = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_counter), sizeof(unsigned int)) == hipSuccess);
    REQUIRE(hip().hipMemset(d_counter, 0, sizeof(unsigned int)) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_add_uint") == hipSuccess);
    
    void* args[] = { &d_counter, &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    unsigned int result = 0;
    REQUIRE(hip().hipMemcpy(&result, d_counter, sizeof(unsigned int), hipMemcpyDeviceToHost) == hipSuccess);
    
    REQUIRE(result == static_cast<unsigned int>(n));
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_counter) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "atomicAdd float", "[kernel][atomic][add][float]") {
    int n = 1024;
    float value = 0.5f;
    
    float* d_result = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_result), sizeof(float)) == hipSuccess);
    
    // Initialize to 0
    float zero = 0.0f;
    REQUIRE(hip().hipMemcpy(d_result, &zero, sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_add_float") == hipSuccess);
    
    void* args[] = { &d_result, const_cast<float*>(&value), &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    float result = 0.0f;
    REQUIRE(hip().hipMemcpy(&result, d_result, sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
    
    float expected = n * value;
    // Allow some floating point tolerance
    REQUIRE(std::abs(result - expected) < 1.0f);
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_result) == hipSuccess);
}

//=============================================================================
// Atomic Sub Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "atomicSub int", "[kernel][atomic][sub]") {
    int n = 1000;
    
    int* d_counter = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_counter), sizeof(int)) == hipSuccess);
    
    // Start at n
    REQUIRE(hip().hipMemcpy(d_counter, &n, sizeof(int), hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_sub_int") == hipSuccess);
    
    void* args[] = { &d_counter, &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    int result = 0;
    REQUIRE(hip().hipMemcpy(&result, d_counter, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
    
    // n - n = 0
    REQUIRE(result == 0);
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_counter) == hipSuccess);
}

//=============================================================================
// Atomic Exchange Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "atomicExch int", "[kernel][atomic][exch]") {
    int n = 100;
    int new_value = 42;
    
    int* d_counter = nullptr;
    int* d_old_values = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_counter), sizeof(int)) == hipSuccess);
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_old_values), n * sizeof(int)) == hipSuccess);
    
    // Initialize counter
    int initial = 0;
    REQUIRE(hip().hipMemcpy(d_counter, &initial, sizeof(int), hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_exch_int") == hipSuccess);
    
    void* args[] = { &d_counter, &d_old_values, const_cast<int*>(&new_value), &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Final value should be new_value
    int final_value = 0;
    REQUIRE(hip().hipMemcpy(&final_value, d_counter, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(final_value == new_value);
    
    // Exactly one thread should have gotten 0 (the initial value)
    // All others should have gotten new_value
    std::vector<int> old_values(n);
    REQUIRE(hip().hipMemcpy(old_values.data(), d_old_values, n * sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
    
    int count_zeros = 0;
    int count_new = 0;
    for (int i = 0; i < n; ++i) {
        if (old_values[i] == 0) count_zeros++;
        else if (old_values[i] == new_value) count_new++;
    }
    REQUIRE(count_zeros == 1);
    REQUIRE(count_new == n - 1);
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_counter) == hipSuccess);
    REQUIRE(hip().hipFree(d_old_values) == hipSuccess);
}

//=============================================================================
// Atomic Min/Max Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "atomicMin int", "[kernel][atomic][min]") {
    int n = 1024;
    
    int* d_result = nullptr;
    int* d_values = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_result), sizeof(int)) == hipSuccess);
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_values), n * sizeof(int)) == hipSuccess);
    
    // Initialize result to INT_MAX
    int max_val = 0x7FFFFFFF;
    REQUIRE(hip().hipMemcpy(d_result, &max_val, sizeof(int), hipMemcpyHostToDevice) == hipSuccess);
    
    // Create values with known minimum
    std::vector<int> h_values(n);
    for (int i = 0; i < n; ++i) {
        h_values[i] = 1000 + i;  // Values: 1000, 1001, 1002, ...
    }
    h_values[n / 2] = -42;  // Insert minimum value in the middle
    
    REQUIRE(hip().hipMemcpy(d_values, h_values.data(), n * sizeof(int), hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_min_int") == hipSuccess);
    
    void* args[] = { &d_result, &d_values, &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    int result = 0;
    REQUIRE(hip().hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
    
    REQUIRE(result == -42);
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_result) == hipSuccess);
    REQUIRE(hip().hipFree(d_values) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "atomicMax int", "[kernel][atomic][max]") {
    int n = 1024;
    
    int* d_result = nullptr;
    int* d_values = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_result), sizeof(int)) == hipSuccess);
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_values), n * sizeof(int)) == hipSuccess);
    
    // Initialize result to INT_MIN
    int min_val = static_cast<int>(0x80000000);
    REQUIRE(hip().hipMemcpy(d_result, &min_val, sizeof(int), hipMemcpyHostToDevice) == hipSuccess);
    
    // Create values with known maximum
    std::vector<int> h_values(n);
    for (int i = 0; i < n; ++i) {
        h_values[i] = i - 500;  // Values: -500, -499, ..., 523
    }
    h_values[0] = 99999;  // Insert maximum at first position
    
    REQUIRE(hip().hipMemcpy(d_values, h_values.data(), n * sizeof(int), hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_max_int") == hipSuccess);
    
    void* args[] = { &d_result, &d_values, &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    int result = 0;
    REQUIRE(hip().hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
    
    REQUIRE(result == 99999);
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_result) == hipSuccess);
    REQUIRE(hip().hipFree(d_values) == hipSuccess);
}

//=============================================================================
// Atomic Bitwise Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "atomicAnd int", "[kernel][atomic][and]") {
    int n = 32;
    
    int* d_result = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_result), sizeof(int)) == hipSuccess);
    
    // Start with all bits set
    int initial = 0xFFFFFFFF;
    REQUIRE(hip().hipMemcpy(d_result, &initial, sizeof(int), hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_and_int") == hipSuccess);
    
    // AND with 0x0F0F0F0F
    int mask = 0x0F0F0F0F;
    void* args[] = { &d_result, &mask, &n };
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, n, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    int result = 0;
    REQUIRE(hip().hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
    
    // After ANDing with mask, result should be the mask
    REQUIRE(result == mask);
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_result) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "atomicOr int", "[kernel][atomic][or]") {
    int n = 32;
    
    int* d_result = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_result), sizeof(int)) == hipSuccess);
    
    // Start with zeros
    REQUIRE(hip().hipMemset(d_result, 0, sizeof(int)) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_or_int") == hipSuccess);
    
    // OR with value that has specific bits set
    int value = 0x12345678;
    void* args[] = { &d_result, &value, &n };
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, n, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    int result = 0;
    REQUIRE(hip().hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
    
    // After ORing, result should have those bits set
    REQUIRE(result == value);
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_result) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "atomicXor int", "[kernel][atomic][xor]") {
    int n = 1024;  // Even number of threads
    
    int* d_result = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_result), sizeof(int)) == hipSuccess);
    
    REQUIRE(hip().hipMemset(d_result, 0, sizeof(int)) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_xor_int") == hipSuccess);
    
    // XOR with same value an even number of times should give 0
    int value = 0x55555555;
    void* args[] = { &d_result, &value, &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    int result = 0;
    REQUIRE(hip().hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
    
    // XOR n times with same value: if n is even, result should be 0
    REQUIRE(result == 0);
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_result) == hipSuccess);
}

//=============================================================================
// Histogram Test (Practical Application)
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Atomic histogram", "[kernel][atomic][histogram]") {
    int n = 4096;
    int num_bins = 256;
    
    unsigned char* d_data = nullptr;
    int* d_histogram = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_data), n * sizeof(unsigned char)) == hipSuccess);
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_histogram), num_bins * sizeof(int)) == hipSuccess);
    
    // Initialize histogram to zeros
    REQUIRE(hip().hipMemset(d_histogram, 0, num_bins * sizeof(int)) == hipSuccess);
    
    // Create test data - all values are the same for easy verification
    std::vector<unsigned char> h_data(n, 42);  // All 42s
    REQUIRE(hip().hipMemcpy(d_data, h_data.data(), n * sizeof(unsigned char), hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_histogram") == hipSuccess);
    
    void* args[] = { &d_data, &d_histogram, &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    std::vector<int> h_histogram(num_bins);
    REQUIRE(hip().hipMemcpy(h_histogram.data(), d_histogram, num_bins * sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
    
    // Bin 42 should have n entries, all others should be 0
    for (int i = 0; i < num_bins; ++i) {
        if (i == 42) {
            REQUIRE(h_histogram[i] == n);
        } else {
            REQUIRE(h_histogram[i] == 0);
        }
    }
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_data) == hipSuccess);
    REQUIRE(hip().hipFree(d_histogram) == hipSuccess);
}

//=============================================================================
// Multi-slot Atomic Test
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Atomic add to multiple slots", "[kernel][atomic][slots]") {
    int n = 4096;
    int num_slots = 8;
    
    int* d_slots = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_slots), num_slots * sizeof(int)) == hipSuccess);
    REQUIRE(hip().hipMemset(d_slots, 0, num_slots * sizeof(int)) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_add_slots") == hipSuccess);
    
    void* args[] = { &d_slots, const_cast<int*>(&num_slots), &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    std::vector<int> h_slots(num_slots);
    REQUIRE(hip().hipMemcpy(h_slots.data(), d_slots, num_slots * sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
    
    // Each slot should have n/num_slots entries
    int expected_per_slot = n / num_slots;
    int total = 0;
    for (int i = 0; i < num_slots; ++i) {
        REQUIRE(h_slots[i] == expected_per_slot);
        total += h_slots[i];
    }
    REQUIRE(total == n);
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_slots) == hipSuccess);
}

//=============================================================================
// Atomic Inc/Dec Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "atomicInc with wraparound", "[kernel][atomic][inc]") {
    int n = 100;
    unsigned int max_val = 9;  // Wrap at 10
    
    unsigned int* d_counter = nullptr;
    unsigned int* d_old_values = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_counter), sizeof(unsigned int)) == hipSuccess);
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_old_values), n * sizeof(unsigned int)) == hipSuccess);
    
    REQUIRE(hip().hipMemset(d_counter, 0, sizeof(unsigned int)) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_atomics_test_atomics_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "atomic_inc_test") == hipSuccess);
    
    void* args[] = { &d_counter, &d_old_values, const_cast<unsigned int*>(&max_val), &n };
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, n, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    unsigned int final_counter = 0;
    REQUIRE(hip().hipMemcpy(&final_counter, d_counter, sizeof(unsigned int), hipMemcpyDeviceToHost) == hipSuccess);
    
    // After 100 increments with max=9, counter should be 100 % 10 = 0
    REQUIRE(final_counter == (n % (max_val + 1)));
    
    // Verify old values - each value 0-9 should appear exactly 10 times
    std::vector<unsigned int> h_old_values(n);
    REQUIRE(hip().hipMemcpy(h_old_values.data(), d_old_values, n * sizeof(unsigned int), hipMemcpyDeviceToHost) == hipSuccess);
    
    std::vector<int> counts(max_val + 1, 0);
    for (int i = 0; i < n; ++i) {
        REQUIRE(h_old_values[i] <= max_val);
        counts[h_old_values[i]]++;
    }
    
    // Each value 0-9 should appear n/(max_val+1) = 10 times
    for (unsigned int i = 0; i <= max_val; ++i) {
        REQUIRE(counts[i] == n / (max_val + 1));
    }
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_counter) == hipSuccess);
    REQUIRE(hip().hipFree(d_old_values) == hipSuccess);
}
