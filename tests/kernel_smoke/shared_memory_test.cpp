// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for kernel shared memory functionality

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

#include "kernel_shared_memory_test_shared_memory.hpp"

#include <vector>
#include <numeric>
#include <algorithm>

//=============================================================================
// Test Constants
//=============================================================================

static constexpr int kBlockSize = 256;

static unsigned int calculateGridSize(int n, int blockSize) {
    return static_cast<unsigned int>((n + blockSize - 1) / blockSize);
}

//=============================================================================
// Static Shared Memory Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Shared memory copy through shared buffer", "[kernel][shared]") {
    int n = GENERATE(256, 512, 1024, 4096);
    
    DYNAMIC_SECTION("N = " << n) {
        size_t size = n * sizeof(int);
        std::vector<int> h_input(n);
        std::vector<int> h_output(n, 0);
        
        // Initialize with sequential values
        std::iota(h_input.begin(), h_input.end(), 1);
        
        int* d_input = nullptr;
        int* d_output = nullptr;
        REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_input), size) == hipSuccess);
        REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_output), size) == hipSuccess);
        
        REQUIRE(hip().hipMemcpy(d_input, h_input.data(), size, hipMemcpyHostToDevice) == hipSuccess);
        REQUIRE(hip().hipMemset(d_output, 0, size) == hipSuccess);
        
        // Load module
        auto kernel_data = hip_cts::kernels::kernel_shared_memory_test_shared_memory_data::get();
        hipModule_t module = nullptr;
        REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
        
        hipFunction_t kernel = nullptr;
        REQUIRE(hip().hipModuleGetFunction(&kernel, module, "shared_copy") == hipSuccess);
        
        void* args[] = { &d_input, &d_output, &n };
        unsigned int gridSize = calculateGridSize(n, kBlockSize);
        
        REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                            0, nullptr, args, nullptr) == hipSuccess);
        REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
        
        REQUIRE(hip().hipMemcpy(h_output.data(), d_output, size, hipMemcpyDeviceToHost) == hipSuccess);
        
        // Verify data was copied correctly through shared memory
        for (int i = 0; i < n; ++i) {
            INFO("Index: " << i);
            REQUIRE(h_output[i] == h_input[i]);
        }
        
        REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
        REQUIRE(hip().hipFree(d_input) == hipSuccess);
        REQUIRE(hip().hipFree(d_output) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "Shared memory neighbor sum with syncthreads", "[kernel][shared][sync]") {
    int n = 1024;
    size_t size = n * sizeof(int);
    
    std::vector<int> h_input(n, 1);  // All ones
    std::vector<int> h_output(n, 0);
    
    int* d_input = nullptr;
    int* d_output = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_input), size) == hipSuccess);
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_output), size) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(d_input, h_input.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_shared_memory_test_shared_memory_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "shared_neighbor_sum") == hipSuccess);
    
    void* args[] = { &d_input, &d_output, &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(h_output.data(), d_output, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify: each element should be sum of self + left + right neighbor
    // For elements in middle of blocks: 1 + 1 + 1 = 3
    // For first element in block: 0 + 1 + 1 = 2
    // For last element in block: 1 + 1 + 0 = 2
    for (int i = 0; i < n; ++i) {
        int local_idx = i % kBlockSize;
        int expected;
        if (local_idx == 0) {
            expected = 2;  // No left neighbor
        } else if (local_idx == kBlockSize - 1) {
            expected = 2;  // No right neighbor
        } else {
            expected = 3;  // Has both neighbors
        }
        INFO("Index: " << i << ", local_idx: " << local_idx);
        REQUIRE(h_output[i] == expected);
    }
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_input) == hipSuccess);
    REQUIRE(hip().hipFree(d_output) == hipSuccess);
}

//=============================================================================
// Dynamic Shared Memory Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Dynamic shared memory allocation", "[kernel][shared][dynamic]") {
    // Dynamic shared memory (extern __shared__) may not be supported on all backends
    if (is_streaming_backend()) {
        SKIP("Dynamic shared memory not yet supported on streaming backend");
    }
    
    int n = 1024;
    size_t size = n * sizeof(int);
    
    std::vector<int> h_output(n, 0);
    
    int* d_output = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_output), size) == hipSuccess);
    REQUIRE(hip().hipMemset(d_output, 0, size) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_shared_memory_test_shared_memory_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "shared_dynamic_fill") == hipSuccess);
    
    void* args[] = { &d_output, &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    // Pass dynamic shared memory size
    size_t dynamicSharedSize = kBlockSize * sizeof(int);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        dynamicSharedSize, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(h_output.data(), d_output, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify: each element should be tid * 10 + blockIdx
    for (int i = 0; i < n; ++i) {
        int block_idx = i / kBlockSize;
        int thread_idx = i % kBlockSize;
        int expected = thread_idx * 10 + block_idx;
        INFO("Index: " << i << ", expected: " << expected << ", got: " << h_output[i]);
        REQUIRE(h_output[i] == expected);
    }
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_output) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Shared memory reduction sum", "[kernel][shared][reduction]") {
    // Dynamic shared memory (extern __shared__) may not be supported on all backends
    if (is_streaming_backend()) {
        SKIP("Dynamic shared memory not yet supported on streaming backend");
    }
    
    int n = GENERATE(256, 1024, 4096);
    
    DYNAMIC_SECTION("N = " << n) {
        size_t input_size = n * sizeof(int);
        int num_blocks = calculateGridSize(n, kBlockSize);
        size_t output_size = num_blocks * sizeof(int);
        
        std::vector<int> h_input(n);
        std::vector<int> h_partial(num_blocks, 0);
        
        // Initialize with 1s for easy verification
        std::fill(h_input.begin(), h_input.end(), 1);
        
        int* d_input = nullptr;
        int* d_output = nullptr;
        REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_input), input_size) == hipSuccess);
        REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_output), output_size) == hipSuccess);
        
        REQUIRE(hip().hipMemcpy(d_input, h_input.data(), input_size, hipMemcpyHostToDevice) == hipSuccess);
        
        // Load module
        auto kernel_data = hip_cts::kernels::kernel_shared_memory_test_shared_memory_data::get();
        hipModule_t module = nullptr;
        REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
        
        hipFunction_t kernel = nullptr;
        REQUIRE(hip().hipModuleGetFunction(&kernel, module, "shared_reduce_sum") == hipSuccess);
        
        void* args[] = { &d_input, &d_output, &n };
        size_t dynamicSharedSize = kBlockSize * sizeof(int);
        
        REQUIRE(hip().hipModuleLaunchKernel(kernel, num_blocks, 1, 1, kBlockSize, 1, 1,
                                            dynamicSharedSize, nullptr, args, nullptr) == hipSuccess);
        REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
        
        REQUIRE(hip().hipMemcpy(h_partial.data(), d_output, output_size, hipMemcpyDeviceToHost) == hipSuccess);
        
        // Sum up partial results
        int total = 0;
        for (int i = 0; i < num_blocks; ++i) {
            total += h_partial[i];
        }
        
        // All elements are 1, so sum should equal n
        REQUIRE(total == n);
        
        REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
        REQUIRE(hip().hipFree(d_input) == hipSuccess);
        REQUIRE(hip().hipFree(d_output) == hipSuccess);
    }
}

//=============================================================================
// Shared Memory Reverse Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Shared memory reverse within blocks", "[kernel][shared]") {
    int n = 512;  // 2 blocks
    size_t size = n * sizeof(int);
    
    std::vector<int> h_data(n);
    std::iota(h_data.begin(), h_data.end(), 0);  // 0, 1, 2, ...
    
    int* d_data = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_data), size) == hipSuccess);
    REQUIRE(hip().hipMemcpy(d_data, h_data.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_shared_memory_test_shared_memory_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "shared_reverse") == hipSuccess);
    
    void* args[] = { &d_data, &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    std::vector<int> h_result(n);
    REQUIRE(hip().hipMemcpy(h_result.data(), d_data, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify: each block should be reversed
    for (int block = 0; block < n / kBlockSize; ++block) {
        for (int i = 0; i < kBlockSize; ++i) {
            int global_idx = block * kBlockSize + i;
            int expected_idx = block * kBlockSize + (kBlockSize - 1 - i);
            INFO("Block: " << block << ", Index: " << i);
            REQUIRE(h_result[global_idx] == expected_idx);
        }
    }
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_data) == hipSuccess);
}

//=============================================================================
// Multiple Syncthreads Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Multiple syncthreads barriers", "[kernel][shared][sync]") {
    int n = 1024;
    size_t size = n * sizeof(int);
    
    std::vector<int> h_data(n);
    std::iota(h_data.begin(), h_data.end(), 0);  // 0, 1, 2, ...
    
    int* d_data = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_data), size) == hipSuccess);
    REQUIRE(hip().hipMemcpy(d_data, h_data.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_shared_memory_test_shared_memory_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "shared_multi_sync") == hipSuccess);
    
    void* args[] = { &d_data, &n };
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    std::vector<int> h_result(n);
    REQUIRE(hip().hipMemcpy(h_result.data(), d_data, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify: each element should have +111 added (1 + 10 + 100)
    for (int i = 0; i < n; ++i) {
        INFO("Index: " << i);
        REQUIRE(h_result[i] == h_data[i] + 111);
    }
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_data) == hipSuccess);
}

//=============================================================================
// Shared Memory on Multiple Streams
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Shared memory kernels on multiple streams", "[kernel][shared][stream]") {
    int n = 1024;
    int num_streams = 4;
    size_t size = n * sizeof(int);
    
    std::vector<hipStream_t> streams(num_streams);
    std::vector<int*> d_inputs(num_streams);
    std::vector<int*> d_outputs(num_streams);
    std::vector<std::vector<int>> h_inputs(num_streams);
    std::vector<std::vector<int>> h_outputs(num_streams);
    
    // Load module once
    auto kernel_data = hip_cts::kernels::kernel_shared_memory_test_shared_memory_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "shared_copy") == hipSuccess);
    
    // Set up resources
    for (int s = 0; s < num_streams; ++s) {
        REQUIRE(hip().hipStreamCreate(&streams[s]) == hipSuccess);
        REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_inputs[s]), size) == hipSuccess);
        REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_outputs[s]), size) == hipSuccess);
        
        h_inputs[s].resize(n);
        h_outputs[s].resize(n, 0);
        
        // Each stream gets different data
        for (int i = 0; i < n; ++i) {
            h_inputs[s][i] = s * 1000 + i;
        }
        
        REQUIRE(hip().hipMemcpy(d_inputs[s], h_inputs[s].data(), size, hipMemcpyHostToDevice) == hipSuccess);
    }
    
    // Launch kernels on all streams
    unsigned int gridSize = calculateGridSize(n, kBlockSize);
    for (int s = 0; s < num_streams; ++s) {
        int count = n;
        void* args[] = { &d_inputs[s], &d_outputs[s], &count };
        REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, kBlockSize, 1, 1,
                                            0, streams[s], args, nullptr) == hipSuccess);
    }
    
    // Sync and verify
    for (int s = 0; s < num_streams; ++s) {
        REQUIRE(hip().hipStreamSynchronize(streams[s]) == hipSuccess);
        REQUIRE(hip().hipMemcpy(h_outputs[s].data(), d_outputs[s], size, hipMemcpyDeviceToHost) == hipSuccess);
        
        for (int i = 0; i < n; ++i) {
            INFO("Stream: " << s << ", Index: " << i);
            REQUIRE(h_outputs[s][i] == h_inputs[s][i]);
        }
    }
    
    // Cleanup
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    for (int s = 0; s < num_streams; ++s) {
        REQUIRE(hip().hipFree(d_inputs[s]) == hipSuccess);
        REQUIRE(hip().hipFree(d_outputs[s]) == hipSuccess);
        REQUIRE(hip().hipStreamDestroy(streams[s]) == hipSuccess);
    }
}
