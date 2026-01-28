// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for HIP occupancy API
// Note: The runtime occupancy API (hipOccupancyMaxActiveBlocksPerMultiprocessor etc.)
// takes host function pointers, not hipFunction_t from module API.
// These tests verify kernel launch correctness with various configurations.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

#include "kernel_occupancy_test_occupancy.hpp"

//=============================================================================
// Test Constants
//=============================================================================

static constexpr int kBlockSize = 256;

//=============================================================================
// Kernel Launch with Various Configurations
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Launch kernel with recommended block sizes", "[kernel][occupancy]") {
    auto kernel_data = hip_cts::kernels::kernel_occupancy_test_occupancy_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "set_value") == hipSuccess);

    // Test with various block sizes - common power-of-2 values
    int blockSizes[] = {64, 128, 256, 512};

    for (int blockSize : blockSizes) {
        DYNAMIC_SECTION("Block size " << blockSize) {
            int n = 1024;
            size_t size = n * sizeof(int);
            std::vector<int> h_output(n, 0);

            int* d_output = nullptr;
            REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_output), size) == hipSuccess);
            REQUIRE(hip().hipMemset(d_output, 0, size) == hipSuccess);

            int value = 42;
            int n_arg = n;
            void* args[] = { &d_output, &value, &n_arg };

            int gridSize = (n + blockSize - 1) / blockSize;
            REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1,
                                                0, nullptr, args, nullptr) == hipSuccess);
            REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

            REQUIRE(hip().hipMemcpy(h_output.data(), d_output, size, hipMemcpyDeviceToHost) == hipSuccess);

            for (int i = 0; i < n; ++i) {
                REQUIRE(h_output[i] == value);
            }

            REQUIRE(hip().hipFree(d_output) == hipSuccess);
        }
    }

    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Launch kernel with various grid sizes", "[kernel][occupancy]") {
    auto kernel_data = hip_cts::kernels::kernel_occupancy_test_occupancy_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "set_value") == hipSuccess);

    int gridSizes[] = {1, 4, 16, 64, 256};
    const int blockSize = 256;

    for (int gridSize : gridSizes) {
        DYNAMIC_SECTION("Grid size " << gridSize) {
            int n = gridSize * blockSize;
            size_t size = n * sizeof(int);
            std::vector<int> h_output(n, 0);

            int* d_output = nullptr;
            REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_output), size) == hipSuccess);
            REQUIRE(hip().hipMemset(d_output, 0, size) == hipSuccess);

            int value = 123;
            int n_arg = n;
            void* args[] = { &d_output, &value, &n_arg };

            REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1,
                                                0, nullptr, args, nullptr) == hipSuccess);
            REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

            REQUIRE(hip().hipMemcpy(h_output.data(), d_output, size, hipMemcpyDeviceToHost) == hipSuccess);

            for (int i = 0; i < n; ++i) {
                REQUIRE(h_output[i] == value);
            }

            REQUIRE(hip().hipFree(d_output) == hipSuccess);
        }
    }

    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Launch kernel with shared memory", "[kernel][occupancy][shared]") {
    auto kernel_data = hip_cts::kernels::kernel_occupancy_test_occupancy_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "shared_mem_kernel") == hipSuccess);

    int n = 1024;
    size_t size = n * sizeof(int);
    std::vector<int> h_output(n, -1);

    int* d_output = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_output), size) == hipSuccess);
    REQUIRE(hip().hipMemset(d_output, 0xFF, size) == hipSuccess);

    int n_arg = n;
    void* args[] = { &d_output, &n_arg };

    // This kernel uses static shared memory (256 ints = 1KB)
    REQUIRE(hip().hipModuleLaunchKernel(kernel, 4, 1, 1, 256, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    REQUIRE(hip().hipMemcpy(h_output.data(), d_output, size, hipMemcpyDeviceToHost) == hipSuccess);

    // Verify: output[gid] = sdata[tid % 256] = (tid % 256)
    for (int i = 0; i < n; ++i) {
        int expected = i % 256;
        REQUIRE(h_output[i] == expected);
    }

    REQUIRE(hip().hipFree(d_output) == hipSuccess);
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Launch simple kernel many times", "[kernel][occupancy][stress]") {
    auto kernel_data = hip_cts::kernels::kernel_occupancy_test_occupancy_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "simple_kernel") == hipSuccess);

    // Launch many kernels in sequence
    const int numLaunches = 1000;
    for (int i = 0; i < numLaunches; ++i) {
        REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, 64, 1, 1,
                                            0, nullptr, nullptr, nullptr) == hipSuccess);
    }
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Launch kernels with various shared memory sizes", "[kernel][occupancy][shared]") {
    auto kernel_data = hip_cts::kernels::kernel_occupancy_test_occupancy_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "simple_kernel") == hipSuccess);

    // Test with various dynamic shared memory sizes
    size_t sharedSizes[] = {0, 1024, 4096, 16384};

    for (size_t sharedSize : sharedSizes) {
        DYNAMIC_SECTION("Shared memory " << sharedSize << " bytes") {
            hipError_t err = hip().hipModuleLaunchKernel(kernel, 1, 1, 1, 64, 1, 1,
                                                         sharedSize, nullptr, nullptr, nullptr);
            // Large shared memory may fail on some GPUs, that's OK
            if (err == hipSuccess) {
                REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
            } else {
                // If launch fails due to resource limits, that's expected
                REQUIRE((err == hipErrorLaunchOutOfResources || 
                         err == hipErrorInvalidConfiguration));
            }
        }
    }

    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}
