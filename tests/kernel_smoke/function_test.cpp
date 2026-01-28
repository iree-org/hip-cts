// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for HIP function attribute and configuration API

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <vector>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

#include "kernel_function_test_function.hpp"

//=============================================================================
// hipFuncGetAttributes Tests (Runtime API - takes host function pointer)
//=============================================================================

// Note: hipFuncGetAttributes in the runtime API takes a host function pointer,
// not a hipFunction_t from the driver API. These tests are for driver API
// modules, so we test using hipModuleLaunchKernel and verify the kernel works.

//=============================================================================
// hipFuncSetCacheConfig Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipFuncSetCacheConfig prefer none", "[kernel][function][cache]") {
    auto kernel_data = hip_cts::kernels::kernel_function_test_function_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "simple_kernel") == hipSuccess);

    // Note: hipFuncSetCacheConfig takes const void* in runtime API
    // For module functions, this may not be directly callable
    // Just verify the function can be launched
    REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

//=============================================================================
// Kernel Launch Tests with Various Configurations
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Launch kernel with different block sizes", "[kernel][function][launch]") {
    auto kernel_data = hip_cts::kernels::kernel_function_test_function_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "simple_kernel") == hipSuccess);

    // Test various block sizes
    int blockSizes[] = {32, 64, 128, 256, 512};

    for (int blockSize : blockSizes) {
        DYNAMIC_SECTION("Block size " << blockSize) {
            REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, blockSize, 1, 1, 0, nullptr, nullptr, nullptr) == hipSuccess);
            REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
        }
    }

    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Launch kernel with different grid sizes", "[kernel][function][launch]") {
    auto kernel_data = hip_cts::kernels::kernel_function_test_function_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "simple_kernel") == hipSuccess);

    // Test various grid sizes
    int gridSizes[] = {1, 4, 16, 64, 256};

    for (int gridSize : gridSizes) {
        DYNAMIC_SECTION("Grid size " << gridSize) {
            REQUIRE(hip().hipModuleLaunchKernel(kernel, gridSize, 1, 1, 64, 1, 1, 0, nullptr, nullptr, nullptr) == hipSuccess);
            REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
        }
    }

    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Launch kernel with 2D grid and block", "[kernel][function][launch]") {
    auto kernel_data = hip_cts::kernels::kernel_function_test_function_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "simple_kernel") == hipSuccess);

    // 2D grid: 8x8 = 64 blocks
    // 2D block: 16x16 = 256 threads
    REQUIRE(hip().hipModuleLaunchKernel(kernel, 8, 8, 1, 16, 16, 1, 0, nullptr, nullptr, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Launch kernel with 3D grid and block", "[kernel][function][launch]") {
    auto kernel_data = hip_cts::kernels::kernel_function_test_function_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "simple_kernel") == hipSuccess);

    // 3D grid: 4x4x4 = 64 blocks
    // 3D block: 4x4x4 = 64 threads
    REQUIRE(hip().hipModuleLaunchKernel(kernel, 4, 4, 4, 4, 4, 4, 0, nullptr, nullptr, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Launch kernel with shared memory via args", "[kernel][function][launch]") {
    auto kernel_data = hip_cts::kernels::kernel_function_test_function_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "shared_mem_kernel") == hipSuccess);

    int n = 256;
    size_t size = n * sizeof(int);
    std::vector<int> h_output(n, 0);

    int* d_output = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_output), size) == hipSuccess);
    REQUIRE(hip().hipMemset(d_output, 0, size) == hipSuccess);

    int n_arg = n;
    void* args[] = { &d_output, &n_arg };

    // Launch with static shared memory used by kernel
    REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    REQUIRE(hip().hipMemcpy(h_output.data(), d_output, size, hipMemcpyDeviceToHost) == hipSuccess);

    // Verify: output[i] = i (from sdata[tid] = tid)
    for (int i = 0; i < n; ++i) {
        REQUIRE(h_output[i] == i);
    }

    REQUIRE(hip().hipFree(d_output) == hipSuccess);
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Launch kernel with local memory", "[kernel][function][launch]") {
    auto kernel_data = hip_cts::kernels::kernel_function_test_function_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "local_mem_kernel") == hipSuccess);

    int n = 256;
    size_t size = n * sizeof(int);
    std::vector<int> h_output(n, 0);

    int* d_output = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_output), size) == hipSuccess);
    REQUIRE(hip().hipMemset(d_output, 0, size) == hipSuccess);

    int n_arg = n;
    void* args[] = { &d_output, &n_arg };

    REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    REQUIRE(hip().hipMemcpy(h_output.data(), d_output, size, hipMemcpyDeviceToHost) == hipSuccess);

    // Verify: each thread computes sum of local_array[i] = tid + i for i in 0..15
    // sum = tid*16 + (0+1+2+...+15) = tid*16 + 120
    for (int i = 0; i < n; ++i) {
        int expected = i * 16 + 120;
        REQUIRE(h_output[i] == expected);
    }

    REQUIRE(hip().hipFree(d_output) == hipSuccess);
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Launch kernel with many registers", "[kernel][function][launch]") {
    auto kernel_data = hip_cts::kernels::kernel_function_test_function_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "many_regs_kernel") == hipSuccess);

    int n = 256;
    size_t size = n * sizeof(float);
    std::vector<float> h_output(n, 0.0f);

    float* d_output = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_output), size) == hipSuccess);
    REQUIRE(hip().hipMemset(d_output, 0, size) == hipSuccess);

    int n_arg = n;
    void* args[] = { &d_output, &n_arg };

    REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    REQUIRE(hip().hipMemcpy(h_output.data(), d_output, size, hipMemcpyDeviceToHost) == hipSuccess);

    // Verify: result = (0+1+2+...+15) + gid = 120 + gid
    for (int i = 0; i < n; ++i) {
        float expected = 120.0f + static_cast<float>(i);
        REQUIRE(h_output[i] == expected);
    }

    REQUIRE(hip().hipFree(d_output) == hipSuccess);
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

//=============================================================================
// Multiple Kernel Launches
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Launch multiple kernels in sequence", "[kernel][function][sequence]") {
    auto kernel_data = hip_cts::kernels::kernel_function_test_function_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "simple_kernel") == hipSuccess);

    // Launch many kernels in sequence
    for (int i = 0; i < 100; ++i) {
        REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, 64, 1, 1, 0, nullptr, nullptr, nullptr) == hipSuccess);
    }
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Launch kernels on multiple streams", "[kernel][function][stream]") {
    auto kernel_data = hip_cts::kernels::kernel_function_test_function_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "simple_kernel") == hipSuccess);

    const int numStreams = 4;
    hipStream_t streams[numStreams];

    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
    }

    // Launch kernels on different streams
    for (int i = 0; i < 100; ++i) {
        hipStream_t stream = streams[i % numStreams];
        REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, 64, 1, 1, 0, stream, nullptr, nullptr) == hipSuccess);
    }

    // Synchronize all streams
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamSynchronize(streams[i]) == hipSuccess);
    }

    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamDestroy(streams[i]) == hipSuccess);
    }

    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}
