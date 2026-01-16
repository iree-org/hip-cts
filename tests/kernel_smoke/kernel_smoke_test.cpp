// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

#include "kernel_smoke_test_vector_add.hpp"
#include "kernel_smoke_test_vector_scale.hpp"
#include "kernel_smoke_test_memset_kernel.hpp"

#include <vector>
#include <cmath>
#include <numeric>

//=============================================================================
// Module Loading Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipModuleLoadData with embedded kernel", "[kernel][module]") {
    hipModule_t module = nullptr;
    
    SECTION("Load vector_add kernel") {
        auto kernel_data = hip_cts::kernels::kernel_smoke_test_vector_add_data::get();
        auto result = hip().hipModuleLoadData(&module, kernel_data.data());
        REQUIRE(result == hipSuccess);
        REQUIRE(module != nullptr);
        
        REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    }
    
    SECTION("Load vector_scale kernel") {
        auto kernel_data = hip_cts::kernels::kernel_smoke_test_vector_scale_data::get();
        auto result = hip().hipModuleLoadData(&module, kernel_data.data());
        REQUIRE(result == hipSuccess);
        REQUIRE(module != nullptr);
        
        REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    }
    
    SECTION("Load memset_kernel") {
        auto kernel_data = hip_cts::kernels::kernel_smoke_test_memset_kernel_data::get();
        auto result = hip().hipModuleLoadData(&module, kernel_data.data());
        REQUIRE(result == hipSuccess);
        REQUIRE(module != nullptr);
        
        REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipModuleGetFunction", "[kernel][module]") {
    auto kernel_data = hip_cts::kernels::kernel_smoke_test_vector_add_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    SECTION("Get existing function") {
        hipFunction_t func = nullptr;
        REQUIRE(hip().hipModuleGetFunction(&func, module, "vector_add") == hipSuccess);
        REQUIRE(func != nullptr);
    }
    
    SECTION("Get non-existent function returns error") {
        hipFunction_t func = nullptr;
        auto result = hip().hipModuleGetFunction(&func, module, "nonexistent_function");
        REQUIRE(result == hipErrorNotFound);
    }
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
}

//=============================================================================
// Kernel Execution Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Vector add kernel execution", "[kernel][execution]") {
    constexpr int N = 1024;
    constexpr size_t size = N * sizeof(float);
    
    // Host data
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N, 0.0f);
    
    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // Device data
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_a), size) == hipSuccess);
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_b), size) == hipSuccess);
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_c), size) == hipSuccess);
    
    // Copy input to device
    REQUIRE(hip().hipMemcpy(d_a, h_a.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    REQUIRE(hip().hipMemcpy(d_b, h_b.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module and get function
    auto kernel_data = hip_cts::kernels::kernel_smoke_test_vector_add_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "vector_add") == hipSuccess);
    
    // Launch kernel
    int n = N;
    void* args[] = { &d_a, &d_b, &d_c, &n };
    
    unsigned int blockSize = 256;
    unsigned int gridSize = (N + blockSize - 1) / blockSize;
    
    REQUIRE(hip().hipModuleLaunchKernel(
        kernel,
        gridSize, 1, 1,    // grid dimensions
        blockSize, 1, 1,   // block dimensions
        0,                 // shared memory
        nullptr,           // stream
        args,              // kernel arguments
        nullptr            // extra
    ) == hipSuccess);
    
    // Synchronize
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Copy result back
    REQUIRE(hip().hipMemcpy(h_c.data(), d_c, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify results
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        REQUIRE(h_c[i] == expected);
    }
    
    // Cleanup
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_a) == hipSuccess);
    REQUIRE(hip().hipFree(d_b) == hipSuccess);
    REQUIRE(hip().hipFree(d_c) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Vector scale kernel execution", "[kernel][execution]") {
    constexpr int N = 1024;
    constexpr size_t size = N * sizeof(float);
    constexpr float scale = 2.5f;
    
    // Host data
    std::vector<float> h_data(N);
    std::vector<float> h_expected(N);
    
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
        h_expected[i] = h_data[i] * scale;
    }
    
    // Device data
    float* d_data = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_data), size) == hipSuccess);
    REQUIRE(hip().hipMemcpy(d_data, h_data.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module
    auto kernel_data = hip_cts::kernels::kernel_smoke_test_vector_scale_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "vector_scale") == hipSuccess);
    
    // Launch kernel
    int n = N;
    float s = scale;
    void* args[] = { &d_data, &s, &n };
    
    unsigned int blockSize = 256;
    unsigned int gridSize = (N + blockSize - 1) / blockSize;
    
    REQUIRE(hip().hipModuleLaunchKernel(
        kernel,
        gridSize, 1, 1,
        blockSize, 1, 1,
        0, nullptr, args, nullptr
    ) == hipSuccess);
    
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(h_data.data(), d_data, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (int i = 0; i < N; ++i) {
        REQUIRE(std::abs(h_data[i] - h_expected[i]) < 1e-5f);
    }
    
    // Cleanup
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_data) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Memset kernel execution", "[kernel][execution]") {
    constexpr int N = 1024;
    constexpr size_t size = N * sizeof(int);
    
    SECTION("memset_int kernel") {
        std::vector<int> h_data(N, 0);
        int* d_data = nullptr;
        
        REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_data), size) == hipSuccess);
        REQUIRE(hip().hipMemcpy(d_data, h_data.data(), size, hipMemcpyHostToDevice) == hipSuccess);
        
        // Load module
        auto kernel_data = hip_cts::kernels::kernel_smoke_test_memset_kernel_data::get();
        hipModule_t module = nullptr;
        REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
        
        hipFunction_t kernel = nullptr;
        REQUIRE(hip().hipModuleGetFunction(&kernel, module, "memset_int") == hipSuccess);
        
        // Launch kernel
        int n = N;
        int value = 42;
        void* args[] = { &d_data, &value, &n };
        
        unsigned int blockSize = 256;
        unsigned int gridSize = (N + blockSize - 1) / blockSize;
        
        REQUIRE(hip().hipModuleLaunchKernel(
            kernel,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, nullptr, args, nullptr
        ) == hipSuccess);
        
        REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
        
        // Verify
        REQUIRE(hip().hipMemcpy(h_data.data(), d_data, size, hipMemcpyDeviceToHost) == hipSuccess);
        
        for (int i = 0; i < N; ++i) {
            REQUIRE(h_data[i] == 42);
        }
        
        REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
        REQUIRE(hip().hipFree(d_data) == hipSuccess);
    }
    
    SECTION("memset_float kernel") {
        std::vector<float> h_data(N, 0.0f);
        float* d_data = nullptr;
        
        REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_data), N * sizeof(float)) == hipSuccess);
        REQUIRE(hip().hipMemcpy(d_data, h_data.data(), N * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
        
        auto kernel_data = hip_cts::kernels::kernel_smoke_test_memset_kernel_data::get();
        hipModule_t module = nullptr;
        REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
        
        hipFunction_t kernel = nullptr;
        REQUIRE(hip().hipModuleGetFunction(&kernel, module, "memset_float") == hipSuccess);
        
        int n = N;
        float value = 3.14f;
        void* args[] = { &d_data, &value, &n };
        
        unsigned int blockSize = 256;
        unsigned int gridSize = (N + blockSize - 1) / blockSize;
        
        REQUIRE(hip().hipModuleLaunchKernel(
            kernel,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, nullptr, args, nullptr
        ) == hipSuccess);
        
        REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
        
        REQUIRE(hip().hipMemcpy(h_data.data(), d_data, N * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
        
        for (int i = 0; i < N; ++i) {
            REQUIRE(std::abs(h_data[i] - 3.14f) < 1e-5f);
        }
        
        REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
        REQUIRE(hip().hipFree(d_data) == hipSuccess);
    }
}

//=============================================================================
// Kernel Launch Parameter Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Kernel launch with various grid sizes", "[kernel][launch]") {
    constexpr size_t size = 1024 * 1024 * sizeof(int); // 1M elements
    constexpr int N = 1024 * 1024;
    
    int* d_data = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_data), size) == hipSuccess);
    
    auto kernel_data = hip_cts::kernels::kernel_smoke_test_memset_kernel_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "memset_int") == hipSuccess);
    
    auto blockSize = GENERATE(32, 64, 128, 256, 512, 1024);
    
    DYNAMIC_SECTION("Block size: " << blockSize) {
        int n = N;
        int value = blockSize; // Use block size as the value for easy verification
        void* args[] = { &d_data, &value, &n };
        
        unsigned int gridSize = (N + blockSize - 1) / blockSize;
        
        REQUIRE(hip().hipModuleLaunchKernel(
            kernel,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, nullptr, args, nullptr
        ) == hipSuccess);
        
        REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
        
        // Verify first few elements
        std::vector<int> h_check(1024);
        REQUIRE(hip().hipMemcpy(h_check.data(), d_data, 1024 * sizeof(int), 
                                 hipMemcpyDeviceToHost) == hipSuccess);
        
        for (int i = 0; i < 1024; ++i) {
            REQUIRE(h_check[i] == blockSize);
        }
    }
    
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_data) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Kernel launch on stream", "[kernel][stream]") {
    constexpr int N = 1024;
    constexpr size_t size = N * sizeof(int);
    
    // Create stream
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    int* d_data = nullptr;
    REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&d_data), size) == hipSuccess);
    
    auto kernel_data = hip_cts::kernels::kernel_smoke_test_memset_kernel_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "memset_int") == hipSuccess);
    
    int n = N;
    int value = 99;
    void* args[] = { &d_data, &value, &n };
    
    // Launch on explicit stream
    REQUIRE(hip().hipModuleLaunchKernel(
        kernel,
        4, 1, 1,
        256, 1, 1,
        0, stream, args, nullptr
    ) == hipSuccess);
    
    // Synchronize stream
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify
    std::vector<int> h_data(N);
    REQUIRE(hip().hipMemcpy(h_data.data(), d_data, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (int i = 0; i < N; ++i) {
        REQUIRE(h_data[i] == 99);
    }
    
    // Cleanup
    REQUIRE(hip().hipModuleUnload(module) == hipSuccess);
    REQUIRE(hip().hipFree(d_data) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

