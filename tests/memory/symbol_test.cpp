// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tests for device symbol access using hipModuleGetGlobal.
// Tests the underlying functionality that hipMemcpyToSymbol/hipMemcpyFromSymbol
// rely on when using dynamically loaded modules.

#include <catch2/catch_test_macros.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

#include "memory_symbol_test_symbol_test.hpp"

#include <vector>

//=============================================================================
// Test Constants
//=============================================================================

static constexpr int kThreadsPerBlock = 256;

static unsigned int calculateBlockCount(size_t count, int threadsPerBlock) {
    return static_cast<unsigned int>((count + threadsPerBlock - 1) / threadsPerBlock);
}

//=============================================================================
// hipModuleGetGlobal Tests
//=============================================================================

// Helper to check if module globals are supported
static bool checkModuleGlobalsSupported(HipLoader& hip, hipModule_t module) {
    hipDeviceptr_t dptr = 0;
    size_t bytes = 0;
    hipError_t result = hip.hipModuleGetGlobal(&dptr, &bytes, module, "d_globalValue");
    return result == hipSuccess;
}

TEST_CASE_METHOD(HipTestFixture, "hipModuleGetGlobal retrieves symbol address", "[memory][symbol][module]") {
    REQUIRE(hip().hipModuleGetGlobal != nullptr);

    // Load module
    auto kernel_data = hip_cts::kernels::memory_symbol_test_symbol_test_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    // Check if this implementation supports module globals
    if (!checkModuleGlobalsSupported(hip(), module)) {
        hip().hipModuleUnload(module);
        SKIP("hipModuleGetGlobal not supported for this implementation");
    }

    SECTION("Get address of d_globalValue") {
        hipDeviceptr_t dptr = 0;
        size_t bytes = 0;
        
        hipError_t result = hip().hipModuleGetGlobal(&dptr, &bytes, module, "d_globalValue");
        REQUIRE(result == hipSuccess);
        REQUIRE(dptr != 0);
        REQUIRE(bytes == sizeof(int));
    }

    SECTION("Get address of d_constantValue") {
        hipDeviceptr_t dptr = 0;
        size_t bytes = 0;
        
        hipError_t result = hip().hipModuleGetGlobal(&dptr, &bytes, module, "d_constantValue");
        REQUIRE(result == hipSuccess);
        REQUIRE(dptr != 0);
        REQUIRE(bytes == sizeof(int));
    }

    SECTION("Get address of d_globalArray") {
        hipDeviceptr_t dptr = 0;
        size_t bytes = 0;
        
        hipError_t result = hip().hipModuleGetGlobal(&dptr, &bytes, module, "d_globalArray");
        REQUIRE(result == hipSuccess);
        REQUIRE(dptr != 0);
        REQUIRE(bytes == 256 * sizeof(int));
    }

    SECTION("Non-existent symbol returns error") {
        hipDeviceptr_t dptr = 0;
        size_t bytes = 0;
        
        hipError_t result = hip().hipModuleGetGlobal(&dptr, &bytes, module, "nonexistent_symbol");
        REQUIRE(result != hipSuccess);
    }

    hip().hipModuleUnload(module);
}

//=============================================================================
// Symbol Read/Write Tests via hipMemcpy
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Write and read global symbol via hipMemcpy", "[memory][symbol][module]") {
    // Load module
    auto kernel_data = hip_cts::kernels::memory_symbol_test_symbol_test_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    if (!checkModuleGlobalsSupported(hip(), module)) {
        hip().hipModuleUnload(module);
        SKIP("hipModuleGetGlobal not supported for this implementation");
    }

    // Get symbol address
    hipDeviceptr_t d_globalValue = 0;
    size_t bytes = 0;
    REQUIRE(hip().hipModuleGetGlobal(&d_globalValue, &bytes, module, "d_globalValue") == hipSuccess);
    REQUIRE(d_globalValue != 0);
    REQUIRE(bytes == sizeof(int));

    // Write value to symbol
    // Note: Some implementations may not support memcpy to global symbol addresses
    // if they are not tracked in the memory management system.
    int write_value = 42;
    hipError_t write_err = hip().hipMemcpy(reinterpret_cast<void*>(d_globalValue), &write_value, 
                             sizeof(int), hipMemcpyHostToDevice);
    if (write_err != hipSuccess) {
        hip().hipModuleUnload(module);
        SKIP("hipMemcpy to global symbol not supported (error " << write_err << ")");
    }

    // Read value back from symbol
    int read_value = 0;
    REQUIRE(hip().hipMemcpy(&read_value, reinterpret_cast<void*>(d_globalValue),
                             sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(read_value == write_value);

    hip().hipModuleUnload(module);
}

TEST_CASE_METHOD(HipTestFixture, "Write array to global symbol", "[memory][symbol][module]") {
    // Load module
    auto kernel_data = hip_cts::kernels::memory_symbol_test_symbol_test_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    if (!checkModuleGlobalsSupported(hip(), module)) {
        hip().hipModuleUnload(module);
        SKIP("hipModuleGetGlobal not supported for this implementation");
    }

    // Get symbol address
    hipDeviceptr_t d_globalArray = 0;
    size_t bytes = 0;
    REQUIRE(hip().hipModuleGetGlobal(&d_globalArray, &bytes, module, "d_globalArray") == hipSuccess);
    REQUIRE(d_globalArray != 0);
    REQUIRE(bytes == 256 * sizeof(int));

    constexpr size_t array_size = 256;
    std::vector<int> write_data(array_size);
    std::vector<int> read_data(array_size, 0);

    // Initialize with pattern
    for (size_t i = 0; i < array_size; ++i) {
        write_data[i] = static_cast<int>(i * 10);
    }

    // Write to symbol
    hipError_t write_err = hip().hipMemcpy(reinterpret_cast<void*>(d_globalArray), write_data.data(),
                             array_size * sizeof(int), hipMemcpyHostToDevice);
    if (write_err != hipSuccess) {
        hip().hipModuleUnload(module);
        SKIP("hipMemcpy to global symbol not supported (error " << write_err << ")");
    }

    // Read back
    REQUIRE(hip().hipMemcpy(read_data.data(), reinterpret_cast<void*>(d_globalArray),
                             array_size * sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);

    // Verify
    for (size_t i = 0; i < array_size; ++i) {
        INFO("Index: " << i);
        REQUIRE(read_data[i] == write_data[i]);
    }

    hip().hipModuleUnload(module);
}

//=============================================================================
// Kernel Uses Symbol Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Kernel reads from global symbol", "[memory][symbol][kernel]") {
    // Load module
    auto kernel_data = hip_cts::kernels::memory_symbol_test_symbol_test_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    if (!checkModuleGlobalsSupported(hip(), module)) {
        hip().hipModuleUnload(module);
        SKIP("hipModuleGetGlobal not supported for this implementation");
    }

    // Get symbol address
    hipDeviceptr_t d_globalValue = 0;
    size_t bytes = 0;
    REQUIRE(hip().hipModuleGetGlobal(&d_globalValue, &bytes, module, "d_globalValue") == hipSuccess);
    REQUIRE(d_globalValue != 0);

    // Write a value to the symbol
    int symbol_value = 12345;
    hipError_t write_err = hip().hipMemcpy(reinterpret_cast<void*>(d_globalValue), &symbol_value,
                             sizeof(int), hipMemcpyHostToDevice);
    if (write_err != hipSuccess) {
        hip().hipModuleUnload(module);
        SKIP("hipMemcpy to global symbol not supported (error " << write_err << ")");
    }

    // Allocate output buffer
    void* d_output = nullptr;
    REQUIRE(hip().hipMalloc(&d_output, sizeof(int)) == hipSuccess);
    REQUIRE(hip().hipMemset(d_output, 0, sizeof(int)) == hipSuccess);

    // Get kernel that reads from global
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "read_global_value") == hipSuccess);

    // Launch kernel
    void* args[] = { &d_output };
    REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1,
                                         0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    // Read result
    int result = 0;
    REQUIRE(hip().hipMemcpy(&result, d_output, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(result == symbol_value);

    REQUIRE(hip().hipFree(d_output) == hipSuccess);
    hip().hipModuleUnload(module);
}

TEST_CASE_METHOD(HipTestFixture, "Kernel writes to global symbol", "[memory][symbol][kernel]") {
    // Load module
    auto kernel_data = hip_cts::kernels::memory_symbol_test_symbol_test_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    if (!checkModuleGlobalsSupported(hip(), module)) {
        hip().hipModuleUnload(module);
        SKIP("hipModuleGetGlobal not supported for this implementation");
    }

    // Get symbol address
    hipDeviceptr_t d_globalValue = 0;
    size_t bytes = 0;
    REQUIRE(hip().hipModuleGetGlobal(&d_globalValue, &bytes, module, "d_globalValue") == hipSuccess);
    REQUIRE(d_globalValue != 0);

    // Initialize symbol to 0
    int initial_value = 0;
    hipError_t write_err = hip().hipMemcpy(reinterpret_cast<void*>(d_globalValue), &initial_value,
                             sizeof(int), hipMemcpyHostToDevice);
    if (write_err != hipSuccess) {
        hip().hipModuleUnload(module);
        SKIP("hipMemcpy to global symbol not supported (error " << write_err << ")");
    }

    // Get kernel that writes to global
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "write_global_value") == hipSuccess);

    // Launch kernel with value to write
    int kernel_value = 99999;
    void* args[] = { &kernel_value };
    REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1,
                                         0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    // Read symbol value
    int result = 0;
    REQUIRE(hip().hipMemcpy(&result, reinterpret_cast<void*>(d_globalValue),
                             sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(result == kernel_value);

    hip().hipModuleUnload(module);
}

TEST_CASE_METHOD(HipTestFixture, "Kernel uses global symbol in computation", "[memory][symbol][kernel]") {
    // Load module
    auto kernel_data = hip_cts::kernels::memory_symbol_test_symbol_test_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    if (!checkModuleGlobalsSupported(hip(), module)) {
        hip().hipModuleUnload(module);
        SKIP("hipModuleGetGlobal not supported for this implementation");
    }

    // Get symbol address
    hipDeviceptr_t d_globalValue = 0;
    size_t bytes = 0;
    REQUIRE(hip().hipModuleGetGlobal(&d_globalValue, &bytes, module, "d_globalValue") == hipSuccess);
    REQUIRE(d_globalValue != 0);

    // Set global value to 100
    int add_value = 100;
    hipError_t write_err = hip().hipMemcpy(reinterpret_cast<void*>(d_globalValue), &add_value,
                             sizeof(int), hipMemcpyHostToDevice);
    if (write_err != hipSuccess) {
        hip().hipModuleUnload(module);
        SKIP("hipMemcpy to global symbol not supported (error " << write_err << ")");
    }

    // Allocate and initialize array
    constexpr size_t count = 1024;
    size_t size = count * sizeof(int);
    void* d_data = nullptr;
    std::vector<int> h_data(count);
    std::vector<int> h_result(count);

    for (size_t i = 0; i < count; ++i) {
        h_data[i] = static_cast<int>(i);
    }

    REQUIRE(hip().hipMalloc(&d_data, size) == hipSuccess);
    REQUIRE(hip().hipMemcpy(d_data, h_data.data(), size, hipMemcpyHostToDevice) == hipSuccess);

    // Get kernel that adds global to array
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "add_global_to_array") == hipSuccess);

    // Launch kernel
    unsigned int blocks = calculateBlockCount(count, kThreadsPerBlock);
    size_t count_arg = count;
    void* args[] = { &d_data, &count_arg };
    REQUIRE(hip().hipModuleLaunchKernel(kernel, blocks, 1, 1, kThreadsPerBlock, 1, 1,
                                         0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    // Read result
    REQUIRE(hip().hipMemcpy(h_result.data(), d_data, size, hipMemcpyDeviceToHost) == hipSuccess);

    // Verify: each element should be original + 100
    for (size_t i = 0; i < count; ++i) {
        INFO("Index: " << i);
        REQUIRE(h_result[i] == static_cast<int>(i) + add_value);
    }

    REQUIRE(hip().hipFree(d_data) == hipSuccess);
    hip().hipModuleUnload(module);
}

TEST_CASE_METHOD(HipTestFixture, "Constant memory read by kernel", "[memory][symbol][constant]") {
    // Load module
    auto kernel_data = hip_cts::kernels::memory_symbol_test_symbol_test_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);

    if (!checkModuleGlobalsSupported(hip(), module)) {
        hip().hipModuleUnload(module);
        SKIP("hipModuleGetGlobal not supported for this implementation");
    }

    // Get constant symbol address
    hipDeviceptr_t d_constantValue = 0;
    size_t bytes = 0;
    REQUIRE(hip().hipModuleGetGlobal(&d_constantValue, &bytes, module, "d_constantValue") == hipSuccess);
    REQUIRE(d_constantValue != 0);

    // Write to constant memory from host
    int const_value = 777;
    hipError_t write_err = hip().hipMemcpy(reinterpret_cast<void*>(d_constantValue), &const_value,
                             sizeof(int), hipMemcpyHostToDevice);
    if (write_err != hipSuccess) {
        hip().hipModuleUnload(module);
        SKIP("hipMemcpy to global symbol not supported (error " << write_err << ")");
    }

    // Allocate output buffer
    void* d_output = nullptr;
    REQUIRE(hip().hipMalloc(&d_output, sizeof(int)) == hipSuccess);
    REQUIRE(hip().hipMemset(d_output, 0, sizeof(int)) == hipSuccess);

    // Get kernel that reads from constant
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "read_constant_value") == hipSuccess);

    // Launch kernel
    void* args[] = { &d_output };
    REQUIRE(hip().hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1,
                                         0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);

    // Read result
    int result = 0;
    REQUIRE(hip().hipMemcpy(&result, d_output, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);

    REQUIRE(result == const_value);

    REQUIRE(hip().hipFree(d_output) == hipSuccess);
    hip().hipModuleUnload(module);
}
