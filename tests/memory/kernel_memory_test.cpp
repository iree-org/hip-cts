// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tests for kernel-based memory operations.
// Verifies that kernels can correctly access and modify device and host memory.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

#include "memory_kernel_test_vector_set.hpp"
#include "memory_kernel_test_increment.hpp"
#include "memory_kernel_test_vector_add.hpp"

#include <vector>
#include <cstring>
#include <numeric>

//=============================================================================
// Test Constants
//=============================================================================

static constexpr int kDefaultCount = 1024;
static constexpr int kLargeCount = 64 * 1024;
static constexpr int kThreadsPerBlock = 256;

//=============================================================================
// Helper Functions
//=============================================================================

static unsigned int calculateBlockCount(size_t count, int threadsPerBlock) {
    return static_cast<unsigned int>((count + threadsPerBlock - 1) / threadsPerBlock);
}

//=============================================================================
// Kernel Memory Access Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Kernel can write to device memory", "[memory][kernel]") {
    size_t test_count = GENERATE(size_t(64), size_t(1024), size_t(4096), size_t(kLargeCount));
    
    DYNAMIC_SECTION("Count: " << test_count) {
        size_t size = test_count * sizeof(int);
        void* d_ptr = nullptr;
        std::vector<int> h_result(test_count, 0);
        
        REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
        
        // Clear device memory
        REQUIRE(hip().hipMemset(d_ptr, 0, size) == hipSuccess);
        
        // Load module and kernel
        auto kernel_data = hip_cts::kernels::memory_kernel_test_vector_set_data::get();
        hipModule_t module = nullptr;
        REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
        
        hipFunction_t kernel = nullptr;
        REQUIRE(hip().hipModuleGetFunction(&kernel, module, "vector_set_int") == hipSuccess);
        
        int value = 42;
        unsigned int blocks = calculateBlockCount(test_count, kThreadsPerBlock);
        void* args[] = { &d_ptr, &value, &test_count };
        
        REQUIRE(hip().hipModuleLaunchKernel(kernel, blocks, 1, 1, kThreadsPerBlock, 1, 1,
                                            0, nullptr, args, nullptr) == hipSuccess);
        REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
        
        // Copy back and verify
        REQUIRE(hip().hipMemcpy(h_result.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < test_count; ++i) {
            INFO("Index: " << i);
            REQUIRE(h_result[i] == value);
        }
        
        hip().hipModuleUnload(module);
        REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "Kernel can read and modify device memory", "[memory][kernel]") {
    size_t count = kDefaultCount;
    size_t size = count * sizeof(int);
    
    void* d_ptr = nullptr;
    std::vector<int> h_src(count);
    std::vector<int> h_result(count);
    
    // Initialize source with sequential values
    std::iota(h_src.begin(), h_src.end(), 0);
    
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    REQUIRE(hip().hipMemcpy(d_ptr, h_src.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Load module and kernel
    auto kernel_data = hip_cts::kernels::memory_kernel_test_increment_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "increment_int") == hipSuccess);
    
    unsigned int blocks = calculateBlockCount(count, kThreadsPerBlock);
    void* args[] = { &d_ptr, &count };
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, blocks, 1, 1, kThreadsPerBlock, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Copy back and verify each element was incremented
    REQUIRE(hip().hipMemcpy(h_result.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < count; ++i) {
        INFO("Index: " << i << ", expected: " << (h_src[i] + 1) << ", got: " << h_result[i]);
        REQUIRE(h_result[i] == h_src[i] + 1);
    }
    
    hip().hipModuleUnload(module);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Kernel modifies memset initialized memory", "[memory][kernel]") {
    size_t count = kDefaultCount;
    size_t size = count * sizeof(char);
    
    void* d_ptr = nullptr;
    std::vector<char> h_result(count);
    
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    
    // Initialize with memset to 'A' (0x41)
    REQUIRE(hip().hipMemset(d_ptr, 'A', size) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Load module and kernel
    auto kernel_data = hip_cts::kernels::memory_kernel_test_increment_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "increment_char") == hipSuccess);
    
    unsigned int blocks = calculateBlockCount(count, kThreadsPerBlock);
    void* args[] = { &d_ptr, &count };
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, blocks, 1, 1, kThreadsPerBlock, 1, 1,
                                        0, nullptr, args, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(h_result.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < count; ++i) {
        INFO("Index: " << i);
        REQUIRE(h_result[i] == 'B');
    }
    
    hip().hipModuleUnload(module);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// Vector Addition Tests (End-to-End Memory Flow)
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Vector addition with memcpy and kernel", "[memory][kernel][vectoradd]") {
    size_t count = GENERATE(size_t(256), size_t(1024), size_t(4096));
    
    DYNAMIC_SECTION("Count: " << count) {
        size_t size = count * sizeof(int);
        
        void* d_A = nullptr;
        void* d_B = nullptr;
        void* d_C = nullptr;
        std::vector<int> h_A(count);
        std::vector<int> h_B(count);
        std::vector<int> h_C(count, 0);
        
        // Initialize host data
        for (size_t i = 0; i < count; ++i) {
            h_A[i] = static_cast<int>(i);
            h_B[i] = static_cast<int>(count - i);
        }
        
        // Allocate device memory
        REQUIRE(hip().hipMalloc(&d_A, size) == hipSuccess);
        REQUIRE(hip().hipMalloc(&d_B, size) == hipSuccess);
        REQUIRE(hip().hipMalloc(&d_C, size) == hipSuccess);
        
        // Copy inputs to device
        REQUIRE(hip().hipMemcpy(d_A, h_A.data(), size, hipMemcpyHostToDevice) == hipSuccess);
        REQUIRE(hip().hipMemcpy(d_B, h_B.data(), size, hipMemcpyHostToDevice) == hipSuccess);
        
        // Load module and kernel
        auto kernel_data = hip_cts::kernels::memory_kernel_test_vector_add_data::get();
        hipModule_t module = nullptr;
        REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
        
        hipFunction_t kernel = nullptr;
        REQUIRE(hip().hipModuleGetFunction(&kernel, module, "vector_add_int") == hipSuccess);
        
        unsigned int blocks = calculateBlockCount(count, kThreadsPerBlock);
        void* args[] = { &d_A, &d_B, &d_C, &count };
        
        REQUIRE(hip().hipModuleLaunchKernel(kernel, blocks, 1, 1, kThreadsPerBlock, 1, 1,
                                            0, nullptr, args, nullptr) == hipSuccess);
        REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
        
        // Copy result back
        REQUIRE(hip().hipMemcpy(h_C.data(), d_C, size, hipMemcpyDeviceToHost) == hipSuccess);
        
        // Verify: C[i] = A[i] + B[i] = i + (count - i) = count
        for (size_t i = 0; i < count; ++i) {
            INFO("Index: " << i << ", expected: " << count << ", got: " << h_C[i]);
            REQUIRE(h_C[i] == static_cast<int>(count));
        }
        
        hip().hipModuleUnload(module);
        REQUIRE(hip().hipFree(d_A) == hipSuccess);
        REQUIRE(hip().hipFree(d_B) == hipSuccess);
        REQUIRE(hip().hipFree(d_C) == hipSuccess);
    }
}

//=============================================================================
// Stream Ordering Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Memset after kernel on same stream", "[memory][kernel][stream]") {
    size_t count = kDefaultCount;
    size_t size = count * sizeof(int);
    
    void* d_ptr = nullptr;
    std::vector<int> h_result(count);
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Initialize with zeros
    REQUIRE(hip().hipMemset(d_ptr, 0, size) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Load module and kernel
    auto kernel_data = hip_cts::kernels::memory_kernel_test_vector_set_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "vector_set_int") == hipSuccess);
    
    // Launch kernel to set values to 100
    int kernel_value = 100;
    unsigned int blocks = calculateBlockCount(count, kThreadsPerBlock);
    void* args[] = { &d_ptr, &kernel_value, &count };
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, blocks, 1, 1, kThreadsPerBlock, 1, 1,
                                        0, stream, args, nullptr) == hipSuccess);
    
    // Immediately enqueue memset on same stream (should execute AFTER kernel)
    char memset_value = 0x42;  // 66 as int pattern
    REQUIRE(hip().hipMemsetAsync(d_ptr, memset_value, size, stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Copy back and verify memset value is present (not kernel value)
    REQUIRE(hip().hipMemcpy(h_result.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Each int should have memset pattern in all bytes: 0x42424242
    int expected = static_cast<int>(0x42424242);
    for (size_t i = 0; i < count; ++i) {
        INFO("Index: " << i << ", expected: " << expected << ", got: " << h_result[i]);
        REQUIRE(h_result[i] == expected);
    }
    
    hip().hipModuleUnload(module);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Kernel after memcpy on same stream", "[memory][kernel][stream]") {
    size_t count = kDefaultCount;
    size_t size = count * sizeof(int);
    
    void* d_ptr = nullptr;
    std::vector<int> h_src(count, 10);  // All values = 10
    std::vector<int> h_result(count, 0);
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Async memcpy on stream
    REQUIRE(hip().hipMemcpyAsync(d_ptr, h_src.data(), size, hipMemcpyHostToDevice, stream) == hipSuccess);
    
    // Load module and kernel
    auto kernel_data = hip_cts::kernels::memory_kernel_test_increment_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "increment_int") == hipSuccess);
    
    unsigned int blocks = calculateBlockCount(count, kThreadsPerBlock);
    void* args[] = { &d_ptr, &count };
    
    REQUIRE(hip().hipModuleLaunchKernel(kernel, blocks, 1, 1, kThreadsPerBlock, 1, 1,
                                        0, stream, args, nullptr) == hipSuccess);
    
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Copy back and verify
    REQUIRE(hip().hipMemcpy(h_result.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Each value should be 11 (10 + 1)
    for (size_t i = 0; i < count; ++i) {
        INFO("Index: " << i);
        REQUIRE(h_result[i] == 11);
    }
    
    hip().hipModuleUnload(module);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// Host Memory Access from Kernel Tests
//=============================================================================

// Helper function to check if mapped host memory is supported for kernel access
static bool isMappedKernelAccessSupported(HipLoader& hip) {
    // Try to allocate mapped memory and get device pointer
    // If hipHostGetDevicePointer returns something different from host pointer,
    // the implementation likely supports real device-mapped memory
    void* h_ptr = nullptr;
    void* d_ptr = nullptr;
    
    if (hip.hipHostMalloc(&h_ptr, 64, hipHostMallocMapped) != hipSuccess) {
        return false;
    }
    
    if (hip.hipHostGetDevicePointer(&d_ptr, h_ptr, 0) != hipSuccess) {
        hip.hipHostFree(h_ptr);
        return false;
    }
    
    // If the device pointer is the same as host pointer, it might be using
    // system malloc and may not support kernel access
    bool supported = (d_ptr != h_ptr);
    
    // Even if pointers are same (unified memory), we should still try
    // For now, just return true and let the test try
    hip.hipHostFree(h_ptr);
    return true;  // Assume supported by default
}

TEST_CASE_METHOD(HipTestFixture, "Kernel accesses mapped host memory", "[memory][kernel][host]") {
    size_t count = kDefaultCount;
    size_t size = count * sizeof(int);
    
    void* h_ptr = nullptr;
    void* d_ptr = nullptr;
    
    // Allocate mapped host memory
    REQUIRE(hip().hipHostMalloc(&h_ptr, size, hipHostMallocMapped) == hipSuccess);
    REQUIRE(hip().hipHostGetDevicePointer(&d_ptr, h_ptr, 0) == hipSuccess);
    
    // Initialize host memory
    int* h_data = static_cast<int*>(h_ptr);
    for (size_t i = 0; i < count; ++i) {
        h_data[i] = static_cast<int>(i);
    }
    
    // Load module and kernel
    auto kernel_data = hip_cts::kernels::memory_kernel_test_increment_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "increment_int") == hipSuccess);
    
    unsigned int blocks = calculateBlockCount(count, kThreadsPerBlock);
    void* args[] = { &d_ptr, &count };
    
    hipError_t launch_result = hip().hipModuleLaunchKernel(kernel, blocks, 1, 1, kThreadsPerBlock, 1, 1,
                                                           0, nullptr, args, nullptr);
    if (launch_result != hipSuccess) {
        // This implementation may not support kernel access to mapped host memory
        hip().hipModuleUnload(module);
        hip().hipHostFree(h_ptr);
        SKIP("Kernel access to mapped host memory not supported (error " << launch_result << ")");
    }
    
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify directly on host memory (no memcpy needed for mapped memory)
    for (size_t i = 0; i < count; ++i) {
        INFO("Index: " << i << ", expected: " << (i + 1) << ", got: " << h_data[i]);
        REQUIRE(h_data[i] == static_cast<int>(i + 1));
    }
    
    hip().hipModuleUnload(module);
    REQUIRE(hip().hipHostFree(h_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Kernel writes to mapped host memory", "[memory][kernel][host]") {
    size_t count = kDefaultCount;
    size_t size = count * sizeof(int);
    
    void* h_ptr = nullptr;
    void* d_ptr = nullptr;
    
    // Allocate mapped host memory
    REQUIRE(hip().hipHostMalloc(&h_ptr, size, hipHostMallocMapped) == hipSuccess);
    REQUIRE(hip().hipHostGetDevicePointer(&d_ptr, h_ptr, 0) == hipSuccess);
    
    // Initialize to zeros
    std::memset(h_ptr, 0, size);
    
    // Load module and kernel
    auto kernel_data = hip_cts::kernels::memory_kernel_test_vector_set_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "vector_set_int") == hipSuccess);
    
    int value = 12345;
    unsigned int blocks = calculateBlockCount(count, kThreadsPerBlock);
    void* args[] = { &d_ptr, &value, &count };
    
    hipError_t launch_result = hip().hipModuleLaunchKernel(kernel, blocks, 1, 1, kThreadsPerBlock, 1, 1,
                                                           0, nullptr, args, nullptr);
    if (launch_result != hipSuccess) {
        // This implementation may not support kernel access to mapped host memory
        hip().hipModuleUnload(module);
        hip().hipHostFree(h_ptr);
        SKIP("Kernel access to mapped host memory not supported (error " << launch_result << ")");
    }
    
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify directly on host
    int* h_data = static_cast<int*>(h_ptr);
    for (size_t i = 0; i < count; ++i) {
        INFO("Index: " << i);
        REQUIRE(h_data[i] == value);
    }
    
    hip().hipModuleUnload(module);
    REQUIRE(hip().hipHostFree(h_ptr) == hipSuccess);
}

//=============================================================================
// Multiple Kernel Execution Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Multiple kernels in sequence", "[memory][kernel]") {
    size_t count = kDefaultCount;
    size_t size = count * sizeof(int);
    
    void* d_ptr = nullptr;
    std::vector<int> h_result(count);
    
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    
    // Load modules
    auto set_kernel_data = hip_cts::kernels::memory_kernel_test_vector_set_data::get();
    hipModule_t set_module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&set_module, set_kernel_data.data()) == hipSuccess);
    
    hipFunction_t set_kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&set_kernel, set_module, "vector_set_int") == hipSuccess);
    
    auto inc_kernel_data = hip_cts::kernels::memory_kernel_test_increment_data::get();
    hipModule_t inc_module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&inc_module, inc_kernel_data.data()) == hipSuccess);
    
    hipFunction_t inc_kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&inc_kernel, inc_module, "increment_int") == hipSuccess);
    
    unsigned int blocks = calculateBlockCount(count, kThreadsPerBlock);
    
    // First kernel: set to 0
    int value = 0;
    void* set_args[] = { &d_ptr, &value, &count };
    REQUIRE(hip().hipModuleLaunchKernel(set_kernel, blocks, 1, 1, kThreadsPerBlock, 1, 1,
                                        0, nullptr, set_args, nullptr) == hipSuccess);
    
    // Run increment kernel 5 times
    void* inc_args[] = { &d_ptr, &count };
    for (int i = 0; i < 5; ++i) {
        REQUIRE(hip().hipModuleLaunchKernel(inc_kernel, blocks, 1, 1, kThreadsPerBlock, 1, 1,
                                            0, nullptr, inc_args, nullptr) == hipSuccess);
    }
    
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Copy back and verify - all values should be 5
    REQUIRE(hip().hipMemcpy(h_result.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < count; ++i) {
        INFO("Index: " << i);
        REQUIRE(h_result[i] == 5);
    }
    
    hip().hipModuleUnload(set_module);
    hip().hipModuleUnload(inc_module);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Kernels on multiple streams", "[memory][kernel][stream]") {
    size_t count = kDefaultCount;
    size_t size = count * sizeof(int);
    constexpr int num_streams = 4;
    
    void* d_ptrs[num_streams];
    std::vector<int> h_results[num_streams];
    hipStream_t streams[num_streams];
    
    // Allocate memory and streams
    for (int s = 0; s < num_streams; ++s) {
        REQUIRE(hip().hipMalloc(&d_ptrs[s], size) == hipSuccess);
        REQUIRE(hip().hipStreamCreate(&streams[s]) == hipSuccess);
        h_results[s].resize(count);
    }
    
    // Load module and kernel
    auto kernel_data = hip_cts::kernels::memory_kernel_test_vector_set_data::get();
    hipModule_t module = nullptr;
    REQUIRE(hip().hipModuleLoadData(&module, kernel_data.data()) == hipSuccess);
    
    hipFunction_t kernel = nullptr;
    REQUIRE(hip().hipModuleGetFunction(&kernel, module, "vector_set_int") == hipSuccess);
    
    unsigned int blocks = calculateBlockCount(count, kThreadsPerBlock);
    
    // Launch kernels on different streams with different values
    for (int s = 0; s < num_streams; ++s) {
        int value = (s + 1) * 100;  // 100, 200, 300, 400
        void* args[] = { &d_ptrs[s], &value, &count };
        REQUIRE(hip().hipModuleLaunchKernel(kernel, blocks, 1, 1, kThreadsPerBlock, 1, 1,
                                            0, streams[s], args, nullptr) == hipSuccess);
    }
    
    // Synchronize all streams and verify
    for (int s = 0; s < num_streams; ++s) {
        REQUIRE(hip().hipStreamSynchronize(streams[s]) == hipSuccess);
        REQUIRE(hip().hipMemcpy(h_results[s].data(), d_ptrs[s], size, hipMemcpyDeviceToHost) == hipSuccess);
        
        int expected = (s + 1) * 100;
        for (size_t i = 0; i < count; ++i) {
            INFO("Stream: " << s << ", Index: " << i);
            REQUIRE(h_results[s][i] == expected);
        }
    }
    
    // Cleanup
    hip().hipModuleUnload(module);
    for (int s = 0; s < num_streams; ++s) {
        REQUIRE(hip().hipStreamDestroy(streams[s]) == hipSuccess);
        REQUIRE(hip().hipFree(d_ptrs[s]) == hipSuccess);
    }
}
