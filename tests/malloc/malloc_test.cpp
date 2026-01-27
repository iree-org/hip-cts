// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <cstring>
#include <set>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// hipMalloc Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMalloc basic allocation", "[malloc]") {
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipMalloc(&ptr, size) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc null pointer fails", "[malloc][negative]") {
    REQUIRE(hip().hipMalloc(nullptr, 4096) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc zero size", "[malloc]") {
    void* ptr = nullptr;
    // Zero size allocation behavior may vary
    hipError_t result = hip().hipMalloc(&ptr, 0);
    if (result == hipSuccess && ptr != nullptr) {
        REQUIRE(hip().hipFree(ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc different sizes", "[malloc]") {
    std::vector<size_t> sizes = {1, 64, 256, 1024, 4096, 65536, 1024*1024};
    
    for (size_t size : sizes) {
        void* ptr = nullptr;
        REQUIRE(hip().hipMalloc(&ptr, size) == hipSuccess);
        REQUIRE(ptr != nullptr);
        REQUIRE(hip().hipFree(ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc multiple allocations", "[malloc]") {
    constexpr int numAllocations = 10;
    constexpr size_t size = 4096;
    std::vector<void*> ptrs(numAllocations, nullptr);
    
    // Allocate multiple buffers
    for (int i = 0; i < numAllocations; ++i) {
        REQUIRE(hip().hipMalloc(&ptrs[i], size) == hipSuccess);
        REQUIRE(ptrs[i] != nullptr);
    }
    
    // Verify all are unique
    std::set<void*> uniquePtrs(ptrs.begin(), ptrs.end());
    REQUIRE(uniquePtrs.size() == static_cast<size_t>(numAllocations));
    
    // Free all
    for (int i = 0; i < numAllocations; ++i) {
        REQUIRE(hip().hipFree(ptrs[i]) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc and use with memset", "[malloc]") {
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    REQUIRE(hip().hipMalloc(&ptr, size) == hipSuccess);
    
    // Use the allocation
    REQUIRE(hip().hipMemset(ptr, 0xAB, size) == hipSuccess);
    REQUIRE(hip().hipMemcpy(hostData.data(), ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xAB);
    }
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc aligned allocation", "[malloc]") {
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipMalloc(&ptr, size) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Device pointers should be aligned to at least 256 bytes
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    REQUIRE((addr % 256) == 0);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

//=============================================================================
// hipFree Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipFree valid pointer", "[free]") {
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, 4096) == hipSuccess);
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipFree null pointer", "[free]") {
    // Freeing null pointer should succeed per HIP spec
    REQUIRE(hip().hipFree(nullptr) == hipSuccess);
}

//=============================================================================
// hipMallocPitch Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMallocPitch basic", "[malloc][pitch]") {
    void* ptr = nullptr;
    size_t pitch = 0;
    constexpr size_t width = 256;
    constexpr size_t height = 128;
    
    REQUIRE(hip().hipMallocPitch(&ptr, &pitch, width, height) == hipSuccess);
    REQUIRE(ptr != nullptr);
    REQUIRE(pitch >= width);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocPitch null ptr fails", "[malloc][pitch][negative]") {
    size_t pitch = 0;
    REQUIRE(hip().hipMallocPitch(nullptr, &pitch, 256, 128) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocPitch null pitch fails", "[malloc][pitch][negative]") {
    void* ptr = nullptr;
    REQUIRE(hip().hipMallocPitch(&ptr, nullptr, 256, 128) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocPitch various sizes", "[malloc][pitch]") {
    struct TestCase {
        size_t width;
        size_t height;
    };
    
    std::vector<TestCase> cases = {
        {64, 64},
        {128, 256},
        {512, 512},
        {1024, 1024},
        {100, 200},  // Non-power-of-2
    };
    
    for (const auto& tc : cases) {
        void* ptr = nullptr;
        size_t pitch = 0;
        
        REQUIRE(hip().hipMallocPitch(&ptr, &pitch, tc.width, tc.height) == hipSuccess);
        REQUIRE(ptr != nullptr);
        REQUIRE(pitch >= tc.width);
        
        REQUIRE(hip().hipFree(ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocPitch and use with memcpy2d", "[malloc][pitch]") {
    void* devicePtr = nullptr;
    size_t pitch = 0;
    constexpr size_t width = 256;
    constexpr size_t height = 128;
    
    REQUIRE(hip().hipMallocPitch(&devicePtr, &pitch, width, height) == hipSuccess);
    
    // Create host data
    std::vector<uint8_t> hostSrc(width * height);
    std::vector<uint8_t> hostDst(width * height, 0);
    
    for (size_t i = 0; i < hostSrc.size(); ++i) {
        hostSrc[i] = static_cast<uint8_t>(i & 0xFF);
    }
    
    // Copy to device using 2D memcpy
    REQUIRE(hip().hipMemcpy2D(devicePtr, pitch, hostSrc.data(), width,
                               width, height, hipMemcpyHostToDevice) == hipSuccess);
    
    // Copy back
    REQUIRE(hip().hipMemcpy2D(hostDst.data(), width, devicePtr, pitch,
                               width, height, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify
    for (size_t i = 0; i < hostSrc.size(); ++i) {
        REQUIRE(hostDst[i] == hostSrc[i]);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

//=============================================================================
// hipMallocManaged Tests
// Note: Managed memory is not supported in the streaming binding
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMallocManaged basic", "[malloc][managed]") {
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    hipError_t result = hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal);
    
    // Streaming binding doesn't support managed memory
    if (result == hipErrorNotSupported) {
        SKIP("Managed memory not supported");
    }
    
    REQUIRE(result == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Managed memory should be accessible from both host and device
    // Write from host
    memset(ptr, 0xCD, size);
    
    // Synchronize to ensure visibility
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify from host
    uint8_t* bytePtr = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(bytePtr[i] == 0xCD);
    }
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocManaged null pointer fails", "[malloc][managed][negative]") {
    REQUIRE(hip().hipMallocManaged(nullptr, 4096, hipMemAttachGlobal) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocManaged with device memset", "[malloc][managed]") {
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    hipError_t result = hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal);
    
    // Streaming binding doesn't support managed memory
    if (result == hipErrorNotSupported) {
        SKIP("Managed memory not supported");
    }
    
    REQUIRE(result == hipSuccess);
    
    // Use device memset
    REQUIRE(hip().hipMemset(ptr, 0xEF, size) == hipSuccess);
    
    // Synchronize
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify from host
    uint8_t* bytePtr = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(bytePtr[i] == 0xEF);
    }
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

//=============================================================================
// Allocation Stress Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMalloc many small allocations", "[malloc][stress]") {
    constexpr int numAllocations = 100;
    constexpr size_t size = 256;
    std::vector<void*> ptrs(numAllocations, nullptr);
    
    // Allocate many small buffers
    for (int i = 0; i < numAllocations; ++i) {
        REQUIRE(hip().hipMalloc(&ptrs[i], size) == hipSuccess);
        REQUIRE(ptrs[i] != nullptr);
    }
    
    // Use each allocation
    for (int i = 0; i < numAllocations; ++i) {
        REQUIRE(hip().hipMemset(ptrs[i], static_cast<int>(i), size) == hipSuccess);
    }
    
    // Verify each
    std::vector<uint8_t> hostData(size);
    for (int i = 0; i < numAllocations; ++i) {
        REQUIRE(hip().hipMemcpy(hostData.data(), ptrs[i], size, hipMemcpyDeviceToHost) == hipSuccess);
        for (size_t j = 0; j < size; ++j) {
            REQUIRE(hostData[j] == static_cast<uint8_t>(i));
        }
    }
    
    // Free all
    for (int i = 0; i < numAllocations; ++i) {
        REQUIRE(hip().hipFree(ptrs[i]) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc allocate free cycle", "[malloc][stress]") {
    constexpr int numCycles = 50;
    constexpr size_t size = 4096;
    
    for (int i = 0; i < numCycles; ++i) {
        void* ptr = nullptr;
        REQUIRE(hip().hipMalloc(&ptr, size) == hipSuccess);
        REQUIRE(ptr != nullptr);
        
        // Use it
        REQUIRE(hip().hipMemset(ptr, i, size) == hipSuccess);
        
        REQUIRE(hip().hipFree(ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc interleaved alloc free", "[malloc][stress]") {
    constexpr size_t size = 4096;
    void* ptr1 = nullptr;
    void* ptr2 = nullptr;
    void* ptr3 = nullptr;
    
    // Allocate and free in interleaved pattern
    REQUIRE(hip().hipMalloc(&ptr1, size) == hipSuccess);
    REQUIRE(hip().hipMalloc(&ptr2, size) == hipSuccess);
    REQUIRE(hip().hipFree(ptr1) == hipSuccess);
    REQUIRE(hip().hipMalloc(&ptr3, size) == hipSuccess);
    REQUIRE(hip().hipFree(ptr2) == hipSuccess);
    REQUIRE(hip().hipFree(ptr3) == hipSuccess);
}
