// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tests for hipMalloc and hipFree APIs
// Migrated from TheRock: rocm-systems/projects/hip-tests/catch/unit/memory/hipMalloc.cc

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

#include <vector>
#include <limits>
#include <cstdint>

//=============================================================================
// hipMalloc Positive Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMalloc basic allocation", "[memory][malloc]") {
    void* ptr = nullptr;
    
    // page_size = 4096
    auto alloc_size = GENERATE(
        size_t(10), 
        size_t(2048),     // page_size / 2
        size_t(4096),     // page_size
        size_t(6144),     // page_size * 3 / 2
        size_t(8192)      // page_size * 2
    );
    
    DYNAMIC_SECTION("Allocation size: " << alloc_size) {
        REQUIRE(hip().hipMalloc(&ptr, alloc_size) == hipSuccess);
        REQUIRE(ptr != nullptr);
        // Check 256-byte alignment
        REQUIRE(reinterpret_cast<intptr_t>(ptr) % 256 == 0);
        REQUIRE(hip().hipFree(ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc zero size returns nullptr", "[memory][malloc]") {
    void* ptr = reinterpret_cast<void*>(0x1);
    REQUIRE(hip().hipMalloc(&ptr, 0) == hipSuccess);
    REQUIRE(ptr == nullptr);
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc alignment check", "[memory][malloc]") {
    void* ptr1 = nullptr;
    void* ptr2 = nullptr;
    
    REQUIRE(hip().hipMalloc(&ptr1, 1) == hipSuccess);
    REQUIRE(hip().hipMalloc(&ptr2, 10) == hipSuccess);
    
    // Check 256-byte alignment for both allocations
    REQUIRE(reinterpret_cast<intptr_t>(ptr1) % 256 == 0);
    REQUIRE(reinterpret_cast<intptr_t>(ptr2) % 256 == 0);
    
    REQUIRE(hip().hipFree(ptr1) == hipSuccess);
    REQUIRE(hip().hipFree(ptr2) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc various sizes", "[memory][malloc]") {
    auto size = GENERATE(
        size_t(1),           // 1 byte
        size_t(256),         // 256 bytes  
        size_t(1024),        // 1 KB
        size_t(65536),       // 64 KB
        size_t(1048576),     // 1 MB
        size_t(16777216)     // 16 MB
    );

    void* ptr = nullptr;
    
    DYNAMIC_SECTION("Size: " << size << " bytes") {
        REQUIRE(hip().hipMalloc(&ptr, size) == hipSuccess);
        REQUIRE(ptr != nullptr);
        REQUIRE(hip().hipFree(ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc multiple allocations", "[memory][malloc]") {
    constexpr int numAllocs = 10;
    std::vector<void*> ptrs(numAllocs, nullptr);
    
    // Allocate
    for (int i = 0; i < numAllocs; ++i) {
        REQUIRE(hip().hipMalloc(&ptrs[i], 1024) == hipSuccess);
        REQUIRE(ptrs[i] != nullptr);
    }
    
    // Verify all pointers are unique
    for (int i = 0; i < numAllocs; ++i) {
        for (int j = i + 1; j < numAllocs; ++j) {
            REQUIRE(ptrs[i] != ptrs[j]);
        }
    }
    
    // Free all
    for (int i = 0; i < numAllocs; ++i) {
        REQUIRE(hip().hipFree(ptrs[i]) == hipSuccess);
    }
}

//=============================================================================
// hipMalloc Negative Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMalloc null pointer returns error", "[memory][malloc][negative]") {
    REQUIRE(hip().hipMalloc(nullptr, 4096) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc max size_t returns out of memory", "[memory][malloc][negative]") {
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, std::numeric_limits<size_t>::max()) == hipErrorOutOfMemory);
}

//=============================================================================
// hipFree Positive Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipFree nullptr is valid", "[memory][free]") {
    // Freeing nullptr should be a no-op and succeed
    REQUIRE(hip().hipFree(nullptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipFree after malloc", "[memory][free]") {
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, 1024) == hipSuccess);
    REQUIRE(ptr != nullptr);
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

//=============================================================================
// hipFree Negative Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipFree invalid pointer returns error", "[memory][free][negative]") {
    char value;
    REQUIRE(hip().hipFree(&value) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipFree double free returns error", "[memory][free][negative]") {
    auto size = GENERATE(size_t(32), size_t(512), size_t(1024));
    
    DYNAMIC_SECTION("Size: " << size) {
        char* ptr = nullptr;
        REQUIRE(hip().hipMalloc(reinterpret_cast<void**>(&ptr), size) == hipSuccess);
        
        REQUIRE(hip().hipFree(ptr) == hipSuccess);
        REQUIRE(hip().hipFree(ptr) == hipErrorInvalidValue);
    }
}

//=============================================================================
// Memory Info Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemGetInfo returns valid values", "[memory][info]") {
    size_t freeMemory = 0;
    size_t totalMemory = 0;
    
    REQUIRE(hip().hipMemGetInfo(&freeMemory, &totalMemory) == hipSuccess);
    
    // Total memory should be non-zero
    REQUIRE(totalMemory > 0);
    
    // Free memory should not exceed total
    REQUIRE(freeMemory <= totalMemory);
    
    INFO("Free memory: " << freeMemory << " bytes");
    INFO("Total memory: " << totalMemory << " bytes");
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc reduces free memory", "[memory][malloc][info]") {
    size_t freeBefore = 0, freeAfter = 0, total = 0;
    
    REQUIRE(hip().hipMemGetInfo(&freeBefore, &total) == hipSuccess);
    
    // Allocate 16 MB
    constexpr size_t allocSize = 16 * 1024 * 1024;
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, allocSize) == hipSuccess);
    
    REQUIRE(hip().hipMemGetInfo(&freeAfter, &total) == hipSuccess);
    
    // Free memory should have decreased (allow some slack for alignment/overhead)
    // Note: The exact decrease depends on implementation, so we just check it decreased
    INFO("Free before: " << freeBefore << ", after: " << freeAfter);
    REQUIRE(freeAfter < freeBefore);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

//=============================================================================
// Allocation with memcpy/memset verification
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMalloc memory can be written and read", "[memory][malloc][memcpy]") {
    constexpr size_t size = 1024;
    std::vector<int> h_src(size / sizeof(int), 42);
    std::vector<int> h_dst(size / sizeof(int), 0);
    
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    REQUIRE(d_ptr != nullptr);
    
    // Copy data to device
    REQUIRE(hip().hipMemcpy(d_ptr, h_src.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Copy data back
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify
    for (size_t i = 0; i < h_dst.size(); ++i) {
        REQUIRE(h_dst[i] == 42);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc memory can be memset", "[memory][malloc][memset]") {
    constexpr size_t size = 1024;
    
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    REQUIRE(d_ptr != nullptr);
    
    // Memset to a pattern
    REQUIRE(hip().hipMemset(d_ptr, 0xAB, size) == hipSuccess);
    
    // Copy back and verify
    std::vector<unsigned char> h_data(size);
    REQUIRE(hip().hipMemcpy(h_data.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_data[i] == 0xAB);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// Async Memory Allocation Tests (Memory Pools)
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMallocAsync basic allocation", "[memory][malloc][async]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    void* ptr = nullptr;
    constexpr size_t size = 1024;
    
    REQUIRE(hip().hipMallocAsync(&ptr, size, stream) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipFreeAsync(ptr, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocAsync with null pointer fails", "[memory][malloc][async][negative]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    REQUIRE(hip().hipMallocAsync(nullptr, 1024, stream) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocAsync zero size", "[memory][malloc][async]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    void* ptr = nullptr;
    // Zero size allocation should succeed (returns nullptr or valid ptr depending on impl)
    hipError_t err = hip().hipMallocAsync(&ptr, 0, stream);
    REQUIRE(err == hipSuccess);
    
    // Free should also succeed (even if ptr is null)
    if (ptr != nullptr) {
        REQUIRE(hip().hipFreeAsync(ptr, stream) == hipSuccess);
    }
    
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocAsync multiple allocations", "[memory][malloc][async]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    constexpr int numAllocs = 10;
    constexpr size_t size = 4096;
    std::vector<void*> ptrs(numAllocs, nullptr);
    
    // Allocate all
    for (int i = 0; i < numAllocs; ++i) {
        REQUIRE(hip().hipMallocAsync(&ptrs[i], size, stream) == hipSuccess);
        REQUIRE(ptrs[i] != nullptr);
    }
    
    // Free all
    for (int i = 0; i < numAllocs; ++i) {
        REQUIRE(hip().hipFreeAsync(ptrs[i], stream) == hipSuccess);
    }
    
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocAsync with memset and memcpy", "[memory][malloc][async]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    constexpr size_t size = 1024;
    void* d_ptr = nullptr;
    std::vector<unsigned char> h_data(size, 0);
    
    REQUIRE(hip().hipMallocAsync(&d_ptr, size, stream) == hipSuccess);
    REQUIRE(d_ptr != nullptr);
    
    // Memset and verify
    REQUIRE(hip().hipMemsetAsync(d_ptr, 0xCD, size, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    REQUIRE(hip().hipMemcpy(h_data.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_data[i] == 0xCD);
    }
    
    REQUIRE(hip().hipFreeAsync(d_ptr, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocAsync on null stream", "[memory][malloc][async]") {
    void* ptr = nullptr;
    constexpr size_t size = 1024;
    
    // Null stream should work (uses default stream)
    REQUIRE(hip().hipMallocAsync(&ptr, size, nullptr) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipFreeAsync(ptr, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipFreeAsync with null pointer", "[memory][malloc][async]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Freeing null should be a no-op (success)
    REQUIRE(hip().hipFreeAsync(nullptr, stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}
