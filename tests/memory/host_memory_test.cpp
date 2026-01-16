// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tests for HIP host (pinned) memory APIs
// Migrated from TheRock: rocm-systems/projects/hip-tests/catch/unit/memory/hipHostMalloc.cc,
//                        hipMallocHost.cc, hipHostRegister.cc, hipHostGetDevicePointer.cc

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

#include <vector>
#include <cstring>
#include <cstdint>

//=============================================================================
// Helper constants
//=============================================================================
static constexpr size_t kDefaultSize = 1024 * 1024;  // 1MB

//=============================================================================
// hipHostMalloc Basic Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc basic allocation", "[memory][host]") {
    void* ptr = nullptr;
    
    REQUIRE(hip().hipHostMalloc(&ptr, kDefaultSize, hipHostMallocDefault) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Verify we can write to the memory
    std::memset(ptr, 0x42, kDefaultSize);
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc with default flag", "[memory][host]") {
    void* ptr = nullptr;
    
    REQUIRE(hip().hipHostMalloc(&ptr, 1024, hipHostMallocDefault) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc with mapped flag", "[memory][host]") {
    void* ptr = nullptr;
    
    REQUIRE(hip().hipHostMalloc(&ptr, 1024, hipHostMallocMapped) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc with write-combined flag", "[memory][host]") {
    void* ptr = nullptr;
    
    REQUIRE(hip().hipHostMalloc(&ptr, 1024, hipHostMallocWriteCombined) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc with portable flag", "[memory][host]") {
    void* ptr = nullptr;
    
    REQUIRE(hip().hipHostMalloc(&ptr, 1024, hipHostMallocPortable) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc combined flags", "[memory][host]") {
    void* ptr = nullptr;
    
    unsigned int flags = hipHostMallocMapped | hipHostMallocPortable;
    REQUIRE(hip().hipHostMalloc(&ptr, 1024, flags) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc various sizes", "[memory][host]") {
    auto size = GENERATE(
        size_t(1),
        size_t(64),
        size_t(1024),
        size_t(4096),
        size_t(1024 * 1024)
    );
    
    DYNAMIC_SECTION("Size: " << size << " bytes") {
        void* ptr = nullptr;
        REQUIRE(hip().hipHostMalloc(&ptr, size, hipHostMallocDefault) == hipSuccess);
        REQUIRE(ptr != nullptr);
        
        // Verify we can access the entire allocation
        std::memset(ptr, 0xAB, size);
        
        REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc zero size", "[memory][host]") {
    void* ptr = nullptr;
    // Zero size allocation - behavior may vary by implementation
    // But it should not crash
    hipError_t result = hip().hipHostMalloc(&ptr, 0, hipHostMallocDefault);
    
    if (result == hipSuccess && ptr != nullptr) {
        REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
    }
    // Otherwise, it's acceptable for the allocation to fail or return nullptr
}

//=============================================================================
// hipHostAlloc Tests (legacy API, equivalent to hipHostMalloc)
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostAlloc basic allocation", "[memory][host]") {
    void* ptr = nullptr;
    
    REQUIRE(hip().hipHostAlloc(&ptr, 1024, hipHostMallocDefault) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

//=============================================================================
// hipHostFree Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostFree nullptr is valid", "[memory][host]") {
    // Freeing nullptr should succeed
    REQUIRE(hip().hipHostFree(nullptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostFree invalid pointer returns error", "[memory][host][negative]") {
    // Freeing an arbitrary pointer should fail
    void* invalid_ptr = reinterpret_cast<void*>(0xDEADBEEF);
    REQUIRE(hip().hipHostFree(invalid_ptr) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostFree double free returns error", "[memory][host][negative]") {
    void* ptr = nullptr;
    REQUIRE(hip().hipHostMalloc(&ptr, 1024, hipHostMallocDefault) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
    
    // Double free should fail
    REQUIRE(hip().hipHostFree(ptr) == hipErrorInvalidValue);
}

//=============================================================================
// hipHostGetDevicePointer Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostGetDevicePointer basic", "[memory][host]") {
    void* host_ptr = nullptr;
    void* device_ptr = nullptr;
    
    // Allocate with mapped flag to ensure we can get device pointer
    REQUIRE(hip().hipHostMalloc(&host_ptr, 1024, hipHostMallocMapped) == hipSuccess);
    REQUIRE(host_ptr != nullptr);
    
    REQUIRE(hip().hipHostGetDevicePointer(&device_ptr, host_ptr, 0) == hipSuccess);
    REQUIRE(device_ptr != nullptr);
    
    REQUIRE(hip().hipHostFree(host_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostGetDevicePointer null device pointer returns error", "[memory][host][negative]") {
    void* host_ptr = nullptr;
    REQUIRE(hip().hipHostMalloc(&host_ptr, 1024, hipHostMallocMapped) == hipSuccess);
    
    REQUIRE(hip().hipHostGetDevicePointer(nullptr, host_ptr, 0) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipHostFree(host_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostGetDevicePointer null host pointer returns error", "[memory][host][negative]") {
    void* device_ptr = nullptr;
    REQUIRE(hip().hipHostGetDevicePointer(&device_ptr, nullptr, 0) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostGetDevicePointer non-zero flags returns error", "[memory][host][negative]") {
    void* host_ptr = nullptr;
    void* device_ptr = nullptr;
    
    REQUIRE(hip().hipHostMalloc(&host_ptr, 1024, hipHostMallocMapped) == hipSuccess);
    
    // Non-zero flags should fail
    REQUIRE(hip().hipHostGetDevicePointer(&device_ptr, host_ptr, 1) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipHostFree(host_ptr) == hipSuccess);
}

//=============================================================================
// hipHostGetFlags Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostGetFlags basic", "[memory][host]") {
    void* ptr = nullptr;
    unsigned int flags = 0;
    
    REQUIRE(hip().hipHostMalloc(&ptr, 1024, hipHostMallocDefault) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipHostGetFlags(&flags, ptr) == hipSuccess);
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostGetFlags with mapped flag", "[memory][host]") {
    void* ptr = nullptr;
    unsigned int flags = 0;
    
    REQUIRE(hip().hipHostMalloc(&ptr, 1024, hipHostMallocMapped) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipHostGetFlags(&flags, ptr) == hipSuccess);
    // The mapped flag should be set
    REQUIRE((flags & hipHostMallocMapped) != 0);
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostGetFlags null flagsPtr returns error", "[memory][host][negative]") {
    void* ptr = nullptr;
    REQUIRE(hip().hipHostMalloc(&ptr, 1024, hipHostMallocDefault) == hipSuccess);
    
    REQUIRE(hip().hipHostGetFlags(nullptr, ptr) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostGetFlags null hostPtr returns error", "[memory][host][negative]") {
    unsigned int flags = 0;
    REQUIRE(hip().hipHostGetFlags(&flags, nullptr) == hipErrorInvalidValue);
}

//=============================================================================
// hipHostRegister / hipHostUnregister Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostRegister basic", "[memory][host][register]") {
    constexpr size_t size = 4096;
    std::vector<unsigned char> buffer(size);
    
    REQUIRE(hip().hipHostRegister(buffer.data(), size, hipHostRegisterDefault) == hipSuccess);
    REQUIRE(hip().hipHostUnregister(buffer.data()) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostRegister with mapped flag", "[memory][host][register]") {
    constexpr size_t size = 4096;
    std::vector<unsigned char> buffer(size);
    
    REQUIRE(hip().hipHostRegister(buffer.data(), size, hipHostRegisterMapped) == hipSuccess);
    REQUIRE(hip().hipHostUnregister(buffer.data()) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostRegister with portable flag", "[memory][host][register]") {
    constexpr size_t size = 4096;
    std::vector<unsigned char> buffer(size);
    
    REQUIRE(hip().hipHostRegister(buffer.data(), size, hipHostRegisterPortable) == hipSuccess);
    REQUIRE(hip().hipHostUnregister(buffer.data()) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostRegister null pointer returns error", "[memory][host][register][negative]") {
    REQUIRE(hip().hipHostRegister(nullptr, 1024, hipHostRegisterDefault) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostRegister zero size returns error", "[memory][host][register][negative]") {
    std::vector<unsigned char> buffer(1024);
    REQUIRE(hip().hipHostRegister(buffer.data(), 0, hipHostRegisterDefault) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostUnregister null pointer returns error", "[memory][host][register][negative]") {
    REQUIRE(hip().hipHostUnregister(nullptr) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostUnregister not registered pointer returns error", "[memory][host][register][negative]") {
    std::vector<unsigned char> buffer(1024);
    // Unregistering memory that was never registered should fail
    REQUIRE(hip().hipHostUnregister(buffer.data()) != hipSuccess);
}

//=============================================================================
// Memory Transfer Tests with Pinned Memory
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy with pinned host memory", "[memory][host][memcpy]") {
    constexpr size_t size = 4096;
    
    void* h_src = nullptr;
    void* h_dst = nullptr;
    void* d_ptr = nullptr;
    
    // Allocate pinned host memory
    REQUIRE(hip().hipHostMalloc(&h_src, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipHostMalloc(&h_dst, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    
    // Initialize source
    std::memset(h_src, 0x55, size);
    std::memset(h_dst, 0x00, size);
    
    // Copy H2D
    REQUIRE(hip().hipMemcpy(d_ptr, h_src, size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Copy D2H
    REQUIRE(hip().hipMemcpy(h_dst, d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify
    REQUIRE(std::memcmp(h_src, h_dst, size) == 0);
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    REQUIRE(hip().hipHostFree(h_dst) == hipSuccess);
    REQUIRE(hip().hipHostFree(h_src) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpyAsync with pinned host memory", "[memory][host][memcpy][async]") {
    constexpr size_t size = 4096;
    
    void* h_src = nullptr;
    void* h_dst = nullptr;
    void* d_ptr = nullptr;
    hipStream_t stream = nullptr;
    
    // Allocate pinned host memory
    REQUIRE(hip().hipHostMalloc(&h_src, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipHostMalloc(&h_dst, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Initialize source
    std::memset(h_src, 0xAA, size);
    std::memset(h_dst, 0x00, size);
    
    // Async copy H2D
    REQUIRE(hip().hipMemcpyAsync(d_ptr, h_src, size, hipMemcpyHostToDevice, stream) == hipSuccess);
    
    // Async copy D2H
    REQUIRE(hip().hipMemcpyAsync(h_dst, d_ptr, size, hipMemcpyDeviceToHost, stream) == hipSuccess);
    
    // Synchronize
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify
    REQUIRE(std::memcmp(h_src, h_dst, size) == 0);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    REQUIRE(hip().hipHostFree(h_dst) == hipSuccess);
    REQUIRE(hip().hipHostFree(h_src) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset on mapped host memory", "[memory][host][memset]") {
    constexpr size_t size = 4096;
    
    void* h_ptr = nullptr;
    void* d_ptr = nullptr;
    
    // Allocate with mapped flag
    REQUIRE(hip().hipHostMalloc(&h_ptr, size, hipHostMallocMapped) == hipSuccess);
    REQUIRE(hip().hipHostGetDevicePointer(&d_ptr, h_ptr, 0) == hipSuccess);
    
    // Memset through device pointer
    REQUIRE(hip().hipMemset(d_ptr, 0xCD, size) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify via host pointer
    unsigned char* h_bytes = static_cast<unsigned char*>(h_ptr);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_bytes[i] == 0xCD);
    }
    
    REQUIRE(hip().hipHostFree(h_ptr) == hipSuccess);
}

//=============================================================================
// Multiple Allocation Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc multiple allocations", "[memory][host]") {
    constexpr int numAllocs = 10;
    constexpr size_t allocSize = 4096;
    
    std::vector<void*> ptrs(numAllocs, nullptr);
    
    // Allocate all
    for (int i = 0; i < numAllocs; ++i) {
        REQUIRE(hip().hipHostMalloc(&ptrs[i], allocSize, hipHostMallocDefault) == hipSuccess);
        REQUIRE(ptrs[i] != nullptr);
        
        // Write pattern to verify later
        std::memset(ptrs[i], i, allocSize);
    }
    
    // Verify patterns
    for (int i = 0; i < numAllocs; ++i) {
        unsigned char* bytes = static_cast<unsigned char*>(ptrs[i]);
        for (size_t j = 0; j < allocSize; ++j) {
            REQUIRE(bytes[j] == static_cast<unsigned char>(i));
        }
    }
    
    // Free all
    for (int i = 0; i < numAllocs; ++i) {
        REQUIRE(hip().hipHostFree(ptrs[i]) == hipSuccess);
    }
}

//=============================================================================
// Registered Memory with Memcpy Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy with registered host memory", "[memory][host][register][memcpy]") {
    constexpr size_t size = 4096;
    
    std::vector<unsigned char> h_src(size, 0x77);
    std::vector<unsigned char> h_dst(size, 0x00);
    void* d_ptr = nullptr;
    
    // Register source and destination
    REQUIRE(hip().hipHostRegister(h_src.data(), size, hipHostRegisterDefault) == hipSuccess);
    REQUIRE(hip().hipHostRegister(h_dst.data(), size, hipHostRegisterDefault) == hipSuccess);
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    
    // Copy H2D
    REQUIRE(hip().hipMemcpy(d_ptr, h_src.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Copy D2H
    REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(h_dst[i] == 0x77);
    }
    
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
    REQUIRE(hip().hipHostUnregister(h_dst.data()) == hipSuccess);
    REQUIRE(hip().hipHostUnregister(h_src.data()) == hipSuccess);
}

//=============================================================================
// Negative Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc null pointer returns error", "[memory][host][negative]") {
    REQUIRE(hip().hipHostMalloc(nullptr, 1024, hipHostMallocDefault) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostAlloc null pointer returns error", "[memory][host][negative]") {
    REQUIRE(hip().hipHostAlloc(nullptr, 1024, hipHostMallocDefault) == hipErrorInvalidValue);
}

//=============================================================================
// Stress Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc allocate and free loop", "[memory][host][stress]") {
    constexpr int iterations = 100;
    constexpr size_t allocSize = 4096;
    
    for (int i = 0; i < iterations; ++i) {
        void* ptr = nullptr;
        REQUIRE(hip().hipHostMalloc(&ptr, allocSize, hipHostMallocDefault) == hipSuccess);
        REQUIRE(ptr != nullptr);
        REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipHostRegister register and unregister loop", "[memory][host][register][stress]") {
    constexpr int iterations = 100;
    constexpr size_t allocSize = 4096;
    std::vector<unsigned char> buffer(allocSize);
    
    for (int i = 0; i < iterations; ++i) {
        REQUIRE(hip().hipHostRegister(buffer.data(), allocSize, hipHostRegisterDefault) == hipSuccess);
        REQUIRE(hip().hipHostUnregister(buffer.data()) == hipSuccess);
    }
}
