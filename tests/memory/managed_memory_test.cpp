// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tests for HIP managed (unified) memory APIs
// Migrated from TheRock: rocm-systems/projects/hip-tests/catch/unit/memory/hipMallocManaged.cc,
//                        hipMemAdvise.cc, hipMemPrefetchAsync.cc, hipMemRangeGetAttribute.cc

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

#include <vector>
#include <cstring>
#include <cstdint>
#include <limits>

//=============================================================================
// Helper to check if managed memory is supported
//=============================================================================

static bool isManagedMemorySupported(HipLoader& hip) {
    int managed = 0;
    hipError_t result = hip.hipDeviceGetAttribute(&managed, 
                                                   hipDeviceAttributeManagedMemory, 0);
    return (result == hipSuccess && managed == 1);
}

//=============================================================================
// hipMallocManaged Basic Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMallocManaged basic allocation", "[memory][managed]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    
    REQUIRE(hip().hipMallocManaged(&ptr, 1024, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Verify we can access from host
    std::memset(ptr, 0x42, 1024);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocManaged with default flags", "[memory][managed]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    
    // hipMemAttachGlobal is the default (0x01)
    REQUIRE(hip().hipMallocManaged(&ptr, 4096, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocManaged with hipMemAttachHost", "[memory][managed]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    
    REQUIRE(hip().hipMallocManaged(&ptr, 4096, hipMemAttachHost) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocManaged various sizes", "[memory][managed]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    auto size = GENERATE(
        size_t(1),
        size_t(64),
        size_t(1024),
        size_t(4096),
        size_t(64 * 1024),
        size_t(1024 * 1024)
    );
    
    DYNAMIC_SECTION("Size: " << size << " bytes") {
        void* ptr = nullptr;
        REQUIRE(hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal) == hipSuccess);
        REQUIRE(ptr != nullptr);
        
        // Verify we can access the entire allocation from host
        std::memset(ptr, 0xAB, size);
        
        REQUIRE(hip().hipFree(ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocManaged zero size", "[memory][managed]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    
    // Zero-size allocation behavior may vary
    hipError_t result = hip().hipMallocManaged(&ptr, 0, hipMemAttachGlobal);
    
    if (result == hipSuccess && ptr != nullptr) {
        REQUIRE(hip().hipFree(ptr) == hipSuccess);
    }
    // Otherwise, it's acceptable for the allocation to fail or return nullptr
}

TEST_CASE_METHOD(HipTestFixture, "hipMallocManaged large allocation fails", "[memory][managed]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    
    // Allocating max size_t should fail with out of memory
    hipError_t result = hip().hipMallocManaged(&ptr, std::numeric_limits<size_t>::max(), 
                                                hipMemAttachGlobal);
    REQUIRE(result == hipErrorOutOfMemory);
}

//=============================================================================
// hipMallocManaged with hipFree Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipFree works with managed memory", "[memory][managed]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    
    REQUIRE(hip().hipMallocManaged(&ptr, 1024, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipFree nullptr after managed allocation", "[memory][managed]") {
    // Free nullptr should succeed
    REQUIRE(hip().hipFree(nullptr) == hipSuccess);
}

//=============================================================================
// hipMallocManaged Negative Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMallocManaged null pointer returns error", "[memory][managed][negative]") {
    REQUIRE(hip().hipMallocManaged(nullptr, 1024, hipMemAttachGlobal) == hipErrorInvalidValue);
}

//=============================================================================
// hipMemAdvise Tests (requires managed memory support)
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemAdvise SetReadMostly", "[memory][managed][advise]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Set read-mostly hint
    REQUIRE(hip().hipMemAdvise(ptr, size, hipMemAdviseSetReadMostly, 0) == hipSuccess);
    
    // Unset read-mostly hint
    REQUIRE(hip().hipMemAdvise(ptr, size, hipMemAdviseUnsetReadMostly, 0) == hipSuccess);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemAdvise SetPreferredLocation device", "[memory][managed][advise]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Set preferred location to device 0
    REQUIRE(hip().hipMemAdvise(ptr, size, hipMemAdviseSetPreferredLocation, 0) == hipSuccess);
    
    // Unset preferred location
    REQUIRE(hip().hipMemAdvise(ptr, size, hipMemAdviseUnsetPreferredLocation, 0) == hipSuccess);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemAdvise SetPreferredLocation CPU", "[memory][managed][advise]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Set preferred location to CPU (hipCpuDeviceId = -1)
    REQUIRE(hip().hipMemAdvise(ptr, size, hipMemAdviseSetPreferredLocation, -1) == hipSuccess);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemAdvise SetAccessedBy", "[memory][managed][advise]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Mark as accessed by device 0
    REQUIRE(hip().hipMemAdvise(ptr, size, hipMemAdviseSetAccessedBy, 0) == hipSuccess);
    
    // Unset accessed by
    REQUIRE(hip().hipMemAdvise(ptr, size, hipMemAdviseUnsetAccessedBy, 0) == hipSuccess);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemAdvise null pointer returns error", "[memory][managed][advise][negative]") {
    hipError_t result = hip().hipMemAdvise(nullptr, 1024, hipMemAdviseSetReadMostly, 0);
    REQUIRE(result == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemAdvise zero size returns error", "[memory][managed][advise][negative]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    REQUIRE(hip().hipMallocManaged(&ptr, 4096, hipMemAttachGlobal) == hipSuccess);
    
    hipError_t result = hip().hipMemAdvise(ptr, 0, hipMemAdviseSetReadMostly, 0);
    // Zero size should be an error
    REQUIRE(result == hipErrorInvalidValue);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

//=============================================================================
// hipMemPrefetchAsync Tests (requires managed memory support)
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemPrefetchAsync to device", "[memory][managed][prefetch]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Initialize data on host
    std::memset(ptr, 0x55, size);
    
    // Prefetch to device 0
    REQUIRE(hip().hipMemPrefetchAsync(ptr, size, 0, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemPrefetchAsync to host", "[memory][managed][prefetch]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Prefetch to CPU (hipCpuDeviceId = -1)
    REQUIRE(hip().hipMemPrefetchAsync(ptr, size, -1, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify we can access from host
    std::memset(ptr, 0xAA, size);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemPrefetchAsync with stream", "[memory][managed][prefetch]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Prefetch to device using stream
    REQUIRE(hip().hipMemPrefetchAsync(ptr, size, 0, stream) == hipSuccess);
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemPrefetchAsync null pointer returns error", "[memory][managed][prefetch][negative]") {
    hipError_t result = hip().hipMemPrefetchAsync(nullptr, 1024, 0, nullptr);
    REQUIRE(result == hipErrorInvalidValue);
}

//=============================================================================
// hipMemRangeGetAttribute Tests (requires managed memory support)
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemRangeGetAttribute ReadMostly", "[memory][managed][range]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Set read-mostly hint
    REQUIRE(hip().hipMemAdvise(ptr, size, hipMemAdviseSetReadMostly, 0) == hipSuccess);
    
    // Query read-mostly attribute
    int read_mostly = 0;
    REQUIRE(hip().hipMemRangeGetAttribute(&read_mostly, sizeof(read_mostly),
                                           hipMemRangeAttributeReadMostly, ptr, size) == hipSuccess);
    REQUIRE(read_mostly == 1);
    
    // Unset and verify
    REQUIRE(hip().hipMemAdvise(ptr, size, hipMemAdviseUnsetReadMostly, 0) == hipSuccess);
    REQUIRE(hip().hipMemRangeGetAttribute(&read_mostly, sizeof(read_mostly),
                                           hipMemRangeAttributeReadMostly, ptr, size) == hipSuccess);
    REQUIRE(read_mostly == 0);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemRangeGetAttribute PreferredLocation", "[memory][managed][range]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Set preferred location to device 0
    REQUIRE(hip().hipMemAdvise(ptr, size, hipMemAdviseSetPreferredLocation, 0) == hipSuccess);
    
    // Query preferred location
    int location = -99;
    REQUIRE(hip().hipMemRangeGetAttribute(&location, sizeof(location),
                                           hipMemRangeAttributePreferredLocation, ptr, size) == hipSuccess);
    REQUIRE(location == 0);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemRangeGetAttribute LastPrefetchLocation", "[memory][managed][range]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Prefetch to device 0
    REQUIRE(hip().hipMemPrefetchAsync(ptr, size, 0, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Query last prefetch location
    int location = -99;
    REQUIRE(hip().hipMemRangeGetAttribute(&location, sizeof(location),
                                           hipMemRangeAttributeLastPrefetchLocation, ptr, size) == hipSuccess);
    REQUIRE(location == 0);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemRangeGetAttribute null data returns error", "[memory][managed][range][negative]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    void* ptr = nullptr;
    REQUIRE(hip().hipMallocManaged(&ptr, 4096, hipMemAttachGlobal) == hipSuccess);
    
    hipError_t result = hip().hipMemRangeGetAttribute(nullptr, sizeof(int),
                                                       hipMemRangeAttributeReadMostly, ptr, 4096);
    REQUIRE(result == hipErrorInvalidValue);
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemRangeGetAttribute null pointer returns error", "[memory][managed][range][negative]") {
    int value = 0;
    hipError_t result = hip().hipMemRangeGetAttribute(&value, sizeof(value),
                                                       hipMemRangeAttributeReadMostly, nullptr, 4096);
    REQUIRE(result == hipErrorInvalidValue);
}

//=============================================================================
// Multiple Managed Allocations Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMallocManaged multiple allocations", "[memory][managed]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    constexpr int numAllocs = 10;
    constexpr size_t allocSize = 4096;
    
    std::vector<void*> ptrs(numAllocs, nullptr);
    
    // Allocate all
    for (int i = 0; i < numAllocs; ++i) {
        REQUIRE(hip().hipMallocManaged(&ptrs[i], allocSize, hipMemAttachGlobal) == hipSuccess);
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
        REQUIRE(hip().hipFree(ptrs[i]) == hipSuccess);
    }
}

//=============================================================================
// Memory Operations with Managed Memory
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy with managed memory source", "[memory][managed][memcpy]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    constexpr size_t size = 4096;
    
    void* managed_ptr = nullptr;
    void* device_ptr = nullptr;
    std::vector<unsigned char> host_dst(size, 0);
    
    REQUIRE(hip().hipMallocManaged(&managed_ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(hip().hipMalloc(&device_ptr, size) == hipSuccess);
    
    // Initialize managed memory on host
    std::memset(managed_ptr, 0x55, size);
    
    // Copy managed to device
    REQUIRE(hip().hipMemcpy(device_ptr, managed_ptr, size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Copy device to host buffer
    REQUIRE(hip().hipMemcpy(host_dst.data(), device_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(host_dst[i] == 0x55);
    }
    
    REQUIRE(hip().hipFree(device_ptr) == hipSuccess);
    REQUIRE(hip().hipFree(managed_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy with managed memory destination", "[memory][managed][memcpy]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    constexpr size_t size = 4096;
    
    void* managed_ptr = nullptr;
    void* device_ptr = nullptr;
    std::vector<unsigned char> host_src(size, 0xAA);
    
    REQUIRE(hip().hipMallocManaged(&managed_ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(hip().hipMalloc(&device_ptr, size) == hipSuccess);
    
    // Copy host to device
    REQUIRE(hip().hipMemcpy(device_ptr, host_src.data(), size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Copy device to managed
    REQUIRE(hip().hipMemcpy(managed_ptr, device_ptr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify managed memory on host
    unsigned char* bytes = static_cast<unsigned char*>(managed_ptr);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(bytes[i] == 0xAA);
    }
    
    REQUIRE(hip().hipFree(device_ptr) == hipSuccess);
    REQUIRE(hip().hipFree(managed_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemset on managed memory", "[memory][managed][memset]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    constexpr size_t size = 4096;
    
    void* ptr = nullptr;
    REQUIRE(hip().hipMallocManaged(&ptr, size, hipMemAttachGlobal) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Memset on managed memory
    REQUIRE(hip().hipMemset(ptr, 0xCD, size) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify on host
    unsigned char* bytes = static_cast<unsigned char*>(ptr);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(bytes[i] == 0xCD);
    }
    
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

//=============================================================================
// Stress Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMallocManaged allocate and free loop", "[memory][managed][stress]") {
    if (!isManagedMemorySupported(hip())) {
        SKIP("Managed memory not supported on this device");
    }
    
    constexpr int iterations = 100;
    constexpr size_t allocSize = 4096;
    
    for (int i = 0; i < iterations; ++i) {
        void* ptr = nullptr;
        REQUIRE(hip().hipMallocManaged(&ptr, allocSize, hipMemAttachGlobal) == hipSuccess);
        REQUIRE(ptr != nullptr);
        REQUIRE(hip().hipFree(ptr) == hipSuccess);
    }
}
