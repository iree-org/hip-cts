// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for HIP pointer attribute API

#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <cstring>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// hipMemGetAddressRange Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemGetAddressRange for device memory", "[memory][pointer]") {
    size_t allocSize = 1024;
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, allocSize) == hipSuccess);

    hipDeviceptr_t base = nullptr;
    size_t size = 0;
    REQUIRE(hip().hipMemGetAddressRange(&base, &size, ptr) == hipSuccess);

    REQUIRE(base == ptr);
    REQUIRE(size >= allocSize);

    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemGetAddressRange with offset pointer", "[memory][pointer]") {
    size_t allocSize = 4096;
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, allocSize) == hipSuccess);

    // Get address range from an offset pointer
    void* offsetPtr = static_cast<char*>(ptr) + 512;

    hipDeviceptr_t base = nullptr;
    size_t size = 0;
    REQUIRE(hip().hipMemGetAddressRange(&base, &size, offsetPtr) == hipSuccess);

    REQUIRE(base == ptr);  // Base should be the original allocation
    REQUIRE(size >= allocSize);

    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemGetAddressRange null base behavior", "[memory][pointer]") {
    // Note: Some HIP implementations crash with null base pointer.
    // Skip this test if native HIP - it's primarily for streaming backend verification.
    if (!is_streaming_backend()) {
        SKIP("Skipping null base test on native HIP - behavior varies by implementation");
    }
    
    size_t allocSize = 1024;
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, allocSize) == hipSuccess);

    size_t size = 0;
    hipError_t err = hip().hipMemGetAddressRange(nullptr, &size, ptr);
    // Accept either success (and size populated) or error
    if (err == hipSuccess) {
        REQUIRE(size >= allocSize);
    }

    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemGetAddressRange null size behavior", "[memory][pointer]") {
    // Note: Some HIP implementations crash with null size pointer.
    // Skip this test if native HIP - it's primarily for streaming backend verification.
    if (!is_streaming_backend()) {
        SKIP("Skipping null size test on native HIP - behavior varies by implementation");
    }
    
    size_t allocSize = 1024;
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, allocSize) == hipSuccess);

    hipDeviceptr_t base = nullptr;
    // Some implementations require non-null size
    hipError_t err = hip().hipMemGetAddressRange(&base, nullptr, ptr);
    REQUIRE((err == hipSuccess || err == hipErrorInvalidValue));
    if (err == hipSuccess) {
        REQUIRE(base == ptr);
    }

    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemGetAddressRange invalid pointer fails", "[memory][pointer][error]") {
    hipDeviceptr_t base = nullptr;
    size_t size = 0;

    // Use an invalid pointer
    void* invalidPtr = reinterpret_cast<void*>(0xDEADBEEF);
    hipError_t err = hip().hipMemGetAddressRange(&base, &size, invalidPtr);
    REQUIRE(err != hipSuccess);
}

//=============================================================================
// hipPointerGetAttributes Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipPointerGetAttributes for device memory", "[memory][pointer][attribute]") {
    if (!hip().hipPointerGetAttributes) {
        SKIP("hipPointerGetAttributes not available");
    }
    
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, 1024) == hipSuccess);

    hipPointerAttribute_t attr;
    memset(&attr, 0, sizeof(attr));
    REQUIRE(hip().hipPointerGetAttributes(&attr, ptr) == hipSuccess);

    REQUIRE(attr.type == hipMemoryTypeDevice);
    REQUIRE(attr.devicePointer == ptr);

    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipPointerGetAttributes for host memory", "[memory][pointer][attribute]") {
    if (!hip().hipPointerGetAttributes) {
        SKIP("hipPointerGetAttributes not available");
    }
    
    void* ptr = nullptr;
    REQUIRE(hip().hipHostMalloc(&ptr, 1024, hipHostMallocDefault) == hipSuccess);

    hipPointerAttribute_t attr;
    memset(&attr, 0, sizeof(attr));
    hipError_t err = hip().hipPointerGetAttributes(&attr, ptr);
    // Some implementations might not support this for host memory
    if (err == hipSuccess) {
        REQUIRE(attr.type == hipMemoryTypeHost);
    }

    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipPointerGetAttributes device ordinal", "[memory][pointer][attribute]") {
    if (!hip().hipPointerGetAttributes) {
        SKIP("hipPointerGetAttributes not available");
    }
    
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, 1024) == hipSuccess);

    hipPointerAttribute_t attr;
    memset(&attr, 0, sizeof(attr));
    REQUIRE(hip().hipPointerGetAttributes(&attr, ptr) == hipSuccess);

    int currentDevice = -1;
    REQUIRE(hip().hipGetDevice(&currentDevice) == hipSuccess);
    REQUIRE(attr.device == currentDevice);

    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipPointerGetAttributes null attr fails", "[memory][pointer][attribute][error]") {
    if (!hip().hipPointerGetAttributes) {
        SKIP("hipPointerGetAttributes not available");
    }
    
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, 1024) == hipSuccess);

    hipError_t err = hip().hipPointerGetAttributes(nullptr, ptr);
    REQUIRE(err == hipErrorInvalidValue);

    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipPointerGetAttributes invalid pointer behavior", "[memory][pointer][attribute]") {
    if (!hip().hipPointerGetAttributes) {
        SKIP("hipPointerGetAttributes not available");
    }
    
    hipPointerAttribute_t attr;
    memset(&attr, 0, sizeof(attr));
    void* invalidPtr = reinterpret_cast<void*>(0xDEADBEEF);
    hipError_t err = hip().hipPointerGetAttributes(&attr, invalidPtr);
    // Note: Behavior varies by implementation.
    // Some return hipErrorInvalidValue, others return hipSuccess with type=hipMemoryTypeUnregistered.
    // Just verify it doesn't crash and check the result is reasonable.
    if (err == hipSuccess) {
        // If success, the memoryType should indicate unregistered or similar
        // (can't strictly verify, just don't crash)
    } else {
        // Error is also acceptable
        REQUIRE((err == hipErrorInvalidValue || err == hipErrorNotFound || err == hipErrorInvalidDevice));
    }
}

//=============================================================================
// hipHostGetDevicePointer Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostGetDevicePointer for mapped memory", "[memory][pointer][host]") {
    void* hostPtr = nullptr;
    REQUIRE(hip().hipHostMalloc(&hostPtr, 1024, hipHostMallocMapped) == hipSuccess);

    void* devicePtr = nullptr;
    REQUIRE(hip().hipHostGetDevicePointer(&devicePtr, hostPtr, 0) == hipSuccess);
    REQUIRE(devicePtr != nullptr);

    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostGetDevicePointer null devicePtr fails", "[memory][pointer][host][error]") {
    void* hostPtr = nullptr;
    REQUIRE(hip().hipHostMalloc(&hostPtr, 1024, hipHostMallocMapped) == hipSuccess);

    hipError_t err = hip().hipHostGetDevicePointer(nullptr, hostPtr, 0);
    REQUIRE(err == hipErrorInvalidValue);

    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostGetDevicePointer for non-mapped memory", "[memory][pointer][host]") {
    void* hostPtr = nullptr;
    REQUIRE(hip().hipHostMalloc(&hostPtr, 1024, hipHostMallocDefault) == hipSuccess);

    void* devicePtr = nullptr;
    // This may or may not succeed depending on whether the memory was implicitly mapped
    hipError_t err = hip().hipHostGetDevicePointer(&devicePtr, hostPtr, 0);
    // Just check it doesn't crash
    (void)err;
    (void)devicePtr;

    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
}

//=============================================================================
// hipHostGetFlags Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostGetFlags for host memory", "[memory][pointer][host]") {
    void* hostPtr = nullptr;
    unsigned int allocFlags = hipHostMallocMapped | hipHostMallocPortable;
    REQUIRE(hip().hipHostMalloc(&hostPtr, 1024, allocFlags) == hipSuccess);

    unsigned int flags = 0;
    REQUIRE(hip().hipHostGetFlags(&flags, hostPtr) == hipSuccess);
    // At minimum, the flags we set should be present
    REQUIRE((flags & hipHostMallocMapped) != 0);

    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostGetFlags null flagsPtr fails", "[memory][pointer][host][error]") {
    void* hostPtr = nullptr;
    REQUIRE(hip().hipHostMalloc(&hostPtr, 1024, hipHostMallocDefault) == hipSuccess);

    hipError_t err = hip().hipHostGetFlags(nullptr, hostPtr);
    REQUIRE(err == hipErrorInvalidValue);

    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostGetFlags invalid pointer fails", "[memory][pointer][host][error]") {
    unsigned int flags = 0;
    void* invalidPtr = reinterpret_cast<void*>(0xDEADBEEF);
    hipError_t err = hip().hipHostGetFlags(&flags, invalidPtr);
    REQUIRE(err != hipSuccess);
}

//=============================================================================
// hipMemGetInfo Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemGetInfo basic", "[memory][info]") {
    size_t free = 0;
    size_t total = 0;

    REQUIRE(hip().hipMemGetInfo(&free, &total) == hipSuccess);

    INFO("Free memory: " << free << " bytes");
    INFO("Total memory: " << total << " bytes");

    REQUIRE(total > 0);
    REQUIRE(free <= total);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemGetInfo after allocation", "[memory][info]") {
    size_t freeBefore = 0;
    size_t total = 0;

    REQUIRE(hip().hipMemGetInfo(&freeBefore, &total) == hipSuccess);

    // Allocate some memory
    size_t allocSize = 64 * 1024 * 1024;  // 64 MB
    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, allocSize) == hipSuccess);

    size_t freeAfter = 0;
    REQUIRE(hip().hipMemGetInfo(&freeAfter, &total) == hipSuccess);

    // Free memory should have decreased
    // Note: The exact amount might vary due to allocation overhead
    REQUIRE(freeAfter < freeBefore);

    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemGetInfo null free is OK", "[memory][info]") {
    size_t total = 0;
    REQUIRE(hip().hipMemGetInfo(nullptr, &total) == hipSuccess);
    REQUIRE(total > 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemGetInfo null total is OK", "[memory][info]") {
    size_t free = 0;
    REQUIRE(hip().hipMemGetInfo(&free, nullptr) == hipSuccess);
}
