// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <vector>
#include <cstring>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// hipHostMalloc Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc basic allocation", "[host_memory][malloc]") {
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipHostMalloc(&ptr, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Verify memory is accessible
    memset(ptr, 0xAB, size);
    uint8_t* bytePtr = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(bytePtr[i] == 0xAB);
    }
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc null pointer fails", "[host_memory][malloc][negative]") {
    REQUIRE(hip().hipHostMalloc(nullptr, 4096, hipHostMallocDefault) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc zero size", "[host_memory][malloc]") {
    void* ptr = nullptr;
    // Zero size allocation - behavior may vary
    hipError_t result = hip().hipHostMalloc(&ptr, 0, hipHostMallocDefault);
    if (result == hipSuccess && ptr != nullptr) {
        REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc with different flags", "[host_memory][malloc][flags]") {
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    SECTION("hipHostMallocDefault") {
        REQUIRE(hip().hipHostMalloc(&ptr, size, hipHostMallocDefault) == hipSuccess);
        REQUIRE(ptr != nullptr);
        REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
    }
    
    SECTION("hipHostMallocPortable") {
        REQUIRE(hip().hipHostMalloc(&ptr, size, hipHostMallocPortable) == hipSuccess);
        REQUIRE(ptr != nullptr);
        REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
    }
    
    SECTION("hipHostMallocMapped") {
        REQUIRE(hip().hipHostMalloc(&ptr, size, hipHostMallocMapped) == hipSuccess);
        REQUIRE(ptr != nullptr);
        REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
    }
    
    SECTION("hipHostMallocWriteCombined") {
        REQUIRE(hip().hipHostMalloc(&ptr, size, hipHostMallocWriteCombined) == hipSuccess);
        REQUIRE(ptr != nullptr);
        REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipHostMalloc multiple allocations", "[host_memory][malloc]") {
    constexpr int numAllocations = 10;
    constexpr size_t size = 4096;
    std::vector<void*> ptrs(numAllocations, nullptr);
    
    // Allocate multiple buffers
    for (int i = 0; i < numAllocations; ++i) {
        REQUIRE(hip().hipHostMalloc(&ptrs[i], size, hipHostMallocDefault) == hipSuccess);
        REQUIRE(ptrs[i] != nullptr);
    }
    
    // Verify all are unique
    for (int i = 0; i < numAllocations; ++i) {
        for (int j = i + 1; j < numAllocations; ++j) {
            REQUIRE(ptrs[i] != ptrs[j]);
        }
    }
    
    // Free all
    for (int i = 0; i < numAllocations; ++i) {
        REQUIRE(hip().hipHostFree(ptrs[i]) == hipSuccess);
    }
}

//=============================================================================
// hipHostFree Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostFree valid pointer", "[host_memory][free]") {
    void* ptr = nullptr;
    REQUIRE(hip().hipHostMalloc(&ptr, 4096, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostFree null pointer", "[host_memory][free]") {
    // Freeing null pointer - behavior may vary by implementation
    hipError_t result = hip().hipHostFree(nullptr);
    // Some implementations accept null, some return error
    REQUIRE((result == hipSuccess || result == hipErrorInvalidValue));
}

//=============================================================================
// hipHostGetDevicePointer Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostGetDevicePointer for mapped memory", "[host_memory][device_pointer]") {
    void* hostPtr = nullptr;
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    
    // Allocate mapped host memory
    REQUIRE(hip().hipHostMalloc(&hostPtr, size, hipHostMallocMapped) == hipSuccess);
    REQUIRE(hostPtr != nullptr);
    
    // Get device pointer
    REQUIRE(hip().hipHostGetDevicePointer(&devicePtr, hostPtr, 0) == hipSuccess);
    REQUIRE(devicePtr != nullptr);
    
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostGetDevicePointer null output fails", "[host_memory][device_pointer][negative]") {
    void* hostPtr = nullptr;
    REQUIRE(hip().hipHostMalloc(&hostPtr, 4096, hipHostMallocMapped) == hipSuccess);
    
    REQUIRE(hip().hipHostGetDevicePointer(nullptr, hostPtr, 0) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
}

//=============================================================================
// hipHostGetFlags Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostGetFlags returns correct flags", "[host_memory][flags]") {
    void* ptr = nullptr;
    unsigned int flags = 0;
    
    SECTION("Default flags") {
        REQUIRE(hip().hipHostMalloc(&ptr, 4096, hipHostMallocDefault) == hipSuccess);
        REQUIRE(hip().hipHostGetFlags(&flags, ptr) == hipSuccess);
        REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
    }
    
    SECTION("Mapped flags") {
        REQUIRE(hip().hipHostMalloc(&ptr, 4096, hipHostMallocMapped) == hipSuccess);
        REQUIRE(hip().hipHostGetFlags(&flags, ptr) == hipSuccess);
        // Should have mapped flag set
        REQUIRE((flags & hipHostMallocMapped) != 0);
        REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipHostGetFlags null pointer fails", "[host_memory][flags][negative]") {
    void* ptr = nullptr;
    REQUIRE(hip().hipHostMalloc(&ptr, 4096, hipHostMallocDefault) == hipSuccess);
    
    REQUIRE(hip().hipHostGetFlags(nullptr, ptr) == hipErrorInvalidValue);
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

//=============================================================================
// Host Memory with Device Operations Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "Host memory to device memcpy", "[host_memory][memcpy]") {
    void* hostPtr = nullptr;
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> result(size, 0);
    
    // Allocate host and device memory
    REQUIRE(hip().hipHostMalloc(&hostPtr, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Initialize host memory
    memset(hostPtr, 0xCD, size);
    
    // Copy to device
    REQUIRE(hip().hipMemcpy(devicePtr, hostPtr, size, hipMemcpyHostToDevice) == hipSuccess);
    
    // Copy back to result
    REQUIRE(hip().hipMemcpy(result.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    
    // Verify
    uint8_t* hostBytes = static_cast<uint8_t*>(hostPtr);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(result[i] == 0xCD);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Host memory async memcpy with stream", "[host_memory][memcpy][async]") {
    void* hostPtr = nullptr;
    void* devicePtr = nullptr;
    hipStream_t stream = nullptr;
    constexpr size_t size = 4096;
    std::vector<uint8_t> result(size, 0);
    
    // Create stream
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Allocate host (pinned) and device memory
    REQUIRE(hip().hipHostMalloc(&hostPtr, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    
    // Initialize host memory
    memset(hostPtr, 0xEF, size);
    
    // Async copy to device
    REQUIRE(hip().hipMemcpyAsync(devicePtr, hostPtr, size, hipMemcpyHostToDevice, stream) == hipSuccess);
    
    // Async copy back
    REQUIRE(hip().hipMemcpyAsync(result.data(), devicePtr, size, hipMemcpyDeviceToHost, stream) == hipSuccess);
    
    // Synchronize
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    // Verify
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(result[i] == 0xEF);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipHostFree(hostPtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

//=============================================================================
// hipHostRegister Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostRegister and hipHostUnregister", "[host_memory][register]") {
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    // Register host memory
    REQUIRE(hip().hipHostRegister(hostData.data(), size, hipHostRegisterDefault) == hipSuccess);
    
    // Unregister
    REQUIRE(hip().hipHostUnregister(hostData.data()) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostRegister null pointer fails", "[host_memory][register][negative]") {
    REQUIRE(hip().hipHostRegister(nullptr, 4096, hipHostRegisterDefault) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostUnregister null pointer fails", "[host_memory][register][negative]") {
    hipError_t result = hip().hipHostUnregister(nullptr);
    // Behavior may vary - some implementations return error, some succeed
    REQUIRE((result == hipSuccess || result == hipErrorInvalidValue || result == hipErrorHostMemoryNotRegistered));
}

TEST_CASE_METHOD(HipTestFixture, "hipHostRegister with mapped flag", "[host_memory][register]") {
    constexpr size_t size = 4096;
    std::vector<uint8_t> hostData(size, 0);
    
    // Register as mapped
    REQUIRE(hip().hipHostRegister(hostData.data(), size, hipHostRegisterMapped) == hipSuccess);
    
    // Get device pointer
    void* devicePtr = nullptr;
    REQUIRE(hip().hipHostGetDevicePointer(&devicePtr, hostData.data(), 0) == hipSuccess);
    REQUIRE(devicePtr != nullptr);
    
    // Unregister
    REQUIRE(hip().hipHostUnregister(hostData.data()) == hipSuccess);
}

// Note: hipPointerGetAttributes tests are skipped as the streaming binding
// implements a different signature for this function.

//=============================================================================
// hipHostAlloc Tests (deprecated but still supported)
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipHostAlloc basic allocation", "[host_memory][alloc]") {
    void* ptr = nullptr;
    constexpr size_t size = 4096;
    
    REQUIRE(hip().hipHostAlloc(&ptr, size, hipHostMallocDefault) == hipSuccess);
    REQUIRE(ptr != nullptr);
    
    // Verify memory is accessible
    memset(ptr, 0x12, size);
    uint8_t* bytePtr = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(bytePtr[i] == 0x12);
    }
    
    REQUIRE(hip().hipHostFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipHostAlloc null pointer fails", "[host_memory][alloc][negative]") {
    REQUIRE(hip().hipHostAlloc(nullptr, 4096, hipHostMallocDefault) == hipErrorInvalidValue);
}
