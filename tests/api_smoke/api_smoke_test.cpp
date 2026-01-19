// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include "hip_loader.hpp"
#include "hip_test_fixture.hpp"

//=============================================================================
// Device Management Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceCount returns valid count", "[device]") {
    int count = -1;
    REQUIRE(hip().hipGetDeviceCount(&count) == hipSuccess);
    REQUIRE(count >= 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipSetDevice and hipGetDevice", "[device]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    REQUIRE(deviceCount > 0);

    SECTION("Can set and get device 0") {
        REQUIRE(hip().hipSetDevice(0) == hipSuccess);
        
        int currentDevice = -1;
        REQUIRE(hip().hipGetDevice(&currentDevice) == hipSuccess);
        REQUIRE(currentDevice == 0);
    }

    SECTION("Setting invalid device returns error") {
        hipError_t err = hip().hipSetDevice(deviceCount + 100);
        REQUIRE(err == hipErrorInvalidDevice);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceProperties", "[device]") {
    hipDeviceProp_t props;
    REQUIRE(hip().hipGetDeviceProperties(&props, 0) == hipSuccess);
    
    SECTION("Device name is non-empty") {
        REQUIRE(props.name[0] != '\0');
    }

    SECTION("Device has valid memory") {
        REQUIRE(props.totalGlobalMem > 0);
    }

    SECTION("Device has valid compute capabilities") {
        REQUIRE(props.major >= 0);
        REQUIRE(props.minor >= 0);
    }
}

//=============================================================================
// Memory Allocation Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMalloc basic allocation", "[memory][malloc]") {
    void* ptr = nullptr;
    
    SECTION("Small allocation") {
        REQUIRE(hip().hipMalloc(&ptr, 1024) == hipSuccess);
        REQUIRE(ptr != nullptr);
        REQUIRE(hip().hipFree(ptr) == hipSuccess);
    }

    SECTION("Zero-size allocation") {
        // Behavior may vary - some implementations return success with nullptr,
        // others return an error
        hipError_t err = hip().hipMalloc(&ptr, 0);
        if (err == hipSuccess && ptr != nullptr) {
            REQUIRE(hip().hipFree(ptr) == hipSuccess);
        }
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipMalloc various sizes", "[memory][malloc]") {
    auto size = GENERATE(
        1ULL,           // 1 byte
        256ULL,         // 256 bytes  
        1024ULL,        // 1 KB
        65536ULL,       // 64 KB
        1048576ULL,     // 1 MB
        16777216ULL     // 16 MB
    );

    void* ptr = nullptr;
    REQUIRE(hip().hipMalloc(&ptr, size) == hipSuccess);
    REQUIRE(ptr != nullptr);
    REQUIRE(hip().hipFree(ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipFree nullptr is valid", "[memory][free]") {
    // Freeing nullptr should be a no-op and succeed
    REQUIRE(hip().hipFree(nullptr) == hipSuccess);
}

//=============================================================================
// Memory Copy Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemcpy host to device and back", "[memory][memcpy]") {
    constexpr size_t size = 1024;
    std::vector<int> h_src(size, 42);
    std::vector<int> h_dst(size, 0);

    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, size * sizeof(int)) == hipSuccess);
    REQUIRE(d_ptr != nullptr);

    SECTION("Host to Device") {
        REQUIRE(hip().hipMemcpy(d_ptr, h_src.data(), size * sizeof(int), 
                                 hipMemcpyHostToDevice) == hipSuccess);
    }

    SECTION("Device to Host") {
        // First copy data to device
        REQUIRE(hip().hipMemcpy(d_ptr, h_src.data(), size * sizeof(int),
                                 hipMemcpyHostToDevice) == hipSuccess);
        
        // Then copy back
        REQUIRE(hip().hipMemcpy(h_dst.data(), d_ptr, size * sizeof(int),
                                 hipMemcpyDeviceToHost) == hipSuccess);
        
        // Verify data
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_dst[i] == 42);
        }
    }

    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// Memset Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemset", "[memory][memset]") {
    constexpr size_t size = 1024;
    void* d_ptr = nullptr;
    
    REQUIRE(hip().hipMalloc(&d_ptr, size) == hipSuccess);
    
    SECTION("Set to zero") {
        REQUIRE(hip().hipMemset(d_ptr, 0, size) == hipSuccess);
        
        // Verify by copying back
        std::vector<unsigned char> h_data(size);
        REQUIRE(hip().hipMemcpy(h_data.data(), d_ptr, size,
                                 hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_data[i] == 0);
        }
    }

    SECTION("Set to pattern") {
        REQUIRE(hip().hipMemset(d_ptr, 0xAB, size) == hipSuccess);
        
        std::vector<unsigned char> h_data(size);
        REQUIRE(hip().hipMemcpy(h_data.data(), d_ptr, size,
                                 hipMemcpyDeviceToHost) == hipSuccess);
        
        for (size_t i = 0; i < size; ++i) {
            REQUIRE(h_data[i] == 0xAB);
        }
    }

    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

//=============================================================================
// Stream Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipStreamCreate and Destroy", "[stream]") {
    hipStream_t stream = nullptr;
    
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    REQUIRE(stream != nullptr);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipStreamSynchronize", "[stream]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Synchronize should succeed even with no pending operations
    REQUIRE(hip().hipStreamSynchronize(stream) == hipSuccess);
    
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "Default stream (nullptr) operations", "[stream]") {
    // Operations on the default stream should work
    void* d_ptr = nullptr;
    REQUIRE(hip().hipMalloc(&d_ptr, 1024) == hipSuccess);
    REQUIRE(hip().hipMemsetAsync(d_ptr, 0, 1024, nullptr) == hipSuccess);
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    REQUIRE(hip().hipFree(d_ptr) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetStreamDeviceId", "[stream]") {
    // Skip if the function is not available
    if (!hip().hipGetStreamDeviceId) {
        SKIP("hipGetStreamDeviceId not available");
    }

    SECTION("Default stream returns current device") {
        int currentDevice = -1;
        REQUIRE(hip().hipGetDevice(&currentDevice) == hipSuccess);
        
        int streamDeviceId = hip().hipGetStreamDeviceId(nullptr);
        REQUIRE(streamDeviceId >= 0);
        REQUIRE(streamDeviceId == currentDevice);
    }

    SECTION("Explicit stream returns correct device") {
        hipStream_t stream = nullptr;
        REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
        
        int currentDevice = -1;
        REQUIRE(hip().hipGetDevice(&currentDevice) == hipSuccess);
        
        int streamDeviceId = hip().hipGetStreamDeviceId(stream);
        REQUIRE(streamDeviceId >= 0);
        REQUIRE(streamDeviceId == currentDevice);
        
        REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    }
}

//=============================================================================
// Event Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipEventCreate and Destroy", "[event]") {
    hipEvent_t event = nullptr;
    
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    REQUIRE(event != nullptr);
    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipEventRecord and Synchronize", "[event]") {
    hipEvent_t event = nullptr;
    REQUIRE(hip().hipEventCreate(&event) == hipSuccess);
    
    SECTION("Record on default stream") {
        REQUIRE(hip().hipEventRecord(event, nullptr) == hipSuccess);
        REQUIRE(hip().hipEventSynchronize(event) == hipSuccess);
    }

    SECTION("Record on explicit stream") {
        hipStream_t stream = nullptr;
        REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
        
        REQUIRE(hip().hipEventRecord(event, stream) == hipSuccess);
        REQUIRE(hip().hipEventSynchronize(event) == hipSuccess);
        
        REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
    }

    REQUIRE(hip().hipEventDestroy(event) == hipSuccess);
}

//=============================================================================
// Error Handling Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGetErrorString", "[error]") {
    const char* str = hip().hipGetErrorString(hipSuccess);
    REQUIRE(str != nullptr);
    
    str = hip().hipGetErrorString(hipErrorOutOfMemory);
    REQUIRE(str != nullptr);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetErrorName", "[error]") {
    const char* name = hip().hipGetErrorName(hipSuccess);
    REQUIRE(name != nullptr);
    
    name = hip().hipGetErrorName(hipErrorInvalidValue);
    REQUIRE(name != nullptr);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetLastError clears error", "[error]") {
    // First, trigger an error (invalid device)
    int deviceCount = 0;
    hip().hipGetDeviceCount(&deviceCount);
    hip().hipSetDevice(deviceCount + 100); // Should fail
    
    // Get the error
    hipError_t err = hip().hipGetLastError();
    REQUIRE(err == hipErrorInvalidDevice);
    
    // Subsequent call should return success (error cleared)
    err = hip().hipGetLastError();
    REQUIRE(err == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipPeekAtLastError does not clear error", "[error]") {
    // First, trigger an error
    int deviceCount = 0;
    hip().hipGetDeviceCount(&deviceCount);
    hip().hipSetDevice(deviceCount + 100); // Should fail
    
    // Peek at the error
    hipError_t err = hip().hipPeekAtLastError();
    REQUIRE(err == hipErrorInvalidDevice);
    
    // Peek again - should still be there
    err = hip().hipPeekAtLastError();
    REQUIRE(err == hipErrorInvalidDevice);
    
    // Clear it
    hip().hipGetLastError();
}

//=============================================================================
// Device Synchronization Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSynchronize", "[device][sync]") {
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
}

