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
// Device Count Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceCount returns valid count", "[device][count]") {
    int count = -1;
    REQUIRE(hip().hipGetDeviceCount(&count) == hipSuccess);
    REQUIRE(count > 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceCount with null pointer fails", "[device][count][negative]") {
    REQUIRE(hip().hipGetDeviceCount(nullptr) == hipErrorInvalidValue);
}

//=============================================================================
// Set/Get Device Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipSetDevice and hipGetDevice basic", "[device][set][get]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    REQUIRE(deviceCount > 0);
    
    // Set and get device 0
    REQUIRE(hip().hipSetDevice(0) == hipSuccess);
    
    int currentDevice = -1;
    REQUIRE(hip().hipGetDevice(&currentDevice) == hipSuccess);
    REQUIRE(currentDevice == 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipSetDevice invalid device negative", "[device][set][negative]") {
    REQUIRE(hip().hipSetDevice(-1) == hipErrorInvalidDevice);
}

TEST_CASE_METHOD(HipTestFixture, "hipSetDevice invalid device out of range", "[device][set][negative]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    
    REQUIRE(hip().hipSetDevice(deviceCount + 100) == hipErrorInvalidDevice);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetDevice with null pointer fails", "[device][get][negative]") {
    REQUIRE(hip().hipGetDevice(nullptr) == hipErrorInvalidValue);
}

//=============================================================================
// Device Properties Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceProperties basic", "[device][properties]") {
    hipDeviceProp_t props;
    memset(&props, 0, sizeof(props));
    
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
    
    SECTION("Device has valid warp size") {
        REQUIRE(props.warpSize > 0);
    }
    
    SECTION("Device has valid max threads per block") {
        REQUIRE(props.maxThreadsPerBlock > 0);
    }
    
    SECTION("Device has non-negative clock rate") {
        // Some devices may report 0 for clock rate
        REQUIRE(props.clockRate >= 0);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceProperties null pointer fails", "[device][properties][negative]") {
    REQUIRE(hip().hipGetDeviceProperties(nullptr, 0) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipGetDeviceProperties invalid device fails", "[device][properties][negative]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    
    hipDeviceProp_t props;
    REQUIRE(hip().hipGetDeviceProperties(&props, deviceCount + 100) == hipErrorInvalidDevice);
}

//=============================================================================
// Device Synchronize Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSynchronize empty", "[device][sync]") {
    // Synchronize with no work should succeed
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSynchronize with stream work", "[device][sync]") {
    hipStream_t stream = nullptr;
    REQUIRE(hip().hipStreamCreate(&stream) == hipSuccess);
    
    // Allocate memory and do some work
    void* devicePtr = nullptr;
    constexpr size_t size = 4096;
    REQUIRE(hip().hipMalloc(&devicePtr, size) == hipSuccess);
    REQUIRE(hip().hipMemsetAsync(devicePtr, 0xAB, size, stream) == hipSuccess);
    
    // Device synchronize should wait for all work
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify data
    std::vector<uint8_t> hostData(size, 0);
    REQUIRE(hip().hipMemcpy(hostData.data(), devicePtr, size, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < size; ++i) {
        REQUIRE(hostData[i] == 0xAB);
    }
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
    REQUIRE(hip().hipStreamDestroy(stream) == hipSuccess);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceSynchronize multiple streams", "[device][sync]") {
    constexpr int numStreams = 4;
    constexpr size_t size = 4096;
    
    std::vector<hipStream_t> streams(numStreams, nullptr);
    std::vector<void*> devicePtrs(numStreams, nullptr);
    std::vector<std::vector<uint8_t>> hostData(numStreams, std::vector<uint8_t>(size, 0));
    
    // Create streams and allocate memory
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipStreamCreate(&streams[i]) == hipSuccess);
        REQUIRE(hip().hipMalloc(&devicePtrs[i], size) == hipSuccess);
    }
    
    // Launch async operations on all streams
    for (int i = 0; i < numStreams; ++i) {
        uint8_t value = static_cast<uint8_t>(i + 1);
        REQUIRE(hip().hipMemsetAsync(devicePtrs[i], value, size, streams[i]) == hipSuccess);
    }
    
    // Device synchronize should wait for all streams
    REQUIRE(hip().hipDeviceSynchronize() == hipSuccess);
    
    // Verify data from all streams
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipMemcpy(hostData[i].data(), devicePtrs[i], size, 
                                 hipMemcpyDeviceToHost) == hipSuccess);
        
        uint8_t expectedValue = static_cast<uint8_t>(i + 1);
        for (size_t j = 0; j < size; ++j) {
            REQUIRE(hostData[i][j] == expectedValue);
        }
    }
    
    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        REQUIRE(hip().hipFree(devicePtrs[i]) == hipSuccess);
        REQUIRE(hip().hipStreamDestroy(streams[i]) == hipSuccess);
    }
}

//=============================================================================
// Driver Version Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDriverGetVersion returns valid version", "[device][driver]") {
    int driverVersion = -1;
    REQUIRE(hip().hipDriverGetVersion(&driverVersion) == hipSuccess);
    REQUIRE(driverVersion > 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipDriverGetVersion null pointer fails", "[device][driver][negative]") {
    REQUIRE(hip().hipDriverGetVersion(nullptr) == hipErrorInvalidValue);
}

//=============================================================================
// Runtime Version Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipRuntimeGetVersion returns valid version", "[device][runtime]") {
    int runtimeVersion = -1;
    REQUIRE(hip().hipRuntimeGetVersion(&runtimeVersion) == hipSuccess);
    REQUIRE(runtimeVersion > 0);
}

TEST_CASE_METHOD(HipTestFixture, "hipRuntimeGetVersion null pointer fails", "[device][runtime][negative]") {
    REQUIRE(hip().hipRuntimeGetVersion(nullptr) == hipErrorInvalidValue);
}

//=============================================================================
// Device Attribute Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetAttribute basic attributes", "[device][attribute]") {
    int value = -1;
    
    SECTION("Max threads per block") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeMaxThreadsPerBlock, 0) == hipSuccess);
        REQUIRE(value > 0);
    }
    
    SECTION("Max block dim X") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeMaxBlockDimX, 0) == hipSuccess);
        REQUIRE(value > 0);
    }
    
    SECTION("Max grid dim X") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimX, 0) == hipSuccess);
        REQUIRE(value > 0);
    }
    
    SECTION("Warp size") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeWarpSize, 0) == hipSuccess);
        REQUIRE(value > 0);
    }
    
    SECTION("Max shared memory per block") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSharedMemoryPerBlock, 0) == hipSuccess);
        REQUIRE(value >= 0);
    }
    
    SECTION("Total constant memory") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeTotalConstantMemory, 0) == hipSuccess);
        REQUIRE(value >= 0);
    }
    
    SECTION("Clock rate") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeClockRate, 0) == hipSuccess);
        REQUIRE(value > 0);
    }
    
    SECTION("Compute capability major") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeComputeCapabilityMajor, 0) == hipSuccess);
        REQUIRE(value >= 0);
    }
    
    SECTION("Compute capability minor") {
        REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeComputeCapabilityMinor, 0) == hipSuccess);
        REQUIRE(value >= 0);
    }
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetAttribute null pointer fails", "[device][attribute][negative]") {
    REQUIRE(hip().hipDeviceGetAttribute(nullptr, hipDeviceAttributeMaxThreadsPerBlock, 0) == hipErrorInvalidValue);
}

TEST_CASE_METHOD(HipTestFixture, "hipDeviceGetAttribute invalid device fails", "[device][attribute][negative]") {
    int deviceCount = 0;
    REQUIRE(hip().hipGetDeviceCount(&deviceCount) == hipSuccess);
    
    int value = -1;
    REQUIRE(hip().hipDeviceGetAttribute(&value, hipDeviceAttributeMaxThreadsPerBlock, deviceCount + 100) == hipErrorInvalidDevice);
}

//=============================================================================
// Memory Info Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipMemGetInfo returns valid values", "[device][memory][info]") {
    size_t freeMem = 0;
    size_t totalMem = 0;
    
    REQUIRE(hip().hipMemGetInfo(&freeMem, &totalMem) == hipSuccess);
    REQUIRE(totalMem > 0);
    REQUIRE(freeMem <= totalMem);
}

TEST_CASE_METHOD(HipTestFixture, "hipMemGetInfo after allocation", "[device][memory][info]") {
    size_t freeMemBefore = 0;
    size_t totalMem = 0;
    REQUIRE(hip().hipMemGetInfo(&freeMemBefore, &totalMem) == hipSuccess);
    
    // Allocate memory
    void* devicePtr = nullptr;
    constexpr size_t allocSize = 16 * 1024 * 1024; // 16 MB
    REQUIRE(hip().hipMalloc(&devicePtr, allocSize) == hipSuccess);
    
    size_t freeMemAfter = 0;
    REQUIRE(hip().hipMemGetInfo(&freeMemAfter, &totalMem) == hipSuccess);
    
    // Free memory should be less after allocation
    REQUIRE(freeMemAfter < freeMemBefore);
    
    REQUIRE(hip().hipFree(devicePtr) == hipSuccess);
}

//=============================================================================
// Init Tests
//=============================================================================

TEST_CASE_METHOD(HipTestFixture, "hipInit with flags 0", "[device][init]") {
    // hipInit(0) should succeed
    REQUIRE(hip().hipInit(0) == hipSuccess);
}
